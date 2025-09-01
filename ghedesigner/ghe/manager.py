from collections.abc import Sequence
from time import time
from typing import cast

from numpy import array, exp, ndarray
from pygfunction.boreholes import Borehole

from ghedesigner.constants import DEG_TO_RAD, MONTHS_IN_YEAR
from ghedesigner.enums import DesignGeomType, FlowConfigType, PipeType, TimestepType
from ghedesigner.ghe.boreholes.single_u_borehole import SingleUTube
from ghedesigner.ghe.coordinates import rectangle
from ghedesigner.ghe.design.base import DesignBase
from ghedesigner.ghe.design.birectangle import DesignBiRectangle, GeometricConstraintsBiRectangle
from ghedesigner.ghe.design.birectangle_constrained import (
    DesignBiRectangleConstrained,
    GeometricConstraintsBiRectangleConstrained,
)
from ghedesigner.ghe.design.bizoned import DesignBiZoned, GeometricConstraintsBiZoned
from ghedesigner.ghe.design.near_square import DesignNearSquare, GeometricConstraintsNearSquare
from ghedesigner.ghe.design.rectangle import DesignRectangle, GeometricConstraintsRectangle
from ghedesigner.ghe.design.rowwise import DesignRowWise, GeometricConstraintsRowWise
from ghedesigner.ghe.gfunction import calculate_g_function
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.utilities import combine_sts_lts, eskilson_log_times, get_loads


class GroundHeatExchanger:  # TODO: Rename this.  Just GHEDesignerManager?  GHEDesigner?
    def __init__(
        self,
        grout_conductivity: float,
        grout_rho_cp: float,
        soil_conductivity: float,
        soil_rho_cp: float,
        soil_undisturbed_temperature: float,
        borehole_buried_depth: float,
        borehole_radius: float,
        pipe_arrangement_type: PipeType,
        pipe_parameters: dict,
        fluid_name: str = "Water",
        fluid_concentration_percent: float = 0.0,
        fluid_temperature: float = 20.0,
    ) -> None:
        self.fluid = Fluid(fluid_name, fluid_temperature, fluid_concentration_percent)
        self.grout = Grout(grout_conductivity, grout_rho_cp)
        self.soil = Soil(soil_conductivity, soil_rho_cp, soil_undisturbed_temperature)
        if pipe_arrangement_type == PipeType.SINGLEUTUBE:
            params = ["conductivity", "rho_cp", "inner_diameter", "outer_diameter", "shank_spacing", "roughness"]
            if not all(x in pipe_parameters for x in params):
                raise ValueError(f"pipe_arrangement_type of {pipe_arrangement_type!s} requires these inputs: {params}")
            pipe_parameters["num_pipes"] = 1
            self.pipe = Pipe.init_single_u_tube(**pipe_parameters)
        elif pipe_arrangement_type == PipeType.DOUBLEUTUBESERIES:
            params = ["conductivity", "rho_cp", "inner_diameter", "outer_diameter", "shank_spacing", "roughness"]
            if not all(x in pipe_parameters for x in params):
                raise ValueError(f"pipe_arrangement_type of {pipe_arrangement_type!s} requires these inputs: {params}")
            self.pipe = Pipe.init_double_u_tube_series(**pipe_parameters)
        elif pipe_arrangement_type == PipeType.DOUBLEUTUBEPARALLEL:
            params = ["conductivity", "rho_cp", "inner_diameter", "outer_diameter", "shank_spacing", "roughness"]
            if not all(x in pipe_parameters for x in params):
                raise ValueError(f"pipe_arrangement_type of {pipe_arrangement_type!s} requires these inputs: {params}")
            self.pipe = Pipe.init_double_u_tube_parallel(**pipe_parameters)
        else:  # Assuming coaxial
            params = [
                "conductivity_inner",
                "rho_cp",
                "conductivity_outer",
                "inner_pipe_d_in",
                "inner_pipe_d_out",
                "outer_pipe_d_in",
                "outer_pipe_d_out",
            ]
            if not all(x in pipe_parameters for x in params):
                raise ValueError(f"pipe_arrangement_type of {PipeType.COAXIAL!s} requires these inputs: {params}")
            pipe_parameters["conductivity"] = (
                pipe_parameters["conductivity_inner"],
                pipe_parameters["conductivity_outer"],
            )
            del pipe_parameters["conductivity_inner"]
            del pipe_parameters["conductivity_outer"]
            self.pipe = Pipe.init_coaxial(**pipe_parameters)
        self.pygfunction_borehole = Borehole(100, borehole_buried_depth, borehole_radius, x=0.0, y=0.0)

    @classmethod
    def init_from_dictionary(cls, ghe_dict: dict, fluid_inputs: dict | None = None) -> "GroundHeatExchanger":
        """
        Initialize a GroundHeatExchanger object from input dictionaries, performing validation and ultimately calling
        the main object constructor.
        :param ghe_dict: Dictionary of ground heat exchanger parameters, see the input schema specification for required
                         inputs in the ground_heat_exchanger schema field.
        :param fluid_inputs: Optional dictionary of fluid input parameters, see the input schema fluid spec for details.
        :return: GroundHeatExchanger object.
        # TODO: Add validation back in to the input fields
        """
        grout_parameters: dict = ghe_dict["grout"]
        g_c: float = grout_parameters["conductivity"]
        g_rho_cp: float = grout_parameters["rho_cp"]

        soil_parameters: dict = ghe_dict["soil"]
        s_k: float = soil_parameters["conductivity"]
        s_rho_cp: float = soil_parameters["rho_cp"]
        s_temp: float = soil_parameters["undisturbed_temp"]

        borehole_parameters: dict = ghe_dict["borehole"]
        buried_depth: float = borehole_parameters["buried_depth"]
        diameter: float = borehole_parameters["diameter"]
        radius: float = diameter / 2.0

        fluid_dict = (
            fluid_inputs if fluid_inputs else {"fluid_name": "Water", "concentration_percent": 0.0, "temperature": 20.0}
        )
        fluid_name = fluid_dict.get("fluid_name", "Water")
        concentration_percent = fluid_dict.get("concentration_percent", 0.0)
        temperature = fluid_dict.get("temperature", 20.0)

        pipe_parameters: dict = ghe_dict["pipe"]
        pipe_type: PipeType = PipeType(pipe_parameters["arrangement"].upper())
        del pipe_parameters["arrangement"]

        ghe: GroundHeatExchanger = cls(
            g_c,
            g_rho_cp,
            s_k,
            s_rho_cp,
            s_temp,
            buried_depth,
            radius,
            pipe_type,
            pipe_parameters,
            fluid_name,
            concentration_percent,
            temperature,
        )
        return ghe

    def design_and_size_ghe(self, ghe_dict: dict, end_month: int, loads_override: list[float] | None = None):
        ghe_loads = loads_override if loads_override else get_loads(ghe_dict["loads"])

        if (end_month % MONTHS_IN_YEAR) > 0:
            raise ValueError(f"end_month must be a multiple of {MONTHS_IN_YEAR}")

        flow_type_str: str = ghe_dict["flow_type"]
        flow_type = FlowConfigType(flow_type_str.upper())
        flow_rate: float = ghe_dict["flow_rate"]

        # grab some design conditions
        design_parameters = ghe_dict["design"]
        continue_if_design_unmet: bool = design_parameters.get("continue_if_design_unmet", False)
        min_eft: float = design_parameters["min_eft"]
        max_eft: float = design_parameters["max_eft"]
        max_height: float = design_parameters["max_height"]
        min_height: float = design_parameters["min_height"]
        max_boreholes: int | None = design_parameters.get("max_boreholes")
        # check_arg_bounds(min_eft, max_eft, "min_eft", "max_eft")

        # set up the geometry constraints section
        geom = ghe_dict["geometric_constraints"]
        geometry_map = {geom.name: geom for geom in DesignGeomType}
        geom_type = geometry_map.get(geom["method"].upper())
        design: DesignBase

        match geom_type:
            case DesignGeomType.RECTANGLE:
                # max_height: float, min_height: float, length: float, width: float, b_min: float, b_max: float
                rect_geometry: GeometricConstraintsRectangle = GeometricConstraintsRectangle(
                    length=geom["length"],
                    width=geom["width"],
                    b_min=geom["b_min"],
                    b_max=geom["b_max"],
                )
                design = DesignRectangle(
                    flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    max_eft,
                    min_eft,
                    max_height,
                    min_height,
                    continue_if_design_unmet,
                    max_boreholes,
                    rect_geometry,
                    ghe_loads,
                    flow_type=flow_type,
                    method=TimestepType.HYBRID,
                )
            case DesignGeomType.NEARSQUARE:
                near_sq_geometry: GeometricConstraintsNearSquare = GeometricConstraintsNearSquare(
                    b=geom["b"],
                    length=geom["length"],
                )
                design = DesignNearSquare(
                    flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    max_eft,
                    min_eft,
                    max_height,
                    min_height,
                    continue_if_design_unmet,
                    max_boreholes,
                    near_sq_geometry,
                    ghe_loads,
                    flow_type=flow_type,
                    method=TimestepType.HYBRID,
                )
            case DesignGeomType.BIRECTANGLE:
                bi_rect_geometry: GeometricConstraintsBiRectangle = GeometricConstraintsBiRectangle(
                    length=geom["length"],
                    width=geom["width"],
                    b_min=geom["b_min"],
                    b_max_x=geom["b_max_x"],
                    b_max_y=geom["b_max_y"],
                )
                design = DesignBiRectangle(
                    flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    max_eft,
                    min_eft,
                    max_height,
                    min_height,
                    continue_if_design_unmet,
                    max_boreholes,
                    bi_rect_geometry,
                    ghe_loads,
                    flow_type=flow_type,
                    method=TimestepType.HYBRID,
                )
            case DesignGeomType.BIZONEDRECTANGLE:
                bi_zoned_geometry: GeometricConstraintsBiZoned = GeometricConstraintsBiZoned(
                    length=geom["length"],
                    width=geom["width"],
                    b_min=geom["b_min"],
                    b_max_x=geom["b_max_x"],
                    b_max_y=geom["b_max_y"],
                )
                design = DesignBiZoned(
                    flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    max_eft,
                    min_eft,
                    max_height,
                    min_height,
                    continue_if_design_unmet,
                    max_boreholes,
                    bi_zoned_geometry,
                    ghe_loads,
                    flow_type=flow_type,
                    method=TimestepType.HYBRID,
                )
            case DesignGeomType.BIRECTANGLECONSTRAINED:
                no_go_boundaries = geom.get("no_go_boundaries", None)
                bi_rect_const_geometry: GeometricConstraintsBiRectangleConstrained = (
                    GeometricConstraintsBiRectangleConstrained(
                        b_min=geom["b_min"],
                        b_max_x=geom["b_max_x"],
                        b_max_y=geom["b_max_y"],
                        property_boundary=geom["property_boundary"],
                        no_go_boundaries=no_go_boundaries,
                    )
                )
                design = DesignBiRectangleConstrained(
                    flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    max_eft,
                    min_eft,
                    max_height,
                    min_height,
                    continue_if_design_unmet,
                    max_boreholes,
                    bi_rect_const_geometry,
                    ghe_loads,
                    flow_type=flow_type,
                    method=TimestepType.HYBRID,
                )
            case DesignGeomType.ROWWISE:
                # use perimeter calculations if present
                perimeter_spacing_ratio = geom.get("perimeter_spacing_ratio", None)
                spacing_step = geom.get("spacing_step", 0)
                no_go_boundaries = geom.get("no_go_boundaries", None)
                geometry_row: GeometricConstraintsRowWise = GeometricConstraintsRowWise(
                    perimeter_spacing_ratio=perimeter_spacing_ratio,
                    max_spacing=geom["max_spacing"],
                    min_spacing=geom["min_spacing"],
                    spacing_step=spacing_step,
                    max_rotation=geom["max_rotation"] * DEG_TO_RAD,
                    min_rotation=geom["min_rotation"] * DEG_TO_RAD,
                    rotate_step=geom["rotate_step"],
                    property_boundary=geom["property_boundary"],
                    no_go_boundaries=no_go_boundaries,
                )
                design = DesignRowWise(
                    flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    max_eft,
                    min_eft,
                    max_height,
                    min_height,
                    continue_if_design_unmet,
                    max_boreholes,
                    geometry_row,
                    ghe_loads,
                    flow_type=flow_type,
                    method=TimestepType.HYBRID,
                )
            case _:
                raise ValueError(f'DesignGeomType "{geom_type}" not supported')

        start_time = time()
        search = design.find_design()  # TODO: I wonder if it would simplify things to just return the GHE object
        search_time = time() - start_time
        found_ghe = cast(GHE, search.ghe)
        found_ghe.compute_g_functions(min_height, max_height)
        found_ghe.size(TimestepType.HYBRID, max_height, min_height, max_eft, min_eft)
        return search, search_time, found_ghe

    def get_g_function(self, ghe_dict: dict, boundary_condition="MIFT") -> tuple[ndarray, ndarray, ndarray]:
        # TODO: Create a SingleUTube class or something in order to get the STS stitched up
        pre_designed = ghe_dict["pre_designed"]
        borehole_height: float = pre_designed["H"]
        if pre_designed["arrangement"] == "MANUAL":
            x_positions: Sequence[float] = pre_designed["x"]
            y_positions: Sequence[float] = pre_designed["y"]
            if len(x_positions) != len(y_positions):
                raise RuntimeError("Borehole location coordinate mismatch, make sure length of x and y are equal")
            locations = list(zip(x_positions, y_positions))
        elif pre_designed["arrangement"] == "RECTANGLE":
            num_bh_x = pre_designed["boreholes_in_x_dimension"]
            num_bh_y = pre_designed["boreholes_in_y_dimension"]
            spacing_x = pre_designed["spacing_in_x_dimension"]
            spacing_y = pre_designed["spacing_in_y_dimension"]
            locations = rectangle(num_bh_x, num_bh_y, spacing_x, spacing_y)
        else:
            raise RuntimeError("Invalid arrangement type for pre_designed borehole field")

        nbh = len(locations)
        flow_rate: float = ghe_dict["flow_rate"]
        flow_type_str: str = str(ghe_dict["flow_type"]).upper()

        if flow_type_str == FlowConfigType.BOREHOLE.name:
            m_flow_borehole = flow_rate * self.fluid.rho / 1000  # conv lps to m3s to kgs
        elif flow_type_str == FlowConfigType.SYSTEM.name:
            m_flow_ghe = flow_rate * self.fluid.rho / 1000  # conv lps to m3s to kgs
            m_flow_borehole = m_flow_ghe / nbh
        else:
            raise NotImplementedError(f"FlowConfigType {flow_type_str} not implemented.")

        self.pygfunction_borehole.H = borehole_height
        ts = borehole_height**2 / (9 * self.soil.alpha)
        log_time_lts = eskilson_log_times()
        time_values = exp(log_time_lts) * ts

        g_lts = calculate_g_function(
            m_flow_borehole,
            self.pipe.type,
            time_values,
            locations,
            self.pygfunction_borehole,
            self.fluid,
            self.pipe,
            self.grout,
            self.soil,
            boundary_condition=boundary_condition,
        )

        single_u_bh = SingleUTube(
            m_flow_borehole, self.fluid, self.pygfunction_borehole, self.pipe, self.grout, self.soil
        )

        log_time_sts, g_sts = single_u_bh.calc_sts_g_functions()
        g_bhw = single_u_bh.g_bhw

        g_interp = combine_sts_lts(
            log_time_lts,
            g_lts.tolist(),
            log_time_sts.tolist(),
            g_sts.tolist(),
        )

        g_bhw_interp = combine_sts_lts(
            log_time_lts,
            g_lts.tolist(),
            log_time_sts.tolist(),
            g_bhw.tolist(),
        )

        log_time_to_write = array(log_time_sts.tolist() + log_time_lts)
        g_to_write = g_interp(log_time_to_write)
        g_bhw_to_write = g_bhw_interp(log_time_to_write)

        return log_time_to_write, g_to_write, g_bhw_to_write

    # def write_input_file(self, output_file_path: Path, simulation_parameters: SimulationParameters) -> None:
    #     """
    #     Writes an input file based on current simulation configuration.
    #
    #     :param output_file_path: output directory to write input file.
    #     :raises AttributeError: If necessary class attributes are not set.
    #     :raises ValueError: If the pipe type is not supported.
    #     """
    #     # TODO: geometric constraints are currently held in two places
    #     #       SimulationParameters and GeometricConstraints
    #     #       these should be consolidated
    #     d_geo = self._geometric_constraints.to_input()
    #     d_geo["max_height"] = simulation_parameters.max_height
    #     d_geo["min_height"] = simulation_parameters.min_height
    #
    #     # TODO: data held in different places
    #     d_des = self._design.to_input()
    #     d_des["max_eft"] = simulation_parameters.max_EFT_allowable
    #     d_des["min_eft"] = simulation_parameters.min_EFT_allowable
    #
    #     if simulation_parameters.max_boreholes is not None:
    #         d_des["max_boreholes"] = simulation_parameters.max_boreholes
    #     if simulation_parameters.continue_if_design_unmet is True:
    #         d_des["continue_if_design_unmet"] = simulation_parameters.continue_if_design_unmet
    #
    #     # pipe data
    #     d_pipe = {"rho_cp": self.pipe.rho_cp, "roughness": self.pipe.roughness}
    #
    #     if self.pipe.type in [BHPipeType.SINGLEUTUBE, BHPipeType.DOUBLEUTUBEPARALLEL, BHPipeType.DOUBLEUTUBESERIES]:
    #         d_pipe["inner_diameter"] = self.pipe.r_in * 2.0
    #         d_pipe["outer_diameter"] = self.pipe.r_out * 2.0
    #         d_pipe["shank_spacing"] = self.pipe.s
    #         d_pipe["conductivity"] = self.pipe.k
    #     elif self.pipe.type == BHPipeType.COAXIAL:
    #         d_pipe["inner_pipe_d_in"] = self.pipe.r_in[0] * 2.0
    #         d_pipe["inner_pipe_d_out"] = self.pipe.r_in[1] * 2.0
    #         d_pipe["outer_pipe_d_in"] = self.pipe.r_out[0] * 2.0
    #         d_pipe["outer_pipe_d_out"] = self.pipe.r_out[1] * 2.0
    #         d_pipe["conductivity_inner"] = self.pipe.k[0]
    #         d_pipe["conductivity_outer"] = self.pipe.k[1]
    #     else:
    #         raise ValueError(f"Invalid pipe type '{self.pipe.type.name if self.pipe.type else 'None'}'")
    #
    #     d_pipe["arrangement"] = self.pipe.type.name
    #
    #     d = {
    #         "fluid": self.fluid.to_input(),
    #         "grout": self.grout.to_input(),
    #         "soil": self.soil.to_input(),
    #         "pipe": d_pipe,
    #         # "borehole": self._borehole.to_input(),
    #         # "simulation": self._simulation_parameters.to_input(),
    #         "geometric_constraints": d_geo,
    #         "design": d_des,
    #         "loads": {"ground_loads": self._ground_loads},
    #     }
    #
    #     output_file_path.parent.mkdir(parents=True, exist_ok=True)
    #     output_file_path.write_text(dumps(d, sort_keys=True, indent=2, separators=(",", ": ")))
