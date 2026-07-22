from collections.abc import Sequence
from time import time
from typing import cast

from numpy import array, average, clip, exp, ndarray
from pygfunction.boreholes import Borehole

from ghedesigner.constants import DEG_TO_RAD, MONTHS_IN_YEAR
from ghedesigner.enums import BHType, DesignGeomType, FlowConfigType, SimCompType, TimestepType
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
from ghedesigner.ghe.gfunction import calc_g_func_for_multiple_lengths, calculate_g_function
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.ghe.shape import get_area
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.utilities import borehole_spacing, combine_sts_lts, eskilson_log_times, get_loads


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
        pipe_arrangement_type: BHType,
        pipe_parameters: dict,
        fluid_name: str = "Water",
        fluid_concentration_percent: float = 0.0,
        fluid_temperature: float = 20.0,
    ) -> None:
        self.fluid = Fluid(fluid_name, fluid_temperature, fluid_concentration_percent)
        self.grout = Grout(grout_conductivity, grout_rho_cp)
        self.soil = Soil(soil_conductivity, soil_rho_cp, soil_undisturbed_temperature)
        if pipe_arrangement_type == BHType.SINGLEUTUBE:
            params = ["conductivity", "rho_cp", "inner_diameter", "outer_diameter", "shank_spacing", "roughness"]
            if not all(x in pipe_parameters for x in params):
                raise ValueError(f"pipe_arrangement_type of {pipe_arrangement_type!s} requires these inputs: {params}")
            self.pipe = Pipe.init_single_u_tube(**pipe_parameters)
        elif pipe_arrangement_type == BHType.DOUBLEUTUBESERIES:
            params = ["conductivity", "rho_cp", "inner_diameter", "outer_diameter", "shank_spacing", "roughness"]
            if not all(x in pipe_parameters for x in params):
                raise ValueError(f"pipe_arrangement_type of {pipe_arrangement_type!s} requires these inputs: {params}")
            self.pipe = Pipe.init_double_u_tube_series(**pipe_parameters)
        elif pipe_arrangement_type == BHType.DOUBLEUTUBEPARALLEL:
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
                raise ValueError(f"pipe_arrangement_type of {BHType.COAXIAL!s} requires these inputs: {params}")
            pipe_parameters["conductivity"] = (
                pipe_parameters["conductivity_inner"],
                pipe_parameters["conductivity_outer"],
            )
            del pipe_parameters["conductivity_inner"]
            del pipe_parameters["conductivity_outer"]
            self.pipe = Pipe.init_coaxial(**pipe_parameters)
        self.pygfunction_borehole = Borehole(100, borehole_buried_depth, borehole_radius, x=0.0, y=0.0)
        self.bhe_type = pipe_arrangement_type
        self.ghe_geometry_set = False
        self.design_parameters_set = False
        self.flow_parameters_set = False
        self.geometric_constraint: (
            GeometricConstraintsBiRectangle
            | GeometricConstraintsBiRectangleConstrained
            | GeometricConstraintsNearSquare
            | GeometricConstraintsRectangle
            | GeometricConstraintsRowWise
            | GeometricConstraintsBiZoned
            | None
        ) = None
        self.geom_type: DesignGeomType | None
        self.pre_designed_area: float = 0.0
        self.pre_designed_locations: list[tuple[float, float]] | None = None
        self.pre_designed_height: float = 20.0
        self.current_ghe: GHE | None = None
        self.design: DesignBase | None = None
        self.continue_if_design_unmet: bool = True
        self.min_eft: float = 0.0
        self.max_eft: float = 35.0
        self.max_height: float = 150.0
        self.min_height: float = 20.0
        self.max_boreholes: int = 200
        self.flow_type: FlowConfigType = FlowConfigType.BOREHOLE
        self.flow_rate: float = 0.0
        self.is_sizable: bool = False

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
        pipe_type: BHType = BHType(pipe_parameters["arrangement"].upper())
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

    def ghe_setup(self, ghe_dict):
        self.configure_ghe_flow(ghe_dict)
        if "pre_designed" in ghe_dict:
            self.is_sizable = False
            self.configure_geometry(ghe_dict["pre_designed"], is_pre_designed=True)
        else:
            self.is_sizable = True
            self.configure_geometry(ghe_dict["geometric_constraints"], is_pre_designed=False)
            self.configure_design(ghe_dict["design"])

    def configure_geometry(self, geom: dict, is_pre_designed=False):

        if not is_pre_designed:
            geometry_map = {geom.name: geom for geom in DesignGeomType}
            self.geom_type = geometry_map.get(geom["method"].upper())
            match self.geom_type:
                case DesignGeomType.RECTANGLE:
                    # max_height: float, min_height: float, length: float, width: float, b_min: float, b_max: float
                    self.geometric_constraint = GeometricConstraintsRectangle(
                        length=geom["length"],
                        width=geom["width"],
                        b_min=geom["b_min"],
                        b_max=geom["b_max"],
                    )
                case DesignGeomType.NEARSQUARE:
                    self.geometric_constraint = GeometricConstraintsNearSquare(
                        b=geom["b"],
                        length=geom["length"],
                    )
                case DesignGeomType.BIRECTANGLE:
                    self.geometric_constraint = GeometricConstraintsBiRectangle(
                        length=geom["length"],
                        width=geom["width"],
                        b_min=geom["b_min"],
                        b_max_x=geom["b_max_x"],
                        b_max_y=geom["b_max_y"],
                    )
                case DesignGeomType.BIZONEDRECTANGLE:
                    self.geometric_constraint = GeometricConstraintsBiZoned(
                        length=geom["length"],
                        width=geom["width"],
                        b_min=geom["b_min"],
                        b_max_x=geom["b_max_x"],
                        b_max_y=geom["b_max_y"],
                    )
                case DesignGeomType.BIRECTANGLECONSTRAINED:
                    no_go_boundaries = geom.get("no_go_boundaries")
                    removal_method = geom.get("borehole_removal_method", "RADIAL")
                    removal_options = geom.get("borehole_removal_options", {})
                    b_max_x = geom.get("b_max_x")
                    b_max_y = geom.get("b_max_y")
                    self.geometric_constraint = GeometricConstraintsBiRectangleConstrained(
                        b_min=geom["b_min"],
                        property_boundary=geom["property_boundary"],
                        b_max_x=b_max_x,
                        b_max_y=b_max_y,
                        no_go_boundaries=no_go_boundaries,
                        borehole_removal_method=removal_method,
                        borehole_removal_options=removal_options,
                    )
                case DesignGeomType.ROWWISE:
                    # use perimeter calculations if present
                    perimeter_spacing_ratio = geom.get("perimeter_spacing_ratio")
                    spacing_step = geom.get("spacing_step", 0)
                    no_go_boundaries = geom.get("no_go_boundaries")
                    self.geometric_constraint = GeometricConstraintsRowWise(
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
                case _:
                    raise ValueError(f'DesignGeomType "{self.geom_type}" not supported')
        else:
            self.pre_designed_height = geom["H"]
            if geom["arrangement"] == "MANUAL":
                x_positions: Sequence[float] = geom["x"]
                y_positions: Sequence[float] = geom["y"]
                if len(x_positions) != len(y_positions):
                    raise RuntimeError("Borehole location coordinate mismatch, make sure length of x and y are equal")
                self.pre_designed_locations = [(coord[0], coord[1]) for coord in zip(x_positions, y_positions)]
                if "area" in geom:
                    self.pre_designed_area = geom["area"]
            elif geom["arrangement"] == "RECTANGLE":
                num_bh_x = geom["boreholes_in_x_dimension"]
                num_bh_y = geom["boreholes_in_y_dimension"]
                spacing_x = geom["spacing_in_x_dimension"]
                spacing_y = geom["spacing_in_y_dimension"]
                self.pre_designed_locations = rectangle(num_bh_x, num_bh_y, spacing_x, spacing_y)
                self.pre_designed_area = (num_bh_x - 1) * spacing_x * (num_bh_y - 1) * spacing_y
            else:
                raise RuntimeError("Invalid arrangement type for pre_designed borehole field")

        self.ghe_geometry_set = True

    def configure_design(self, design_parameters):
        # grab some design conditions
        self.continue_if_design_unmet = design_parameters.get("continue_if_design_unmet", False)
        self.min_eft = design_parameters["min_eft"]
        self.max_eft = design_parameters["max_eft"]
        self.max_height = design_parameters["max_height"]
        self.min_height = design_parameters["min_height"]
        self.max_boreholes = design_parameters.get("max_boreholes")
        end_month = 2
        ghe_loads = []
        match self.geometric_constraint:
            case GeometricConstraintsRectangle():
                # max_height: float, min_height: float, length: float, width: float, b_min: float, b_max: float
                design = DesignRectangle(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case GeometricConstraintsNearSquare():
                design = DesignNearSquare(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case GeometricConstraintsBiZoned():
                design = DesignBiZoned(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case GeometricConstraintsBiRectangle():
                design = DesignBiRectangle(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case GeometricConstraintsBiRectangleConstrained():
                design = DesignBiRectangleConstrained(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case GeometricConstraintsRowWise():
                design = DesignRowWise(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case _:
                raise ValueError(f'DesignGeomType "{self.geom_type}" not supported')
        self.design = design
        self.design_parameters_set = True

    def configure_ghe_flow(self, ghe_dict: dict):
        flow_type_str = ghe_dict["flow_type"]
        self.flow_type = FlowConfigType(flow_type_str.upper())
        self.flow_rate = ghe_dict["flow_rate"]
        self.flow_parameters_set = True

    def retrieve_flow(self, coordinates, rho):
        if self.flow_type == FlowConfigType.BOREHOLE:
            v_flow_system = self.flow_rate * len(coordinates)
            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = self.flow_rate / 1000.0 * rho
        elif self.flow_type == FlowConfigType.SYSTEM:
            v_flow_system = self.flow_rate
            v_flow_borehole = self.flow_rate / len(coordinates)
            m_flow_borehole = v_flow_borehole / 1000.0 * rho
        else:
            raise ValueError("The flow argument should be either `borehole` or `system`.")
        return v_flow_system, m_flow_borehole

    def new_nbh_design(self, design_nbh):
        design_nbh = clip(design_nbh, *self.design.get_bounds())
        new_coords = self.design.closest_nbh(design_nbh)
        self.pre_designed_locations = new_coords
        self.pre_designed_height = self.max_height
        self.initialize_pre_designed_ghe()

    def new_ts_design(self, target_spacing):
        if self.geom_type != DesignGeomType.ROWWISE:
            raise ValueError('"new_ts_design" can only be used on GHEs which have RowWisegeometric constraints.')
        new_coords = self.design.get_field_by_target_spacing(target_spacing)
        self.pre_designed_locations = new_coords
        self.pre_designed_height = self.max_height
        self.initialize_pre_designed_ghe()

    def average_bound_nbh(self):
        return average(self.design.get_bounds())

    def initialize_pre_designed_ghe(
        self, log_time=eskilson_log_times(), start_month=0, end_month=0, hourly_extraction_ground_loads=[]
    ):
        v_flow_system, m_flow_borehole = self.retrieve_flow(self.pre_designed_locations, self.fluid.rho)
        self.log_time = log_time
        self.pygfunction_borehole.H = self.pre_designed_height
        borehole = self.pygfunction_borehole
        fluid = self.fluid
        pipe = self.pipe
        grout = self.grout
        soil = self.soil

        b = borehole_spacing(borehole, self.pre_designed_locations)

        g_function = calc_g_func_for_multiple_lengths(
            b,
            [borehole.H],
            borehole.r_b,
            borehole.D,
            m_flow_borehole,
            self.bhe_type,
            self.log_time,
            self.pre_designed_locations,
            fluid,
            pipe,
            grout,
            soil,
        )

        # Initialize the GHE object
        self.current_ghe = GHE(
            v_flow_system,
            b,
            self.bhe_type,
            fluid,
            borehole,
            pipe,
            grout,
            soil,
            g_function,
            start_month,
            end_month,
            hourly_extraction_ground_loads,
        )

    def design_and_size_ghe(
        self, end_month: int, loads_override: list[float] | None = None, ghe_dict: dict | None = None
    ):
        ghe_loads: list[float]
        if loads_override is not None:
            ghe_loads = loads_override
        elif ghe_dict is not None:
            ghe_loads = get_loads(ghe_dict["name"], SimCompType.GROUND_HEAT_EXCHANGER.name, ghe_dict["loads"])
        else:
            raise ValueError(
                'Either a load override or a "loads" must exist in the GHE dictionary to design/size a GHE.'
            )

        if (end_month % MONTHS_IN_YEAR) > 0:
            raise ValueError(f"end_month must be a multiple of {MONTHS_IN_YEAR}")

        if not self.flow_parameters_set:
            if ghe_dict is not None:
                self.configure_ghe_flow(ghe_dict)
            else:
                raise ValueError("GHE dictionary is required if flow parameters have not been set.")

        # set up the geometry constraints section
        if not self.ghe_geometry_set:
            if ghe_dict is not None:
                self.configure_geometry(ghe_dict["geometric_constraints"])
            else:
                raise ValueError("GHE dictionary is required if geometry constraints have not been set.")

        # Make sure that necessary design conditions are set
        if not self.design_parameters_set:
            if ghe_dict is not None:
                self.configure_design(ghe_dict["design"])
            else:
                raise ValueError("GHE dictionary is required if design parameters have not been set.")

        design: DesignBase
        match self.geometric_constraint:
            case GeometricConstraintsRectangle():
                # max_height: float, min_height: float, length: float, width: float, b_min: float, b_max: float
                design = DesignRectangle(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case GeometricConstraintsNearSquare():
                design = DesignNearSquare(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case GeometricConstraintsBiZoned():
                design = DesignBiZoned(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case GeometricConstraintsBiRectangle():
                design = DesignBiRectangle(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case GeometricConstraintsBiRectangleConstrained():
                design = DesignBiRectangleConstrained(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case GeometricConstraintsRowWise():
                design = DesignRowWise(
                    self.flow_rate,
                    self.pygfunction_borehole,
                    self.fluid,
                    self.pipe,
                    self.grout,
                    self.soil,
                    1,
                    end_month,
                    self.max_eft,
                    self.min_eft,
                    self.max_height,
                    self.min_height,
                    self.continue_if_design_unmet,
                    self.max_boreholes,
                    self.geometric_constraint,
                    ghe_loads,
                    flow_type=self.flow_type,
                    method=TimestepType.HYBRID,
                )
            case _:
                raise ValueError(f'DesignGeomType "{self.geom_type}" not supported')

        start_time = time()
        search = design.find_design()  # TODO: I wonder if it would simplify things to just return the GHE object
        search_time = time() - start_time
        found_ghe = cast(GHE, search.ghe)
        found_ghe.compute_g_functions(self.min_height, self.max_height)
        found_ghe.size(TimestepType.HYBRID, self.max_height, self.min_height, self.max_eft, self.min_eft)
        self.current_ghe = found_ghe
        self.search = search
        self.design = design
        return search, search_time, found_ghe

    def get_design_area(self) -> float:

        if not self.ghe_geometry_set:
            raise ValueError("A set of geometric constraints needs to be set before a design area can be defined.")

        if self.pre_designed_area:
            return self.pre_designed_area

        match self.geometric_constraint:
            case GeometricConstraintsRectangle():
                # max_height: float, min_height: float, length: float, width: float, b_min: float, b_max: float
                area = self.geometric_constraint.length * self.geometric_constraint.width
            case GeometricConstraintsNearSquare():
                area = self.geometric_constraint.length * self.geometric_constraint.length
            case GeometricConstraintsBiZoned():
                area = self.geometric_constraint.length * self.geometric_constraint.width
            case GeometricConstraintsBiRectangle():
                area = self.geometric_constraint.length * self.geometric_constraint.width
            case GeometricConstraintsBiRectangleConstrained():
                area = 0
                for prop_bound in self.geometric_constraint.property_boundary:
                    area += get_area(prop_bound)  # This presumes that property polygons are non-intersecting
                ng = self.geometric_constraint.no_go_boundaries
                if ng is not None:
                    for ng_zone in ng:
                        area -= get_area(ng_zone)  # This presumes that no-go zone polygons are non-intersecting
            case GeometricConstraintsRowWise():
                area = get_area(self.geometric_constraint.property_boundary)
                if self.geometric_constraint.no_go_boundaries is not None:
                    for ng_zone in self.geometric_constraint.no_go_boundaries:
                        area -= get_area(ng_zone)  # This presumes that no-go zone polygons are non-intersecting
            case _:
                raise ValueError(f'DesignGeomType "{self.geom_type}" not supported')
        return area

    def get_design_volume(self):
        area = self.get_design_area()
        if not self.design_parameters_set and self.pre_designed_height is None:
            raise ValueError("Design parameters must be known before the design volume can be determined.")
        elif self.pre_designed_height is not None:
            return area * self.pre_designed_height
        else:
            return area * self.max_height

    def get_g_function(
        self, ghe_dict: (dict | None) = None, boundary_condition="MIFT"
    ) -> tuple[ndarray, ndarray, ndarray]:
        # TODO: Create a SingleUTube class or something in order to get the STS stitched up
        if ghe_dict is not None:
            pre_designed = ghe_dict["pre_designed"]
            self.configure_geometry(pre_designed, is_pre_designed=True)
            self.configure_ghe_flow(ghe_dict)
        elif not self.ghe_geometry_set:
            raise ValueError(
                "The field geometry either needs to be set before calling this function or provided in a"
                " GHE dictionary provided."
            )
        if self.pre_designed_locations is None:
            raise ValueError(
                'Pre-designed locations are not already defined ghe_dict must be given to call "get_g_function"'
            )
        if not self.flow_parameters_set:
            raise ValueError(
                "Flow parameters either need to be set before calling this function or provided in a GHE dictionary."
            )

        nbh = len(self.pre_designed_locations)
        if self.flow_type == FlowConfigType.BOREHOLE:
            m_flow_borehole = self.flow_rate * self.fluid.rho / 1000  # conv lps to m3s to kgs
        elif self.flow_type == FlowConfigType.SYSTEM:
            m_flow_ghe = self.flow_rate * self.fluid.rho / 1000  # conv lps to m3s to kgs
            m_flow_borehole = m_flow_ghe / nbh
        else:
            raise NotImplementedError(f"FlowConfigType {self.flow_type} not implemented.")

        self.pygfunction_borehole.H = self.pre_designed_height
        ts = self.pre_designed_height**2 / (9 * self.soil.alpha)
        log_time_lts = eskilson_log_times()
        time_values = exp(log_time_lts) * ts

        g_lts = calculate_g_function(
            m_flow_borehole,
            self.pipe.type,
            time_values,
            self.pre_designed_locations,
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

    # def write_input_file(self, output_file_path: Path, simulation_parameters:
    # SimulationParameters) -> None:
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
