from enum import Enum, auto
from json import loads, dumps
from pathlib import Path
from sys import exit, stderr
from time import time
from typing import List, Optional

import click

from ghedesigner import VERSION
from ghedesigner.borehole import GHEBorehole
from ghedesigner.constants import DEG_TO_RAD
from ghedesigner.design import AnyBisectionType, DesignBase, DesignNearSquare, DesignRectangle, DesignBiRectangle
from ghedesigner.design import DesignBiZoned, DesignBiRectangleConstrained, DesignRowWise
from ghedesigner.enums import BHPipeType, DesignMethodTimeStep
from ghedesigner.geometry import GeometricConstraints, GeometricConstraintsRectangle, GeometricConstraintsNearSquare
from ghedesigner.geometry import GeometricConstraintsBiRectangle, GeometricConstraintsBiZoned
from ghedesigner.geometry import GeometricConstraintsBiRectangleConstrained, GeometricConstraintsRowWise
from ghedesigner.media import GHEFluid, Grout, Pipe, Soil
from ghedesigner.output import OutputManager
from ghedesigner.simulation import SimulationParameters
from ghedesigner.validate import validate_input_file


class GHEManager:
    class DesignGeomType(Enum):
        NearSquare = auto()
        Rectangle = auto()
        BiRectangle = auto()
        BiZonedRectangle = auto()
        BiRectangleConstrained = auto()
        RowWise = auto()

    def __init__(self):
        self._fluid: Optional[GHEFluid] = None
        self._grout: Optional[Grout] = None
        self._soil: Optional[Soil] = None
        self._pipe: Optional[Pipe] = None
        self._u_tube_type: BHPipeType = None
        self._borehole: Optional[GHEBorehole] = None
        self._simulation_parameters: Optional[SimulationParameters] = None
        self._ground_loads: Optional[List[float]] = None
        # OK so geometric_constraints is tricky.  We have base classes, yay!
        # Unfortunately, the functionality between the child classes is not actually
        # collapsed into a base class function ... yet.  So there will be complaints
        # about types temporarily.  It's going in the right direction though.
        self._geometric_constraints: Optional[GeometricConstraints] = None
        self._design: Optional[DesignBase] = None
        self._search: Optional[AnyBisectionType] = None
        self.results: Optional[OutputManager] = None

        # some things for results
        self._search_time: int = 0
        self.summary_results: dict = {}

    def get_design_geometry_type(self, design_geometry_str: str):
        design_geometry_str = str(design_geometry_str).upper()
        if design_geometry_str == "RECTANGLE":
            return self.DesignGeomType.Rectangle
        if design_geometry_str == "NEARSQUARE":
            return self.DesignGeomType.NearSquare
        if design_geometry_str == "BIRECTANGLE":
            return self.DesignGeomType.BiRectangle
        if design_geometry_str == "BIZONEDRECTANGLE":
            return self.DesignGeomType.BiZonedRectangle
        if design_geometry_str == "BIRECTANGLECONSTRAINED":
            return self.DesignGeomType.BiRectangleConstrained
        if design_geometry_str == "ROWWISE":
            return self.DesignGeomType.RowWise
        raise ValueError("Geometry constraint method not supported.")

    def get_bh_pipe_type(self, bh_pipe_str: str):
        bh_pipe_str = str(bh_pipe_str).upper()
        if bh_pipe_str == "SINGLEUTUBE":
            return BHPipeType.SingleUType
        if bh_pipe_str == "DOUBLEUTUBE":
            return BHPipeType.DoubleUType
        if bh_pipe_str == "COAXIAL":
            return BHPipeType.CoaxialType
        raise ValueError("Borehole pipe type not supported.")

    def set_fluid(self, fluid_name: str = "Water", concentration_percent: float = 0.0):
        """
        fluid_name - convert to an enum
        concentration_percent %
        """
        self._fluid = GHEFluid(fluid_str=fluid_name, percent=concentration_percent)

    def set_grout(self, conductivity: float, rho_cp: float):
        """
        conductivity W/mK
        rho_cp J/K-m3
        """
        self._grout = Grout(conductivity, rho_cp)

    def set_soil(self, conductivity: float, rho_cp: float, undisturbed_temp: float):
        """
        conductivity W/mK
        rho_cp J/K-m3
        undisturbed_temp Celsius
        """
        self._soil = Soil(conductivity, rho_cp, undisturbed_temp)

    def set_single_u_tube_pipe(self, inner_radius: float, outer_radius: float, shank_spacing: float,
                               roughness: float, conductivity: float, rho_cp: float):
        """
        inner_radius m
        outer_radius m
        shank_spacing m
        roughness m
        conductivity W/mK
        rho_cp J/K-m3
        """

        # TODO: Convert scalar properties if double or coax
        self._u_tube_type = BHPipeType.SingleUType
        pipe_positions = Pipe.place_pipes(shank_spacing, outer_radius, 1)
        self._pipe = Pipe(pipe_positions, inner_radius, outer_radius, shank_spacing, roughness, conductivity, rho_cp)

    def set_double_u_tube_pipe(self, inner_radius: float, outer_radius: float, roughness: float, shank_spacing: float,
                               conductivity: float, rho_cp: float):

        # TODO: Convert scalar properties if double or coax
        self._u_tube_type = BHPipeType.DoubleUType
        pipe_positions = Pipe.place_pipes(shank_spacing, outer_radius, 2)
        self._pipe = Pipe(pipe_positions, inner_radius, outer_radius, shank_spacing, roughness, conductivity, rho_cp)

    def set_coaxial_pipe(self, inner_pipe_r_in: float, inner_pipe_r_out: float, outer_pipe_r_in: float,
                         outer_pipe_r_out: float, roughness: float, conductivity_inner: float,
                         conductivity_outer: float,
                         rho_cp: float):

        # TODO: Convert scalar properties if double or coax
        self._u_tube_type = BHPipeType.CoaxialType
        # Note: This convention is different from pygfunction
        r_inner = [inner_pipe_r_in, inner_pipe_r_out]  # The radii of the inner pipe from in to out
        r_outer = [outer_pipe_r_in, outer_pipe_r_out]  # The radii of the outer pipe from in to out
        k_p = [conductivity_inner, conductivity_outer]
        self._pipe = Pipe((0, 0), r_inner, r_outer, 0, roughness, k_p, rho_cp)

    def set_borehole(self, height: float, buried_depth: float, radius: float):
        """
        length m (borehole ?length?)
        buried_depth m (burial depth?)
        radius m radius of borehole itself
        """
        self._borehole = GHEBorehole(height, buried_depth, radius, x=0.0, y=0.0)

    def set_simulation_parameters(
            self, num_months: int, max_eft: float, min_eft: float, max_height: float, min_height: float
    ):
        """
        num_months for now assuming we start at month 1
        max_eft, min_eft Celsius on limits of loop temperature
        max_height, min_height m heat exchanger limits
        """
        # TODO: Should max height be limited by the GHEBorehole length?
        self._simulation_parameters = SimulationParameters(
            1,
            num_months,
            max_eft,
            min_eft,
            max_height,
            min_height,
        )

    def set_ground_loads_from_hourly_list(self, hourly_ground_loads: List[float]):
        """
        hourly_ground_loads list of hourly load float values
        """
        # TODO: Add API methods for different load inputs
        # TODO: Define load direction positive/negative
        self._ground_loads = hourly_ground_loads

    def set_geometry_constraints_near_square(self, b: float, length: float):
        self._geometric_constraints = GeometricConstraintsNearSquare(b, length)

    def set_geometry_constraints_rectangle(self, length: float, width: float, b_min: float, b_max: float):
        self._geometric_constraints = GeometricConstraintsRectangle(width, length, b_min, b_max)

    def set_geometry_constraints_bi_rectangle(self, length: float, width: float, b_min: float,
                                              b_max_x: float, b_max_y: float):
        self._geometric_constraints = GeometricConstraintsBiRectangle(width, length, b_min, b_max_x, b_max_y)

    def set_geometry_constraints_bi_zoned_rectangle(self, length: float, width: float, b_min: float,
                                                    b_max_x: float, b_max_y: float):
        self._geometric_constraints = GeometricConstraintsBiZoned(width, length, b_min, b_max_x, b_max_y)

    def set_geometry_constraints_bi_rectangle_constrained(self, b_min: float, b_max_x: float, b_max_y: float,
                                                          property_boundary: list, no_go_boundaries: list):
        self._geometric_constraints = GeometricConstraintsBiRectangleConstrained(b_min, b_max_x, b_max_y,
                                                                                 property_boundary, no_go_boundaries)

    def set_geometry_constraints_rowwise(self, perimeter_spacing_ratio: float,
                                         spacing_start: float, spacing_stop: float, spacing_step: float,
                                         rotate_start: float, rotate_stop: float, rotate_step: float,
                                         property_boundary: list, no_go_boundaries: list):

        # convert from degrees to radians
        rotate_start = rotate_start * DEG_TO_RAD
        rotate_stop = rotate_stop * DEG_TO_RAD

        self._geometric_constraints = GeometricConstraintsRowWise(perimeter_spacing_ratio,
                                                                  spacing_start, spacing_stop, spacing_step,
                                                                  rotate_start, rotate_stop, rotate_step,
                                                                  property_boundary, no_go_boundaries)

    def set_design(self, flow_rate: float, flow_type: str, design_method_geo: DesignGeomType):
        """
        system_flow_rate L/s total system flow rate
        flow_type string, for now either "system" or "borehole"
        """

        if design_method_geo == self.DesignGeomType.NearSquare:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignNearSquare(
                flow_rate,
                self._borehole,
                self._u_tube_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=DesignMethodTimeStep.Hybrid,
            )
        elif design_method_geo == self.DesignGeomType.Rectangle:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignRectangle(
                flow_rate,
                self._borehole,
                self._u_tube_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=DesignMethodTimeStep.Hybrid,
            )
        elif design_method_geo == self.DesignGeomType.BiRectangle:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignBiRectangle(
                flow_rate,
                self._borehole,
                self._u_tube_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=DesignMethodTimeStep.Hybrid,
            )
        elif design_method_geo == self.DesignGeomType.BiZonedRectangle:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignBiZoned(
                flow_rate,
                self._borehole,
                self._u_tube_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=DesignMethodTimeStep.Hybrid,
            )
        elif design_method_geo == self.DesignGeomType.BiRectangleConstrained:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignBiRectangleConstrained(
                flow_rate,
                self._borehole,
                self._u_tube_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=DesignMethodTimeStep.Hybrid,
            )
        elif design_method_geo == self.DesignGeomType.RowWise:
            # temporary disable of the type checker because of the _geometric_constraints member
            # noinspection PyTypeChecker
            self._design = DesignRowWise(
                flow_rate,
                self._borehole,
                self._u_tube_type,
                self._fluid,
                self._pipe,
                self._grout,
                self._soil,
                self._simulation_parameters,
                self._geometric_constraints,
                self._ground_loads,
                flow_type=flow_type,
                method=DesignMethodTimeStep.Hybrid,
            )
        else:
            raise NotImplementedError("This design method has not been implemented")

    def find_design(self):
        if any([x is None for x in [
            self._fluid,
            self._grout,
            self._soil,
            self._pipe,
            self._borehole,
            self._simulation_parameters,
            self._ground_loads,
            self._geometric_constraints,
            self._design,
        ]]):
            raise Exception("didn't set something")
        start_time = time()
        self._search = self._design.find_design()
        self._search.ghe.compute_g_functions()
        self._search_time = time() - start_time
        # TODO: Don't hard-wire Hybrid here
        self._search.ghe.size(method=DesignMethodTimeStep.Hybrid)

    def prepare_results(self, project_name: str, note: str, author: str, iteration_name: str) -> None:
        self.results = OutputManager(
            self._search,
            self._search_time,
            project_name,
            note,
            author,
            iteration_name,
            load_method=DesignMethodTimeStep.Hybrid,
        )

    def write_output_files(self, output_directory: Path, output_file_suffix: str = ""):
        self.results.write_all_output_files(output_directory=output_directory, file_suffix=output_file_suffix)

    def write_input_file(self, output_file_path: Path):

        # TODO: geometric constraints are currently held in two places
        #       SimulationParameters and GeometricConstraints
        #       these should be consolidated
        d_geo = self._geometric_constraints.to_input()
        d_geo['max_height'] = self._simulation_parameters.max_height
        d_geo['min_height'] = self._simulation_parameters.min_height

        # TODO: data held in different places. consolodate
        d_des = self._design.to_input()
        d_des['max_eft'] = self._simulation_parameters.max_EFT_allowable
        d_des['min_eft'] = self._simulation_parameters.min_EFT_allowable

        # pipe data
        d_pipe = {'rho_cp': self._pipe.rhoCp, 'roughness': self._pipe.roughness}

        if self._u_tube_type in [BHPipeType.SingleUType, BHPipeType.DoubleUType]:
            d_pipe['inner_radius'] = self._pipe.r_in
            d_pipe['outer_radius'] = self._pipe.r_out
            d_pipe['shank_spacing'] = self._pipe.s
            d_pipe['conductivity'] = self._pipe.k
        elif self._u_tube_type == BHPipeType.CoaxialType:
            d_pipe['inner_pipe_r_in'] = self._pipe.r_in[0]
            d_pipe['inner_pipe_r_out'] = self._pipe.r_in[1]
            d_pipe['outer_pipe_r_in'] = self._pipe.r_out[0]
            d_pipe['outer_pipe_r_out'] = self._pipe.r_out[1]
            d_pipe['conductivity_inner'] = self._pipe.k[0]
            d_pipe['conductivity_outer'] = self._pipe.k[1]
        else:
            raise TypeError('Invalid pipe type')

        if self._u_tube_type == BHPipeType.SingleUType:
            d_pipe['arrangement'] = 'singleutube'
        elif self._u_tube_type == BHPipeType.DoubleUType:
            d_pipe['arrangement'] = 'doubleutube'
        elif self._u_tube_type == BHPipeType.CoaxialType:
            d_pipe['arrangement'] = 'coaxial'
        else:
            raise TypeError('Invalid pipe type')

        d = {
            'version': VERSION,
            'fluid': self._fluid.to_input(),
            'grout': self._grout.to_input(),
            'soil': self._soil.to_input(),
            'pipe': d_pipe,
            'borehole': self._borehole.to_input(),
            'simulation': self._simulation_parameters.to_input(),
            'geometric_constraints': d_geo,
            'design': d_des,
            'ground_loads': self._ground_loads
        }

        with open(output_file_path, 'w') as f:
            f.write(dumps(d, sort_keys=True, indent=2, separators=(',', ': ')))


def run_manager_from_cli_worker(input_file_path: Path, output_directory: Path):
    # TODO: need better input and runtime error handling

    if not input_file_path.exists():
        print(f"No input file found at {input_file_path}, aborting")
        exit(1)

    # validate inputs against schema before doing anything
    validate_input_file(input_file_path)

    inputs = loads(input_file_path.read_text())

    ghe = GHEManager()

    version = inputs['version']

    if version != VERSION:
        print("Mismatched version, could be a problem", file=stderr)

    fluid_props = inputs['fluid']  # type: dict
    grout_props = inputs['grout']  # type: dict
    soil_props = inputs['soil']  # type: dict
    pipe_props = inputs['pipe']  # type: dict
    borehole_props = inputs['borehole']  # type: dict
    sim_props = inputs['simulation']  # type: dict
    constraint_props = inputs['geometric_constraints']  # type: dict
    design_props = inputs['design']  # type: dict
    ground_load_props = inputs['ground_loads']    # type: list

    ghe.set_fluid(**fluid_props)
    ghe.set_grout(**grout_props)
    ghe.set_soil(**soil_props)

    pipe_type = ghe.get_bh_pipe_type(pipe_props["arrangement"])
    if pipe_type == BHPipeType.SingleUType:
        ghe.set_single_u_tube_pipe(
            inner_radius=pipe_props["inner_radius"],
            outer_radius=pipe_props["outer_radius"],
            shank_spacing=pipe_props["shank_spacing"],
            roughness=pipe_props["roughness"],
            conductivity=pipe_props["conductivity"],
            rho_cp=pipe_props["rho_cp"]
        )
    elif pipe_type == BHPipeType.DoubleUType:
        ghe.set_double_u_tube_pipe(
            inner_radius=pipe_props["inner_radius"],
            outer_radius=pipe_props["outer_radius"],
            shank_spacing=pipe_props["shank_spacing"],
            roughness=pipe_props["roughness"],
            conductivity=pipe_props["conductivity"],
            rho_cp=pipe_props["rho_cp"]
        )
    elif pipe_type == BHPipeType.CoaxialType:
        ghe.set_coaxial_pipe(
            inner_pipe_r_in=pipe_props["inner_pipe_r_in"],
            inner_pipe_r_out=pipe_props["inner_pipe_r_out"],
            outer_pipe_r_in=pipe_props["outer_pipe_r_in"],
            outer_pipe_r_out=pipe_props["outer_pipe_r_out"],
            roughness=pipe_props["roughness"],
            conductivity_inner=pipe_props["conductivity_inner"],
            conductivity_outer=pipe_props["conductivity_outer"],
            rho_cp=pipe_props["rho_cp"]
        )

    ghe.set_borehole(
        height=constraint_props["max_height"],
        buried_depth=borehole_props["buried_depth"],
        radius=borehole_props["radius"]
    )

    ghe.set_ground_loads_from_hourly_list(ground_load_props)
    ghe.set_simulation_parameters(
        num_months=sim_props["num_months"],
        max_eft=design_props["max_eft"],
        min_eft=design_props["min_eft"],
        max_height=constraint_props["max_height"],
        min_height=constraint_props["min_height"]
    )

    geom_type = ghe.get_design_geometry_type(constraint_props["method"])
    if geom_type == ghe.DesignGeomType.Rectangle:
        ghe.set_geometry_constraints_rectangle(
            length=constraint_props["length"],
            width=constraint_props["width"],
            b_min=constraint_props["b_min"],
            b_max=constraint_props["b_max"],
        )
    elif geom_type == ghe.DesignGeomType.NearSquare:
        ghe.set_geometry_constraints_near_square(
            b=constraint_props["b"],
            length=constraint_props["length"]
        )
    elif geom_type == ghe.DesignGeomType.BiRectangle:
        ghe.set_geometry_constraints_bi_rectangle(
            length=constraint_props["length"],
            width=constraint_props["width"],
            b_min=constraint_props["b_min"],
            b_max_x=constraint_props["b_max_x"],
            b_max_y=constraint_props["b_max_y"]
        )
    elif geom_type == ghe.DesignGeomType.BiZonedRectangle:
        ghe.set_geometry_constraints_bi_zoned_rectangle(
            length=constraint_props["length"],
            width=constraint_props["width"],
            b_min=constraint_props["b_min"],
            b_max_x=constraint_props["b_max_x"],
            b_max_y=constraint_props["b_max_y"]
        )
    elif geom_type == ghe.DesignGeomType.BiRectangleConstrained:
        ghe.set_geometry_constraints_bi_rectangle_constrained(
            b_min=constraint_props["b_min"],
            b_max_x=constraint_props["b_max_x"],
            b_max_y=constraint_props["b_max_y"],
            property_boundary=constraint_props["property_boundary"],
            no_go_boundaries=constraint_props["no_go_boundaries"]
        )
    elif geom_type == ghe.DesignGeomType.RowWise:

        # if present, we are using perimeter calculations
        if "perimeter_spacing_ratio" in constraint_props.keys():
            perimeter_spacing_ratio = constraint_props["perimeter_spacing_ratio"]
        else:
            perimeter_spacing_ratio = None

        ghe.set_geometry_constraints_rowwise(
            perimeter_spacing_ratio=perimeter_spacing_ratio,
            spacing_start=constraint_props["spacing_start"],
            spacing_stop=constraint_props["spacing_stop"],
            spacing_step=constraint_props["spacing_step"],
            rotate_start=constraint_props["rotate_start"],
            rotate_stop=constraint_props["rotate_stop"],
            rotate_step=constraint_props["rotate_step"],
            property_boundary=constraint_props["property_boundary"],
            no_go_boundaries=constraint_props["no_go_boundaries"]
        )
    else:
        raise ValueError("Geometry constraint method not supported.")

    ghe.set_design(
        flow_rate=design_props["flow_rate"],
        flow_type=design_props["flow_type"],
        design_method_geo=geom_type
    )

    ghe.find_design()
    ghe.prepare_results("GHEDesigner Run from CLI", "Notes", "Author", "Iteration Name")
    ghe.write_output_files(output_directory)


@click.command(name="GHEDesignerCommandLine")
@click.argument("input-path", type=click.Path(exists=True))
@click.argument("output-directory", type=click.Path(exists=True))
def run_manager_from_cli(input_path, output_directory):
    input_path = Path(input_path).resolve()
    output_path = Path(output_directory).resolve()

    if not input_path.exists():
        print(f'Input file does not exist. Input file path: "{str(input_path)}"')

    run_manager_from_cli_worker(input_path, output_path)
