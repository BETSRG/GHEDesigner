from enum import Enum, auto
from json import dumps, loads
from pathlib import Path
from sys import exit, stderr
from typing import List, Optional, Type, Union

import click

from ghedesigner import VERSION
from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import CoaxialPipe, MultipleUTube, SingleUTube
from ghedesigner.design import AnyBisectionType, DesignBase, DesignNearSquare, DesignRectangle
from ghedesigner.geometry import GeometricConstraints
from ghedesigner.media import GHEFluid, Grout, Pipe, SimulationParameters, Soil
from ghedesigner.utilities import DesignMethodTimeStep


class DesignMethodGeometry(Enum):
    NearSquare = auto()
    Rectangular = auto()


class BoreholeType(Enum):
    SingleUTubeType = auto()
    DoubleUTubeType = auto()
    CoaxialType = auto()


class GHEManager:
    """
    TODO: Add docs guiding all the steps
    """

    def __init__(self):
        self._fluid: Optional[GHEFluid] = None
        self._grout: Optional[Grout] = None
        self._soil: Optional[Soil] = None
        self._pipe: Optional[Pipe] = None
        self._u_tube_type: Optional[
            Union[Type[SingleUTube], Type[MultipleUTube], Type[CoaxialPipe]]
        ] = None
        self._borehole: Optional[GHEBorehole] = None
        self._simulation_parameters: Optional[SimulationParameters] = None
        self._ground_loads: Optional[List[float]] = None
        self._geometric_constraints: Optional[GeometricConstraints] = None
        self._design: Optional[DesignBase] = None
        self._search: Optional[AnyBisectionType] = None

        # outputs after design is found
        self.u_tube_height = -1.0

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
        self._u_tube_type = SingleUTube  # for now just store the type on the class here
        pipe_positions = Pipe.place_pipes(shank_spacing, outer_radius, 1)
        self._pipe = Pipe(pipe_positions, inner_radius, outer_radius, shank_spacing, roughness, conductivity, rho_cp)

    def set_double_u_tube_pipe(self, inner_radius: float, outer_radius: float, roughness: float, shank_spacing: float,
                               conductivity: float, rho_cp: float):

        # TODO: Convert scalar properties if double or coax
        self._u_tube_type = MultipleUTube  # for now just store the type on the class here
        pipe_positions = Pipe.place_pipes(shank_spacing, outer_radius, 2)
        self._pipe = Pipe(pipe_positions, inner_radius, outer_radius, shank_spacing, roughness, conductivity, rho_cp)

    def set_coaxial_pipe(self, inner_pipe_r_in: float, inner_pipe_r_out: float, outer_pipe_r_in: float,
                         outer_pipe_r_out: float,
                         roughness: float, conductivity_inner: float, conductivity_outer: float, rho_cp: float):

        # TODO: Convert scalar properties if double or coax
        self._u_tube_type = CoaxialPipe  # for now just store the type on the class here
        # Note: This convention is different from pygfunction
        r_inner = [inner_pipe_r_in, inner_pipe_r_out]  # The radii of the inner pipe from in to out
        r_outer = [outer_pipe_r_in, outer_pipe_r_out]  # The radii of the outer pipe from in to out
        k_p = [conductivity_inner, conductivity_outer]
        self._pipe = Pipe((0, 0), r_inner, r_outer, 0, roughness, k_p, rho_cp)

    def set_borehole(self, length: float, buried_depth: float, radius: float):
        """
        length m (borehole ?length?)
        buried_depth m (burial depth?)
        radius m radius of borehole itself
        """
        self._borehole = GHEBorehole(length, buried_depth, radius, x=0.0, y=0.0)

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

    def set_geometry_constraints(self, **kwargs):
        # TODO: Figure out how the user should know which constraints are needed
        # Probably just need to add a few methods for set_geometry_*
        self._geometric_constraints = GeometricConstraints(**kwargs)

    def set_geometry_constraints_rectangular(self, length: float, width: float, b_min: float, b_max: float):
        self._geometric_constraints = GeometricConstraints(length=length, width=width, b_min=b_min, b_max_x=b_max)

    def set_design(self, flow_rate: float, flow_type: str, design_method_geo: DesignMethodGeometry):
        """
        system_flow_rate L/s total system flow rate
        flow_type string, for now either "system" or "borehole"
        """

        # TODO: Allow setting flow and method dynamically

        if design_method_geo == DesignMethodGeometry.NearSquare:
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
                flow=flow_type,
                method=DesignMethodTimeStep.Hybrid,
            )
        elif design_method_geo == DesignMethodGeometry.Rectangular:
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
                flow=flow_type,
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
        self._search = self._design.find_design()
        self._search.ghe.compute_g_functions()
        # TODO: Don't hard-wire Hybrid here
        self._search.ghe.size(method=DesignMethodTimeStep.Hybrid)
        self.u_tube_height = self._search.ghe.bhe.b.H

    def get_g_function(self):
        lts = self._search.ghe.gFunction.log_time
        # TODO: handle the different keys in the g_vals dict, 60, 97.5, 135
        g_vals = list(self._search.ghe.gFunction.g_lts.values())[0]
        return list(zip(lts, g_vals))

    def get_borehole_locations(self):
        return self._search.ghe.gFunction.bore_locations


def run_manager_from_cli_worker(input_file_path: Path, output_file_path: Path):
    if not input_file_path.exists():
        print(f"No input file found at {input_file_path}, aborting")
        exit(1)
    inputs = loads(input_file_path.read_text())
    manager = GHEManager()
    version = inputs['version']
    if version != VERSION:
        print("Mismatched version, could be a problem", file=stderr)
    fluid_props = inputs['fluid']
    manager.set_fluid(**fluid_props)
    grout_props = inputs['grout']
    manager.set_grout(**grout_props)
    soil_props = inputs['soil']
    manager.set_soil(**soil_props)
    pipe_props = inputs['pipe']
    manager.set_single_u_tube_pipe(**pipe_props)
    borehole_props = inputs['borehole']
    manager.set_borehole(**borehole_props)
    sim_props = inputs['simulation']
    manager.set_simulation_parameters(**sim_props)
    ground_load_props = inputs['ground_loads']  # TODO: Modify this to allow different spec types
    manager.set_ground_loads_from_hourly_list(ground_load_props)
    constraint_props = inputs['geometric_constraints']
    manager.set_geometry_constraints(**constraint_props)
    design_props = inputs['design']
    manager.set_design(**design_props)
    manager.find_design()
    with open(output_file_path, 'w') as f:
        f.write(dumps(
            {
                'design_borehole_height': manager.u_tube_height,
                'g_function': manager.get_g_function(),
                'borehole_locations': manager.get_borehole_locations(),
            },
            indent=2
        ))


@click.command(name="GHEDesignerCommandLine")
@click.argument("input-path", type=click.Path(exists=True))
@click.argument("output-path", type=click.Path(exists=True))
def run_manager_from_cli(input_path, output_path):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.exists():
        print(f'Input file does not exist. Input file path: "{str(input_path)}"')

    run_manager_from_cli_worker(input_path, output_path)
