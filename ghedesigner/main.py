#!/usr/bin/env python
import logging
import sys
from json import loads
from pathlib import Path

import click
from jsonschema import ValidationError
from pygfunction.gfunction import evaluate_g_function_MIFT
from pygfunction.pipes import PipeTypes

from ghedesigner.constants import VERSION
from ghedesigner.enums import BHPipeType, DesignGeomType
from ghedesigner.ghe.manager import GroundHeatExchanger
from ghedesigner.heat_pump import HeatPump
from ghedesigner.utilities import write_idf
from ghedesigner.validate import validate_input_file

logging.basicConfig(level=logging.WARN, format="%(message)s", datefmt="[%X]")
logger = logging.getLogger(__name__)


def _run_manager_from_cli_worker(input_file_path: Path, output_directory: Path) -> int:
    """
    Worker function to run simulation.

    :param input_file_path: path to input file. Input file must exist.
    :param output_directory: path to write output files. Output directory must be a valid path.
    """

    # validate inputs against schema before doing anything
    if validate_input_file(input_file_path) != 0:
        return 1

    inputs = loads(input_file_path.read_text())

    # Read in all the inputs into small dicts
    # it is possible to define multiple fluids, GHEs, and boreholes in the inputs, I'm just taking the first for now
    input_file_version: int = inputs["version"]
    if input_file_version != 1:
        print("Bad input file version, right now we support these versions: 1")
        return 1

    # Process the load source, it should be a building object or a GHE with loads specified
    ghe_contains_loads = ["loads" in ghe_dict for _, ghe_dict in inputs["ground-heat-exchanger"].items()]
    all_ghe_has_loads = all(ghe_contains_loads)
    no_ghe_has_loads = all(not x for x in ghe_contains_loads)
    building_input = "building" in inputs
    valid_load_source = all_ghe_has_loads ^ (building_input and no_ghe_has_loads)  # XOR because we don't want both
    if not valid_load_source:
        print("Bad load specified, need exactly one of: loads in each ghe, or building object")
        return 1

    # Get a couple required objects from the inputs
    sim_control = inputs["simulation-control"]
    fluid_props = inputs["fluid"]

    # Loop over the topology and init the found objects, for now just the GHE and an optional building
    topology_props: list[dict] = inputs["topology"]
    heat_pump: HeatPump
    ghe = GroundHeatExchanger()
    for component in topology_props:
        comp_type = component["type"]
        comp_name = component["name"]
        if comp_type == "building":
            single_building = inputs["building"][comp_name]
            heat_pump = HeatPump(single_building["name"])
            heat_pump.set_fixed_cop(single_building["cop"])
            loads_file_path = Path(single_building["loads"]).resolve()
            if not loads_file_path.exists():  # TODO: I'll try to find it relative to repo/tests/ for now...
                this_file = Path(__file__).resolve()
                ghe_designer_dir = this_file.parent
                tests_dir = ghe_designer_dir / "tests"
                loads_file_path = tests_dir / single_building["loads"]
            heat_pump.set_loads_from_file(loads_file_path)
            # TODO: Actually calculate the ground load using COP
            ghe_loads_raw = loads_file_path.read_text().strip().split("\n")
            ghe_loads_float = [float(x) for x in ghe_loads_raw]
            ghe.set_ground_loads_from_hourly_list(ghe_loads_float)
        elif comp_type == "ground-heat-exchanger":
            ghe_dict = inputs["ground-heat-exchanger"][comp_name]
            if "loads" in ghe_dict:
                ground_load_props: list = ghe_dict["loads"]
                ghe.set_ground_loads_from_hourly_list(ground_load_props)
            pipe_props: dict = ghe_dict["pipe"]
            ghe.set_fluid(**fluid_props)
            ghe.set_grout(**ghe_dict["grout"])
            ghe.set_soil(**ghe_dict["soil"])

            ghe.set_pipe_type(pipe_props["arrangement"])
            if ghe.pipe_type == BHPipeType.SINGLEUTUBE:
                ghe.set_single_u_tube_pipe(
                    inner_diameter=pipe_props["inner_diameter"],
                    outer_diameter=pipe_props["outer_diameter"],
                    shank_spacing=pipe_props["shank_spacing"],
                    roughness=pipe_props["roughness"],
                    conductivity=pipe_props["conductivity"],
                    rho_cp=pipe_props["rho_cp"],
                )
            elif ghe.pipe_type == BHPipeType.DOUBLEUTUBEPARALLEL:
                ghe.set_double_u_tube_pipe_parallel(
                    inner_diameter=pipe_props["inner_diameter"],
                    outer_diameter=pipe_props["outer_diameter"],
                    shank_spacing=pipe_props["shank_spacing"],
                    roughness=pipe_props["roughness"],
                    conductivity=pipe_props["conductivity"],
                    rho_cp=pipe_props["rho_cp"],
                )
            elif ghe.pipe_type == BHPipeType.DOUBLEUTUBESERIES:
                ghe.set_double_u_tube_pipe_series(
                    inner_diameter=pipe_props["inner_diameter"],
                    outer_diameter=pipe_props["outer_diameter"],
                    shank_spacing=pipe_props["shank_spacing"],
                    roughness=pipe_props["roughness"],
                    conductivity=pipe_props["conductivity"],
                    rho_cp=pipe_props["rho_cp"],
                )
            elif ghe.pipe_type == BHPipeType.COAXIAL:
                ghe.set_coaxial_pipe(
                    inner_pipe_d_in=pipe_props["inner_pipe_d_in"],
                    inner_pipe_d_out=pipe_props["inner_pipe_d_out"],
                    outer_pipe_d_in=pipe_props["outer_pipe_d_in"],
                    outer_pipe_d_out=pipe_props["outer_pipe_d_out"],
                    roughness=pipe_props["roughness"],
                    conductivity_inner=pipe_props["conductivity_inner"],
                    conductivity_outer=pipe_props["conductivity_outer"],
                    rho_cp=pipe_props["rho_cp"],
                )

            borehole_props: dict = ghe_dict["borehole"]
            ghe.set_borehole(
                buried_depth=borehole_props["buried_depth"],
                diameter=borehole_props["diameter"],
            )

            need_to_design = bool(ghe_dict["do-sizing"])
            if need_to_design:
                design_props: dict = ghe_dict["design"]
                max_bh = design_props.get("max_boreholes")
                continue_if_design_unmet = design_props.get("continue_if_design_unmet", False)
                ghe.set_simulation_parameters(
                    num_months=sim_control["simulation-months"],
                    max_boreholes=max_bh,
                    continue_if_design_unmet=continue_if_design_unmet,
                )

                constraint_props: dict = ghe_dict["geometric_constraints"]
                try:
                    ghe.set_design_geometry_type(constraint_props["method"])
                except ValueError:
                    return 1
                if ghe.geom_type == DesignGeomType.RECTANGLE:
                    # max_height: float, min_height: float, length: float, width: float, b_min: float, b_max: float
                    ghe.set_geometry_constraints_rectangle(
                        min_height=constraint_props["min_height"],
                        max_height=constraint_props["max_height"],
                        length=constraint_props["length"],
                        width=constraint_props["width"],
                        b_min=constraint_props["b_min"],
                        b_max=constraint_props["b_max"],
                    )
                elif ghe.geom_type == DesignGeomType.NEARSQUARE:
                    ghe.set_geometry_constraints_near_square(
                        min_height=constraint_props["min_height"],
                        max_height=constraint_props["max_height"],
                        b=constraint_props["b"],
                        length=constraint_props["length"],
                    )
                elif ghe.geom_type == DesignGeomType.BIRECTANGLE:
                    ghe.set_geometry_constraints_bi_rectangle(
                        min_height=constraint_props["min_height"],
                        max_height=constraint_props["max_height"],
                        length=constraint_props["length"],
                        width=constraint_props["width"],
                        b_min=constraint_props["b_min"],
                        b_max_x=constraint_props["b_max_x"],
                        b_max_y=constraint_props["b_max_y"],
                    )
                elif ghe.geom_type == DesignGeomType.BIZONEDRECTANGLE:
                    ghe.set_geometry_constraints_bi_zoned_rectangle(
                        min_height=constraint_props["min_height"],
                        max_height=constraint_props["max_height"],
                        length=constraint_props["length"],
                        width=constraint_props["width"],
                        b_min=constraint_props["b_min"],
                        b_max_x=constraint_props["b_max_x"],
                        b_max_y=constraint_props["b_max_y"],
                    )
                elif ghe.geom_type == DesignGeomType.BIRECTANGLECONSTRAINED:
                    ghe.set_geometry_constraints_bi_rectangle_constrained(
                        min_height=constraint_props["min_height"],
                        max_height=constraint_props["max_height"],
                        b_min=constraint_props["b_min"],
                        b_max_x=constraint_props["b_max_x"],
                        b_max_y=constraint_props["b_max_y"],
                        property_boundary=constraint_props["property_boundary"],
                        no_go_boundaries=constraint_props["no_go_boundaries"],
                    )
                elif ghe.geom_type == DesignGeomType.ROWWISE:
                    # use perimeter calculations if present
                    perimeter_spacing_ratio = constraint_props.get("perimeter_spacing_ratio", 0.0)
                    ghe.set_geometry_constraints_rowwise(
                        min_height=constraint_props["min_height"],
                        max_height=constraint_props["max_height"],
                        perimeter_spacing_ratio=perimeter_spacing_ratio,
                        max_spacing=constraint_props["max_spacing"],
                        min_spacing=constraint_props["min_spacing"],
                        spacing_step=constraint_props["spacing_step"],
                        max_rotation=constraint_props["max_rotation"],
                        min_rotation=constraint_props["min_rotation"],
                        rotate_step=constraint_props["rotate_step"],
                        property_boundary=constraint_props["property_boundary"],
                        no_go_boundaries=constraint_props["no_go_boundaries"],
                    )
                else:
                    print("Geometry constraint method not supported.", file=sys.stderr)
                    return 1

                # self, flow_rate: float, flow_type_str: str, max_eft: float, min_eft: float, throw: bool = True
                ghe.set_design(
                    flow_rate=design_props["flow_rate"],
                    flow_type_str=design_props["flow_type"],
                    min_eft=design_props["min_eft"],
                    max_eft=design_props["max_eft"],
                )
                ghe.find_design()
                ghe.prepare_results("GHEDesigner Run from CLI", "Notes", "Author", "Iteration Name")
                ghe.write_output_files(output_directory)
            else:
                pre_designed_dict: dict = ghe_dict["pre_designed"]
                borehole_height: float = pre_designed_dict["H"]
                x_positions: list[float] = pre_designed_dict["x"]
                y_positions: list[float] = pre_designed_dict["y"]
                m_flow_network = 0.05
                from ghedesigner.media import Pipe

                pipe_positions = Pipe.place_pipes(0.04, ghe._pipe.r_out, 2)
                if len(x_positions) != len(y_positions):
                    pass  # TODO: Emit error
                alpha = 1e-6
                ts = borehole_height**2 / (9 * alpha)
                from numpy import array

                t = [
                    0.1,
                    0.144,
                    0.207,
                    0.298,
                    0.428,
                    0.616,
                    0.886,
                    1.274,
                    1.833,
                    2.637,
                    3.793,
                    5.456,
                    7.848,
                    11.288,
                    16.238,
                    23.357,
                    33.598,
                    48.329,
                    69.519,
                    100.0,
                ]
                time = array(t) * ts
                g_values = evaluate_g_function_MIFT(
                    H=borehole_height,
                    D=4,  # ghe._borehole.D,
                    r_b=0.075,  # ghe._borehole.r_b,
                    x=x_positions,
                    y=y_positions,
                    alpha=alpha,  # ghe._soil.k / ghe._soil.rhoCp,
                    time=time,
                    pos=pipe_positions,
                    r_in=ghe._pipe.r_in,
                    r_out=ghe._pipe.r_out,
                    k_s=ghe._soil.k,
                    k_g=ghe._grout.k,
                    k_p=ghe._pipe.k,
                    epsilon=ghe._pipe.roughness,
                    pipe_type=PipeTypes.SINGLEUTUBE,
                    m_flow_network=m_flow_network,
                    fluid_name=fluid_props["fluid_name"],
                    fluid_concentration_pct=fluid_props["concentration_percent"],
                )
                from ghedesigner.ghe.gfunction import GFunction
                from ghedesigner.ghe.ground_heat_exchangers import GHE
                from ghedesigner.utilities import borehole_spacing

                b = borehole_spacing(ghe._borehole, coordinates=[[0, 0]])
                g_function = GFunction(b, 4, {100: 0.075}, {100: g_values}, list(time), [[0, 0]])
                ghe.set_simulation_parameters(
                    num_months=sim_control["simulation-months"],
                    max_boreholes=1,
                    continue_if_design_unmet=False,
                )
                this_ghe = GHE(
                    m_flow_network,
                    b,
                    BHPipeType.SINGLEUTUBE,
                    fluid=ghe._fluid,
                    borehole=ghe._borehole,
                    pipe=ghe._pipe,
                    grout=ghe._grout,
                    soil=ghe._soil,
                    g_function=g_function,
                    sim_params=ghe._simulation_parameters,
                    hourly_extraction_ground_loads=[],
                )
                from ghedesigner.output import OutputManager

                results = OutputManager("GHEDesigner Run from CLI", "Notes", "Author", "Iteration Name")
                results.write_presized_output_files(output_directory=output_directory, ghe=this_ghe)

    return 0


@click.command(name="GHEDesignerCommandLine")
@click.argument("input-path", type=click.Path(exists=True), required=True)
@click.argument("output-directory", type=click.Path(exists=False), required=False)
@click.version_option(VERSION)
@click.option("--validate-only", default=False, is_flag=True, show_default=False, help="Validate input file and exit.")
@click.option("-c", "--convert", help="Convert output to specified format. Options supported: 'IDF'.")
def run_manager_from_cli(input_path, output_directory, validate_only, convert):
    input_path = Path(input_path).resolve()

    if validate_only:
        try:
            validate_input_file(input_path)
            logger.info("Valid input file.")
            return 0
        except ValidationError:
            logger.error("Schema validation error. See previous error message for details.")
            return 1

    if convert:
        if convert == "IDF":
            try:
                write_idf(input_path)
                print("Output converted to IDF objects.")
                return 0
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Conversion to IDF error: {e}")
                return 1

        else:
            print(f"Unsupported conversion format type: {format}", file=sys.stderr)
            return 1

    if output_directory is None:
        print("Output directory path must be passed as an argument, aborting", file=sys.stderr)
        return 1

    output_path = Path(output_directory).resolve()

    return _run_manager_from_cli_worker(input_path, output_path)


if __name__ == "__main__":
    sys.exit(run_manager_from_cli())
