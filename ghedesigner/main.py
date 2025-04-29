#!/usr/bin/env python
import logging
import sys
from json import loads
from pathlib import Path

import click
from jsonschema import ValidationError
from numpy import array
from pygfunction.gfunction import evaluate_g_function_MIFT
from pygfunction.pipes import PipeTypes

from ghedesigner.constants import VERSION
from ghedesigner.enums import BHPipeType, TimestepType
from ghedesigner.ghe.gfunction import GFunction
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.ghe.manager import GroundHeatExchanger
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.heat_pump import HeatPump
from ghedesigner.output import OutputManager
from ghedesigner.utilities import borehole_spacing, write_idf
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

    # read all inputs into a dict
    full_inputs = loads(input_file_path.read_text())

    # Read in all the inputs into small dicts
    # it is possible to define multiple fluids, GHEs, and boreholes in the inputs, I'm just taking the first for now
    input_file_version: int = full_inputs["version"]
    if input_file_version != 1:
        print("Bad input file version, right now we support these versions: 1")
        return 1

    # Validate the load source, it should be a building object or a GHE with loads specified
    ghe_contains_loads = ["loads" in ghe_dict for _, ghe_dict in full_inputs["ground-heat-exchanger"].items()]
    all_ghe_has_loads = all(ghe_contains_loads)
    no_ghe_has_loads = all(not x for x in ghe_contains_loads)
    building_input = "building" in full_inputs
    valid_load_source = all_ghe_has_loads ^ (building_input and no_ghe_has_loads)  # XOR because we don't want both
    if not valid_load_source:
        print("Bad load specified, need exactly one of: loads in each ghe, or building object")
        return 1

    # Loop over the topology and init the found objects, for now just the GHE or a GHE with a HP
    topology_props: list[dict] = full_inputs["topology"]
    ghe_names = []
    building_names = []
    for component in topology_props:
        if component["type"] == "building":
            building_names.append(component["name"])
        elif component["type"] == "ground-heat-exchanger":
            ghe_names.append(component["name"])

    # do actions based on what is provided in input
    if len(ghe_names) >= 1 and len(building_names) == 0:
        # we are just doing a GHE design/sizing/simulation alone
        for ghe_name in ghe_names:
            ghe_dict = full_inputs["ground-heat-exchanger"][ghe_name]
            ghe = GroundHeatExchanger.init_from_dictionary(ghe_dict, full_inputs["fluid"])
            if "pre_designed" in ghe_dict:
                pass  # TODO: What would we want to do here if it is already pre-designed and by itself?
            else:
                # TODO: Assert that "design" data is in the ghe object
                search, search_time, _ = ghe.design_and_size_ghe(full_inputs, ghe_name)
                results = OutputManager("GHEDesigner Run from CLI", "Notes", "Author", "Iteration Name")
                results.set_design_data(search, search_time, load_method=TimestepType.HYBRID)
                results.write_all_output_files(output_directory=output_directory, file_suffix="")
    elif len(ghe_names) == 1 and len(building_names) == 1:
        # we have a GHE and a building, grab both
        ghe_dict = full_inputs["ground-heat-exchanger"][ghe_names[0]]
        ghe = GroundHeatExchanger.init_from_dictionary(ghe_dict, full_inputs["fluid"])
        single_building = full_inputs["building"][building_names[0]]
        heat_pump = HeatPump(single_building["name"])
        heat_pump.set_fixed_cop(single_building["cop"])
        loads_file_path = Path(single_building["loads"]).resolve()
        if not loads_file_path.exists():  # TODO: I'll try to find it relative to repo/tests/ for now...
            this_file = Path(__file__).resolve()
            ghe_designer_dir = this_file.parent
            tests_dir = ghe_designer_dir / "tests"
            loads_file_path = tests_dir / single_building["loads"]
        heat_pump.set_loads_from_file(loads_file_path)
        ghe_loads = heat_pump.get_ground_loads()
        if "pre_designed" in ghe_dict:
            pre_designed = ghe_dict["pre_designed"]
            borehole_height: float = pre_designed["H"]
            x_positions: list[float] = pre_designed["x"]
            y_positions: list[float] = pre_designed["y"]
            # burial_depth: float = ghe.pygfunction_borehole.D
            # borehole_radius: float = ghe.pygfunction_borehole.r_b
            m_flow_network = 0.05
            pipe_positions = Pipe.place_pipes(0.04, ghe.pipe.r_out, 2)
            if len(x_positions) != len(y_positions):
                pass  # TODO: Emit error
            alpha = 1e-6
            ts = borehole_height**2 / (9 * alpha)

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
            ]
            time_array = array(t) * ts
            g_values = evaluate_g_function_MIFT(
                H=borehole_height,
                D=4,  # ghe._borehole.D,
                r_b=0.075,  # ghe._borehole.r_b,
                x=x_positions,
                y=y_positions,
                alpha=alpha,  # ghe._soil.k / ghe._soil.rhoCp,
                time=time_array,
                pos=pipe_positions,
                r_in=ghe.pipe.r_in,
                r_out=ghe.pipe.r_out,
                k_s=ghe.soil.k,
                k_g=ghe.grout.k,
                k_p=ghe.pipe.k,
                epsilon=ghe.pipe.roughness,
                pipe_type=PipeTypes.SINGLEUTUBE,
                m_flow_network=m_flow_network,
                fluid_name=ghe.fluid.name,
                fluid_concentration_pct=ghe.fluid.concentration_percent,
            )

            b = borehole_spacing(ghe.pygfunction_borehole, coordinates=[[0, 0]])
            g_function = GFunction(
                b, 4, {borehole_height: 0.075}, {borehole_height: g_values}, list(time_array), [[0, 0]]
            )
            # simulation_parameters = SimulationParameters(simulation_parameters['simulation-months'], 1, False)
            this_ghe = GHE(
                m_flow_network,
                b,
                BHPipeType.SINGLEUTUBE,
                fluid=ghe.fluid,
                borehole=ghe.pygfunction_borehole,
                pipe=ghe.pipe,
                grout=ghe.grout,
                soil=ghe.soil,
                g_function=g_function,
                start_month=1,
                end_month=1,
                hourly_extraction_ground_loads=ghe_loads,
            )
            this_ghe.simulate(method=TimestepType.HYBRID)
            results = OutputManager("GHEDesigner Run from CLI", "Notes", "Author", "Iteration Name")
            results.write_presized_output_files(output_directory=output_directory, ghe=this_ghe)
        else:
            search, search_time, _ = ghe.design_and_size_ghe(full_inputs, ghe_names[0], loads_override=ghe_loads)
            results = OutputManager("GHEDesigner Run from CLI", "Notes", "Author", "Iteration Name")
            results.set_design_data(search, search_time, load_method=TimestepType.HYBRID)
            results.write_all_output_files(output_directory=output_directory, file_suffix="")
    else:
        print("Bad input file, for now the only available configs are: 1 GLHE alone, or 1 GLHE and 1 building")
        return 1
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
