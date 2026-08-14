#!/usr/bin/env python
"""Command-line and file-based entry points for GHEDesigner workflows."""

import sys
from pathlib import Path

import click
from jsonschema.exceptions import ValidationError

from ghedesigner.constants import INPUT_VERSION, MONTHS_IN_YEAR, VERSION
from ghedesigner.district_system import GHEHPSystem
from ghedesigner.enums import TimestepType
from ghedesigner.ghe.manager import GroundHeatExchanger
from ghedesigner.heat_pump_fixed_cop import HeatPumpFixedCOP
from ghedesigner.output.manager import OutputManager
from ghedesigner.utilities import load_input_file, write_idf
from ghedesigner.validate import validate_input_file


def run(input_file_path: Path, output_directory: Path) -> int:
    """Run the design or simulation described by a validated JSON input file.

    :param input_file_path: Existing GHEDesigner JSON input file.
    :param output_directory: Directory in which result files will be written.
    :return: Zero on success and one when validation or workflow selection fails.
    """

    # validate inputs against the schema before doing anything
    try:
        validate_input_file(input_file_path)
    except ValidationError:
        return 1

    # read all inputs into a dict
    full_inputs = load_input_file(input_file_path)

    # Read in all the inputs into small dicts
    # it is possible to define multiple fluids, GHEs, and boreholes in the inputs, I'm just taking the first for now
    input_file_version: int = full_inputs["version"]
    if input_file_version != INPUT_VERSION:
        print(f"Bad input file version; supported version is: {INPUT_VERSION}")
        return 1

    # Validate the load source, it should be a building object or a GHE with loads specified
    # any GHE instances found with pre_designed will just be ignored since they don't need anything added
    unsized_ghe_contains_loads = []
    for _, ghe_dict in full_inputs["ground_heat_exchanger"].items():
        if "loads" in ghe_dict:
            unsized_ghe_contains_loads.append(True)
        else:
            unsized_ghe_contains_loads.append(False)
    all_ghe_has_loads = all(unsized_ghe_contains_loads)
    no_ghe_has_loads = not any(unsized_ghe_contains_loads)
    building_input = "building" in full_inputs
    valid_load_source = all_ghe_has_loads ^ (building_input and no_ghe_has_loads)  # XOR because we don't want both
    if not valid_load_source:
        print("Bad load specified, need exactly one of: loads in each ghe, or building object")

    # Loop over the topology and init the found objects, for now just the GHE or a GHE with an HP
    topology_props: list[dict] = full_inputs["topology"]
    ghe_names = []
    building_names = []
    central_loop = "central_loop" in full_inputs
    for component in topology_props:
        if component["type"] == "building":
            building_names.append(component["name"])
        elif component["type"] == "ground_heat_exchanger":
            ghe_names.append(component["name"])

    # do actions depending on what is provided in input
    if len(ghe_names) >= 1 and len(building_names) == 0:
        # we are just doing a GHE design/sizing/simulation alone
        for ghe_name in ghe_names:
            ghe_dict = full_inputs["ground_heat_exchanger"][ghe_name]

            if "loads" in ghe_dict and "file_path" in ghe_dict["loads"]:
                if "column_name" in ghe_dict["loads"]:
                    column = ghe_dict["loads"]["column_name"]
                elif "column_number" in ghe_dict["loads"]:
                    column = ghe_dict["loads"]["column_number"]
                else:
                    raise ValueError(f"Column name or number must be provided for loads in GHE '{ghe_name}'")

                ghe_dict["loads"]["column"] = column

            ghe = GroundHeatExchanger.init_from_dictionary(
                ghe_dict,
                full_inputs["fluid"],
                soil_inputs=full_inputs["soil"],
            )
            if "pre_designed" in ghe_dict:
                log_time, g_values, g_bhw_values = ghe.get_g_function(ghe_dict)
                results = OutputManager("GHEDesigner Run from CLI", "Just Calculate G", "", "")
                results.just_write_g_function(output_directory, log_time, g_values, g_bhw_values)
            else:
                # TODO: Assert that "design" data is in the ghe object
                ghe_dict["name"] = ghe_name
                end_month = full_inputs["simulation_control"]["sizing_years"] * MONTHS_IN_YEAR
                search, search_time, _ = ghe.design_and_size_ghe(end_month, ghe_dict=ghe_dict)
                results = OutputManager("GHEDesigner Run from CLI", "Notes", "Author", "Iteration Name")
                results.set_design_data(search, search_time, load_method=TimestepType.HYBRID)
                results.write_all_output_files(output_directory=output_directory, file_suffix="")
    elif len(ghe_names) == 1 and len(building_names) == 1 and not central_loop:
        # we have a GHE and a building, grab both
        ghe_dict = full_inputs["ground_heat_exchanger"][ghe_names[0]]
        ghe_dict["name"] = ghe_names[0]
        ghe = GroundHeatExchanger.init_from_dictionary(
            ghe_dict,
            full_inputs["fluid"],
            soil_inputs=full_inputs["soil"],
        )
        single_building_data = full_inputs["building"][building_names[0]]
        heat_pump = HeatPumpFixedCOP(building_names[0], single_building_data)
        ghe_loads = heat_pump.get_ground_loads()
        if "pre_designed" in ghe_dict:
            log_time, g_values, g_bhw_values = ghe.get_g_function(ghe_dict)
            print(g_values, g_bhw_values)
        else:
            end_month = full_inputs["simulation_control"]["sizing_years"] * MONTHS_IN_YEAR
            search, search_time, _ = ghe.design_and_size_ghe(end_month, loads_override=ghe_loads, ghe_dict=ghe_dict)
            results = OutputManager("GHEDesigner Run from CLI", "Notes", "Author", "Iteration Name")
            results.set_design_data(search, search_time, load_method=TimestepType.HYBRID)
            results.write_all_output_files(output_directory=output_directory, file_suffix="")
    elif central_loop:
        system = GHEHPSystem(input_file_path)
        system.size_and_simulate()

        if len(system.nbh_selections) != 0:
            system.create_output(
                output_directory / f"{input_file_path.stem}.csv",
                output_path_2=output_directory / "Search_Summary.csv",
                output_path_coordinates=output_directory / "coordinates.json",
            )
        else:
            system.create_output(output_directory / f"{input_file_path.stem}.csv")
    else:
        print("Bad input file, for now only the following configurations are available:")
        print("1 GHE; 1 GHE + 1 Building; or N GHE + M Buildings + 1 Central Loop")
        return 1
    return 0


@click.command(name="GHEDesignerCommandLine")
@click.argument("input-path", type=click.Path(exists=True), required=True)
@click.argument("output-directory", type=click.Path(exists=False), required=False)
@click.version_option(VERSION)
@click.option("--validate-only", default=False, is_flag=True, show_default=False, help="Validate input file and exit.")
@click.option("-c", "--convert", help="Convert output to specified format. Options supported: 'IDF'.")
def run_manager_from_cli(input_path, output_directory, validate_only, convert):
    """Run, validate, or convert a GHEDesigner input or result file."""

    # Note that since this is wrapped in click, it should use the exit(code) instead of return.
    # Click will absorb the return code and not return it.
    # If we use exit(code), it will return the code properly.
    input_path = Path(input_path).resolve()

    if validate_only:
        try:
            validate_input_file(input_path)
            print("Valid input file.")
            sys.exit(0)
        except ValidationError as ve:
            print(ve)
            sys.exit(1)

    if convert:
        if convert == "IDF":
            try:
                write_idf(input_path)
                print("Output converted to IDF objects.")
                sys.exit(0)
            except Exception as e:  # noqa: BLE001
                print(f"Conversion to IDF error: {e}")
                sys.exit(1)

        else:
            print(f"Unsupported conversion format type: {convert}", file=sys.stderr)
            sys.exit(1)

    if output_directory is None:
        print("Output directory path must be passed as an argument, aborting", file=sys.stderr)
        sys.exit(1)

    output_path = Path(output_directory).resolve()

    return_code = run(input_path, output_path)
    sys.exit(return_code)


if __name__ == "__main__":
    exit_code = run_manager_from_cli()
    sys.exit(exit_code)
