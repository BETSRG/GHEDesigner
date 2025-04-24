#!/usr/bin/env python
import logging
import sys
from json import loads
from pathlib import Path
from time import time
from typing import cast

import click
from jsonschema import ValidationError
from pygfunction.gfunction import evaluate_g_function_MIFT
from pygfunction.pipes import PipeTypes

from ghedesigner.constants import DEG_TO_RAD, MONTHS_IN_YEAR, VERSION
from ghedesigner.enums import BHPipeType, DesignGeomType, FlowConfigType, TimestepType
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


def design_and_size_ghe(
    inputs: dict, ghe_dict: dict, design_parameters: dict, ghe: GroundHeatExchanger, ghe_loads: list[float]
):
    num_months: int = inputs["simulation-control"]["simulation-months"]
    if (num_months % MONTHS_IN_YEAR) > 0:
        raise ValueError(f"num_months must be a multiple of {MONTHS_IN_YEAR}")
    start_month: int = 1
    end_month: int = num_months

    # grab some design conditions
    continue_if_design_unmet: bool = ghe_dict["design"].get("continue_if_design_unmet", False)
    flow_type_str: str = ghe_dict["design"]["flow_type"]
    flow_type = FlowConfigType(flow_type_str.upper())
    flow_rate: float = ghe_dict["design"]["flow_rate"]
    min_eft: float = ghe_dict["design"]["min_eft"]
    max_eft: float = ghe_dict["design"]["max_eft"]
    max_height: float = ghe_dict["geometric_constraints"]["max_height"]  # TODO: Move min/max height to design?
    min_height: float = ghe_dict["geometric_constraints"]["min_height"]
    max_boreholes: int | None = design_parameters.get("max_boreholes")
    # check_arg_bounds(min_eft, max_eft, "min_eft", "max_eft")

    # set up the geometry constraints section
    geom_parameters = ghe_dict["geometric_constraints"]
    geometry_map = {geom.name: geom for geom in DesignGeomType}
    geom_type = geometry_map.get(geom_parameters["method"].upper())
    design: DesignBase
    # simulation_parameters.set_design_heights()
    if geom_type == DesignGeomType.RECTANGLE:
        # max_height: float, min_height: float, length: float, width: float, b_min: float, b_max: float
        rect_geometry: GeometricConstraintsRectangle = GeometricConstraintsRectangle(
            length=geom_parameters["length"],
            width=geom_parameters["width"],
            b_min=geom_parameters["b_min"],
            b_max_x=geom_parameters["b_max"],
        )
        design = DesignRectangle(
            flow_rate,
            ghe.pygfunction_borehole,
            ghe.fluid,
            ghe.pipe,
            ghe.grout,
            ghe.soil,
            start_month,
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
    elif geom_type == DesignGeomType.NEARSQUARE:
        near_sq_geometry: GeometricConstraintsNearSquare = GeometricConstraintsNearSquare(
            b=geom_parameters["b"],
            length=geom_parameters["length"],
        )
        design = DesignNearSquare(
            flow_rate,
            ghe.pygfunction_borehole,
            ghe.fluid,
            ghe.pipe,
            ghe.grout,
            ghe.soil,
            start_month,
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
    elif geom_type == DesignGeomType.BIRECTANGLE:
        bi_rect_geometry: GeometricConstraintsBiRectangle = GeometricConstraintsBiRectangle(
            length=geom_parameters["length"],
            width=geom_parameters["width"],
            b_min=geom_parameters["b_min"],
            b_max_x=geom_parameters["b_max_x"],
            b_max_y=geom_parameters["b_max_y"],
        )
        design = DesignBiRectangle(
            flow_rate,
            ghe.pygfunction_borehole,
            ghe.fluid,
            ghe.pipe,
            ghe.grout,
            ghe.soil,
            start_month,
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
    elif geom_type == DesignGeomType.BIZONEDRECTANGLE:
        bi_zoned_geometry: GeometricConstraintsBiZoned = GeometricConstraintsBiZoned(
            length=geom_parameters["length"],
            width=geom_parameters["width"],
            b_min=geom_parameters["b_min"],
            b_max_x=geom_parameters["b_max_x"],
            b_max_y=geom_parameters["b_max_y"],
        )
        design = DesignBiZoned(
            flow_rate,
            ghe.pygfunction_borehole,
            ghe.fluid,
            ghe.pipe,
            ghe.grout,
            ghe.soil,
            start_month,
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
    elif geom_type == DesignGeomType.BIRECTANGLECONSTRAINED:
        bi_rect_const_geometry: GeometricConstraintsBiRectangleConstrained = GeometricConstraintsBiRectangleConstrained(
            b_min=geom_parameters["b_min"],
            b_max_x=geom_parameters["b_max_x"],
            b_max_y=geom_parameters["b_max_y"],
            property_boundary=geom_parameters["property_boundary"],
            no_go_boundaries=geom_parameters["no_go_boundaries"],
        )
        design = DesignBiRectangleConstrained(
            flow_rate,
            ghe.pygfunction_borehole,
            ghe.fluid,
            ghe.pipe,
            ghe.grout,
            ghe.soil,
            start_month,
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
    else:  # geom_type == DesignGeomType.ROWWISE:
        # use perimeter calculations if present
        perimeter_spacing_ratio = geom_parameters.get("perimeter_spacing_ratio", 0.0)
        geometry_row: GeometricConstraintsRowWise = GeometricConstraintsRowWise(
            perimeter_spacing_ratio=perimeter_spacing_ratio,
            max_spacing=geom_parameters["max_spacing"],
            min_spacing=geom_parameters["min_spacing"],
            spacing_step=geom_parameters["spacing_step"],
            max_rotation=geom_parameters["max_rotation"] * DEG_TO_RAD,
            min_rotation=geom_parameters["min_rotation"] * DEG_TO_RAD,
            rotate_step=geom_parameters["rotate_step"],
            property_boundary=geom_parameters["property_boundary"],
            no_go_boundaries=geom_parameters["no_go_boundaries"],
        )
        design = DesignRowWise(
            flow_rate,
            ghe.pygfunction_borehole,
            ghe.fluid,
            ghe.pipe,
            ghe.grout,
            ghe.soil,
            start_month,
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
    start_time = time()
    search = design.find_design()
    search_time = time() - start_time
    if not search.ghe:
        pass
    found_ghe = cast(GHE, search.ghe)
    found_ghe.compute_g_functions(min_height, max_height)
    found_ghe.size(
        method=TimestepType.HYBRID,
        max_height=max_height,
        min_height=min_height,
        design_max_eft=max_eft,
        design_min_eft=min_eft,
    )
    return search, search_time, found_ghe


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
    inputs = loads(input_file_path.read_text())

    # Read in all the inputs into small dicts
    # it is possible to define multiple fluids, GHEs, and boreholes in the inputs, I'm just taking the first for now
    input_file_version: int = inputs["version"]
    if input_file_version != 1:
        print("Bad input file version, right now we support these versions: 1")
        return 1

    # Validate the load source, it should be a building object or a GHE with loads specified
    ghe_contains_loads = ["loads" in ghe_dict for _, ghe_dict in inputs["ground-heat-exchanger"].items()]
    all_ghe_has_loads = all(ghe_contains_loads)
    no_ghe_has_loads = all(not x for x in ghe_contains_loads)
    building_input = "building" in inputs
    valid_load_source = all_ghe_has_loads ^ (building_input and no_ghe_has_loads)  # XOR because we don't want both
    if not valid_load_source:
        print("Bad load specified, need exactly one of: loads in each ghe, or building object")
        return 1

    # Loop over the topology and init the found objects, for now just the GHE or a GHE with a HP
    topology_props: list[dict] = inputs["topology"]
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
            ghe_dict = inputs["ground-heat-exchanger"][ghe_name]
            ghe = GroundHeatExchanger.init_from_dictionary(ghe_dict, inputs["fluid"])
            ghe_loads = ghe_dict["loads"]
            if "do-sizing" not in ghe_dict or (ghe_dict.get("do-sizing")):
                search, search_time, _ = design_and_size_ghe(inputs, ghe_dict, ghe_dict["design"], ghe, ghe_loads)
                results = OutputManager("GHEDesigner Run from CLI", "Notes", "Author", "Iteration Name")
                results.set_design_data(search, search_time, load_method=TimestepType.HYBRID)
                results.write_all_output_files(output_directory=output_directory, file_suffix="")
    elif len(ghe_names) == 1 and len(building_names) == 1:
        # we have a GHE and a building, grab both
        ghe_dict = inputs["ground-heat-exchanger"][ghe_names[0]]
        ghe = GroundHeatExchanger.init_from_dictionary(ghe_dict, inputs["fluid"])
        single_building = inputs["building"][building_names[0]]
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
        ghe_loads = [float(x) for x in ghe_loads_raw]
        if "do-sizing" not in ghe_dict or (ghe_dict.get("do-sizing")):
            search, search_time, _ = design_and_size_ghe(inputs, ghe_dict, ghe_dict["design"], ghe, ghe_loads)
            results = OutputManager("GHEDesigner Run from CLI", "Notes", "Author", "Iteration Name")
            results.set_design_data(search, search_time, load_method=TimestepType.HYBRID)
            results.write_all_output_files(output_directory=output_directory, file_suffix="")
        else:
            pre_designed = ghe_dict["pre_designed"]
            borehole_height: float = pre_designed["H"]
            x_positions: list[float] = pre_designed["x"]
            y_positions: list[float] = pre_designed["y"]
            m_flow_network = 0.05
            pipe_positions = Pipe.place_pipes(0.04, ghe.pipe.r_out, 2)
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
            g_function = GFunction(b, 4, {100: 0.075}, {100: g_values}, list(time_array), [[0, 0]])
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
