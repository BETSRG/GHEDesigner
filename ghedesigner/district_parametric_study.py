import csv
from copy import deepcopy
from itertools import chain, product
from pathlib import Path
from typing import Any

import numpy as np

from ghedesigner.district_system import GHEHPSystem
from ghedesigner.enums import ParametricStudyParameters
from ghedesigner.utilities import load_input_file

NUMBER_OF_LENGTHS_ALLOWED_IN_ENUMERATED_STUDY = 2
STUDY_OUTPUT_HEADER = [
    "NBH (-)",
    "Design Height (m)",
    "Total Drilling (m)",
    "Excess Temperature (°C)",
    "Total Energy Consumption (MWhr)",
]

PARAMETER_INPUT_KEYS = {
    ParametricStudyParameters.MAX_EFT_MODIFICATION: "max_eft_modifications",
    ParametricStudyParameters.MIN_EFT_MODIFICATION: "min_eft_modifications",
}


class SystemParametricStudySupervisor:
    def __init__(self, f_path_json: Path):

        # Get full input file
        json_data = load_input_file(f_path_json)
        self.initial_dict = json_data

        # Get original system data
        building_keys = list(json_data["building"])
        parametric_dict = self.initial_dict.pop("parametric_study")
        self.system_dict = deepcopy(json_data)
        ghe_keys = parametric_dict.get("ghes_to_modify", list(json_data["ground_heat_exchanger"]))
        original_pipe_inner_diameter = json_data["ground_heat_exchanger"][ghe_keys[0]]["pipe"]["inner_diameter"]
        original_pipe_outer_diameter = json_data["ground_heat_exchanger"][ghe_keys[0]]["pipe"]["outer_diameter"]
        original_grout_conductivity = json_data["ground_heat_exchanger"][ghe_keys[0]]["grout"]["conductivity"]
        self.original_min_efts = [json_data["building"][building_key]["min_eft"] for building_key in building_keys]
        self.original_max_efts = [json_data["building"][building_key]["max_eft"] for building_key in building_keys]
        self.original_system_topology = json_data["topology"]
        original_borehole_height = (
            json_data["ground_heat_exchanger"][ghe_keys[0]]["pre_designed"]["H"]
            if "pre_designed" in json_data["ground_heat_exchanger"][ghe_keys[0]]
            else json_data["ground_heat_exchanger"][ghe_keys[0]]["design"]["max_height"]
        )
        self.ghe_keys = ghe_keys
        self.building_keys = building_keys

        # Get parametric study data
        self.study_type = parametric_dict.get("study_type", "combination")
        self.parameter_ranges: dict[str, Any] = {}
        self.parameters_to_modify: set[ParametricStudyParameters] = set()
        for parameter_key in ParametricStudyParameters:
            input_key = PARAMETER_INPUT_KEYS.get(parameter_key, parameter_key.value)
            if parameter_key == ParametricStudyParameters.UPDATED_TOPOLOGY:
                self.parameter_ranges[parameter_key] = parametric_dict.get(input_key, [[]])
                if input_key in parametric_dict:
                    self.parameters_to_modify.add(parameter_key)
            elif input_key in parametric_dict:
                self.parameters_to_modify.add(parameter_key)
                if parameter_key == ParametricStudyParameters.PIPE_SIZES:
                    self.parameter_ranges[parameter_key] = {
                        "inner_diameter": {
                            "values": parametric_dict[input_key][0]["values"],
                            "parameter_range": parametric_dict[input_key][0].get("parameter_range", False),
                        },
                        "outer_diameter": {
                            "values": parametric_dict[input_key][1]["values"],
                            "parameter_range": parametric_dict[input_key][1].get("parameter_range", False),
                        },
                    }
                else:
                    self.parameter_ranges[parameter_key] = {
                        "values": parametric_dict[input_key]["values"],
                        "parameter_range": parametric_dict[input_key].get("parameter_range", False),
                    }
            else:
                self.parameter_ranges[parameter_key] = {"parameter_range": False}
                if parameter_key in (
                    ParametricStudyParameters.MAX_EFT_MODIFICATION,
                    ParametricStudyParameters.MIN_EFT_MODIFICATION,
                ):
                    self.parameter_ranges[parameter_key]["values"] = [0.0]
                elif parameter_key == ParametricStudyParameters.GROUT_CONDUCTIVITIES:
                    self.parameter_ranges[parameter_key]["values"] = [original_grout_conductivity]
                elif parameter_key == ParametricStudyParameters.PIPE_SIZES:
                    self.parameter_ranges[parameter_key]["inner_diameter"] = {
                        "values": [original_pipe_inner_diameter],
                        "parameter_range": False,
                    }
                    self.parameter_ranges[parameter_key]["outer_diameter"] = {
                        "values": [original_pipe_outer_diameter],
                        "parameter_range": False,
                    }
                elif parameter_key == ParametricStudyParameters.BOREHOLE_HEIGHTS:
                    self.parameter_ranges[parameter_key]["values"] = [original_borehole_height]

        # Finish initialization
        self.iterator: list[dict[str | int | float, str | int | float]] = []
        self.system = GHEHPSystem(Path(""), initialization_dict=self.system_dict)
        self.study_input_values: list[list[str | int | float]] = []
        self.study_output_values: list[list[int | float]] = []
        self.component_types = {component["name"]: component["type"] for component in self.initial_dict["topology"]}
        self.minimum_total_drilling = float("inf")
        self.minimum_td_system = self.system

    def generate_study_iterator(self):

        parameter_arrays = {}
        for parameter, p_entry in self.parameter_ranges.items():
            if parameter == ParametricStudyParameters.UPDATED_TOPOLOGY:
                parameter_arrays[parameter] = p_entry
            elif parameter == "pipe_sizes":
                p_range = p_entry["inner_diameter"]["values"]
                is_range = p_entry["inner_diameter"]["parameter_range"]
                inner_diameter_list = np.linspace(*p_range) if is_range else p_range

                p_range = p_entry["outer_diameter"]["values"]
                is_range = p_entry["outer_diameter"]["parameter_range"]
                outer_diameter_list = np.linspace(*p_range) if is_range else p_range

                parameter_arrays[parameter] = list(zip(inner_diameter_list, outer_diameter_list))
            else:
                p_range = p_entry["values"]
                is_range = p_entry["parameter_range"]
                parameter_arrays[parameter] = np.linspace(*p_range) if is_range else p_range

        if self.study_type == "combination":
            parameter_keys = parameter_arrays.keys()
            parameter_values = parameter_arrays.values()
            self.iterator = [dict(zip(parameter_keys, combination)) for combination in product(*parameter_values)]
        elif self.study_type == "enumerate":
            parametric_study_lengths = {}
            unique_parametric_study_lengths = set()
            max_length = 0
            for parameter, p_array in parameter_arrays.items():
                p_length = len(list(p_array))
                parametric_study_lengths[parameter] = p_length
                unique_parametric_study_lengths.add(p_length)
                max_length = max_length if p_length < max_length else p_length
            if len(unique_parametric_study_lengths) > NUMBER_OF_LENGTHS_ALLOWED_IN_ENUMERATED_STUDY or (
                len(unique_parametric_study_lengths) == NUMBER_OF_LENGTHS_ALLOWED_IN_ENUMERATED_STUDY
                and 1 not in unique_parametric_study_lengths
            ):
                raise ValueError(
                    "With the 'enumerate' parametric study type,"
                    " all parameter ranges must have the same number of entries (or only one entry)."
                )
            self.iterator = []
            for i in range(max_length):
                self.iterator.append(
                    {
                        parameter: parameter_arrays[parameter][i % parametric_study_lengths[parameter]]
                        for parameter in parameter_arrays
                    }
                )
        else:
            raise ValueError(f"Unrecognized study type: {self.study_type}")

    def design_single_system(self):
        self.system = GHEHPSystem(Path(""), initialization_dict=self.system_dict)
        self.system.size_and_simulate()

    def get_study_results(self, rounding_decimals=2):
        min_total_drilling = float("inf")
        min_td_system = None
        for parameters in self.iterator:
            self.prepare_design_dict(parameters)
            self.design_single_system()
            input_vals = []
            for parameter_key in parameters:
                if parameter_key == ParametricStudyParameters.UPDATED_TOPOLOGY:
                    if parameters[parameter_key] is not None:
                        input_vals.append(
                            " | ".join(
                                [
                                    "_".join(parameters[parameter_key][ind])
                                    for ind in range(len(parameters[parameter_key]))
                                ]
                            )
                        )
                    else:
                        input_vals.append("None")
                else:
                    input_vals.append(str(parameters[parameter_key]))
            self.study_input_values.append(input_vals)
            nbh, td, height = self.system.get_nbh_and_td()
            output_vals = [
                str(nbh),
                str(round(height, rounding_decimals)),
                str(round(td, rounding_decimals)),
                str(round(self.system.calculate_building_excess(), rounding_decimals)),
                str(round(self.system.get_total_energy_consumption() * 1e-6, rounding_decimals)),
            ]
            self.study_output_values.append(output_vals)
            if td < min_total_drilling:
                min_total_drilling = td
                min_td_system = self.system
        self.minimum_total_drilling = min_total_drilling
        self.minimum_td_system = min_td_system

    def prepare_design_dict(self, design_parameters):
        design_dict = deepcopy(self.initial_dict)
        initial_dict = self.initial_dict
        for parameter in design_parameters:
            if parameter not in self.parameters_to_modify:
                continue
            match parameter:
                case ParametricStudyParameters.MIN_EFT_MODIFICATION:
                    for building_key in self.building_keys:
                        design_dict["building"][building_key]["min_eft"] = (
                            initial_dict["building"][building_key]["min_eft"]
                            + design_parameters[ParametricStudyParameters.MIN_EFT_MODIFICATION]
                        )
                case ParametricStudyParameters.MAX_EFT_MODIFICATION:
                    for building_key in self.building_keys:
                        design_dict["building"][building_key]["max_eft"] = (
                            initial_dict["building"][building_key]["max_eft"]
                            + design_parameters[ParametricStudyParameters.MAX_EFT_MODIFICATION]
                        )
                case ParametricStudyParameters.GROUT_CONDUCTIVITIES:
                    for ghe_key in self.ghe_keys:
                        design_dict["ground_heat_exchanger"][ghe_key]["grout"]["conductivity"] = design_parameters[
                            ParametricStudyParameters.GROUT_CONDUCTIVITIES
                        ]
                case ParametricStudyParameters.PIPE_SIZES:
                    for ghe_key in self.ghe_keys:
                        design_dict["ground_heat_exchanger"][ghe_key]["pipe"]["inner_diameter"] = design_parameters[
                            ParametricStudyParameters.PIPE_SIZES
                        ][0]
                        design_dict["ground_heat_exchanger"][ghe_key]["pipe"]["outer_diameter"] = design_parameters[
                            ParametricStudyParameters.PIPE_SIZES
                        ][1]
                case ParametricStudyParameters.BOREHOLE_HEIGHTS:
                    for ghe_key in self.ghe_keys:
                        ghe_data = design_dict["ground_heat_exchanger"][ghe_key]
                        new_height = design_parameters[ParametricStudyParameters.BOREHOLE_HEIGHTS]
                        if "pre_designed" in ghe_data:
                            ghe_data["pre_designed"]["H"] = new_height
                        else:
                            ghe_data["design"]["max_height"] = new_height
                case ParametricStudyParameters.UPDATED_TOPOLOGY:
                    topology_updates = design_parameters[ParametricStudyParameters.UPDATED_TOPOLOGY]
                    if len(topology_updates) == 0:
                        continue
                    moved_components = [component_key for component_key, _ in topology_updates]
                    if len(moved_components) != len(set(moved_components)):
                        raise ValueError("Each component can only be moved once in an updated topology.")

                    known_components = set(self.component_types)
                    for component_key, component_previous_element in topology_updates:
                        if component_key not in known_components:
                            raise ValueError(f"Unknown topology component to move: {component_key}")
                        if component_previous_element and component_previous_element not in known_components:
                            raise ValueError(f"Unknown preceding topology component: {component_previous_element}")
                        if component_key == component_previous_element:
                            raise ValueError(f"A topology component cannot be moved after itself: {component_key}")

                    children: dict[str, list[str]] = {}
                    for component_key, component_previous_element in topology_updates:
                        children.setdefault(component_previous_element, []).append(component_key)

                    moved_component_set = set(moved_components)
                    original_topology = design_dict["topology"]
                    component_data = {component["name"]: component for component in original_topology}
                    updated_topology = []
                    emitted: set[str] = set()
                    active_path: set[str] = set()

                    def emit_component(component_name: str):
                        if component_name in active_path:
                            raise ValueError("A cycle was found in the updated topology.")
                        if component_name in emitted:
                            return
                        active_path.add(component_name)
                        updated_topology.append(deepcopy(component_data[component_name]))
                        emitted.add(component_name)
                        for child_name in children.get(component_name, []):
                            emit_component(child_name)
                        active_path.remove(component_name)

                    for component_name in children.get("", []):
                        emit_component(component_name)
                    for component in original_topology:
                        if component["name"] not in moved_component_set:
                            emit_component(component["name"])

                    if len(emitted) != len(original_topology):
                        raise ValueError("The updated topology contains a cycle or an unreachable component.")
                    design_dict["topology"] = updated_topology
                case _:
                    raise ValueError("Invalid keyword given to 'prepare_design_dict'.")
        self.system_dict = design_dict

    def get_best_design(self, output_directory: Path):
        system = self.minimum_td_system
        if len(system.nbh_selections) != 0:
            system.create_output(
                output_directory / "minimum_system_simulation.csv",
                output_path_2=output_directory / "minimum_system_search.csv",
                output_path_coordinates=output_directory / "minimum_system_coordinates.json",
            )
        else:
            system.create_output(
                output_directory / "minimum_system_simulation.csv",
            )

    def output_study_results(self, output_path: Path):
        input_parameter_header = list(self.parameter_ranges.keys())
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        with output_path.open("w", newline="") as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(chain(input_parameter_header, STUDY_OUTPUT_HEADER))
            csv_writer.writerows([row[0] + row[1] for row in zip(self.study_input_values, self.study_output_values)])
