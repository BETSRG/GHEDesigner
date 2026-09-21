import json
from copy import deepcopy
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import pytest
from jsonschema import Draft7Validator

from ghedesigner.district_parametric_study import SystemParametricStudySupervisor
from ghedesigner.enums import ParametricStudyParameters
from ghedesigner.tests.test_base_case import GHEBaseTest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMOS_DIRECTORY = PROJECT_ROOT / "demos"
SCHEMA_FILE = PROJECT_ROOT / "ghedesigner" / "schemas" / "ghedesigner.schema.json"


class TestParametricStudy(GHEBaseTest):
    def setUp(self) -> None:
        super().setUp()
        # Reference Values
        reference_data_file = self.test_data_directory / "parametric_study_reference_values.csv"
        self.reference_values = pd.read_csv(reference_data_file)

        # Load Combinatorial File
        combinatorial_file = DEMOS_DIRECTORY / "Network_Sizing_Study_3GHE_6HP_BUPCRS_Combinatorial.json"

        # Load Enumerated File
        enumerated_file = DEMOS_DIRECTORY / "Network_Sizing_Study_3GHE_6HP_BUPCRS_Enumerated.json"

        # Establish Properties
        self.combinatorial_supervisor = SystemParametricStudySupervisor(combinatorial_file)
        self.enumerated_supervisor = SystemParametricStudySupervisor(enumerated_file)

    def test_iterator_methods(self):
        reference_values = self.reference_values
        self.combinatorial_supervisor.generate_study_iterator()
        self.enumerated_supervisor.generate_study_iterator()

        for i, entry in enumerate(self.combinatorial_supervisor.iterator):
            self.assertEqual(str(entry), reference_values["test_iterator_methods_combinatorial"][i])

        for i, entry in enumerate(self.enumerated_supervisor.iterator):
            self.assertEqual(str(entry), reference_values["test_iterator_methods_enumerated"][i])

    def test_study_methods(self):
        reference_values = self.reference_values
        self.combinatorial_supervisor.generate_study_iterator()
        self.enumerated_supervisor.generate_study_iterator()
        self.combinatorial_supervisor.get_study_results()
        self.enumerated_supervisor.get_study_results()

        for i, entry in enumerate(self.combinatorial_supervisor.study_output_values):
            self.assertEqual(str(entry), reference_values["test_study_methods_combinatorial"][i])

        for i, entry in enumerate(self.enumerated_supervisor.study_output_values):
            self.assertEqual(str(entry), reference_values["test_study_methods_enumerated"][i])

        self.assertAlmostEqual(
            self.combinatorial_supervisor.minimum_total_drilling,
            float(reference_values["test_minimum_drilling_combinatorial"][0]),
            delta=0.001,
        )
        self.assertAlmostEqual(
            self.enumerated_supervisor.minimum_total_drilling,
            float(reference_values["test_minimum_drilling_enumerated"][0]),
            delta=0.001,
        )


class TestParametricStudyInputs(TestCase):
    @staticmethod
    def input_data(parametric_study=None):
        return {
            "building": {"A": {"min_eft": 5.0, "max_eft": 30.0}},
            "ground_heat_exchanger": {
                "g1": {
                    "grout": {"conductivity": 1.0},
                    "pipe": {"inner_diameter": 0.03, "outer_diameter": 0.04},
                    "design": {"max_height": 100.0},
                }
            },
            "topology": [
                {"type": "building", "name": "A"},
                {"type": "ground_heat_exchanger", "name": "g1"},
            ],
            "parametric_study": parametric_study or {},
        }

    @staticmethod
    def create_supervisor(input_data):
        with (
            patch(
                "ghedesigner.district_parametric_study.load_input_file",
                return_value=deepcopy(input_data),
            ),
            patch("ghedesigner.district_parametric_study.GHEHPSystem"),
        ):
            return SystemParametricStudySupervisor(Path("unused.json"))

    def test_plural_eft_input_keys_are_applied(self):
        input_data = self.input_data(
            {
                "min_eft_modifications": {"values": [-1.0]},
                "max_eft_modifications": {"values": [2.0]},
            }
        )
        supervisor = self.create_supervisor(input_data)
        supervisor.generate_study_iterator()
        supervisor.prepare_design_dict(supervisor.iterator[0])

        self.assertEqual(supervisor.system_dict["building"]["A"]["min_eft"], 4.0)
        self.assertEqual(supervisor.system_dict["building"]["A"]["max_eft"], 32.0)

    def test_omitted_ghe_parameters_preserve_each_ghe(self):
        input_data = self.input_data()
        input_data["ground_heat_exchanger"]["g2"] = {
            "grout": {"conductivity": 2.0},
            "pipe": {"inner_diameter": 0.05, "outer_diameter": 0.06},
            "design": {"max_height": 120.0},
        }
        input_data["topology"].append({"type": "ground_heat_exchanger", "name": "g2"})
        supervisor = self.create_supervisor(input_data)
        supervisor.generate_study_iterator()
        supervisor.prepare_design_dict(supervisor.iterator[0])

        g2 = supervisor.system_dict["ground_heat_exchanger"]["g2"]
        self.assertEqual(g2["grout"]["conductivity"], 2.0)
        self.assertEqual(g2["pipe"], {"inner_diameter": 0.05, "outer_diameter": 0.06})
        self.assertEqual(g2["design"]["max_height"], 120.0)

    def test_ranged_pipe_sizes_generate_paired_values(self):
        input_data = self.input_data(
            {
                "pipe_sizes": [
                    {"values": [0.03, 0.05, 3], "parameter_range": True},
                    {"values": [0.04, 0.06, 3], "parameter_range": True},
                ]
            }
        )
        supervisor = self.create_supervisor(input_data)
        supervisor.generate_study_iterator()

        pipe_sizes = [entry[ParametricStudyParameters.PIPE_SIZES] for entry in supervisor.iterator]
        self.assertEqual(pipe_sizes, [(0.03, 0.04), (0.04, 0.05), (0.05, 0.06)])

    def test_borehole_height_updates_predesigned_ghe(self):
        input_data = self.input_data({"borehole_heights": {"values": [80.0]}})
        ghe_data = input_data["ground_heat_exchanger"]["g1"]
        ghe_data["pre_designed"] = {"H": 100.0}
        del ghe_data["design"]
        supervisor = self.create_supervisor(input_data)
        supervisor.generate_study_iterator()
        supervisor.prepare_design_dict(supervisor.iterator[0])

        self.assertEqual(supervisor.system_dict["ground_heat_exchanger"]["g1"]["pre_designed"]["H"], 80.0)

    def test_dependent_topology_moves_use_updated_positions(self):
        input_data = self.input_data({"updated_topology": [[["C", "A"], ["D", "C"]]]})
        input_data["building"] = {name: {"min_eft": 5.0, "max_eft": 30.0} for name in ("A", "B", "C", "D")}
        input_data["topology"] = [{"type": "building", "name": name} for name in ("A", "B", "C", "D")]
        supervisor = self.create_supervisor(input_data)
        supervisor.generate_study_iterator()
        supervisor.prepare_design_dict(supervisor.iterator[0])

        self.assertEqual([component["name"] for component in supervisor.system_dict["topology"]], list("ACDB"))

    def test_duplicate_topology_moves_are_rejected(self):
        input_data = self.input_data({"updated_topology": [[["A", "g1"], ["A", "g1"]]]})
        supervisor = self.create_supervisor(input_data)
        supervisor.generate_study_iterator()

        with pytest.raises(ValueError, match="only be moved once"):
            supervisor.prepare_design_dict(supervisor.iterator[0])

    def test_schema_rejects_non_array_pipe_sizes(self):
        schema = json.loads(SCHEMA_FILE.read_text())
        input_data = json.loads(
            (DEMOS_DIRECTORY / "Network_Sizing_Study_3GHE_6HP_BUPCRS_Combinatorial.json").read_text()
        )
        input_data["parametric_study"]["pipe_sizes"] = "bad"

        self.assertFalse(Draft7Validator(schema).is_valid(input_data))
