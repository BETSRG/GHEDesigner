import pandas as pd

from ghedesigner.district_parametric_study import SystemParametricStudySupervisor
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestParametricStudy(GHEBaseTest):
    def setUp(self) -> None:
        super().setUp()
        # Reference Values
        reference_data_file = self.test_data_directory / "parametric_study_reference_values.csv"
        self.reference_values = pd.read_csv(str(reference_data_file))

        # Load Combinatorial File
        combinatorial_file = (
            self.test_data_directory / "../../../demos/Network_Sizing_Study_3GHE_6HP_BUPCRS_Combinatorial.json"
        )

        # Load Enumerated File
        enumerated_file = (
            self.test_data_directory / "../../../demos/Network_Sizing_Study_3GHE_6HP_BUPCRS_Enumerated.json"
        )

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
