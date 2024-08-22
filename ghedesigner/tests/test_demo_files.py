import os
from datetime import datetime
from json import loads
from pathlib import Path

from ghedesigner.manager import _run_manager_from_cli_worker
from ghedesigner.tests.ghe_base_case import GHEBaseTest

# results can be updated with the update_demo_results.py file in /scripts
# comment the 'self.assert' statements below to generate an updated set of results first
expected_results_path = Path(__file__).parent / "expected_demo_results.json"
expected_demo_results_dict = loads(expected_results_path.read_text())


def abs_error_within_tolerance(val_1, val_2, delta=0):
    return True if abs(val_1 - val_2) <= delta else False


class TestDemoFiles(GHEBaseTest):

    def test_demo_files(self):

        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # run demo files first
        for _, _, files in os.walk(self.demos_path):
            for f in files:
                demo_file_path = self.demos_path / f
                out_dir = self.demo_output_parent_dir / time_str / f.replace('.json', '')
                os.makedirs(out_dir)
                print(f"Running: {demo_file_path}")
                self.assertEqual(0, _run_manager_from_cli_worker(input_file_path=demo_file_path,
                                                                 output_directory=out_dir))

        failed_tests = []

        # check the outputs
        for _, _, files in os.walk(self.demos_path):
            out_dir = self.demo_output_parent_dir / time_str / f.replace('.json', '')
            results_path = out_dir / 'SimulationSummary.json'

            actual_results = loads(results_path.read_text())
            actual_length = actual_results['ghe_system']['active_borehole_length']['value']
            actual_nbh = actual_results['ghe_system']['number_of_boreholes']

            expected_results = expected_demo_results_dict[out_dir.stem]
            expected_length = expected_results['active_borehole_length']
            expected_nbh = expected_results['number_of_boreholes']

            len_passes = abs_error_within_tolerance(actual_length, expected_length, delta=0.1)
            nbh_passes = abs_error_within_tolerance(actual_nbh, expected_nbh)

            if not len_passes or not nbh_passes:
                failed_tests.append(out_dir.stem)

        if failed_tests:
            print(f"Demo tests failed: {failed_tests}")
            self.assertTrue(False)
        else:
            self.assertTrue(True)
