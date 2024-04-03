import os
from datetime import datetime
from json import loads
from pathlib import Path

from ghedesigner.manager import run_manager_from_cli_worker
from ghedesigner.tests.ghe_base_case import GHEBaseTest

# results can be updated with the update_demo_results.py file in /scripts
expected_results_path = Path(__file__).parent / "expected_demo_results.json"
expected_demo_results_dict = loads(expected_results_path.read_text())


class TestDemoFiles(GHEBaseTest):

    def test_demo_files(self):

        time_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        for _, _, files in os.walk(self.demos_path):
            for f in files:
                demo_file_path = self.demos_path / f
                out_dir = self.demo_output_parent_dir / time_str / f.replace('.json', '')
                os.makedirs(out_dir)
                print(f"Running: {demo_file_path}")
                self.assertEqual(0, run_manager_from_cli_worker(input_file_path=demo_file_path,
                                                                output_directory=out_dir))

                results_path = out_dir / 'SimulationSummary.json'

                actual_results = loads(results_path.read_text())
                actual_length = actual_results['ghe_system']['active_borehole_length']['value']
                actual_nbh = actual_results['ghe_system']['number_of_boreholes']

                expected_results = expected_demo_results_dict[out_dir.stem]
                expected_length = expected_results['active_borehole_length']
                expected_nbh = expected_results['number_of_boreholes']

                self.assertAlmostEqual(actual_length, expected_length, delta=0.01)
                self.assertEqual(expected_nbh, actual_nbh)
