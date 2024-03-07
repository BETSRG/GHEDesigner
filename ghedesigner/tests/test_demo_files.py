import os
from datetime import datetime
from json import loads

from ghedesigner.manager import run_manager_from_cli_worker
from ghedesigner.tests.ghe_base_case import GHEBaseTest

expected_demo_results_dict = {
    'find_design_bi_rectangle_constrained_single_u_tube':
        {
            'active_borehole_length': 134.79,
            'number_of_boreholes': 72
        },
    'find_design_bi_rectangle_double_u_tube_series':
        {
            'active_borehole_length': 124.45,
            'number_of_boreholes': 77
        },
    'find_design_bi_rectangle_single_u_tube':
        {
            'active_borehole_length': 117.57,
            'number_of_boreholes': 88
        },
    'find_design_bi_zoned_rectangle_single_u_tube':
        {
            'active_borehole_length': 128.87,
            'number_of_boreholes': 68
        },
    'find_design_near_square_coaxial':
        {
            'active_borehole_length': 134.89,
            'number_of_boreholes': 132
        },
    'find_design_near_square_double_u_tube':
        {
            'active_borehole_length': 128.39,
            'number_of_boreholes': 144
        },
    'find_design_near_square_single_u_tube':
        {
            'active_borehole_length': 123.86,
            'number_of_boreholes': 156
        },
    'find_design_rectangle_coaxial':
        {
            'active_borehole_length': 92.34,
            'number_of_boreholes': 99
        },
    'find_design_rectangle_double_u_tube':
        {
            'active_borehole_length': 129.69,
            'number_of_boreholes': 77
        },
    'find_design_rectangle_single_u_tube':
        {
            'active_borehole_length': 121.49,
            'number_of_boreholes': 88
        },
    'find_design_rowwise_single_u_tube':
        {
            'active_borehole_length': 133.07,
            'number_of_boreholes': 63
        },
    'input_bldg0000056_odd_loads':
        {
            'active_borehole_length': 74.29,
            'number_of_boreholes': 2
        },
}


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
