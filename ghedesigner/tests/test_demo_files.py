from json import loads
from pathlib import Path

import pytest

from ghedesigner.main import _run_manager_from_cli_worker

# results can be updated with the update_demo_results.py file in /scripts
# comment the 'self.assert' statements below to generate an updated set of results first
expected_results_path = Path(__file__).parent / "expected_demo_results.json"
expected_demo_results_dict = loads(expected_results_path.read_text())

# override this with a list of Paths to JSON config files to run, or set to None to run all demo files
files_to_debug: list[Path] = [
    # Path(
    #     "/home/edwin/Projects/GHEDesigner/demos/issue_97_interp_out_of_range.json"
    # ),
    # Path(
    #     "/home/edwin/Projects/GHEDesigner/demos/find_design_simple_system.json"
    # )
    # Path(__file__).parent.parent.parent / "demos" / "find_design_bi_rectangle_constrained_single_u_tube.json"
]

limit_debug_file_count = 0


def abs_error_within_tolerance(val_1, val_2, delta: float = 0):
    return bool(abs(val_1 - val_2) <= delta)


def get_test_input_files() -> list[Path]:
    if files_to_debug:
        return files_to_debug
    demos_path = Path(__file__).parent.parent.parent / "demos"
    demo_files = demos_path.glob("*.json")
    demo_file_list = list(demo_files)
    if limit_debug_file_count > 0:
        return demo_file_list[:limit_debug_file_count]
    return demo_file_list


@pytest.mark.parametrize("demo_file_path", get_test_input_files(), ids=lambda f: "Demo: " + f.stem)
def test_demo_files(demo_file_path: Path, time_str: str):
    failed_tests = []

    # run demo files first
    demo_output_parent_dir = Path(__file__).parent.parent.parent / "demo_outputs"
    out_dir = demo_output_parent_dir / time_str / demo_file_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running: {demo_file_path}")
    assert _run_manager_from_cli_worker(input_file_path=demo_file_path, output_directory=out_dir) == 0

    # check the outputs
    results_path = out_dir / "SimulationSummary.json"

    actual_results = loads(results_path.read_text())
    actual_length = actual_results["ghe_system"]["active_borehole_length"]["value"]
    actual_nbh = actual_results["ghe_system"]["number_of_boreholes"]

    expected_results = expected_demo_results_dict[out_dir.stem]
    expected_length = expected_results["active_borehole_length"]
    expected_nbh = expected_results["number_of_boreholes"]

    len_passes = abs_error_within_tolerance(actual_length, expected_length, delta=0.1)
    nbh_passes = abs_error_within_tolerance(actual_nbh, expected_nbh)

    if not len_passes or not nbh_passes:
        failed_tests.append(out_dir.stem)
