from json import loads
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ghedesigner.main import run

# results can be updated with the update_demo_results.py file in /scripts
# entries may contain one expected result or a list of accepted results for
# validated dependency baselines
expected_results_path = Path(__file__).parent / "expected_demo_results.json"
timeseries_baseline_path = Path(__file__).parent / "test_data"
expected_demo_results_dict = loads(expected_results_path.read_text())

# override this with a list of Paths to JSON config files to run, or set to `None` to run all demo files
files_to_debug: list[Path] = [
    # Path(__file__).parent.parent.parent / "demos" / "find_design_rowwise_single_u_tube.json"
]

limit_debug_file_count = 0


def assert_timeseries_csv_matches_baseline(actual_path: Path, baseline_path: Path) -> None:
    actual = pd.read_csv(actual_path)
    expected = pd.read_csv(baseline_path)
    assert_frame_equal(actual, expected, check_dtype=False, check_exact=False, rtol=0.0, atol=1e-2)


def assert_demo_result_matches_any(
    actual_length: float,
    actual_nbh: int,
    expected_results: dict | list[dict],
    delta: float = 0.1,
) -> None:
    accepted_results = expected_results if isinstance(expected_results, list) else [expected_results]

    for expected_result in accepted_results:
        expected_length = expected_result["active_borehole_length"]
        expected_nbh = expected_result["number_of_boreholes"]
        if actual_nbh == expected_nbh and abs(actual_length - expected_length) <= delta:
            return

    expected_summary = ", ".join(
        "(length={length:.6f}+/-{delta:.2f}, boreholes={boreholes})".format(
            length=result["active_borehole_length"],
            delta=delta,
            boreholes=result["number_of_boreholes"],
        )
        for result in accepted_results
    )
    raise AssertionError(
        "Unexpected demo result: "
        f"length={actual_length:.6f}, boreholes={actual_nbh}; "
        f"expected one of {expected_summary}"
    )


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
    expected_results = expected_demo_results_dict.get(demo_file_path.stem)
    assert expected_results is not None, f"Missing expected demo results for {demo_file_path.stem}"

    # run demo files first
    demo_output_parent_dir = Path(__file__).parent.parent.parent / "demo_outputs"
    out_dir = demo_output_parent_dir / time_str / demo_file_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running: {demo_file_path}")
    try:
        assert run(input_file_path=demo_file_path, output_directory=out_dir) == 0
    except Exception as err:
        if isinstance(expected_results, dict) and expected_results.get("xfail_run", False):
            pytest.xfail("{}: {!r}".format(expected_results["xfail_reason"], err))
        raise

    if isinstance(expected_results, dict) and expected_results.get("xfail_run", False):
        pytest.fail(f"Demo {demo_file_path.stem} no longer fails; remove xfail_run metadata")

    timeseries_baseline = timeseries_baseline_path / f"{demo_file_path.stem}.csv"
    if timeseries_baseline.exists():
        timeseries_output = out_dir / f"{demo_file_path.stem}.csv"
        assert timeseries_output.exists(), f"Missing timeseries output CSV: {timeseries_output}"
        assert_timeseries_csv_matches_baseline(timeseries_output, timeseries_baseline)

    if isinstance(expected_results, dict) and expected_results.get("skip_checks", False):
        return

    # check the outputs
    results_path = out_dir / "SimulationSummary.json"

    actual_results = loads(results_path.read_text())
    if "ghe_system" in actual_results:
        actual_length = actual_results["ghe_system"]["active_borehole_length"]["value"]
        actual_nbh = actual_results["ghe_system"]["number_of_boreholes"]

        assert_demo_result_matches_any(actual_length, actual_nbh, expected_results)

    else:
        # TODO: Verify it was intentionally predesigned
        assert "log_time" in actual_results
        assert "g_values" in actual_results
