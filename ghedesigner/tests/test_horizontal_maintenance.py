import itertools
import json
from importlib import resources

import pandas as pd
import pytest

from ghedesigner.constants import HORZ_LIBRARY_FILENAME
from ghedesigner.ghe.build_horizontal_pipe_table import (
    STANDARD_BETAS,
    STANDARD_DEPTHS,
    STANDARD_RADII,
    STANDARD_SOIL_CONDUCTIVITIES,
    STANDARD_SPACINGS,
    table_contains_case,
    write_library_atomically,
)
from ghedesigner.horizontal_component_simulation import init_worker, run_horizontal_simulation


def test_existing_horizontal_library_standard_cases_are_recognized():
    with resources.files("ghedesigner.ghe").joinpath(HORZ_LIBRARY_FILENAME).open("r", encoding="utf-8") as file:
        library = json.load(file)

    single_cases = itertools.product(STANDARD_DEPTHS, STANDARD_BETAS, STANDARD_RADII, STANDARD_SOIL_CONDUCTIVITIES)
    parallel_cases = itertools.product(
        STANDARD_DEPTHS,
        STANDARD_SPACINGS,
        STANDARD_BETAS,
        STANDARD_RADII,
        STANDARD_SOIL_CONDUCTIVITIES,
    )

    assert all(table_contains_case(library["table_single"], case) for case in single_cases)
    assert all(table_contains_case(library["table_parallel"], case) for case in parallel_cases)


def test_horizontal_library_write_replaces_destination_after_serialization(tmp_path):
    output_path = tmp_path / "library.json"
    output_path.write_text("original")
    library = {"axes": {"depths": [1.0]}, "table_single": {}, "table_parallel": {}}

    write_library_atomically(library, output_path)

    assert json.loads(output_path.read_text()) == library
    assert not list(tmp_path.glob(".library.json.*.tmp"))


def test_horizontal_library_write_preserves_destination_on_serialization_failure(tmp_path):
    output_path = tmp_path / "library.json"
    output_path.write_text("original")

    with pytest.raises(TypeError):
        write_library_atomically({"not_serializable": object()}, output_path)

    assert output_path.read_text() == "original"
    assert not list(tmp_path.glob(".library.json.*.tmp"))


@pytest.mark.parametrize("case_type", ["ISOLATED", "COUPLED"])
def test_horizontal_component_simulator_uses_json_library(tmp_path, case_type):
    init_worker()
    run_name = f"short_{case_type.lower()}"
    config = {
        "run_name": run_name,
        "output_dir": str(tmp_path),
        "type": case_type,
        "length": 10.0,
        "segments": 1,
        "depth": 1.5,
        "spacing": 0.5,
        "inner_diameter": 0.07,
        "outer_diameter": 0.0762,
        "beta": 0.344,
        "soil_k": 1.5,
        "mass_flow": 0.5,
        "num_hours": 3,
    }

    name, success, error = run_horizontal_simulation(config)

    assert (name, success, error) == (run_name, True, None)
    output = pd.read_csv(tmp_path / f"{run_name}.csv")
    assert len(output) == 3
    assert output["Time [hr]"].tolist() == pytest.approx([1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("steps_per_hour", "expected_inlet_temperatures"),
    [
        (1, [10.0, 20.0, 30.0]),
        (2, [10.0, 10.0, 15.0, 20.0, 25.0, 30.0]),
    ],
)
def test_horizontal_component_csv_rows_align_with_solved_intervals(
    tmp_path, steps_per_hour, expected_inlet_temperatures
):
    init_worker()
    run_name = f"csv_inlet_{steps_per_hour}"
    inlet_path = tmp_path / "inlet.csv"
    pd.DataFrame({"T_in": [10.0, 20.0, 30.0]}).to_csv(inlet_path, index=False)
    config = {
        "run_name": run_name,
        "output_dir": str(tmp_path),
        "type": "ISOLATED",
        "length": 10.0,
        "segments": 1,
        "depth": 1.5,
        "inner_diameter": 0.07,
        "outer_diameter": 0.0762,
        "beta": 0.344,
        "soil_k": 1.5,
        "mass_flow": 0.5,
        "num_hours": 3,
        "steps_per_hour": steps_per_hour,
        "t_in_csv_path": str(inlet_path),
    }

    name, success, error = run_horizontal_simulation(config)

    assert (name, success, error) == (run_name, True, None)
    output = pd.read_csv(tmp_path / f"{run_name}.csv")
    assert output[f"{run_name}_pipe_Inlet [C]"].tolist() == pytest.approx(expected_inlet_temperatures)


def test_horizontal_component_rejects_missing_inlet_csv(tmp_path):
    init_worker()
    config = {
        "run_name": "missing_inlet",
        "output_dir": str(tmp_path),
        "type": "ISOLATED",
        "length": 10.0,
        "segments": 1,
        "depth": 1.5,
        "inner_diameter": 0.07,
        "outer_diameter": 0.0762,
        "beta": 0.344,
        "soil_k": 1.5,
        "mass_flow": 0.5,
        "num_hours": 3,
        "t_in_csv_path": str(tmp_path / "missing.csv"),
    }

    name, success, error = run_horizontal_simulation(config)

    assert name == "missing_inlet"
    assert not success
    assert error is not None
    assert "Inlet-temperature CSV does not exist" in error
    assert not (tmp_path / "missing_inlet.csv").exists()


@pytest.mark.parametrize(
    ("csv_data", "expected_error"),
    [
        ({"wrong_column": [10.0, 20.0, 30.0]}, "must contain a 'T_in' column"),
        ({"T_in": [10.0, "invalid", 30.0]}, "'T_in' values must be numeric"),
        ({"T_in": [10.0, float("nan"), 30.0]}, "'T_in' values must be finite"),
    ],
)
def test_horizontal_component_rejects_malformed_inlet_csv(tmp_path, csv_data, expected_error):
    init_worker()
    inlet_path = tmp_path / "malformed.csv"
    pd.DataFrame(csv_data).to_csv(inlet_path, index=False)
    config = {
        "run_name": "malformed_inlet",
        "output_dir": str(tmp_path),
        "type": "ISOLATED",
        "length": 10.0,
        "segments": 1,
        "depth": 1.5,
        "inner_diameter": 0.07,
        "outer_diameter": 0.0762,
        "beta": 0.344,
        "soil_k": 1.5,
        "mass_flow": 0.5,
        "num_hours": 3,
        "t_in_csv_path": str(inlet_path),
    }

    name, success, error = run_horizontal_simulation(config)

    assert name == "malformed_inlet"
    assert not success
    assert error is not None
    assert expected_error in error
    assert not (tmp_path / "malformed_inlet.csv").exists()
