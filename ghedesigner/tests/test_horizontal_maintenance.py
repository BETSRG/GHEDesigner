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
    assert len(output) == 2
