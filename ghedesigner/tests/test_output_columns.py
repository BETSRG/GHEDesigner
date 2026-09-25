import csv
import re
from pathlib import Path

import numpy as np

from ghedesigner.district_parametric_study import (
    STUDY_INPUT_HEADERS,
    STUDY_OUTPUT_HEADER,
    _format_study_input_values,
)
from ghedesigner.enums import ParametricStudyParameters
from ghedesigner.output import columns as csv_columns
from ghedesigner.output.manager import OutputManager

CANONICAL_HEADER = re.compile(r"^.+: .+ \[[^]]+]$")


def test_output_column_uses_canonical_format() -> None:
    assert csv_columns.output_column("ghe_a", "Heat Transfer Rate", "W/m") == ("ghe_a: Heat Transfer Rate [W/m]")


def test_predesigned_g_function_csv_uses_configured_object_name(tmp_path: Path) -> None:
    OutputManager.just_write_g_function(
        tmp_path,
        np.array([1.0]),
        np.array([2.0]),
        np.array([3.0]),
        object_name="ghe_a",
    )

    with (tmp_path / "Gfunction.csv").open(newline="") as output_file:
        rows = list(csv.reader(output_file))

    assert rows == [
        [
            "ghe_a: Log Time Ratio [-]",
            "ghe_a: G-Function [-]",
            "ghe_a: Borehole-Wall G-Function [-]",
        ],
        ["1.0", "2.0", "3.0"],
    ]


def test_parametric_study_headers_split_pipe_diameters_and_use_canonical_units() -> None:
    assert STUDY_INPUT_HEADERS[ParametricStudyParameters.PIPE_SIZES] == [
        "Parametric Study: Pipe Inner Diameter [m]",
        "Parametric Study: Pipe Outer Diameter [m]",
    ]
    assert "System: Total Energy Consumption [MWh]" in STUDY_OUTPUT_HEADER
    assert "System: Excess Temperature [C]" in STUDY_OUTPUT_HEADER
    assert _format_study_input_values({ParametricStudyParameters.PIPE_SIZES: (0.02, 0.03)}) == ["0.02", "0.03"]


def test_district_result_fixture_headers_follow_canonical_format() -> None:
    fixture = Path(__file__).parent / "test_data" / "simulate_1_pipe_1_ghe_1_bldg_district.csv"
    with fixture.open(newline="") as input_file:
        header = next(csv.reader(input_file))

    assert all(CANONICAL_HEADER.fullmatch(column) for column in header)
    assert "building1: Heating Heat Pump Power [W]" in header
    assert "building1: Cooling Heat Pump Power [W]" in header
    assert "building1: Source-Side Net Heat Transfer Rate [W]" in header
    assert all("°C" not in column for column in header)
