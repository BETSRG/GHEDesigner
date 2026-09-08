from pathlib import Path

import pytest

from ghedesigner.tests.app import init_or_sanitize_panes, register_data_file


def write_results(path: Path, columns: str = "Time [hr],Temperature [C]") -> None:
    path.write_text(f"{columns}\n0,10\n1,11\n", encoding="utf-8")


def test_register_data_file_returns_the_selected_results_file(tmp_path: Path) -> None:
    results_path = tmp_path / "custom_results.csv"
    write_results(results_path)

    selected_file, label = register_data_file(results_path)

    assert label == "custom results"
    assert selected_file == {label: str(results_path.resolve())}


def test_register_data_file_formats_simulation_filename_for_display(tmp_path: Path) -> None:
    results_path = tmp_path / "simulate_1_ghe_bldg_w_loads.csv"
    write_results(results_path)

    _, label = register_data_file(results_path)

    assert label == "1 GHE bldg with loads"


def test_register_data_file_rejects_a_csv_without_time_column(tmp_path: Path) -> None:
    results_path = tmp_path / "invalid_results.csv"
    write_results(results_path, columns="Hour,Temperature [C]")

    with pytest.raises(ValueError, match="Missing required column 'Time \\[hr\\]'"):
        register_data_file(results_path)


def test_first_selected_file_initializes_default_panes() -> None:
    datasets = {
        "selected": [
            {"Time [hr]": 0, "building1:EFT [C]": 12.0, "ghe1:EFT [C]": 11.0},
        ]
    }

    panes = init_or_sanitize_panes("selected", datasets, [{"title": "Pane 1", "columns": []}])

    assert panes == [
        {"title": "Buildings — EFT [C]", "columns": ["building1:EFT [C]"]},
        {"title": "GHEs — EFT [C]", "columns": ["ghe1:EFT [C]"]},
    ]
