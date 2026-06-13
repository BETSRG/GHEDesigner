import json
from pathlib import Path

from jsonschema import Draft7Validator

from ghedesigner.gui.ghedesigner_adapter import export_to_ghedesigner
from ghedesigner.gui.models import NetworkDocument
from ghedesigner.gui.run_paths import build_run_paths, demo_style_stem


def test_gui_run_paths_follow_demo_stem_layout(tmp_path: Path):
    output_parent = tmp_path / "demo_outputs" / "20260613_120000"

    output_dir, input_path, stem = build_run_paths(
        output_parent,
        "Simulate 1 Pipe 1 GHE 1 Bldg District",
    )

    assert stem == "simulate_1_pipe_1_ghe_1_bldg_district"
    assert output_dir == output_parent / stem
    assert input_path == output_dir / f"{stem}.json"
    assert output_dir / f"{input_path.stem}.csv" == output_dir / f"{stem}.csv"


def test_gui_run_paths_do_not_repeat_existing_stem_directory(tmp_path: Path):
    output_parent = tmp_path / "simulate_1_pipe_1_ghe_1_bldg_district"

    output_dir, input_path, stem = build_run_paths(
        output_parent,
        "simulate_1_pipe_1_ghe_1_bldg_district",
    )

    assert output_dir == output_parent
    assert input_path == output_parent / f"{stem}.json"


def test_demo_style_stem_has_stable_fallback():
    assert demo_style_stem("  ---  ") == "ghedesigner_gui_run"


def test_starter_gui_export_matches_schema_and_fixed_cop_pattern():
    document = NetworkDocument.starter()
    exported = export_to_ghedesigner(document)

    assert exported["version"] == 2
    assert exported["topology"] == [
        {"type": "building", "name": "building_1"},
        {"type": "ground_heat_exchanger", "name": "ground_heat_exchanger_1"},
    ]
    assert exported["simulation_control"]["constant_cop"] is True
    assert "building" in exported
    assert "ground_heat_exchanger" in exported

    schema_path = Path(__file__).parents[1] / "schemas" / "ghedesigner.schema.json"
    schema = json.loads(schema_path.read_text())
    errors = sorted(Draft7Validator(schema).iter_errors(exported), key=lambda error: list(error.path))
    assert errors == []
