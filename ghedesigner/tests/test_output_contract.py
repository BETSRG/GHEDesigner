import csv
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from ghedesigner.output.contracts import (
    DESIGN_EVALUATION_COLUMNS,
    SCHEMA_VERSION,
    ArtifactManifest,
    BoreholeCoordinate,
    ComponentDesign,
    ConstraintCheck,
    DesignArtifactManifest,
    DesignEvaluation,
    DesignPass,
    DesignStatus,
    GHEParameters,
    InputProvenance,
    NetworkPipeParameters,
    RunStatus,
    ScopeType,
    SimulationArtifactManifest,
    SimulationMetrics,
    SystemDesign,
    validate_design_history,
)

PROJECT_ROOT = Path(__file__).parents[2]
SCHEMA_DIRECTORY = PROJECT_ROOT / "ghedesigner" / "schemas" / "output"
FIXTURE_DIRECTORY = Path(__file__).parent / "test_data" / "output_contract"
SHA256 = "a" * 64
CREATED_AT = "2026-01-01T00:00:00Z"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def validate(instance: dict, schema_name: str) -> None:
    validator = Draft202012Validator(
        load_json(SCHEMA_DIRECTORY / schema_name),
        format_checker=FormatChecker(),
    )
    validator.validate(instance)


def selected_evaluation(evaluation_id: str, component_id: str, component_type: str) -> DesignEvaluation:
    return DesignEvaluation(
        evaluation_id=evaluation_id,
        sequence=1,
        design_pass_id=f"pass-{component_id}",
        parent_evaluation_id=None,
        scope_type=ScopeType.COMPONENT,
        scope_id=component_id,
        component_type=component_type,
        component_id=component_id,
        design_method="fixture-method",
        candidate_summary="selected fixture candidate",
        objective_name="total_cost",
        objective_value=100.0,
        constraint_excess=-0.1,
        feasible=True,
        is_selected=True,
        detail_ref=f"#/details/{evaluation_id}",
    )


def test_output_schemas_are_valid():
    schema_paths = sorted(SCHEMA_DIRECTORY.glob("*.schema.json"))
    assert {path.name for path in schema_paths} == {
        "design-evaluation-details.schema.json",
        "design.schema.json",
        "run-summary.schema.json",
        "simulation-summary.schema.json",
    }
    for schema_path in schema_paths:
        Draft202012Validator.check_schema(load_json(schema_path))


@pytest.mark.parametrize(
    "fixture_name",
    [
        "standalone_design.json",
        "multi_ghe_design.json",
        "ghe_and_network_design.json",
        "predesigned_design.json",
    ],
)
def test_representative_designs_share_one_system_schema(fixture_name):
    validate(load_json(FIXTURE_DIRECTORY / fixture_name), "design.schema.json")


def test_standalone_fixture_preserves_reference_numerical_result():
    standalone = load_json(FIXTURE_DIRECTORY / "standalone_design.json")
    parameters = standalone["components"][0]["parameters"]

    assert parameters["borehole_count"] == 88
    assert parameters["borehole_height_m"] == pytest.approx(126.90579549671773)
    assert parameters["total_drilling_m"] == pytest.approx(
        parameters["borehole_count"] * parameters["borehole_height_m"]
    )


def test_network_pipe_uses_the_common_evaluation_columns():
    ghe_row = selected_evaluation("eval-ghe", "ghe-1", "ground_heat_exchanger").csv_row()
    pipe_row = selected_evaluation("eval-pipe", "pipe-1", "network_pipe").csv_row()

    assert tuple(ghe_row) == DESIGN_EVALUATION_COLUMNS
    assert tuple(pipe_row) == DESIGN_EVALUATION_COLUMNS
    assert set(ghe_row) == set(pipe_row)
    assert "nominal_diameter_m" not in pipe_row


def test_typed_system_design_validates_selected_references_and_coordinates():
    ghe_evaluation = selected_evaluation("eval-ghe", "ghe-1", "ground_heat_exchanger")
    pipe_evaluation = selected_evaluation("eval-pipe", "pipe-1", "network_pipe")
    ghe = ComponentDesign(
        component_id="ghe-1",
        component_type="ground_heat_exchanger",
        design_method="ROW_WISE",
        method_version="1",
        selected_evaluation_ids=(ghe_evaluation.evaluation_id,),
        status=DesignStatus.COMPLETED,
        governing_constraint="maximum_entering_fluid_temperature_c",
        feasibility_margin=0.2,
        parameters=GHEParameters(
            borehole_count=2,
            borehole_height_m=100.0,
            borehole_spacing_m=5.0,
            total_drilling_m=200.0,
        ),
    )
    network_pipe = ComponentDesign(
        component_id="pipe-1",
        component_type="network_pipe",
        design_method="NETWORK_PIPE_PRESSURE_DROP",
        method_version="1",
        selected_evaluation_ids=(pipe_evaluation.evaluation_id,),
        status=DesignStatus.COMPLETED,
        governing_constraint="maximum_velocity_m_s",
        feasibility_margin=0.1,
        parameters=NetworkPipeParameters(
            nominal_diameter_m=0.05,
            inner_diameter_m=0.04,
            circuit_length_m=100.0,
            design_flow_kg_s=1.0,
            design_velocity_m_s=1.2,
            design_pressure_loss_pa=25000.0,
        ),
    )
    design = SystemDesign(
        schema_version=SCHEMA_VERSION,
        run_id="run-1",
        design_id="design-1",
        status=DesignStatus.COMPLETED,
        workflow="district_system",
        input_file="input.json",
        input_sha256=SHA256,
        created_at_utc=CREATED_AT,
        design_criteria={"maximum_velocity_m_s": 1.3},
        governing_constraints=("maximum_velocity_m_s",),
        selected_evaluation_ids=(ghe_evaluation.evaluation_id, pipe_evaluation.evaluation_id),
        components=(ghe, network_pipe),
    )
    coordinates = (
        BoreholeCoordinate("design-1", "ghe-1", "bh-1", 0.0, 0.0, 100.0),
        BoreholeCoordinate("design-1", "ghe-1", "bh-2", 5.0, 0.0, 100.0),
    )

    passes = (
        DesignPass(
            "pass-ghe-1",
            1,
            "ROW_WISE",
            "1",
            ScopeType.COMPONENT,
            "ghe-1",
            ("eval-ghe",),
            DesignStatus.COMPLETED,
            "selected feasible candidate",
        ),
        DesignPass(
            "pass-pipe-1",
            2,
            "NETWORK_PIPE_PRESSURE_DROP",
            "1",
            ScopeType.COMPONENT,
            "pipe-1",
            ("eval-pipe",),
            DesignStatus.COMPLETED,
            "selected feasible diameter",
        ),
    )

    validate_design_history(design, passes, (ghe_evaluation, pipe_evaluation), coordinates)
    validate(design.to_dict(), "design.schema.json")


def test_typed_design_rejects_inconsistent_drilling_and_unselected_references():
    with pytest.raises(ValueError, match=r"borehole_count \* borehole_height_m"):
        GHEParameters(
            borehole_count=2,
            borehole_height_m=100.0,
            borehole_spacing_m=5.0,
            total_drilling_m=199.0,
        )

    evaluation = selected_evaluation("eval-ghe", "ghe-1", "ground_heat_exchanger")
    unselected = DesignEvaluation(**{**evaluation.__dict__, "is_selected": False})
    component = ComponentDesign(
        component_id="ghe-1",
        component_type="ground_heat_exchanger",
        design_method="RECTANGLE",
        method_version="1",
        selected_evaluation_ids=(evaluation.evaluation_id,),
        status=DesignStatus.COMPLETED,
        governing_constraint=None,
        feasibility_margin=0.0,
        parameters=GHEParameters(1, 100.0, 5.0, 100.0),
    )
    design = SystemDesign(
        SCHEMA_VERSION,
        "run-1",
        "design-1",
        DesignStatus.COMPLETED,
        "standalone_ghe",
        "input.json",
        SHA256,
        CREATED_AT,
        {},
        (),
        (evaluation.evaluation_id,),
        (component,),
    )

    with pytest.raises(ValueError, match="is not marked is_selected"):
        design.validate_references((unselected,))


def test_manifest_and_simulation_typed_records_match_published_schemas():
    manifest = ArtifactManifest(
        schema_version=SCHEMA_VERSION,
        run_id="run-1",
        status=RunStatus.COMPLETED,
        workflow="standalone_ghe",
        created_at_utc=CREATED_AT,
        input=InputProvenance("input.json", SHA256),
        design=DesignArtifactManifest(
            status=DesignStatus.COMPLETED,
            design_id="design-1",
            methods=("RECTANGLE",),
            artifact="Design.json",
            evaluations="DesignEvaluations.csv",
            evaluation_details="DesignEvaluationDetails.json",
            coordinates="BoreholeCoordinates.csv",
        ),
        simulation=SimulationArtifactManifest(
            status=RunStatus.NOT_RUN,
            design_id="design-1",
            summary=None,
            timeseries=None,
            constraints_passed=None,
        ),
    )
    metrics = SimulationMetrics(
        schema_version=SCHEMA_VERSION,
        run_id="run-1",
        design_id="design-1",
        status=RunStatus.COMPLETED,
        workflow="standalone_ghe",
        created_at_utc=CREATED_AT,
        period_start_utc="2026-01-01T00:00:00Z",
        period_end_utc="2026-12-31T23:00:00Z",
        timestep_seconds=3600.0,
        extrema={
            "maximum_entering_fluid_temperature_c": {
                "value": 34.8,
                "units": "C",
                "timestamp_utc": "2026-08-01T14:00:00Z",
            }
        },
        energy_totals={"pump_energy_kwh": 1200.0},
        constraint_checks=(ConstraintCheck("maximum_eft", 35.0, 34.8, 0.2, "C", True),),
        constraints_passed=True,
    )

    validate(manifest.to_dict(), "run-summary.schema.json")
    validate(metrics.to_dict(), "simulation-summary.schema.json")


def test_evaluation_details_and_csv_dictionary_match_contract():
    validate(
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": "run-1",
            "created_at_utc": CREATED_AT,
            "details": {
                "eval-pipe": {
                    "nominal_diameter_m": 0.05,
                    "design_pressure_loss_pa": 25000.0,
                }
            },
        },
        "design-evaluation-details.schema.json",
    )

    dictionary_path = PROJECT_ROOT / "docs" / "output_contract_csv_data_dictionary.csv"
    with dictionary_path.open(newline="") as dictionary_file:
        rows = list(csv.DictReader(dictionary_file))
    evaluation_columns = tuple(row["column"] for row in rows if row["artifact"] == "DesignEvaluations.csv")

    assert evaluation_columns == DESIGN_EVALUATION_COLUMNS
