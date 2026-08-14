"""Typed records for the version 1 output contract.

These records describe the handoff between design methods and simulation.  They
deliberately do not know how a GHE or network component is sized; output writers
can therefore adopt the contract without changing the numerical algorithms.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isclose
from re import fullmatch
from typing import Any, cast

SCHEMA_VERSION = "1.0"

DESIGN_EVALUATION_COLUMNS = (
    "evaluation_id",
    "sequence",
    "design_pass_id",
    "parent_evaluation_id",
    "scope_type",
    "scope_id",
    "component_type",
    "component_id",
    "design_method",
    "candidate_summary",
    "objective_name",
    "objective_value",
    "constraint_excess",
    "feasible",
    "is_selected",
    "detail_ref",
)


class ScopeType(StrEnum):
    """The part of a design affected by an evaluation."""

    COMPONENT = "component"
    COMPONENT_GROUP = "component_group"
    SYSTEM = "system"


class DesignStatus(StrEnum):
    """Allowed outcomes for a design or design pass."""

    COMPLETED = "completed"
    INFEASIBLE = "infeasible"
    FAILED = "failed"
    PROVIDED = "provided"


class RunStatus(StrEnum):
    """Allowed outcomes for a run or simulation."""

    COMPLETED = "completed"
    INFEASIBLE = "infeasible"
    FAILED = "failed"
    NOT_RUN = "not_run"


def _require_text(value: str, name: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _require_nonnegative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_identity(schema_version: str, run_id: str, created_at_utc: str) -> None:
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    _require_text(run_id, "run_id")
    try:
        timestamp = datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("created_at_utc must be an ISO 8601 timestamp") from error
    utc_offset = timestamp.utcoffset()
    if utc_offset is None or utc_offset.total_seconds() != 0:
        raise ValueError("created_at_utc must include the UTC offset")


def _validate_sha256(input_sha256: str) -> None:
    if fullmatch(r"[0-9a-f]{64}", input_sha256) is None:
        raise ValueError("input_sha256 must be a lowercase SHA-256 digest")


class ContractRecord:
    """Mixin providing a JSON-compatible representation for contract records."""

    def to_dict(self) -> dict[str, Any]:
        """Return a recursively JSON-compatible dictionary."""

        # Every concrete ContractRecord is a dataclass, but the dataclass
        # decorator on subclasses cannot be expressed on this mixin's type.
        return _json_value(asdict(cast(Any, self)))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, StrEnum):
        return str(value)
    return value


@dataclass(frozen=True)
class DesignEvaluation(ContractRecord):
    """One method-independent row in ``DesignEvaluations.csv``."""

    evaluation_id: str
    sequence: int
    design_pass_id: str
    parent_evaluation_id: str | None
    scope_type: ScopeType
    scope_id: str
    component_type: str
    component_id: str | None
    design_method: str
    candidate_summary: str
    objective_name: str
    objective_value: float | None
    constraint_excess: float | None
    feasible: bool
    is_selected: bool
    detail_ref: str | None

    def __post_init__(self) -> None:
        for name in (
            "evaluation_id",
            "design_pass_id",
            "scope_id",
            "component_type",
            "design_method",
            "candidate_summary",
            "objective_name",
        ):
            _require_text(getattr(self, name), name)
        if self.sequence < 1:
            raise ValueError("sequence must be at least 1")
        if self.scope_type == ScopeType.COMPONENT and self.component_id is None:
            raise ValueError("component_id is required for component-scoped evaluations")
        if self.component_id is not None:
            _require_text(self.component_id, "component_id")
        if self.parent_evaluation_id == self.evaluation_id:
            raise ValueError("an evaluation cannot be its own parent")

    def csv_row(self) -> dict[str, Any]:
        """Return a row in the normative column order."""
        return {column: getattr(self, column) for column in DESIGN_EVALUATION_COLUMNS}


@dataclass(frozen=True)
class DesignPass(ContractRecord):
    """A single invocation of one design method."""

    design_pass_id: str
    sequence: int
    design_method: str
    method_version: str
    scope_type: ScopeType
    scope_id: str
    evaluation_ids: tuple[str, ...]
    status: DesignStatus
    termination_reason: str

    def __post_init__(self) -> None:
        for name in ("design_pass_id", "design_method", "method_version", "scope_id", "termination_reason"):
            _require_text(getattr(self, name), name)
        if self.sequence < 1:
            raise ValueError("sequence must be at least 1")
        if len(set(self.evaluation_ids)) != len(self.evaluation_ids):
            raise ValueError("evaluation_ids must be unique")
        if self.status == DesignStatus.PROVIDED:
            raise ValueError("provided is a design status, not a design-pass status")


@dataclass(frozen=True)
class GHEParameters(ContractRecord):
    """Initial selected-parameter schema for a vertical GHE component."""

    borehole_count: int
    borehole_height_m: float
    borehole_spacing_m: float
    total_drilling_m: float
    design_flow_kg_s: float | None = None
    field_type: str | None = None
    field_specifier: str | None = None

    def __post_init__(self) -> None:
        if self.borehole_count < 1:
            raise ValueError("borehole_count must be at least 1")
        _require_positive(self.borehole_height_m, "borehole_height_m")
        _require_positive(self.borehole_spacing_m, "borehole_spacing_m")
        _require_positive(self.total_drilling_m, "total_drilling_m")
        if self.design_flow_kg_s is not None:
            _require_positive(self.design_flow_kg_s, "design_flow_kg_s")
        expected_drilling = self.borehole_count * self.borehole_height_m
        if not isclose(self.total_drilling_m, expected_drilling, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("total_drilling_m must equal borehole_count * borehole_height_m")


@dataclass(frozen=True)
class NetworkPipeParameters(ContractRecord):
    """Initial selected-parameter schema for a network-pipe component."""

    nominal_diameter_m: float
    inner_diameter_m: float
    circuit_length_m: float
    design_flow_kg_s: float
    design_velocity_m_s: float
    design_pressure_loss_pa: float
    roughness_m: float | None = None
    pump_power_w: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "nominal_diameter_m",
            "inner_diameter_m",
            "circuit_length_m",
            "design_flow_kg_s",
            "design_velocity_m_s",
        ):
            _require_positive(getattr(self, name), name)
        _require_nonnegative(self.design_pressure_loss_pa, "design_pressure_loss_pa")
        if self.inner_diameter_m > self.nominal_diameter_m:
            raise ValueError("inner_diameter_m cannot exceed nominal_diameter_m")
        if self.roughness_m is not None:
            _require_nonnegative(self.roughness_m, "roughness_m")
        if self.pump_power_w is not None:
            _require_nonnegative(self.pump_power_w, "pump_power_w")


ComponentParameters = GHEParameters | NetworkPipeParameters | Mapping[str, Any]


@dataclass(frozen=True)
class ComponentDesign(ContractRecord):
    """Common envelope around a selected component-specific design."""

    component_id: str
    component_type: str
    design_method: str
    method_version: str
    selected_evaluation_ids: tuple[str, ...]
    status: DesignStatus
    governing_constraint: str | None
    feasibility_margin: float | None
    parameters: ComponentParameters

    def __post_init__(self) -> None:
        for name in ("component_id", "component_type", "design_method", "method_version"):
            _require_text(getattr(self, name), name)
        if len(set(self.selected_evaluation_ids)) != len(self.selected_evaluation_ids):
            raise ValueError("selected_evaluation_ids must be unique")
        if self.status != DesignStatus.PROVIDED and not self.selected_evaluation_ids:
            raise ValueError("a computed component must reference at least one selected evaluation")
        if self.component_type == "ground_heat_exchanger" and not isinstance(self.parameters, GHEParameters):
            raise TypeError("ground_heat_exchanger parameters must use GHEParameters")
        if self.component_type == "network_pipe" and not isinstance(self.parameters, NetworkPipeParameters):
            raise TypeError("network_pipe parameters must use NetworkPipeParameters")


@dataclass(frozen=True)
class BoreholeCoordinate(ContractRecord):
    """One selected row in ``BoreholeCoordinates.csv``."""

    design_id: str
    component_id: str
    borehole_id: str
    x_m: float
    y_m: float
    borehole_height_m: float

    def __post_init__(self) -> None:
        for name in ("design_id", "component_id", "borehole_id"):
            _require_text(getattr(self, name), name)
        _require_positive(self.borehole_height_m, "borehole_height_m")


@dataclass(frozen=True)
class SystemDesign(ContractRecord):
    """The complete system design consumed by simulation."""

    schema_version: str
    run_id: str
    design_id: str
    status: DesignStatus
    workflow: str
    input_file: str
    input_sha256: str
    created_at_utc: str
    design_criteria: Mapping[str, Any]
    governing_constraints: tuple[str, ...]
    selected_evaluation_ids: tuple[str, ...]
    components: tuple[ComponentDesign, ...]

    def __post_init__(self) -> None:
        _validate_identity(self.schema_version, self.run_id, self.created_at_utc)
        _validate_sha256(self.input_sha256)
        for name in ("design_id", "workflow", "input_file"):
            _require_text(getattr(self, name), name)
        if not self.components:
            raise ValueError("components must not be empty")
        component_ids = [component.component_id for component in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("component_id values must be unique within a system design")
        if len(set(self.selected_evaluation_ids)) != len(self.selected_evaluation_ids):
            raise ValueError("selected_evaluation_ids must be unique")
        component_evaluations = {
            evaluation_id for component in self.components for evaluation_id in component.selected_evaluation_ids
        }
        if not component_evaluations.issubset(set(self.selected_evaluation_ids)):
            raise ValueError("component selections must also appear in the system selected_evaluation_ids")
        if self.status == DesignStatus.PROVIDED and self.selected_evaluation_ids:
            raise ValueError("provided designs cannot reference design evaluations")

    def validate_references(
        self,
        evaluations: Iterable[DesignEvaluation],
        coordinates: Sequence[BoreholeCoordinate] = (),
    ) -> None:
        """Validate cross-artifact selection and selected-coordinate invariants."""
        evaluation_list = list(evaluations)
        evaluation_by_id = {evaluation.evaluation_id: evaluation for evaluation in evaluation_list}
        if len(evaluation_by_id) != len(evaluation_list):
            raise ValueError("evaluation_id values must be unique")
        for evaluation_id in self.selected_evaluation_ids:
            if evaluation_id not in evaluation_by_id:
                raise ValueError(f"selected evaluation {evaluation_id!r} does not exist")
            if not evaluation_by_id[evaluation_id].is_selected:
                raise ValueError(f"selected evaluation {evaluation_id!r} is not marked is_selected")

        coordinates_by_component: dict[str, list[BoreholeCoordinate]] = {}
        for coordinate in coordinates:
            if coordinate.design_id != self.design_id:
                raise ValueError("all coordinates must reference the selected design_id")
            coordinates_by_component.setdefault(coordinate.component_id, []).append(coordinate)

        for component in self.components:
            if not isinstance(component.parameters, GHEParameters):
                continue
            component_coordinates = coordinates_by_component.get(component.component_id, [])
            if coordinates and len(component_coordinates) != component.parameters.borehole_count:
                raise ValueError(f"coordinate count does not match {component.component_id!r} borehole_count")
            if any(
                not isclose(
                    coordinate.borehole_height_m,
                    component.parameters.borehole_height_m,
                    rel_tol=1e-9,
                    abs_tol=1e-6,
                )
                for coordinate in component_coordinates
            ):
                raise ValueError(f"coordinate height does not match {component.component_id!r} design")


@dataclass(frozen=True)
class ConstraintCheck(ContractRecord):
    """One simulation constraint and its observed margin."""

    name: str
    limit: float
    observed: float
    margin: float
    units: str
    passed: bool

    def __post_init__(self) -> None:
        _require_text(self.name, "name")
        _require_text(self.units, "units")
        if self.passed != (self.margin >= 0):
            raise ValueError("passed must agree with the sign of margin")


@dataclass(frozen=True)
class SimulationMetrics(ContractRecord):
    """Aggregate results written to ``SimulationSummary.json``."""

    schema_version: str
    run_id: str
    design_id: str
    status: RunStatus
    workflow: str
    created_at_utc: str
    period_start_utc: str
    period_end_utc: str
    timestep_seconds: float
    extrema: Mapping[str, Mapping[str, Any]]
    energy_totals: Mapping[str, float]
    constraint_checks: tuple[ConstraintCheck, ...]
    constraints_passed: bool

    def __post_init__(self) -> None:
        _validate_identity(self.schema_version, self.run_id, self.created_at_utc)
        for name in ("design_id", "workflow", "period_start_utc", "period_end_utc"):
            _require_text(getattr(self, name), name)
        _require_positive(self.timestep_seconds, "timestep_seconds")
        if self.status == RunStatus.NOT_RUN:
            raise ValueError("SimulationMetrics cannot represent a simulation that was not run")
        if self.constraints_passed != all(check.passed for check in self.constraint_checks):
            raise ValueError("constraints_passed must equal the aggregate constraint check result")


@dataclass(frozen=True)
class DesignArtifactManifest(ContractRecord):
    """Design-stage status and artifact references in ``RunSummary.json``."""

    status: DesignStatus
    design_id: str
    methods: tuple[str, ...]
    artifact: str
    evaluations: str | None = None
    evaluation_details: str | None = None
    coordinates: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.design_id, "design_id")
        _require_text(self.artifact, "artifact")
        if self.status == DesignStatus.PROVIDED and self.methods:
            raise ValueError("provided designs must not list computed design methods")


@dataclass(frozen=True)
class SimulationArtifactManifest(ContractRecord):
    """Simulation-stage status and artifact references in ``RunSummary.json``."""

    status: RunStatus
    design_id: str
    summary: str | None
    timeseries: str | None
    constraints_passed: bool | None

    def __post_init__(self) -> None:
        _require_text(self.design_id, "design_id")
        if self.status == RunStatus.NOT_RUN:
            if any(value is not None for value in (self.summary, self.timeseries, self.constraints_passed)):
                raise ValueError("not_run simulations must omit simulation artifacts and results")
        elif self.summary is None or self.timeseries is None or self.constraints_passed is None:
            raise ValueError("a completed simulation stage must reference its artifacts and constraint result")


@dataclass(frozen=True)
class InputProvenance(ContractRecord):
    """Input identity embedded in ``RunSummary.json``."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _require_text(self.path, "path")
        _validate_sha256(self.sha256)


@dataclass(frozen=True)
class ArtifactManifest(ContractRecord):
    """Authoritative ``RunSummary.json`` stage and artifact index."""

    schema_version: str
    run_id: str
    status: RunStatus
    workflow: str
    created_at_utc: str
    input: InputProvenance
    design: DesignArtifactManifest
    simulation: SimulationArtifactManifest
    timeseries_fields: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identity(self.schema_version, self.run_id, self.created_at_utc)
        _require_text(self.workflow, "workflow")
        if self.status == RunStatus.NOT_RUN:
            raise ValueError("ArtifactManifest cannot represent a run that was not started")
        if self.design.design_id != self.simulation.design_id:
            raise ValueError("design and simulation must reference the same design_id")
        if self.design.status == DesignStatus.PROVIDED and (
            self.design.evaluations is not None or self.design.evaluation_details is not None
        ):
            raise ValueError("provided designs must omit evaluation artifacts")


def validate_design_history(
    design: SystemDesign,
    passes: Iterable[DesignPass],
    evaluations: Iterable[DesignEvaluation],
    coordinates: Sequence[BoreholeCoordinate] = (),
) -> None:
    """Validate pass membership, parent links, selections, and design references."""
    pass_list = list(passes)
    evaluation_list = list(evaluations)
    pass_by_id = {design_pass.design_pass_id: design_pass for design_pass in pass_list}
    evaluation_by_id = {evaluation.evaluation_id: evaluation for evaluation in evaluation_list}
    if len(pass_by_id) != len(pass_list):
        raise ValueError("design_pass_id values must be unique")
    if len(evaluation_by_id) != len(evaluation_list):
        raise ValueError("evaluation_id values must be unique")

    for evaluation in evaluation_list:
        if evaluation.design_pass_id not in pass_by_id:
            raise ValueError(f"evaluation {evaluation.evaluation_id!r} references an unknown design pass")
        if evaluation.evaluation_id not in pass_by_id[evaluation.design_pass_id].evaluation_ids:
            raise ValueError(f"evaluation {evaluation.evaluation_id!r} is missing from its design pass")
        if evaluation.parent_evaluation_id is not None and evaluation.parent_evaluation_id not in evaluation_by_id:
            raise ValueError(f"evaluation {evaluation.evaluation_id!r} references an unknown parent")

    for design_pass in pass_list:
        recorded_ids = {
            evaluation.evaluation_id
            for evaluation in evaluation_list
            if evaluation.design_pass_id == design_pass.design_pass_id
        }
        if recorded_ids != set(design_pass.evaluation_ids):
            raise ValueError(f"design pass {design_pass.design_pass_id!r} evaluation_ids do not match its evaluations")

    marked_selected = {evaluation.evaluation_id for evaluation in evaluation_list if evaluation.is_selected}
    if marked_selected != set(design.selected_evaluation_ids):
        raise ValueError("is_selected evaluations must exactly match the system design selections")
    design.validate_references(evaluation_list, coordinates)
