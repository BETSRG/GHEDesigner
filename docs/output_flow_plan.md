# Output flow plan: design and simulation

## Goal

Make every run tell one traceable story with two stages:

1. **Design** evaluates alternatives and selects a buildable system configuration that satisfies capacity and operating constraints.
2. **Simulation** evaluates that selected configuration and reports whether it meets those constraints over time.

“Design” intentionally includes what the current code calls sizing. Borehole count, borehole depth, field layout, and future pipe diameter or pump selections are all design decisions. They must share one evaluation history and one selected system design rather than pass through separate sizing and design artifacts.

The contract must support:

- standalone design of one GHE;
- system-level design of multiple GHEs;
- predesigned systems that proceed directly to simulation;
- multiple design methods in one run, including future network-pipe, pump, and other component selection methods.

## Current output review

There are currently three output paths:

- A standalone GHE run writes `SimulationSummary.json`, text, and CSV reports through `OutputManager`, even when the primary activity is design.
- A district run writes `<input-stem>.csv`, plus `Search_Summary.csv` and `coordinates.json` when a search occurs.
- The standalone horizontal-pipe runner writes another time-series CSV format with different component and node column names.

The district artifacts contain enough information to partially reconstruct a run, but their relationships and scope are implicit:

| Artifact                 | Present role                                                  | Alignment issue                                                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Search_Summary.csv`     | Search evaluations for GHE field selection and borehole depth | It treats sizing as a separate concept, has no stable run/evaluation ID or selected flag, and assumes GHE-specific metrics. It cannot naturally record a network-pipe selection. |
| `coordinates.json`       | GHE coordinate snapshots captured during search               | Consumers must infer which snapshot matches an evaluation and which is selected. It represents only GHE geometry, not a complete system design.                                  |
| `<input-stem>.csv`       | Final district simulation time series                         | It identifies neither the selected system design nor the evaluations that produced it. Identity, quantity, and units are encoded in column strings.                              |
| `SimulationSummary.json` | Structured summary for standalone GHE workflows               | The name conflates design and simulation, and the artifact is not produced consistently for district runs.                                                                       |

The primary gap is a missing **design handoff**: no machine-readable record states “these component-level evaluations produced system design X, and simulation Y evaluated X.”

## Target output contract

Use one authoritative manifest, one design stage, and one simulation stage:

```text
output-directory/
├── RunSummary.json                  # authoritative stage and artifact index
├── DesignEvaluations.csv            # normalized history across design methods
├── Design.json                      # selected complete system design
├── BoreholeCoordinates.csv          # selected GHE coordinates, when applicable
├── DesignEvaluationDetails.json     # optional method-specific evaluation data
├── SimulationSummary.json           # aggregate outcomes and constraint checks
└── SimulationTimeseries.csv         # detailed hourly/sub-hourly results
```

The aligned contract replaces `Search_Summary.csv`, `coordinates.json`, input-stem time-series files, and workflow-specific summary shapes. Consumers begin with `RunSummary.json`; no compatibility aliases are emitted.

### Common identity and provenance

Every artifact must carry, or be referenced through, these fields:

- `schema_version` — output-contract version, independent of the input schema version;
- `run_id` — UUID generated once at run start;
- `input_file` and `input_sha256` — input provenance;
- `created_at_utc` — ISO 8601 timestamp;
- `workflow` — for example `standalone_ghe`, `district_system`, or `component_simulation`;
- `design_id` — identifier shared by the selected design and its simulation;
- `status` — `completed`, `infeasible`, `failed`, or `not_run`.

## Stage 1: design

### Design model

The selected `Design.json` represents the complete design used by simulation. It is a system-level container even when it contains only one standalone GHE:

```text
Design
├── identity and provenance
├── design criteria and governing constraints
├── selected evaluation references
└── components[]
    ├── GHE design(s)
    ├── network-pipe design(s)
    ├── pump design(s)
    └── future component designs
```

Each component entry has a common envelope:

- `component_id` and `component_type`;
- `design_method` and `method_version`;
- `selected_evaluation_id` or IDs;
- `status`, governing constraint, and feasibility margin;
- `parameters` containing the type-specific selected values.

For a standalone GHE, `components` contains one GHE. For system-level multi-GHE design, it contains every GHE plus any other designed network components. This keeps the output shape stable as scope grows.

### Design evaluation history

`DesignEvaluations.csv` uses one row per method evaluation. Its core columns are method- and component-independent:

```text
evaluation_id,sequence,design_pass_id,parent_evaluation_id,
scope_type,scope_id,component_type,component_id,design_method,
candidate_summary,objective_name,objective_value,
constraint_excess,feasible,is_selected,detail_ref
```

Key semantics:

- `scope_type` distinguishes `component`, `component_group`, and `system` evaluations.
- `scope_id` identifies a standalone GHE, a group of GHEs, the distribution network, or the whole system.
- `design_pass_id` groups evaluations from one method invocation.
- `parent_evaluation_id` links nested decisions, such as pipe sizing performed for a candidate GHE layout.
- `is_selected` is explicit; consumers never infer selection from the last row.
- `detail_ref` points to method-specific data without adding GHE-only columns to the common table.

`DesignEvaluationDetails.json` stores extensible method payloads keyed by `evaluation_id`. Examples include:

- GHE target spacing, borehole count, borehole height, total drilling, and coordinate-set reference;
- network-pipe nominal diameter, circuit lengths, design flow, velocity, pressure loss, and pump-power consequence;
- optimizer vectors, bounds, penalties, and termination diagnostics.

A new sizing method can therefore add a component parameter schema and evaluation-detail schema without changing the common history columns or manifest.

### Selected GHE coordinates

`BoreholeCoordinates.csv` contains only coordinates belonging to the selected `design_id`:

```text
design_id,component_id,borehole_id,x_m,y_m,borehole_height_m
```

Candidate coordinate sets belong in `DesignEvaluationDetails.json`, keyed by evaluation. The selected coordinate count for each GHE must equal its selected borehole count, and summed `count × height` must agree with total drilling within tolerance.

### Multiple methods and coupled design

A run may execute multiple design passes. For example:

1. select candidate layouts for a multi-GHE system;
2. size borehole heights for the candidate layouts;
3. size network-pipe diameters and pumps for the resulting design flow;
4. re-evaluate system constraints and cost;
5. select the complete system design.

These remain one **design stage**. `design_pass_id`, parent evaluation links, and component selections preserve the internal sequence without exposing “sizing” as a separate stage. `Design.json` is written only from the final in-memory component states, never reconstructed from the final CSV row.

## Stage 2: simulation

Simulation consumes exactly one `design_id` and produces:

1. `SimulationSummary.json`: simulated period and timestep, extrema with timestamps, energy totals, pump energy, load/ground balance, and constraint checks with limits, observed values, margins, and pass/fail state.
2. `SimulationTimeseries.csv`: time-indexed component data with standardized quantity names and units.

Standardize terminology across district and standalone horizontal-pipe output:

- use `entering_fluid_temperature_c` and `exiting_fluid_temperature_c` in structured data;
- use one node convention (`node_1`, `node_2`, ...) and define whether a node value is outlet or mean temperature;
- correct ambiguous or misspelled quantities such as `Q_src_het`;
- keep display labels in CSV headers while recording canonical field metadata in `RunSummary.json`.

## `RunSummary.json` outline

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "status": "completed",
  "workflow": "district_system",
  "input": { "path": "...", "sha256": "..." },
  "design": {
    "status": "completed",
    "design_id": "design-...",
    "methods": ["GLOBAL_ROWWISE", "NETWORK_PIPE_PRESSURE_DROP"],
    "artifact": "Design.json",
    "evaluations": "DesignEvaluations.csv",
    "evaluation_details": "DesignEvaluationDetails.json",
    "coordinates": "BoreholeCoordinates.csv"
  },
  "simulation": {
    "status": "completed",
    "design_id": "design-...",
    "summary": "SimulationSummary.json",
    "timeseries": "SimulationTimeseries.csv",
    "constraints_passed": true
  }
}
```

A predesigned run records `design.status` as `provided`, writes the supplied design into the common `Design.json` shape, and omits evaluation artifacts. A design-only run records simulation as `not_run`. This preserves the same two-stage model for every workflow.

## Implementation plan

### Phase 1 — define the extensible design contract

1. Inventory representative standalone-GHE, multi-GHE, predesigned, and horizontal-pipe outputs.
2. Define typed records for `DesignEvaluation`, `DesignPass`, `ComponentDesign`, `SystemDesign`, `SimulationMetrics`, and `ArtifactManifest`.
3. Define a common component envelope plus initial parameter schemas for GHEs and network piping.
4. Publish JSON Schemas for the manifest, selected design, evaluation details, and simulation summary, plus a CSV data dictionary.
5. Add contract tests that protect numerical results and invariants without preserving historical filenames or layouts.

**Exit criterion:** standalone and multi-GHE results map to the same system-design schema, and a network-pipe method can be represented without changing the common evaluation table.

### Phase 2 — unify standalone and system design recording

1. Replace parallel search lists with `DesignEvaluation` records in standalone and district workflows.
2. Assign evaluation, design-pass, scope, component, and parent IDs when each design method runs.
3. Mark selected evaluations explicitly after all coupled method passes finish.
4. Build `SystemDesign` from final in-memory components, whether the run contains one GHE or many.
5. Emit `DesignEvaluations.csv`, `DesignEvaluationDetails.json`, `Design.json`, and selected coordinates.
6. Validate component counts, coordinates, drilling totals, selected-evaluation references, and feasibility margins.

**Exit criterion:** both standalone GHE and system-level multi-GHE workflows produce the same design artifacts and require no “last row” assumptions.

### Phase 3 — add other component design methods

1. Implement a design-method interface that accepts criteria and system context, then returns evaluations and a selected component design.
2. Adapt the existing GHE search and borehole-height logic to that interface without changing numerical algorithms.
3. Add network-pipe design parameters and an initial method adapter, keeping hydraulic details in evaluation-detail payloads.
4. Allow methods to declare dependencies so network sizing can consume selected flows and trigger a coupled system re-evaluation when needed.
5. Record method versions, termination reasons, governing constraints, and selections in the common contract.

**Exit criterion:** GHE and network-pipe methods coexist in one design stage and produce one internally consistent `Design.json`.

### Phase 4 — link design to simulation and cut over

1. Generate `design_id` only after the complete system design is selected and store it before the final detailed solve.
2. Write district and standalone `SimulationSummary.json` files against the same schema.
3. Normalize district and horizontal-component time-series naming and metadata.
4. Generate `RunSummary.json` last using atomic replacement so it advertises only complete artifacts.
5. Update demos, baselines, converters, documentation, and first-party analysis tools to start from the manifest.
6. Remove code and tests for `Search_Summary.csv`, `coordinates.json`, input-stem time-series filenames, and workflow-specific summary shapes.

**Exit criterion:** every first-party workflow produces one contract, and simulation can prove which complete standalone or system-level design it evaluated.

## Verification matrix

| Scenario                     | Required assertions                                                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Standalone GHE design        | One-component `Design.json`; explicit selected evaluations; simulation may be `not_run`.                                            |
| Multi-GHE system design      | All GHEs share one `design_id`; component and system evaluations remain distinguishable; coordinate/count/drilling invariants pass. |
| GHE plus network-pipe design | Both methods appear under one design stage; pipe decisions reference the flows/design evaluations they consumed.                    |
| Predesigned simulation       | Design is `provided`; no evaluation history is required; simulation references the normalized `design_id`.                          |
| Infeasible design            | Design reports `infeasible`; the best candidate is retained but not mislabeled as compliant.                                        |
| Horizontal pipe enabled      | Segments use the canonical node convention and simulation metadata identifies their selected design parameters.                     |
| Interrupted or failed run    | No completed manifest advertises missing or partial artifacts.                                                                      |

## Recommended first delivery slice

Implement the unified design handoff before adding new numerical methods:

1. Add the typed design records and common component envelope.
2. Adapt existing standalone and district GHE searches to emit `DesignEvaluation` records.
3. Write `RunSummary.json`, `Design.json`, `DesignEvaluations.csv`, and selected coordinates.
4. Link the final simulation to the complete `design_id` and emit the common `SimulationSummary.json`.
5. Replace legacy files and update demo/test consumers in the same change.
6. Add the design-method interface seam and a network-pipe schema fixture to prove extensibility, while deferring the production pipe-sizing algorithm.

This slice removes the artificial sizing/design boundary, aligns standalone and multi-GHE workflows, and creates an extension point for network and other component design methods without changing current GHE numerical behavior.
