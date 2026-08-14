# Output contract inventory and phase 1 mapping

Phase 1 defines the version 1 output shapes without changing the numerical paths that currently produce results. The
representative inputs and outputs below anchor the contract tests and identify what later adapters must preserve.

| Workflow                   | Representative input/output                                                                                | Numerical values that map forward                                                                | Contract representation                                                                    |
| -------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Standalone GHE design      | `demos/find_design_rectangle_single_u_tube.json`; current `SimulationSummary.json` and `BoreFieldData.csv` | selected field type, borehole count, height, spacing, drilling, coordinates, temperature excess  | one `SystemDesign` containing one `ground_heat_exchanger` component                        |
| Multi-GHE design           | `demos/Network_Sizing_3GHE_6HP_BUPCRS.json`; current `Search_Summary.csv` and `coordinates.json`           | per-GHE counts and heights, total drilling, search objective, selected coordinate sets           | one `SystemDesign` containing all GHE components and component/system-scoped evaluations   |
| Predesigned simulation     | `demos/pre_designed_rectangle.json`                                                                        | supplied count, height, spacing, and coordinates                                                 | `design.status = provided`, an empty selected-evaluation list, and no evaluation artifacts |
| Horizontal-pipe simulation | `demos/simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json`; current input-stem CSV               | pipe geometry, segment temperatures, heat transfer, flow, and pump energy                        | component parameters in `Design.json`; canonical quantities indexed by `RunSummary.json`   |
| Future network-pipe design | existing district loop pressure-loss and horizontal-pipe inputs                                            | nominal/inner diameter, length, flow, velocity, pressure loss, roughness, pump-power consequence | a `network_pipe` component using the same evaluation columns as GHE methods                |

The normative, machine-readable schemas are in `ghedesigner/schemas/output/`:

- `run-summary.schema.json` validates the authoritative artifact manifest.
- `design.schema.json` validates both standalone and multi-component selected designs. It defines initial GHE and network-pipe parameter payloads while leaving new component types extensible.
- `design-evaluation-details.schema.json` validates method-owned details keyed by evaluation ID.
- `simulation-summary.schema.json` validates aggregate simulation outcomes and constraint checks.

[The CSV data dictionary](output_contract_csv_data_dictionary.csv) defines the two normalized CSV artifacts.
Cross-artifact rules that JSON Schema
cannot express are enforced by the typed records in `ghedesigner.output.contracts`: selected references must exist and be
marked selected; component IDs must be unique; GHE drilling equals count times height; and selected coordinate counts and
heights must agree with the component design.

The schemas use output contract version `1.0`, independently of the input-file schema version. Phase 1 does not emit or
replace runtime artifacts; adapting current writers is phase 2.
