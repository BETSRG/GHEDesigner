# Input schema

GHEDesigner accepts JSON input files that conform to schema version 3. The authoritative schema is
`ghedesigner/schemas/ghedesigner.schema.json`; the documentation build also publishes a generated **Schema** reference
containing every property, constraint, and default.

Validate an input without running a design or simulation:

```bash
ghedesigner --validate-only path/to/input.json
```

Version 3 does not accept legacy field names. Objects generally reject unknown properties, enum values are lowercase,
and physical quantities include their units in the property name. Common suffixes include `_m`, `_m2`, `_kg_per_s`,
`_l_per_s`, `_pa`, `_w`, `_c`, `_w_per_m_k`, and `_j_per_m3_k`. Fractions use `_fraction`; dimensionless coefficients
have no unit suffix.

## Top-level sections

| Property                      | Purpose                                                                                             |
| ----------------------------- | --------------------------------------------------------------------------------------------------- |
| `schema_version`              | Required integer with the value `3`.                                                                |
| `fluid`                       | Required loop-fluid type, concentration, and property evaluation temperature.                       |
| `soil`                        | Required ground thermal properties shared by all vertical GHEs and horizontal pipes.                |
| `ground_heat_exchangers`      | Required collection of vertical GHE definitions.                                                    |
| `simulation_control`          | Simulation duration, sizing duration, load method, search method, and optional horizontal settings. |
| `buildings`                   | Building heating, cooling, or combined-load components.                                             |
| `heat_pumps`                  | Heat pump performance-map library referenced by building load sources.                              |
| `network`                     | Ordered one-pipe or two-pipe district network.                                                      |
| `source_sink_heat_exchangers` | Controlled external source or sink heat exchangers.                                                 |
| `horizontal_piping`           | Thermal models that distribution segments can reference.                                            |
| `parametric_study`            | Optional network-design parametric study controls.                                                  |

The first four sections are required by the JSON schema. Other sections become operationally required when the selected
design or simulation workflow uses them.

## Fluid and soil

The `fluid` object requires:

- `fluid_type`: `water`, `ethyl_alcohol`, `ethylene_glycol`, `methyl_alcohol`, or `propylene_glycol`.
- `concentration_percent`: antifreeze concentration by mass, from 0 through 60.
- `property_evaluation_temperature_c`: temperature used to evaluate fluid properties, in C.

The `soil` object requires:

- `thermal_conductivity_w_per_m_k`.
- `volumetric_heat_capacity_j_per_m3_k`.
- `undisturbed_ground_temperature_c`.

Horizontal pipe heat transfer additionally uses `soil.ground_temperature_model`, which contains
`annual_temperature_amplitude_c`, `semiannual_temperature_amplitude_c`, `annual_phase_lag_days`, and
`semiannual_phase_lag_days`.

## Simulation controls

`simulation_control` uses unit-explicit duration fields and lowercase enum values:

| Property                           | Accepted values or meaning                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
| `simulation_years`                 | District simulation duration in years.                                                      |
| `sizing_years`                     | GHE sizing duration in years.                                                               |
| `load_method`                      | `hybrid`, `hourly`, or `load_aggregation_hourly`.                                           |
| `search_method`                    | `global_bupcrs`, `global_bupcrs_br`, `global_rowwise`, `nelder_mead`, or `simulation_only`. |
| `constant_cop`                     | Use fixed-COP load conversion instead of full performance-map calculations.                 |
| `exhaustive_search`                | Search all applicable GHE design candidates.                                                |
| `horizontal_simulation_considered` | Include thermal effects from `horizontal_piping`.                                           |
| `horizontal_segments`              | Positive integer number of calculation segments per horizontal pipe.                        |

Provide exactly one of `simulation_years` or `sizing_years`, according to the workflow.

## Loads and buildings

A load source supplies values in one of three ways:

- Inline values: `heat_transfer_rate_values_w`.
- CSV column by name: `file_path` and `column_name`.
- CSV column by zero-based number: `file_path` and `column_number`.

Every form also requires `value_units: "W"`. A source may include either `heat_pump_id`, which references an entry in
the top-level `heat_pumps` collection, or `heat_pump_cop` for fixed-COP conversion. Do not provide both.

A building contains one of these load arrangements:

- `heating_load_source` and `cooling_load_source`.
- `heating_load_source` only.
- `cooling_load_source` only.
- `total_load_source` for a signed combined load.

Separate heating and cooling series must contain non-negative values. For a signed `total_load_source` or direct GHE
`loads`, positive values represent heating or heat extraction and negative values represent cooling or heat rejection.

Optional building temperature fields are `minimum_entering_fluid_temperature_c`,
`maximum_entering_fluid_temperature_c`, `heating_cop_evaluation_temperature_c`, and
`cooling_cop_evaluation_temperature_c`.

## Heat pump performance maps

Each object in `heat_pumps` requires `cooling_performance`, `heating_performance`,
`design_mass_flow_rate_kg_per_s`, `design_pressure_drop_pa`, and `pump_efficiency_fraction`.

Both performance objects contain two quadratic curves evaluated against entering fluid temperature in C:

| Curve                       | Coefficients                                                                                    | Result                             |
| --------------------------- | ----------------------------------------------------------------------------------------------- | ---------------------------------- |
| `heat_transfer_ratio_curve` | `quadratic_coefficient_per_c_squared`, `linear_coefficient_per_c`, `constant_coefficient`       | Dimensionless heat-transfer ratio. |
| `capacity_curve`            | `quadratic_coefficient_w_per_c_squared`, `linear_coefficient_w_per_c`, `constant_coefficient_w` | Heat pump capacity in W.           |

For coefficients `a`, `b`, and `c`, GHEDesigner evaluates `a * T^2 + b * T + c`. The capacity polynomial returns watts
directly; no separate reference-capacity input is used. Optional `minimum_curve_temperature_c` and
`maximum_curve_temperature_c` fields hold a curve at its endpoint value outside the fitted temperature range.

## Ground heat exchangers

Each GHE requires these fields:

- `grout`, with `thermal_conductivity_w_per_m_k` and `volumetric_heat_capacity_j_per_m3_k`.
- `pipe`, with an `arrangement` and arrangement-specific, unit-explicit geometry and material properties.
- `borehole`, with `top_depth_below_grade_m` and `diameter_m`.
- `design_volumetric_flow_rate_per_borehole_l_per_s`, which is a volumetric design flow for one borehole, not the
  entire field.
- Either `fixed_borefield` for a pre-designed field or both `borefield_layout_constraints` and `design` for sizing.

`fixed_borefield.arrangement` is `manual` or `rectangle`. Manual fields use `active_borehole_length_m` and
`borehole_coordinates_m`. Rectangular fields use `active_borehole_length_m`, `borehole_count_x`, `borehole_count_y`,
`borehole_spacing_x_m`, and `borehole_spacing_y_m`.

Sizing constraints select a lowercase `method`: `bi_rectangle`, `bi_rectangle_constrained`, `bi_zoned_rectangle`,
`near_square`, `rectangle`, or `rowwise`. Geometry, spacing, rotation, and boundary fields use explicit `_m` or
`_degrees` suffixes.

### GHE circulation pumps

Every GHE in a one-pipe network requires `circulation_pump` with:

- `reference_pressure_drop_pa`: pressure loss at the full GHE design flow for the borefield and local headers only.
  Distribution piping is excluded.
- `wire_to_water_efficiency_fraction`.
- `pressure_drop_multiplier`.
- Optional `minimum_flow_fraction`, which defaults to `0.05`.

The GHE design mass flow is derived from `design_volumetric_flow_rate_per_borehole_l_per_s`, fluid density, and the
number of boreholes. The circulation-pump reference pressure drop is applied at that total design flow.

## District network

`network.type` is `one_pipe` or `two_pipe`. Both use an ordered compact representation:

- `stations`: each physical component exactly once, in distribution order, as a `component_id`.
- `segments`: physical connections with `id`, `from_component_id`, `to_component_id`, and `length_m`. Optional overrides
  are `diameter_m`, `surface_roughness_m`, and `minor_loss_coefficient`.
- `distribution_pipe_defaults`: a required `diameter_m` plus optional `surface_roughness_m` and
  `minor_loss_coefficient` defaults.
- `pumps`: pump models keyed by ID, each with `wire_to_water_efficiency_fraction`.
- `component_pumps`: component IDs mapped to IDs in `pumps`.

A segment may set `thermal_model_id` to an object in `horizontal_piping`. One-pipe networks additionally require
`distribution_pump` and `mass_flow_control`. `distribution_pump` references a pump model through `pump_id`;
`mass_flow_control` requires `distribution_flow_multiplier` and may set
`minimum_distribution_mass_flow_rate_kg_per_s`.

The compact one-pipe representation automatically creates zero-loss station bypasses. Segment pressure losses and
configured component pressure losses are evaluated for reporting and pump power; they do not allocate network flow.

## Horizontal piping and source/sink heat exchangers

Each `horizontal_piping` object requires `length_m`, `trench_depth_m`, and a single-U-tube `pipe` definition. A coupled
pair also supplies `coupled_to_id` and `spacing_m`; `counter_flow` selects the paired-flow orientation.

Each `source_sink_heat_exchangers` object requires `effectiveness_fraction`, `source_temperature_c`,
`source_mass_flow_rate_kg_per_s`, `cut_in_temperature_c`, and `cut_out_temperature_c`. Optional component `hydraulics`
uses `type: "passive"` and a paired `reference_mass_flow_rate_kg_per_s` and `reference_pressure_drop_pa` design point.
