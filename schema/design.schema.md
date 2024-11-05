# JSON Schema

## Properties

- **`flow_rate`** *(number, format: Liters/Second, required)*: Volumetric design flow rate. Value specified will be either the system or per-borehole flow rate depending on the 'flow_type' set. Minimum: `0`.
- **`flow_type`** *(string, required)*: Indicates whether the design volumetric flow rate set on on a per-borehole or system basis. Must be one of: `["BOREHOLE", "SYSTEM"]`.
- **`max_eft`** *(number, format: Centigrade, required)*: Maximum heat pump entering fluid temperature.
- **`min_eft`** *(number, format: Centigrade, required)*: Minimum heat pump entering fluid temperature.
- **`max_boreholes`** *(number)*: Maximum number of boreholes in search. Optional. Applies to rectangular and near-square design algorithms. If unspecified, the search space will be bounded by the size of the GHE boundary.
- **`continue_if_design_unmet`** *(boolean)*: Causes to return the best available borehole  field configuration rather than fail if design conditions  are unmet.  Optional. Default False.
