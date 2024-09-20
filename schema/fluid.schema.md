# JSON Schema

## Properties

- **`fluid_name`** *(string, required)*: Circulation fluid type. Must be one of: `["WATER", "ETHYLALCOHOL", "ETHYLENEGLYCOL", "METHYLALCOHOL", "PROPYLENEGLYCOL"]`.
- **`concentration_percent`** *(number, format: Percent, required)*: Mass fraction concentration percent of circulation fluid. e.g.: '0' indicates pure water; '20' indicates 20% antifreeze, 80% pure water. Minimum: `0`. Maximum: `60`.
- **`temperature`** *(number, format: Centigrade, required)*: Average design fluid temperature at peak conditions.
