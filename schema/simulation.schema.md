# JSON Schema

## Properties

- **`start_month`** *(string)*: This field is currently unused. Must be one of: `["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]`. Default: `"JANUARY"`.
- **`num_months`** *(number, format: Months, required)*: Number of months used in ground heat exchanger sizing. Minimum: `1`.
- **`timestep`** *(string)*: This field is currently unused. Simulation timestep used in ground heat exchanger sizing. 'HYBRID' is the only option currently available. Must be one of: `["HYBRID", "HOURLY"]`. Default: `"HYBRID"`.
