# JSON Schema

## Properties

- **`ground_loads`** *(array, format: Watts, required)*: Annual, hourly heat extraction and heat rejection loads of the ground heat exchanger. Positive value indicate heat extraction, negative values indicate heat rejection. Length must be equal to 8760.
  - **Items** *(number)*
- **`heat_pump_loads`** *(array, format: Watts)*: This field is currently unused. Length must be equal to 8760.
  - **Items** *(number)*
