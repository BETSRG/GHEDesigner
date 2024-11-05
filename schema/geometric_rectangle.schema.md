# JSON Schema

## Properties

- **`length`** *(number, format: Meters, required)*: Horizontal length of property boundary defining surface area available for ground heat exchanger. Minimum: `0`.
- **`width`** *(number, format: Meters, required)*: Horizontal width of property boundary defining surface area available for ground heat exchanger. Minimum: `0`.
- **`b_min`** *(number, format: Meters, required)*: Minimum borehole-to-borehole spacing. Minimum: `0`.
- **`b_max`** *(number, format: Meters, required)*: Maximum borehole-to-borehole spacing. Minimum: `0`.
- **`max_height`** *(number, format: Meters, required)*: Maximum height, or active length, of each borehole heat exchanger. Minimum: `0`.
- **`min_height`** *(number, format: Meters, required)*: Minimum height, or active length, of each borehole heat exchanger. Minimum: `0`.
- **`method`** *(string, required)*: Design algorithm specified. Must be: `"RECTANGLE"`.
