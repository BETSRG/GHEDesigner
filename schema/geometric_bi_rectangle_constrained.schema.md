# JSON Schema

## Properties

- **`b_min`** *(number, format: Meters, required)*: Minimum borehole-to-borehole spacing. Minimum: `0`.
- **`b_max_x`** *(number, format: Meters, required)*: Maximum borehole-to-borehole spacing in x-direction. Minimum: `0`.
- **`b_max_y`** *(number, format: Meters, required)*: Maximum borehole-to-borehole spacing in y-direction. Minimum: `0`.
- **`max_height`** *(number, format: Meters, required)*: Maximum height, or active length, of each borehole heat exchanger. Minimum: `0`.
- **`min_height`** *(number, format: Meters, required)*: Minimum height, or active length, of each borehole heat exchanger. Minimum: `0`.
- **`property_boundary`** *(array, format: Meters, required)*: (x, y) coordinate points of closed polygon defining property boundary. Points should be entered in a counter-clockwise fashion.
  - **Items** *(array)*
    - **Items** *(array)*: Length must be equal to 2.
      - **Items** *(number)*: Minimum: `0`.
- **`no_go_boundaries`** *(array, format: Meters, required)*: (x, y) coordinate points of closed polygon defining go/no-go boundaries. Go/no-go zones must lie within the area defined in 'property_boundary'. Points should be entered in a counter-clockwise fashion.
  - **Items** *(array)*
    - **Items** *(array)*: Length must be equal to 2.
      - **Items** *(number)*: Minimum: `0`.
- **`method`** *(string, required)*: Design algorithm specified. Must be: `"BIRECTANGLECONSTRAINED"`.
