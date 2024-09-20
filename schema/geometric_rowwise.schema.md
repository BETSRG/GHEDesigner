# JSON Schema

## Properties

- **`perimeter_spacing_ratio`** *(number, format: fraction, required)*: The ratio between the minimum spacing between boreholes placed along the property and no-go zones and the standard borehole-to-borehole spacing used for internal boreholes. Minimum: `0`.
- **`max_spacing`** *(number, format: Meters, required)*: The largest minimum spacing that will be used to generate a RowWise field. Minimum: `0`.
- **`min_spacing`** *(number, format: Meters, required)*: The smallest minimum spacing that will be used to generate a RowWise field. Minimum: `0`.
- **`spacing_step`** *(number, format: Meters, required)*: The distance in spacing from the design found in the first part of first search to exhaustively check in the second part. Minimum: `0`.
- **`max_rotation`** *(number, format: Degrees, required)*: The maximum rotation of the rows of each field relative to horizontal that will be used in the search. Minimum: `-90`. Maximum: `90`.
- **`min_rotation`** *(number, format: Degrees, required)*: The minimum rotation of the rows of each field relative to horizontal that will be used in the search. Minimum: `-90`. Maximum: `90`.
- **`rotate_step`** *(number, format: Degrees, required)*: Step size for field rotation search.
- **`max_height`** *(number, format: Meters, required)*: Maximum height, or active length, of each borehole heat exchanger. Minimum: `0`.
- **`min_height`** *(number, format: Meters, required)*: Minimum height, or active length, of each borehole heat exchanger. Minimum: `0`.
- **`property_boundary`** *(array, format: Meters, required)*: (x, y) coordinate points of closed polygon defining property boundary. Points should be entered in a counter-clockwise fashion.
  - **Items** *(array)*: Length must be equal to 2.
    - **Items** *(number)*: Minimum: `0`.
- **`no_go_boundaries`** *(array, format: Meters, required)*: (x, y) coordinate points of closed polygon defining go/no-go boundaries. Go/no-go zones must lie within the area defined in 'property_boundary'. Points should be entered in a counter-clockwise fashion.
  - **Items** *(array)*
    - **Items** *(array)*: Length must be equal to 2.
      - **Items** *(number)*: Minimum: `0`.
- **`method`** *(string, required)*: Design algorithm specified. Must be: `"ROWWISE"`.
