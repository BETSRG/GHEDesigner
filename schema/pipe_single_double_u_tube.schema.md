# JSON Schema

## Properties

- **`inner_diameter`** *(number, format: Meters, required)*: Inner diameter of pipe. Minimum: `0`.
- **`outer_diameter`** *(number, format: Meters, required)*: Outer diameter of pipe. Minimum: `0`.
- **`shank_spacing`** *(number, format: Meters, required)*: Spacing between up/down legs of u-tube pipe, as measured from nearest outer surfaces of each pipe (o<-- s -->o). Minimum: `0`.
- **`roughness`** *(number, format: Meters, required)*: Surface roughness of pipe. Minimum: `0`.
- **`conductivity`** *(number, format: Watts/Meter-Kelvin, required)*: Thermal conductivity. Minimum: `0`.
- **`rho_cp`** *(number, format: Joules/Meter^3-Kelvin, required)*: Volumetric heat capacity. Minimum: `0`.
- **`arrangement`** *(string, required)*: Pipe arrangement specified. Must be one of: `["SINGLEUTUBE", "DOUBLEUTUBESERIES", "DOUBLEUTUBEPARALLEL"]`.
