# JSON Schema

## Properties

- **`inner_pipe_d_in`** *(number, format: Meters, required)*: Inner pipe inner diameter. Minimum: `0`.
- **`inner_pipe_d_out`** *(number, format: Meters, required)*: Inner pipe outer diameter. Minimum: `0`.
- **`outer_pipe_d_in`** *(number, format: Meters, required)*: Outer pipe inner diameter. Minimum: `0`.
- **`outer_pipe_d_out`** *(number, format: Meters, required)*: Outer pipe outer diameter. Minimum: `0`.
- **`roughness`** *(number, format: Meters, required)*: Surface roughness. Minimum: `0`.
- **`conductivity_inner`** *(number, format: W/Meters-K, required)*: Thermal conductivity of inner pipe. Minimum: `0`.
- **`conductivity_outer`** *(number, format: Watts/Meter-Kelvin, required)*: Thermal conductivity of outer pipe. Minimum: `0`.
- **`rho_cp`** *(number, format: Joules/Meter^3-Kelvin, required)*: Volumetric heat capacity. Minimum: `0`.
- **`arrangement`** *(string, required)*: Pipe arrangement specified. Must be: `"COAXIAL"`.
