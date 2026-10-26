# Modelica GHE comparison references

These files are hourly reductions of the four five-minute Modelica result files
in NREL's `FY26/TRNSYS Comparison/GHE Comparisons/dtdz_zero` shared directory.
The Modelica borefield's geothermal gradient is disabled (`dT/dz = 0`) so both
models use a uniform 10 C undisturbed ground temperature. Each row contains the
trapezoidally energy-averaged heater load for the preceding hour and the
Modelica GHE outlet temperature at the end of that hour.

The four cases are:

1. 4 by 5 rectangular field, constant 90 kW heat rejection
2. 4 by 5 rectangular field, variable load profile
3. 20-borehole L-shaped field, constant 90 kW heat rejection
4. 20-borehole L-shaped field, variable load profile

The tests compare all 8,760 hourly outlet temperatures. The acceptance limits
are 0.20 C RMSE, 0.15 C absolute mean error, and 0.40 C maximum absolute error.
The source document does not specify pipe volumetric heat capacity, so the tests
use GHEDesigner's standard HDPE value of 1,542,000 J/(m3 K).
