# PLAT - History of changes

The peak load analysis tool (PLAT) was in its own repository, but has now been located inside of `ghedt`. The changes given here are the changes in the `PLAT` repository. 

## Version 0.1

### Deprecates 

* [Issue 35](https://github.com/j-c-cook/PLAT/issues/35) - The convection coefficient can no longer be over-ridden for inputs to the borehole heat exchanger objects. The original intent was for equivalent BHE's, but EnergyPlus does not have convection coefficient as an input. 

### Enhancements

* [Issue 15](https://github.com/j-c-cook/PLAT/issues/15) - Adds a media module that contains a thermal properties object and a pipe object that inherits from thermal properties. The thermal properties are no longer passed around alone as float, but rather as members of objects.
* [Issue 16](https://github.com/j-c-cook/PLAT/issues/16) - The equivalent single U-tube borehole heat exchanger now varies the flow rate to match the convection coefficient. The previous version forced the convection coefficient. This provides only a small difference, but should be a better solution.
* [Issue 22](https://github.com/j-c-cook/PLAT/issues/22) - The radial numerical short-time step calculation is vectorized, resulting in more than a 22x speed increase, and the root mean squared error between the old and new implementation over multiple tests was less than 1.0e-10. Prior to vectorization, the short-time step computation averaged around 2.5 seconds. After vectorization (via Numpy arrays and scipy.LAPACK API call), the code runs in 0.11 seconds.
* [Issue 51](https://github.com/j-c-cook/PLAT/issues/51) - The two-day detailed simulations that work to find the peak duration is vectorized, resulting in a speed increase for peak load analysis. 

### Fixes

* [Issue 14](https://github.com/j-c-cook/PLAT/issues/14) - The volumetric heat capacities are now passed into the radial numerical short time step. The volumetric heat capacities are stored in the equivalent single U-tube borehole heat exchanger.
* [Issue 19](https://github.com/j-c-cook/PLAT/issues/19) - Fix mass flow rate not back propagating (provides no difference to any results, only for when checking final equivalent values).
* [Issue 29](https://github.com/j-c-cook/PLAT/issues/29) - Mass flow rate is no longer varied to set the convective resistances equal for the equivalent borehole. The lumped convection and pipe resistance are set equal by varying the pipe thermal conductivity.
* [Issue 44](https://github.com/j-c-cook/PLAT/issues/44) - Fixes the possible code break when the peak load duration is found to be 0.0. The short-time step calculation is adjusted so that the first time step occurs at 1.0e-12 seconds, and the peak duration for 0.0 is changed to 1.0e-6. The interpolation for the short-time step simulating the hybrid time step loads will not be out of bounds now.
* [Issue 48](https://github.com/j-c-cook/PLAT/issues/48) - Fixes the two-day simulation to pull the day before and day of rather than the day of and day after. 
* [Issue 53](https://github.com/j-c-cook/PLAT/issues/53) - Fixes the issue of interpolation error that could occasionally occur in the initialization of the design procedure. The specific portion of the code has to do with the interpolation for peak load duration. Previously, values below the interpolation range (less than 0) were handled. The error handling is improved by considering values above the interpolation range. 

### New features

* [Issue 1](https://github.com/j-c-cook/PLAT/issues/1) - Adds functionality to compute equivalent single U-tube given a borehole with an n-number of U-tubes.
* [Issue 3](https://github.com/j-c-cook/PLAT/issues/3) - Adds single and multiple U-tube effective borehole resistance calculations from `pygfunction`. A mediocre coaxial effective resistance calculation is implemented, but is planned to be replaced once `pygfunction` has Coaxial pipe functionality.
* [Issue 11](https://github.com/j-c-cook/PLAT/issues/11) - The Coaxial object in `pipes` now inherits from pygfunction.
* [Issue 13](https://github.com/j-c-cook/PLAT/issues/13) - Adds equivalent functionality for Coaxial pipe to single U-tube.
* [Issue 4](https://github.com/j-c-cook/PLAT/issues) - Adds radial numerical short time step that takes in `PLAT.pipes` (inherited from pygfunction) for Single U-tube, Multiple U-tube and Coaxial borehole heat exchanger models.
* [Issue 5](https://github.com/j-c-cook/PLAT/issues/5) - Adds hybrid load object along with methods to process hourly ground heat extraction loads into a hybrid time step simulation format.
* [Issue 20](https://github.com/j-c-cook/PLAT/issues/20) - A function to create a doubling synthetic hybrid time step load is implemented for testing the hybrid load object.
* [Issue 31](https://github.com/j-c-cook/PLAT/issues/32) - A soil object is defined that inherits from `ThermalProperties`, and has a required input of undisturbed ground temperature.
* [Issue 33](https://github.com/j-c-cook/PLAT/issues/33) - A simulation parameters object is created that holds all instances unrelated to other aspects of GLHE design. The simulation time, heat pump entering fluid temperature constraints and the height constraints are required inputs.
* [Issue 38](https://github.com/j-c-cook/PLAT/issues/38) - Adds a method to the `HybridLoad` object that can create an hourly (8760 hours) load profile from the peak load analysis outputs, `PLAT.ground_loads.HybridLoad.hourly_load_representation`.
* [Issue 41](https://github.com/j-c-cook/PLAT/issues/40) - Adds `__repr__` string functions to each object so that creating an output of the parameters is made simple and reusable.
* [Issue 43](https://github.com/j-c-cook/PLAT/pull/43) - Adds update borehole thermal resistance functions to multiple U-tube and coaxial pipe borehole heat exchanger objects.

### Maintenance

* [Issue 10](https://github.com/j-c-cook/PLAT/issues/10) - The `pygfunction` package becomes a git submodule linked to a "long-lived" GLHEDT branch at https://github.com/j-c-cook/pygfunction/tree/GLHEDT.

### Refactors
* [Issue 30](https://github.com/j-c-cook/PLAT/issues/30) - Make the volumetric thermal capacity a required input for the soil, grout and pipe objects.
* [Issue 34](https://github.com/j-c-cook/PLAT/issues/34) - PLAT.borehole_heat_exchanger objects get type def recommendations for the inputs to the objects.
* [Issue 36](https://github.com/j-c-cook/PLAT/issues/36) - A `compute_resistance` function is created for the BHE objects, and is called from the __init__ class. This helps with updating the borehole resistance when values change.
* [Issue 37](https://github.com/j-c-cook/PLAT/issues/37) - The radial numerical short time step calculation takes the maximum of log time -8.6 and 49 hours for the calculation time. 

### Tests and CI

* [Issue 6](https://github.com/j-c-cook/PLAT/issues/6) - A github workflow file for running tests at each push is created. A self-hosted server runner is used to run the tests upon each push. The server is maintained by the developer.
* [Issue 7](https://github.com/j-c-cook/PLAT/issues/7) - An effective borehole resistance file has been created. Single and double U-tubes and coaxial heat exchangers are tested over a range of volumetric flow rates.
* [Issue 8](https://github.com/j-c-cook/PLAT/issues/8) - An equivalent geometries test file has been created. A test for a double U-tube to single U-tube equivalent is added.
* [Issue 17](https://github.com/j-c-cook/PLAT/issues/17) - A short time step testing module is created. Borehole heat exchangers containing single and double U-tube as well as concentric piping are computed.
* [Issue 25](https://github.com/j-c-cook/PLAT/issues/25) - A peak load analysis test file is created to check the differences of peak duration days for heating and cooling for all months. The load input is synthetic doubling, and the borehole heat exchanger is a single U-tube. Adequate coverage of equivalent borehole heat exchangers are already covered.