# GHEDT - History of changes
 
## Current version

### Enhancements

* [Issue 12](https://github.com/j-c-cook/ghedt/issues/12) - The `GFunction` object can be passed into the `GHE` object with only one g-function calculated, as long as the `B/H` value being "interpolated" for is the same one that it was computed for. 

### New features

* [Issue 1](https://github.com/j-c-cook/ghedt/issues/1) - The `GFunction` object used for interpolation of g-functions based on B/H and borehole radius correction is pulled from the g-FunctionDatabase. 
* [Issue 3](https://github.com/j-c-cook/GLHEDT/issues/3) - Adds a GHE object with the ability to simulate and size with a hybrid time step using information found by the peak load analysis tool.
* [Issue 8](https://github.com/j-c-cook/GLHEDT/issues/8) - Adds a detailed hourly simulation so that the hybrid time step can be validated. 
* [Issue 9](https://github.com/j-c-cook/ghedt/issues/11) - A function that computes a live g-function using `pygfunction` is created. The functions default arguments are unequal segments, the equivalent solver and the mixed inlet fluid temperature boundary condition.
* [Issue 14](https://github.com/j-c-cook/ghedt/issues/14) - The `coordinates.py` module is moved from the g-Function Database repository to this package. 

### Tests and CI

* [Issue 15](https://github.com/j-c-cook/ghedt/issues/15) - A file is added for testing the sizing of a GHE using live-time g-function calculations that utilize the equivalent solver method and 8 unequal segment lengths along the borehole for Single U-tube, Double U-tube and Coaxial tube BHEs. 

