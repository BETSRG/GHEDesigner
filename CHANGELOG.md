# GHEDT - History of changes
 
## Current version

### Enhancements

* [Issue 12](https://github.com/j-c-cook/ghedt/issues/12) - The `GFunction` object can be passed into the `GHE` object with only one g-function calculated, as long as the `B/H` value being "interpolated" for is the same one that it was computed for.

### Fixes

* [Issue 23](https://github.com/j-c-cook/ghedt/issues/23) - Fixes the possibility for an extrapolation error to be thrown on the interpolation function at the outer bounds when floating point rounding causes the number to be out of bounds by 1e-16 or less. A check is put in place to compare the absolute value of the difference for the outer bounds, and if the value is within 1.0e-06, then the outer bound is made use of. 

### New features

* [Issue 1](https://github.com/j-c-cook/ghedt/issues/1) - The `GFunction` object used for interpolation of g-functions based on B/H and borehole radius correction is pulled from the g-FunctionDatabase. 
* [Issue 3](https://github.com/j-c-cook/GLHEDT/issues/3) - Adds a GHE object with the ability to simulate and size with a hybrid time step using information found by the peak load analysis tool.
* [Issue 8](https://github.com/j-c-cook/GLHEDT/issues/8) - Adds a detailed hourly simulation so that the hybrid time step can be validated. 
* [Issue 9](https://github.com/j-c-cook/ghedt/issues/11) - A function that computes a live g-function using `pygfunction` is created. The functions default arguments are unequal segments, the equivalent solver and the mixed inlet fluid temperature boundary condition.
* [Issue 14](https://github.com/j-c-cook/ghedt/issues/14) - The `coordinates.py` module is moved from the g-Function Database repository to this package.
* [Issue 16](https://github.com/j-c-cook/ghedt/issues/16) - An integer bisection search routine is implemented that, given a domain of coordinates specified at a target height, selects the borefield that results in an excess temperature less than or equal to 0.
* [Issue 17](https://github.com/j-c-cook/ghedt/issues/17) - A design routine that can determine a square or near square routine based on a range (ie. 1 to 3 is 1x1, 1x2, 2x2, 2x3, 3x3), and a target depth (maximum height).
* [Issue 18](https://github.com/j-c-cook/ghedt/issues/18) - The methodology for determining the `B` spacing for a borefield is one that compares the first two boreholes in the list. If there is one borehole, the borehole radius is set as the borehole spacing.
* [Issue 10](https://github.com/j-c-cook/ghedt/issues) - A rectangular design routine that requires length x width land area constraint, and minimum and maximum borehole spacings.
* [Issue 25](https://github.com/j-c-cook/ghedt/tree/issue25_BiRectangle) - A bi-rectangular search routine is implemented that requires the arguments in the rectangular design in addition to the maximum spacing along the shortest length.

### Tests and CI

* [Issue 15](https://github.com/j-c-cook/ghedt/issues/15) - A file is added for testing the sizing of a GHE using live-time g-function calculations that utilize the equivalent solver method and 8 unequal segment lengths along the borehole for Single U-tube, Double U-tube and Coaxial tube BHEs. 

