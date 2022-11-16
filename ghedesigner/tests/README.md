# Design (Automated borehole configuration selection)

The examples provided here are described in section 4.4 of Cook (2021).

## Find near square

The search routine utilized in the example file `near_square.py` is described
in section 4.3.2 from pages 123-129 in Cook (2021).

![near-square](../../../images/near-square.png)

## Find rectangle

The search routine utilized in the example file `find_rectangle.py` is
described in section 4.4.1 from pages 129-133 in Cook (2021).

![rectangular](../../../images/rectangular.png)

## Find bi-rectangle

The search routine utilized in the example file `find_bi_rectangle.py` is
described in section 4.4.2 from pages 134-138 in Cook (2021).

![bi-rectangular](../../../images/bi-rectangular.png)

## Find bi-zoned rectangle

The search routine utilized in the example file `find_bi_zoned_rectangle.py`
is described in section 4.4.3 from pages 138-143 in Cook (2021).

![bi-zoned](../../../images/bi-zoned.png)

# Examples

Welcome to `GHEDesigner` examples. The following is a file structure overview of what
is contained here. See below for a description of what is contained in the
current levels documents and folders.

```angular2html
├── Atlanta_Office_Building_Loads.csv
├── BHE
│.... ├── BoreholeResistance
│.... │.... ├── Coaxial.py
│.... │.... ├── DoubleUTube.py
│.... │.... ├── GLHEPRO.xlsx
│.... │.... ├── SingleUTube.py
│.... │.... └── validation.py
│.... └── EquivalentPipes
│....     ├── coaxial_to_single_u_tube.py
│....     └── double_to_single_u_tube.py
├── Design
│.... ├── find_bi_rectangle.py
│.... ├── find_bi_zoned_rectangle.py
│.... ├── find_near_square.py
│.... ├── find_rectangle.py
│.... └── FromFile (For Oak Ridge National Lab internal use)
│....     ├── ...
├── gFunctions
│.... ├── computed_g_function_sim_and_size.py
│.... ├── GLHEPRO_gFunctions_12x13.json
│.... └── live_g_function_sim_and_size.py
```

## Ground Loads File

This package takes hourly ground loads as ann input.

```angular2html
├── Atlanta_Office_Building_Loads.csv - A load file containing 8760 hourly ground loads.
```

## Borehole Heat Exchangers

A borehole heat exchanger object inherits from `pygfunction/pipes.py` objects.
The required inputs for a borehole heat exchanger (BHE) are mass flow rate,
fluid thermal properties, borehole geometric values (x, y, D, H, r_b), pipe
thermal properties and geometry, grout thermal properties and soil thermal
properties. The multiple u-tubes can be configured with serial or parallel flow.

```angular2html
├── BHE - Borehole Heat Exchanger
│.... ├── BoreholeResistance - These examples show how to compute effective borehole thermal resistances with various geometries.
│.... │.... ├── Coaxial.py
│.... │.... ├── DoubleUTube.py
│.... │.... ├── GLHEPRO.xlsx
│.... │.... ├── SingleUTube.py
│.... │.... └── validation.py
│.... └── EquivalentPipes - Equivalent pipes are currently necessary for EnergyPlus to simulate non-single U-tube BHEs.
│....     ├── coaxial_to_single_u_tube.py
│....     └── double_to_single_u_tube.py
```

## Design

The design examples showcase the automatic borehole selection process of
`GHEDesigner`. The following search routines are currently show cased:

- Find near square - The algorithm utilized in this example finds a square or
  near-square field (N x N or N x N+1).
- Find rectangle - Shows how to automatically find a rectangle within length and
  width geometric constraints and will contain uniform spacing in the x- and
  y-direction.
- Find bi-rectangle - Shows how to automatically select a bi-rectangular
  (different x- and y-spacing) borehole layout based on length and width
  constraints.
- Find bi-zoned rectangle - Shows how to automatically select a bi-rectangular
  zoned layout. The zoned rectangle contains more boreholes around the perimeter
  and less in the interior.

```angular2html
├── Design
│.... ├── find_bi_rectangle.py
│.... ├── find_bi_zoned_rectangle.py
│.... ├── find_near_square.py
│.... ├── find_rectangle.py
│.... └── FromFile (For Oak Ridge National Lab internal use)
│....     ├── ...
```

## g-Functions

These examples showcase how to use a custom g-function to simulate and size a
particular ground heat exchanger. The g-functions can be computed live time
via `pygfunction` or can be read from a json file (The format of the json file
was the result of the output of `cpgfunction`. The g-functions can come from any
g-function generation program, but must be properly formatted.)

```angular2html
├── gFunctions
│.... ├── computed_g_function_sim_and_size.py
│.... ├── GLHEPRO_gFunctions_12x13.json
│.... └── live_g_function_sim_and_size.py
```
