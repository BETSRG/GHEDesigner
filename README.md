# GHEDesigner - A Flexible and Automatic Ground Heat Exchanger Design Tool

GHEDesigner is a Python package for designing ground heat exchangers (GHE) used with ground source heat pump (GSHP)
systems. Compared to currently available tools such
as [GLHEPRO](https://betsrg.org/ground-loop-heat-exchanger-design-software), GHEDesigner:

- is flexible. It can synthesize borehole fields that are custom fit to the user's property description,
- implements the RowWise algorithm ([Spitler, et al. 2022a](https://doi.org/10.22488/okstate.22.000016)) for
  automatically placing and sizing boreholes in any land area with complex geometry,
- is highly automated. It can select library configurations or custom configurations and determine the final number and
  depth requirement of boreholes,
- can make automated conversion of hourly loads to an improved hybrid time step
  representation ([Cullin and Spitler 2011](https://doi.org/10.1016/j.geothermics.2011.01.001)), and
- is under continuing development at Oklahoma State University (OSU), Oak Ridge National Laboratory (ORNL), and the
  National Renewable Energy Laboratory (NREL). (GLHEPRO remains under development at OSU.)

## Background

GHEDesigner was originally funded through US Department of Energy contract DE‐AC05‐00OR22725 via a subcontract from Oak
Ridge National Laboratory. The project led by Dr. Xiaobing Liu developed an
online [screening tool](https://gshp.ornl.gov/login) ([Liu, et al. 2022](http://dx.doi.org/10.22488/okstate.22.000042))
for techno-economic analysis and design of ground-source heat pump systems. The Oklahoma State University team led by
Prof. Jeffrey D. Spitler was contracted to investigate fast methods for computing g-functions. An outgrowth of this
research was a tool for automatically selecting and sizing borehole configurations. This tool, originally called GHEDT,
is described in an MS thesis ([Cook 2021](https://hdl.handle.net/11244/335489)). Since that time, the tool has been
renamed GHEDesigner, and work has continued at Oklahoma State University, Oak Ridge National Laboratory, and the
National Renewable Energy Laboratory.

Updates since [Cook (2021)](https://hdl.handle.net/11244/335489) include:

- Development and addition of RowWise algorithm to efficiently place boreholes in the available land area.
- Extensive refactoring for creating a user-focused, stable API.
- Simplification of library dependencies.
- Development of automated testing and deployment procedures.

## Borehole Field Design Algorithms

- Long time-step g-functions are calculated using pygfunction (Cimmino 2018) using the equivalent borehole
  method ([Prieto and Cimmino 2021](https://doi.org/10.1080/19401493.2021.1968953)). It's also possible to read
  g-functions from a library ([Spitler, et al. 2021](https://doi.org/10.15121/1811518)).
- Borehole thermal resistance is computed for single and double U-tube configurations via the multipole
  method ([Claesson and Hellström 2011](https://doi.org/10.1080/10789669.2011.609927)). For coaxial ground heat
  exchangers, it is computed from fundamental heat transfer relationships.
- Short time-step g-functions are computed using the
  [Xu and Spitler (2006)](https://hvac.okstate.edu/sites/default/files/pubs/papers/2006/07-Xu_Spitler_06.pdf) method.
- GHEDesigner contains a novel design methodology for automated selection of borehole fields. The advanced methodology
  performs optimization based on a target drilling depth. An integer bisection routine is utilized to quickly search
  over a uni-modal domain of boreholes. GHEDesigner can consider the available land area for drilling and no-drilling
  zones defined as polygons.
- GHEDesigner can synthesize a range of regularly shaped borehole configurations, including previously available
  shapes (rectangles, open rectangles, L-shape, U-shape, line) and shapes not previously available (C-shapes and zoned
  rectangles). More information about these shapes can be found in the documentation for a publicly available g-function
  library. ([Spitler, et al. 2021](https://doi.org/10.15121/1811518), \
  [2022b](https://doi.org/10.22488/okstate.22.000040))
- GHEDesigner can synthesize on the fly irregularly shaped borehole configurations using the RowWise
  algorithm ([Spitler, et al. 2022a](https://doi.org/10.22488/okstate.22.000016)) or the bi-uniform polygonal constrained
  rectangular search (BUPCRS) ([Cook 2021](https://hdl.handle.net/11244/335489)). Both configurations are adapted to the
  user-specified property boundaries and no-drill zones, if any. [Spitler, et al. 2022a](https://doi.org/10.22488/okstate.22.000016)
  gives an example where the RowWise algorithm saves 12-18% compared to the BUPCRS algorithm. The RowWise algorithm takes
  longer to run, though.
- A set of search routines can be used to size different types of configurations:
    - The unconstrained square/near-square search will search a domain of square (*n* x *n*) and near-square
      (*n-1* x *n*) boreholes fields, with uniform spacing between the boreholes.
    - Uniform and bi-uniform constrained rectangular searches will search domains of rectangular configurations that
      have either uniform spacing or "bi-uniform" spacing – that is, uniform in the x direction and uniform in the y
      direction, but the two spacings may be different.
    - The bi-uniform constrained zoned rectangular search allows for rectangular configurations with different interior
      and perimeter spacings.
    - The bi-uniform polygonal constrained rectangular search (BUPCRS) can search configurations with an outer perimeter
      and no-go zones described as irregular polygons. This is still referred to as a rectangular search because it is
      still based on a rectangular grid, from which boreholes that are outside the perimeter or inside a no-go zone are
      removed.
    - The RowWise method generates and searches custom borehole fields that make full use of the available property. The
      RowWise algorithms are described by [Spitler et al. (2022a)](https://shareok.org/handle/11244/336846).

## Limitations

GHEDesigner does not have every feature that is found in a tool like GLHEPRO. Features that are currently missing
include:

- Heat pumps are not modeled. Users input heat rejection/extraction rates.
- An hourly simulation is available, but it doesn't make use of load aggregation, so is very slow.
- GHEDesigner only covers vertical borehole ground heat exchangers. Horizontal ground heat exchangers are not treated.
- GHEDesigner does not calculate the head loss in the ground heat exchanger or warn the user that head loss may be
  excessive.
- GHEDesigner does not have a graphical user interface.
- GHEDesigner is a Python package and requires some Python knowledge to use.

## Requirements

GHEDesigner is supported for Python versions >= 3.8, and is tested with Python 3.8 and 3.9. GHEDesigner is dependent on
the following packages:

- [click][#click] (>=8.1.3)
- [jsonschema][#jsonschema] (>=4.17.3)
- [numpy][#numpy] (>=1.24.2)
- [opencv-python][#opencv] (==4.7.0.68)
- [pygfunction][#pygfunction] (>=2.2.2)
- [scipy][#scipy] (>=1.10.0)

## Quick Start

**Users** - Install `GHEDesigner` via the package installer for Python ([pip][#pip]):

```
pip install ghedesigner
```

**Developers** - Clone the repository via git:

```
git clone https://github.com/betsrg/ghedesigner
```

## Questions

If there are any questions, comments or concerns please [create][#create] an issue, comment on an [open][#issue] issue,
comment on a [closed][#closed] issue.

## Acknowledgements

The initial release of this work was financially supported by the U.S. Department of Energy through research
subcontracts from Oak Ridge National Laboratory and the National Renewable Energy Laboratory, and by OSU through the
Center for Integrated Building Systems, the OG&E Energy Technology Chair, and Oklahoma State University via return of
indirect costs to Dr. Jeffrey D. Spitler.

## References

Cimmino, M. 2018. pygfunction: an open-source toolbox for the evaluation of thermal. eSim 2018, Montreál, IBPSA Canada.
492-501. http://www.ibpsa.org/proceedings/eSimPapers/2018/2-3-A-4.pdf

Claesson, J. and G. Hellström. 2011. Multipole method to calculate borehole thermal resistances in a borehole heat
exchanger. HVAC&R Research 17(6): 895-911. https://doi.org/10.1080/10789669.2011.609927

Cook, J.C. (2021). Development of Computer Programs for Fast Computation of g-Functions and Automated Ground Heat
Exchanger Design. Master's Thesis, Oklahoma State University, Stillwater, OK. https://hdl.handle.net/11244/335489

Cullin, J.R. and J.D. Spitler. 2011. A Computationally Efficient Hybrid Time Step Methodology for Simulation of Ground
Heat Exchangers. Geothermics. 40(2): 144-156. https://doi.org/10.1016/j.geothermics.2011.01.001

Liu X., J. Degraw, M. Malhotra, W. Forman, M. Adams, G. Accawi, B. Brass, N. Kunwar, J. New, J. Guo. 2022. Development
of a Web-based Screening Tool for Ground Source Heat Pump Applications. 2022. IGSHPA Research Conference Proceedings.
Pp. 280-290. December 6-8. Las Vegas. http://dx.doi.org/10.22488/okstate.22.000042

Prieto, C. and M. Cimmino. 2021. Thermal interactions in large irregular fields of geothermal boreholes: the method of
equivalent boreholes. Journal of Building Performance Simulation 14(4):
446-460. https://doi.org/10.1080/19401493.2021.1968953

Spitler, J. D., J. Cook, T. West and X. Liu 2021. G-Function Library for Modeling Vertical Bore Ground Heat Exchanger,
Oak Ridge National Laboratory. https://doi.org/10.15121/1811518

Spitler, J.D., T.N. West and X. Liu. 2022a. Ground Heat Exchanger Design Tool with RowWise Placement of Boreholes.
IGSHPA Research Conference Proceedings. Pp. 53-60. Las Vegas. Dec. 6-8. https://doi.org/10.22488/okstate.22.000016

Spitler, J.D., T.N. West, X. Liu and I. Borshon. 2022b. An open library of g-functions for 34,321 configurations. IGSHPA
Research Conference Proceedings. Pp. 264-271. Las Vegas. Dec. 6-8  https://doi.org/10.22488/okstate.22.000040

Xu, X. and J. D. Spitler. 2006. Modelling of Vertical Ground Loop Heat Exchangers with Variable Convective Resistance
and Thermal Mass of the Fluid. 10th International Conference on Thermal Energy Storage - Ecostock 2006, Pomona,
NJ. https://hvac.okstate.edu/sites/default/files/pubs/papers/2006/07-Xu_Spitler_06.pdf

[#pygfunction]: https://github.com/MassimoCimmino/pygfunction

[#numpy]: https://numpy.org/doc/stable/

[#scipy]: https://docs.scipy.org/doc/scipy/

[#opencv]: https://pypi.org/project/opencv-python/

[#click]: https://click.palletsprojects.com/en/8.1.x/

[#jsonschema]: https://pypi.org/project/jsonschema/

[#pip]: https://pip.pypa.io/en/latest/

[#create]: https://github.com/betsrg/ghedesigner/issues/new

[#issue]: https://github.com/betsrg/ghedesigner/issues

[#closed]: https://github.com/betsrg/ghedesigner/issues?q=is%3Aissue+is%3Aclosed
