# Ground Heat Exchanger Design Toolbox

A package with the novel capability of automatically selecting ground heat 
exchanger configurations based on polygonal land constraints. This package 
contains advanced methods that are the first of their kind. The results are 
validated against the world renowned `GLHEPRO` (Cook 2021).

The Ground Heat Exchanger Design Toolbox (GHEDT) is a Python package that can 
quantify the short- and long-term thermal interaction in a ground heat exchanger 
(GHE). The long-term thermal response g-functions are computed live-time with 
[pygfunction][#pygfunction]. The GHEDT contains a fast monthly hybrid time step 
simulation for prediction of heat pump entering fluid temperatures over a design 
life. GHEDT can vary the size (or borehole height) of a specified borehole field 
to ensure the heat pump entering fluid temperature remains within specified 
bounds. GHEDT contains a novel borehole configuration selection algorithm.

## Novel Design Algorithms

GHEDT contains a novel design methodology for automated selection of borehole 
fields. The advanced methodology performs optimization based on a target 
drilling depth. An integer bisection routine is utilized to quickly search 
over a unimodal domain of boreholes. GHEDT can consider available drilling and 
no-drilling zones defined as polygons. 

The selection process shown below is performed in less than half a minute on an 
11th Gen Intel Core i9-11900K @ 3.50GHz. Refer to `Cook (2021)` for more 
information.

![polygonal](images/find_bi_alternative_03.gif)

## Requirements 

GHEDT requires at least Python 3.7 and is tested with Python 3.7 and 3.8. GHEDT 
is dependent on the following packages:

- pygfunction (>=2.1)
- numpy (>=1.19.2)
- scipy (>=1.6.2)
- matplotlib (>=3.3.4)
- coolprop (>=6.4.1)
- pandas (>=1.3.2)
- openpyxl (>=3.0.8)
- opencv-python (==4.5.4.58)

## Quick Start

**Users** - Install `ghedt` via the package installer for Python ([pip][#pip]):
```angular2html
pip install ghedt
```

**Developers** - Clone the repository to via git:
```angular2html
git clone https://github.com/j-c-cook/ghedt
```

See [installation](https://github.com/j-c-cook/ghedt/blob/main/INSTALLATION.md) 
for more notes on installing. See [ghedt/examples/](https://github.com/j-c-cook/ghedt/tree/main/ghedt/examples) 
for usage.   

## Citing GHEDT 

GHEDT and other related work is described in the following thesis: 

```angular2html
Cook, J.C. (2021). Development of Computer Programs for Fast Computation of 
    g-Functions and Automated Ground Heat Exchanger Design. Master's Thesis, 
    Oklahoma State University, Stillwater, OK.
```

Here is an example of a BibTeX entry:
```angular2html
@mastersthesis{Cook_2021,
school = "{Oklahoma State University, Stillwater, OK}",
author = {Cook, J C.},
language = {eng},
title = "{Development of Computer Programs for Fast Computation of g-Functions 
and Automated Ground Heat Exchanger Design}",
year = {2021},
}
```

## Questions?

If there are any questions, comments or concerns please [create][#create] an 
issue, comment on an [open][#issue] issue, comment on a [closed][#closed] issue, 
or [start][#start] a [discussion][#discussion]. 
  

## Acknowledgements
The initial release of this work (`ghedt-v0.1`) was financially supported by the 
U.S. Department of Energy through research subcontracts from Oak Ridge National 
Laboratory and the National Renewable Energy Laboratory, and by OSU through the 
Center for Integrated Building Systems, the OG&E Energy Technology Chair, and 
Oklahoma State University via return of indirect costs to Dr. Jeffrey D. 
Spitler.

[#pygfunction]: https://github.com/MassimoCimmino/pygfunction
[#pip]: https://pip.pypa.io/en/latest/
[#create]: https://github.com/j-c-cook/ghedt/issues/new
[#issue]: https://github.com/j-c-cook/ghedt/issues
[#closed]: https://github.com/j-c-cook/ghedt/issues?q=is%3Aissue+is%3Aclosed
[#start]: https://github.com/j-c-cook/ghedt/discussions/new
[#discussion]: https://github.com/j-c-cook/ghedt/discussions
