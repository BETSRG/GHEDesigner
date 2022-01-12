# INSTALLATION

## Installation from source

This section is for developers who wish to modify and contribute to `ghedt`.

### Install git

Version control of this repository is maintained via [git][#git]. To install 
`git`, see [git-downloads][#git-downloads]. For information about git not 
provided later in this section, refer to the free online [book][#git-book].

### Clone the repository

Clone and then change directories into the repository.
```angular2html
git clone https://github.com/j-c-cook/ghedt && cd ghedt
```
The Python package `pygfunction` is maintained as a `git submodule` and is 
pointing to a long-lived branch at [j-c-cook:ghedt][#pyg-branch]. 

Recursively update the submodule.
```angular2html
 git submodule update --init --recursive
```

### Create a Virtual Environment for ghedt

Python [Virtual Environments][#VirtualEnvironments] are how packages are 
accessed. 

Create environment that GHEDT can be installed to. 
```angular2html
conda create -n ENV python=3.7
```
Activate the environment. 
```angular2html
conda activate ENV
```
Install GHEDT to the environment.
```angular2html
cd ghedt/
pip install .
```
or if the package is zipped. 
```angular2html
pip install ghedt.zip
```

## Project Overview

```angular2html
├── ghedt
│   ├── __init__.py 
│   ├── coordinates.py
│   ├── design.py
│   ├── domains.py
│   ├── feature_recognition.py
│   ├── gfunction.py
│   ├── ground_heat_exchangers.py
│   ├── media.py
│   ├── peak_load_analysis_tool
│   │   ├── ...
│   ├── pygfunction
│   │   ├── ...
│   ├── search_routines.py
│   └── utilities.py
├── LICENSE
├── README.md
├── requirements.txt
├── setup.py
```

[#git]: https://en.wikipedia.org/wiki/Git
[#git-downloads]: https://git-scm.com/downloads
[#git-book]: https://git-scm.com/book/en/v2
[#VirtualEnvironments]: https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/
[#pyg-branch]: https://github.com/j-c-cook/pygfunction/tree/ghedt