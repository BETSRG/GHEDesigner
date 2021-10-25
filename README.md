# Ground Heat Exchanger Design Toolbox

The Ground Heat Exchanger Design Toolbox (GHEDT) is a Python package that 
provides the tools necessary to quantify the short and long-term thermal 
interaction in a GHE. GHEDT can predict heat pump entering fluid temperatures 
for design over the design life. It can determine the required borehole depth 
of a specified borehole field to ensure the heat pump entering fluid temperature 
remains within specified bounds. GHEDT is novel in its ability to select a 
borehole field layout based on a target depth.

## Usage
Clone the repository.
```angular2html
git clone https://github.com/j-c-cook/GLHEDT
```
Recursively update the submodules.
```angular2html
 git submodule update --init --recursive
```
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
pip install GHEDT.zip
```