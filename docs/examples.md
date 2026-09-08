# Examples

## Command Line Usage

Demo input files are available in the repository's [`demos`](https://github.com/BETSRG/GHEDesigner/tree/main/demos)
directory. A standalone GHE design can be run with:

```bash
ghedesigner demos/find_design_rectangle_single_u_tube.json ./tmp
```

Validate an input file without running a design or simulation:

```bash
ghedesigner --validate-only demos/find_design_rectangle_single_u_tube.json
```

Convert a GHEDesigner output summary to EnergyPlus IDF objects:

```bash
ghedesigner --convert IDF path/to/SimulationSummary.json
```

## Programmatic Usage

The public input-file path used by the command line interface can also be used from Python. This example loads one of
the demo files, creates a `GroundHeatExchanger` from the JSON dictionary, and runs the same design workflow used by the
CLI for standalone GHE sizing.

```python
from pathlib import Path

from ghedesigner.constants import MONTHS_IN_YEAR
from ghedesigner.ghe.manager import GroundHeatExchanger
from ghedesigner.utilities import load_input_file

inputs = load_input_file(Path("demos/find_design_rectangle_single_u_tube.json"))
ghe_name = next(iter(inputs["ground_heat_exchanger"]))
ghe_dict = inputs["ground_heat_exchanger"][ghe_name]
ghe_dict["name"] = ghe_name

manager = GroundHeatExchanger.init_from_dictionary(ghe_dict, inputs["fluid"], soil_inputs=inputs["soil"])
end_month = inputs["simulation_control"]["sizing_years"] * MONTHS_IN_YEAR
search, search_time, found_ghe = manager.design_and_size_ghe(end_month, ghe_dict=ghe_dict)

design_summary = found_ghe.as_dict()
print(design_summary["number_of_boreholes"], design_summary["borehole_depth"], search_time)
```

For district systems, use the same JSON files through the CLI or create a `GHEHPSystem` with an input file path.

## Useful Demo Files

| File                                                           | Feature shown                                                                         |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `find_design_rectangle_single_u_tube.json`                     | Standalone vertical borehole GHE sizing from direct GHE loads.                        |
| `find_design_rectangle_single_u_tube_bldg_loads.json`          | Single-building sizing with fixed COP conversion from building loads to GHE loads.    |
| `simulate_1_pipe_1_ghe_1_bldg_district.json`                   | One-pipe district simulation using heat pump performance data and a pre-designed GHE. |
| `simulate_2_pipe_3_ghe_6_bldg_district_HOURLY.json`            | Two-pipe district simulation using hourly loads and fixed COP conversion.             |
| `simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json` | Canonical-network schema example with isolated and coupled buried horizontal piping.  |

## Input Features

Version 4 input files use the JSON schema in `ghedesigner/schemas/ghedesigner.schema.json`. Common feature switches
include:

- `simulation_control.load_method`: `HYBRID` aggregates loads for faster GHE calculations; `HOURLY` solves hourly loads
  directly.
- `soil`: defines the ground conductivity, volumetric heat capacity, and undisturbed temperature shared by every
  vertical GHE and horizontal pipe in the input. Horizontal simulations add a nested `soil.ground_temperature_model`
  containing the seasonal amplitudes and phase lags; `soil.undisturbed_temp` is its annual-average temperature.
- `simulation_control.search_method`: `GLOBAL_BUPCRS` and `NELDER-MEAD` size system GHEs; `SIMULATION_ONLY` simulates
  pre-designed GHEs.
- `network.type`: `one_pipe` and `two_pipe` select an ordered, unidirectional district-loop topology.
- Network flow is prescribed from building loads. GHEDesigner allocates GHE flow in proportion to each GHE's total
  design flow and evaluates Darcy-Weisbach pressure losses afterward for reporting and pump-energy calculations.
  Pressure loss does not allocate network flow.
- A one-pipe network uses `network.mass_flow_control` to set the distribution-flow multiplier and
  minimum circulation rate.
- Compact one-pipe bypass branches are generated automatically for every station and are treated as zero-loss paths.
  Distribution-segment losses are calculated from pipe geometry. A GHE's `circulation_pump` data describes only its
  local borefield and header pressure loss.
- Building loads can reference a heat pump performance map with `heat_pump_name` or use fixed COP conversion with
  `heat_pump_cop`.
- Ordered distribution segments can reference an isolated model in `horizontal_piping` when
  `soil.ground_temperature_model` and `simulation_control.horizontal_simulation_considered` are set.
