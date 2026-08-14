# District systems

District simulations connect buildings, vertical GHEs, source/sink heat exchangers, and optional buried horizontal
piping through a one-pipe or two-pipe central loop. The complete topology and simulation controls are defined in a
schema-validated JSON input file.

For normal use, pass that file to the [command-line interface](cli.md). Programmatic callers can construct a
`GHEHPSystem`, call `size_and_simulate`, and then write the detailed simulation and optional search results with
`create_output`.

```python
from pathlib import Path

from ghedesigner.district_system import GHEHPSystem

input_path = Path("demos/simulate_1_pipe_1_ghe_1_bldg_district.json")
output_directory = Path("tmp")
output_directory.mkdir(exist_ok=True)

system = GHEHPSystem(input_path)
system.size_and_simulate()
system.create_output(output_directory / f"{input_path.stem}.csv")
```

## Public interface

::: ghedesigner.district_system.GHEHPSystem
options:
members: - size_and_simulate - create_output
show_source: false
