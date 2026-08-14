# Command Line Interface

This library comes with a command line interface. Once this library is [pip installed](index.md#quick-start), a new
binary executable is available with the name `ghedesigner`. The command accepts an input JSON file and, for runs that
produce output, an output directory.

```bash
$ ghedesigner --help
Usage: ghedesigner [OPTIONS] INPUT_PATH [OUTPUT_DIRECTORY]

  Run, validate, or convert a GHEDesigner input or result file.

Options:
  --version           Show the version and exit.
  --validate-only     Validate input file and exit.
  -c, --convert TEXT  Convert output to specified format. Options supported:
                      'IDF'.
  --help              Show this message and exit.
```

Run a design or simulation:

```bash
ghedesigner demos/find_design_rectangle_single_u_tube.json ./tmp
```

Validate an input file without creating outputs:

```bash
ghedesigner --validate-only demos/find_design_rectangle_single_u_tube.json
```

Convert a GHEDesigner simulation summary output to EnergyPlus IDF objects:

```bash
ghedesigner --convert IDF path/to/SimulationSummary.json
```
