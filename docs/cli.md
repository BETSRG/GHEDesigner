# Command Line Interface

This library comes with a command line interface. Once this library is [pip installed](index.md#quick-start), a new binary executable will be available with the name `ghedesigner`. The command has a help argument with output similar to this (execute manually to verify latest syntax):
```bash
  $ ghedesigner --help
  Usage: ghedesigner [OPTIONS] INPUT_PATH [OUTPUT_DIRECTORY]

  Options:
    --version           Show the version and exit.
    --validate-only     Validate input file and exit.
    -c, --convert TEXT  Convert output to specified format. Options supported:
                        'IDF'.
    --help              Show this message and exit.
```
