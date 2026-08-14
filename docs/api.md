# Python API

The supported high-level Python interfaces mirror the command-line workflows. Modules beneath `ghedesigner.ghe.design`,
`ghedesigner.ghe.search`, and the district component classes are implementation details and may change as the numerical
methods evolve.

## Run and validate input files

::: ghedesigner.main.run
options:
show_source: false

::: ghedesigner.validate.validate_input_file
options:
show_source: false

## Media properties

::: ghedesigner.media.Fluid
options:
members: - get_fluid_type - update_props_with_new_temp
show_source: false

::: ghedesigner.media.Grout
options:
show_source: false

::: ghedesigner.media.Soil
options:
members: - as_dict - to_input
show_source: false

## Enumerations

::: ghedesigner.enums
options:
members: true
show_source: false
