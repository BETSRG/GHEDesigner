# Standalone GHE design

`GroundHeatExchanger` coordinates the geometry search, thermal model, and borehole sizing for a standalone vertical
ground heat exchanger. Most users should create it from a validated input dictionary with
`GroundHeatExchanger.init_from_dictionary` instead of calling the constructor directly.

The [programmatic example](examples.md#programmatic-usage) shows a complete input-to-design workflow. The command-line
interface remains the simplest supported entry point when the inputs already live in a JSON file.

## Public interface

::: ghedesigner.ghe.manager.GroundHeatExchanger
options:
members: - init_from_dictionary - design_and_size_ghe - get_design_area - get_design_volume - get_g_function
show_source: false
