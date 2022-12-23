Command Line Interface
======================

This library comes with a command line interface. Once this library is pip installed, a new binary executable will be available with the name ``ghedesigner``. The command has a help argument with output similar to this (execute manually to verify latest syntax)::

  $ ghedesigner --help
  Usage: ghedesigner [OPTIONS] INPUT_PATH OUTPUT_PATH

  Options:
    --help  Show this message and exit.

Example input file::

    {
      "version": "0.1",
      "fluid": {
        "fluid_name": "Water",
        "concentration_percent": 0.0
      },
      "grout": {
        "conductivity": 1.0,
        "rho_cp": 3901000
      },
      "soil": {
        "conductivity": 2.0,
        "rho_cp": 2343493,
        "undisturbed_temp": 18.3
      },
      "pipe": {
        "inner_radius": 0.0108,
        "outer_radius": 0.0133,
        "shank_spacing": 0.0323,
        "roughness": 0.000001,
        "conductivity": 0.4,
        "rho_cp": 1542000
      },
      "borehole": {
        "length": 96.0,
        "buried_depth": 2.0,
        "radius": 0.075
      },
      "simulation": {
        "num_months": 240,
        "max_eft": 35.0,
        "min_eft": 5.0,
        "max_height": 135.0,
        "min_height": 60.0
      },
      "geometric_constraints": {
        "b": 5.0,
        "length": 155
      },
      "design": {
        "flow_rate": 31.2,
        "flow_type": "system"
      },
      "ground_loads": [
        0,
        100,
        0,
        -100,
        ...
        ]
    }


Most input fields are self explanatory. However, a few details are given here to clarify what may not be readily apparent.

``fluid.fluid_name`` -- The name of the fluid mixture used. Valid options are "Water", "Propylene Glycol", "Ethylene Glycol", "Ethyl Alcohol", and "Methyl Alcohol". See the `SCP <https://secondarycoolantprops.readthedocs.io/en/latest/index.html>`_ fluid properties library for other supported options.

``fluid.concentration_percent`` -- The fluid mixture mass concentration in percent of mixed fluid relative to water. Values of 0-60 are expected to be supported as of current writing. See the `SCP <https://secondarycoolantprops.readthedocs.io/en/latest/index.html>`_ fluid properties library for further clarification.

``grout.conductivity`` -- The conductivity of the grouting material, in units of W/m-K.

``grout.rho_cp`` -- The volumetric heat capacity of the grouting material, in units of J/m^3-K

``soil.conductivity`` -- The conductivity of the soil, in units of W/m-K.

``soil.rho_cp`` -- The volumetric heat capacity of the soil, in units of J/m^3-K

``soil.undisturbed_temp`` -- The undisturbed average soil temperature, in units of degrees Celsius.

``pipe.inner_radius`` -- The radius of the inner pipe surface, in meters.

``pipe.outer_radius`` -- The radius of the outer pipe surface, in meters.

``pipe.shank_spacing`` -- The spacing between the U-tube legs, as referenced from outer surface of the pipes (i.e. not referenced from each pipes respective centerline), in meters.

``pipe.roughness`` -- The surface roughness of the pipe, in meters.

``pipe.conductivity`` -- The conductivity of the pipe material, in W/m-K.

``pipe.rho_cp`` -- The volumetric heat capacity of the pipe material, in J/m^3-K

``borehole.length`` -- The length of the borehole, in meters. (TODO: need to investigate why this is here when design length is an output of the sizing.)

``borehole.buried_depth`` -- The depth below the ground surface to the top of the borehole, in meters.

``borehole.radius`` -- The radius of the borehole in

``simulation.num_months`` -- The duration of the sizing period, in months.

``simulation.max_eft`` -- The maximum heat pump entering fluid temperature, in degrees Celsius.

``simulation.min_eft`` -- The minimum heat pump entering fluid temperature, in degrees Celsius.

``simulation.max_height`` -- The maximum active borehole length, in meters.

``simulation.min_height`` -- The minimum active borehole length, in meters.

``geometric_constraints.b`` -- The borehole-to-borehole spacing used in system design, in meters.

``geometric_constraints.length`` -- (TODO: figure out why is this length also needed)

``design.flow_rate`` -- The nominal fluid flow rate, in kg/s, on a per-borehole or system basis, as specified by the ``design.flow_type`` field.

``design.flow_type`` -- The fluid flow rate type. Accepted options are "System", or "Borehole". (TODO: verify)

``ground_loads`` -- The ground heat exchanger loads, in Watts. Positive values indicate heat extraction.
