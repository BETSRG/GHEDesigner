# Examples

## API Example Usage

A basic programmatic example is shown below. Additional examples are found at [https://github.com/BETSRG/GHEDesigner/tree/main/ghedesigner/tests](https://github.com/BETSRG/GHEDesigner/tree/main/ghedesigner/tests)

```python
ghe = GHEManager()
ghe.set_single_u_tube_pipe(
    inner_diameter=0.03404,
    outer_diameter=0.04216,
    shank_spacing=0.01856,
    roughness=1.0e-6,
    conductivity=0.4,
    rho_cp=1542000.0,
)
ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
ghe.set_fluid()
ghe.set_borehole(height=96.0, buried_depth=2.0, diameter=0.140)
ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
ghe.set_geometry_constraints_rectangle(length=85.0, width=36.5, b_min=3.0, b_max=10.0)
ghe.set_design(flow_rate=0.5, flow_type_str="borehole")
ghe.find_design()
```

## Command Line Example Usage

A basic command line example is shown below. Demo files can be found at [https://github.com/BETSRG/GHEDesigner/tree/main/demos](https://github.com/BETSRG/GHEDesigner/tree/main/demos)
```bash
  $ ghedesigner path/to/my_file.json path/to/output_dir/
```
