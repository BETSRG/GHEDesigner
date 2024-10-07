from pathlib import Path

from ghedesigner.building import Building
from ghedesigner.heat_pump import HeatPump
from ghedesigner.manager import GroundHeatExchanger
from ghedesigner.system import System
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestNewWorkflows(GHEBaseTest):

    def test_new_workflow(self):

        system = System()
        system.set_simulation_parameters(num_months=240)

        # read building loads
        heat_pump = HeatPump("load 1")
        building_loads_path = Path(__file__).parent / "test_data" / "ground_loads.csv"
        heat_pump.set_loads_from_file(building_loads_path)
        heat_pump.set_fixed_cop(3)
        heat_pump.do_sizing()

        # size building heat pump
        building = Building("building")
        building.add_heat_pump(heat_pump)

        # size ghe
        ghe = GroundHeatExchanger()
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
        ghe.set_simulation_parameters(num_months=240)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_rectangle(max_height=135, min_height=60, length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        ghe.set_design(flow_rate=0.5, flow_type_str="borehole", max_eft=35, min_eft=5)
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindRectangleDesignSingleUTube"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")

        # simulate hourly
        system.simulate()
