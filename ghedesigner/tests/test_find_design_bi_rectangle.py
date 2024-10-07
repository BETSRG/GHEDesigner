# Purpose: Design a constrained bi-rectangular field using the common design
# interface with a single U-tube, multiple U-tube and coaxial tube borehole
# heat exchanger.

# This search is described in section 4.4.2 of Cook (2021) from pages 134-138.

from ghedesigner.manager import GroundHeatExchanger
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestFindBiRectangleDesign(GHEBaseTest):
    def test_single_u_tube(self):
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
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_bi_rectangle(length=85.0, width=40.0, b_min=3.0, b_max_x=10.0, b_max_y=12.0)
        ghe.set_design(flow_rate=0.5, flow_type_str="borehole")
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindBiRectangleDesignSingleUTube"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(132.7, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(136, len(nbh))

    def test_double_u_tube(self):
        ghe = GroundHeatExchanger()
        ghe.set_double_u_tube_pipe_parallel(
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
        ghe.set_geometry_constraints_bi_rectangle(length=85.0, width=40.0, b_min=3.0, b_max_x=10.0, b_max_y=12.0)
        ghe.set_design(flow_rate=0.5, flow_type_str="borehole")
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindBiRectangleDesignDoubleUTube"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(133.0, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(116, len(nbh))

    def test_coaxial(self):
        ghe = GroundHeatExchanger()
        ghe.set_coaxial_pipe(
            inner_pipe_d_in=0.0442,
            inner_pipe_d_out=0.050,
            outer_pipe_d_in=0.0974,
            outer_pipe_d_out=0.11,
            roughness=1.0e-6,
            conductivity_inner=0.4,
            conductivity_outer=0.4,
            rho_cp=1542000.0,
        )
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, diameter=0.140)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_bi_rectangle(length=85.0, width=40.0, b_min=3.0, b_max_x=10.0, b_max_y=12.0)
        ghe.set_design(flow_rate=0.8, flow_type_str="borehole")
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindBiRectangleDesignCoaxial"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(133.0, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(101, len(nbh))
