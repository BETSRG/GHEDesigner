from ghedesigner.manager import GHEManager
from ghedesigner.tests.ghe_base_case import GHEBaseTest


class TestLoadAggregation(GHEBaseTest):
    def setUp(self):
        super().setUp()
        self.ghe_manager = GHEManager()
        self.ghe_manager.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        self.ghe_manager.set_grout(conductivity=1.0, rho_cp=3901000.0)
        self.ghe_manager.set_fluid()
        self.ghe_manager.set_borehole(height=96.0, buried_depth=2.0, diameter=0.140)
        self.ghe_manager.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        self.ghe_manager.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        self.ghe_manager.set_geometry_constraints_rectangle(length=85.0, width=36.5, b_min=3.0, b_max=10.0)

    def test_single_u_tube(self):
        self.ghe_manager.set_single_u_tube_pipe(
            inner_diameter=0.03404, outer_diameter=0.04216, shank_spacing=0.01856,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        self.ghe_manager.set_design(flow_rate=0.5, flow_type_str="borehole", timestep="HOURLY")
        self.ghe_manager.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindRectangleDesignSingleUTubeLoadAggregation"
        self.ghe_manager.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        self.ghe_manager.write_output_files(output_file_directory, "")
        u_tube_height = self.ghe_manager.results.output_dict["ghe_system"]["active_borehole_length"]["value"]
        self.assertAlmostEqual(134.12, u_tube_height, delta=0.1)
        nbh = self.ghe_manager.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(145, len(nbh))

    def test_double_u_tube(self):
        self.ghe_manager.set_double_u_tube_pipe_parallel(
            inner_diameter=0.03404, outer_diameter=0.04216, shank_spacing=0.01856,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        self.ghe_manager.set_design(flow_rate=0.5, flow_type_str="borehole", timestep="HOURLY")
        self.ghe_manager.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindRectangleDesignDoubleUTubeLoadAggregation"
        self.ghe_manager.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        self.ghe_manager.write_output_files(output_file_directory, "")
        u_tube_height = self.ghe_manager.results.output_dict["ghe_system"]["active_borehole_length"]["value"]
        self.assertAlmostEqual(127.10, u_tube_height, delta=0.1)
        nbh = self.ghe_manager.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(145, len(nbh))

    def test_coaxial_pipe(self):
        self.ghe_manager.set_coaxial_pipe(
            inner_pipe_d_in=0.0442, inner_pipe_d_out=0.050, outer_pipe_d_in=0.0974, outer_pipe_d_out=0.11,
            roughness=1.0e-6, conductivity_inner=0.4, conductivity_outer=0.4, rho_cp=1542000.0)
        self.ghe_manager.set_design(flow_rate=0.8, flow_type_str="borehole", timestep="HOURLY")
        self.ghe_manager.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindRectangleDesignCoaxialUTubeLoadAggregation"
        self.ghe_manager.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        self.ghe_manager.write_output_files(output_file_directory, "")
        u_tube_height = self.ghe_manager.results.output_dict["ghe_system"]["active_borehole_length"]["value"]
        self.assertAlmostEqual(134.80, u_tube_height, delta=0.1)
        nbh = self.ghe_manager.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(106, len(nbh))

    def test_single_u_tube_no_load_agg(self):
        self.ghe_manager.set_simulation_parameters(num_months=12, max_eft=35, min_eft=5, max_height=135, min_height=60)
        self.ghe_manager.set_single_u_tube_pipe(
            inner_diameter=0.03404, outer_diameter=0.04216, shank_spacing=0.01856,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        self.ghe_manager.set_design(flow_rate=0.5, flow_type_str="borehole", timestep="HOURLYNOLOADAGG")
        self.ghe_manager.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindRectangleDesignSingleUTubeNoLoadAggregation"
        self.ghe_manager.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        self.ghe_manager.write_output_files(output_file_directory, "")
        u_tube_height = self.ghe_manager.results.output_dict["ghe_system"]["active_borehole_length"]["value"]
        self.assertAlmostEqual(119.72, u_tube_height, delta=0.1)
        nbh = self.ghe_manager.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(56, len(nbh))
