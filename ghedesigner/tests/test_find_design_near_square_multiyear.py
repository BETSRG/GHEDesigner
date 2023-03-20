# Purpose: Design a square or near-square field using the common design
# interface with a single U-tube, multiple U-tube and coaxial tube.

# This search is described in section 4.3.2 of Cook (2021) from pages 123-129.

from ghedesigner.manager import GHEManager
from ghedesigner.tests.ghe_base_case import GHEBaseTest


class TestFindNearSquareMultiyearDesign(GHEBaseTest):

    # This file contains three examples utilizing the square-near-square design algorithm
    # (utilizing a multi-year loading) for a single U, double U, and coaxial tube  The
    # results from these examples are exported to the "DesignExampleOutput" folder.

    def test_multiyear_loading_single_u_tube(self):
        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.013335, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0
        )
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=48, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_multiyear_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "borehole" flow?
        ghe.set_design(flow_rate=0.2, flow_type_str="borehole")
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindNearSquareMultiyearDesignSingleU"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(128.61, u_tube_height, delta=0.01)
        selected_coordinates = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(16 + 1, len(selected_coordinates))

    def test_multiyear_loading_double_u_tube(self):
        ghe = GHEManager()
        ghe.set_double_u_tube_pipe_parallel(
            inner_radius=0.0108, outer_radius=0.013335, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=48, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_multiyear_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "borehole" flow?
        ghe.set_design(flow_rate=0.2, flow_type_str="borehole")
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindNearSquareMultiyearDesignDoubleU"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(116.12, u_tube_height, delta=0.01)
        selected_coordinates = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(16 + 1, len(selected_coordinates))

    def test_multiyear_loading_coaxial(self):
        ghe = GHEManager()
        ghe.set_coaxial_pipe(
            inner_pipe_r_in=0.0221, inner_pipe_r_out=0.025, outer_pipe_r_in=0.0487, outer_pipe_r_out=0.055,
            roughness=1.0e-6, conductivity_inner=0.4, conductivity_outer=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=48, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_multiyear_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "borehole" flow?
        ghe.set_design(flow_rate=0.2, flow_type_str="borehole")
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindNearSquareMultiyearDesignCoaxial"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(114.76, u_tube_height, delta=0.01)
        selected_coordinates = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(16 + 1, len(selected_coordinates))
