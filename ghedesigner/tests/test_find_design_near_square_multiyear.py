# Purpose: Design a square or near-square field using the common design
# interface with a single U-tube, multiple U-tube and coaxial tube.

# This search is described in section 4.3.2 of Cook (2021) from pages 123-129.

from ghedesigner.manager import GHEManager
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestFindNearSquareMultiyearDesign(GHEBaseTest):
    # This file contains three examples utilizing the square-near-square design algorithm
    # (utilizing a multi-year loading) for a single U, double U, and coaxial tube  The
    # results from these examples are exported to the "DesignExampleOutput" folder.

    def test_multiyear_loading_single_u_tube(self):
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
        ghe.set_simulation_parameters(num_months=48, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_multiyear_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "borehole" flow?
        ghe.set_design(flow_rate=0.5, flow_type_str="borehole")
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindNearSquareMultiyearDesignSingleU"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(132.7, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(17, len(nbh))

    def test_multiyear_loading_double_u_tube(self):
        ghe = GHEManager()
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
        ghe.set_simulation_parameters(num_months=48, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_multiyear_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "borehole" flow?
        ghe.set_design(flow_rate=0.5, flow_type_str="borehole")
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindNearSquareMultiyearDesignDoubleU"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(119.16, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(17, len(nbh))

    def test_multiyear_loading_coaxial(self):
        ghe = GHEManager()
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
        ghe.set_simulation_parameters(num_months=48, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_multiyear_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "borehole" flow?
        ghe.set_design(flow_rate=0.8, flow_type_str="borehole")
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindNearSquareMultiyearDesignCoaxial"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(109.8, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(13, len(nbh))
