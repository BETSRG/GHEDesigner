# Purpose: Design a square or near-square field using the common design
# interface with a single U-tube, multiple U-tube and coaxial tube.

# This search is described in section 4.3.2 of Cook (2021) from pages 123-129.

from ghedesigner.manager import GHEManager
from ghedesigner.tests.ghe_base_case import GHEBaseTest
from ghedesigner.utilities import length_of_side


class TestFindNearSquareDesign(GHEBaseTest):

    def test_find_single_u_tube_design(self):
        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_diameter=0.03404, outer_diameter=0.04216,
            shank_spacing=0.01856, roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, diameter=0.140)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())

        b = 5.0
        number_of_boreholes = 32
        length = length_of_side(number_of_boreholes, b)
        ghe.set_geometry_constraints_near_square(b=b, length=length)  # borehole spacing and field side length
        ghe.set_design(flow_rate=0.3, flow_type_str="borehole")
        ghe.find_design()

        project_name = "Atlanta Office Building: Design Example"
        note = "Square-Near-Square Usage Example: Single U Tube"
        author = "John Doe"
        iteration_name = "Example 1"
        output_file_directory = self.test_outputs_directory / "TestFindDesignNearSquareSingleU"
        ghe.prepare_results(project_name, note, author, iteration_name)
        ghe.write_output_files(output_file_directory, "_SU")
        # can grab data off the outputs dict
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(125.0, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(157, len(nbh))

    def test_find_double_u_tube_parallel_design(self):
        ghe = GHEManager()
        ghe.set_double_u_tube_pipe_parallel(
            inner_diameter=0.03404, outer_diameter=0.04216, shank_spacing=0.01856,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, diameter=0.140)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())

        b = 5.0
        number_of_boreholes = 32
        length = length_of_side(number_of_boreholes, b)
        ghe.set_geometry_constraints_near_square(b=b, length=length)
        ghe.set_design(flow_rate=0.5, flow_type_str="borehole")
        ghe.find_design()

        project_name = "Atlanta Office Building: Design Example"
        note = "Square-Near-Square Usage Example: Double U Tube"
        author = "John Doe"
        iteration_name = "Example 1"
        output_file_directory = self.test_outputs_directory / "TestFindDesignNearSquareDoubleUParallel"
        ghe.prepare_results(project_name, note, author, iteration_name)
        ghe.write_output_files(output_file_directory, "_DU")

        # can grab data off the outputs dict
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(u_tube_height, 130.5, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(145, len(nbh))

    def test_find_double_u_tube_series_design(self):
        ghe = GHEManager()
        ghe.set_double_u_tube_pipe_series(
            inner_diameter=0.03404, outer_diameter=0.04216, shank_spacing=0.01856,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, diameter=0.140)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())

        b = 5.0
        number_of_boreholes = 32
        length = length_of_side(number_of_boreholes, b)
        ghe.set_geometry_constraints_near_square(b=b, length=length)
        ghe.set_design(flow_rate=0.5, flow_type_str="borehole")
        ghe.find_design()

        project_name = "Atlanta Office Building: Design Example"
        note = "Square-Near-Square Usage Example: Double U Tube"
        author = "John Doe"
        iteration_name = "Example 1"
        output_file_directory = self.test_outputs_directory / "TestFindDesignNearSquareDoubleUSeries"
        ghe.prepare_results(project_name, note, author, iteration_name)
        ghe.write_output_files(output_file_directory, "_DU")

        # can grab data off the outputs dict
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(u_tube_height, 130.7, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(145, len(nbh))

    def test_find_coaxial_pipe_design(self):
        ghe = GHEManager()
        ghe.set_coaxial_pipe(
            inner_pipe_d_in=0.0442, inner_pipe_d_out=0.050, outer_pipe_d_in=0.0974, outer_pipe_d_out=0.11,
            roughness=1.0e-6, conductivity_inner=0.4, conductivity_outer=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, diameter=0.140)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())

        b = 5.0
        number_of_boreholes = 32
        length = length_of_side(number_of_boreholes, b)
        ghe.set_geometry_constraints_near_square(b=b, length=length)
        ghe.set_design(flow_rate=0.8, flow_type_str="borehole")
        ghe.find_design()

        output_file_directory = self.test_outputs_directory / "TestFindRectangleDesignCoaxialUTube"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(122.7, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(145, len(nbh))
