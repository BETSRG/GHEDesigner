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
            inner_radius=0.0108, outer_radius=0.013335,
            shank_spacing=0.0323, roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0
        )
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())

        # TODO: need to understand why this is here. this was just pulled from the original example file
        b = 5.0
        number_of_boreholes = 32
        length = length_of_side(number_of_boreholes, b)
        ghe.set_geometry_constraints_near_square(b=b, length=length)  # borehole spacing and field side length
        # perform a design search assuming "system" flow?
        ghe.set_design(flow_rate=6.4, flow_type="system", design_method_geo=ghe.DesignGeomType.NearSquare)
        ghe.find_design()

        # Output File Configuration  # TODO: Could add these to the manager constructor, or a set_meta_data method
        project_name = "Atlanta Office Building: Design Example"
        note = "Square-Near-Square Usage Example: Single U Tube"
        author = "John Doe"
        iteration_name = "Example 1"
        output_file_directory = self.test_outputs_directory / "TestFindDesignNearSquareSingleU"
        outputs = ghe.collect_outputs(project_name, note, author, iteration_name, output_file_directory, "_SU")

        # can grab data off the outputs dict
        u_tube_height = outputs['ghe_system']['active_borehole_length']
        self.assertAlmostEqual(u_tube_height, 124.92, delta=1e-2)
        # TODO: This was being checked, but I don't see this in the output structure, need to mine it out
        # selected_coordinates = outputs['ghe_system']['selected_coordinates']
        self.assertEqual(144, len(ghe._search.selected_coordinates))

    def test_find_double_u_tube_design(self):

        ghe = GHEManager()
        ghe.set_double_u_tube_pipe(inner_radius=0.0108, outer_radius=0.013335,
                                   shank_spacing=0.0323,
                                   roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())

        # TODO: need to understand why this is here. this was just pulled from the original example file
        b = 5.0
        number_of_boreholes = 32
        length = length_of_side(number_of_boreholes, b)
        ghe.set_geometry_constraints_near_square(b=b, length=length)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.NearSquare)
        ghe.find_design()

        # Output File Configuration  # TODO: Could add these to the manager constructor, or a set_meta_data method
        project_name = "Atlanta Office Building: Design Example"
        note = "Square-Near-Square Usage Example: Double U Tube"
        author = "John Doe"
        iteration_name = "Example 1"
        output_file_directory = self.test_outputs_directory / "TestFindDesignNearSquareDoubleU"
        outputs = ghe.collect_outputs(project_name, note, author, iteration_name, output_file_directory, "_SU")

        # can grab data off the outputs dict
        u_tube_height = outputs['ghe_system']['active_borehole_length']
        self.assertAlmostEqual(u_tube_height, 131.57, delta=1e-2)
        # TODO: This was being checked, but I don't see this in the output structure, need to mine it out
        # selected_coordinates = outputs['ghe_system']['selected_coordinates']
        self.assertEqual(144, len(ghe._search.selected_coordinates))

    def test_find_coaxial_pipe_design(self):

        ghe = GHEManager()
        ghe.set_coaxial_pipe(
            inner_pipe_r_in=0.0221, inner_pipe_r_out=0.025, outer_pipe_r_in=0.0487, outer_pipe_r_out=0.055,
            roughness=1.0e-6, conductivity_inner=0.4, conductivity_outer=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())

        # TODO: need to understand why this is here. this was just pulled from the original example file
        b = 5.0
        number_of_boreholes = 32
        length = length_of_side(number_of_boreholes, b)
        ghe.set_geometry_constraints_near_square(b=b, length=length)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.NearSquare)
        ghe.find_design()

        output_file_directory = self.test_outputs_directory / "TestFindRectangleDesignCoaxialUTube"
        outputs = ghe.collect_outputs("Project Name", "Notes", "Author", "Iteration Name", output_file_directory)
        u_tube_height = outputs['ghe_system']['active_borehole_length']
        self.assertAlmostEqual(124.78, u_tube_height, delta=0.01)
        # TODO: This was being checked, but I don't see this in the output structure, need to mine it out
        # selected_coordinates = outputs['ghe_system']['selected_coordinates']
        self.assertEqual(156, len(ghe._search.selected_coordinates))
