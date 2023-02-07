from ghedesigner.manager import GHEManager
from ghedesigner.tests.ghe_base_case import GHEBaseTest


class TestFindRectangleDesign(GHEBaseTest):
    def test_single_u_tube(self):
        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.013335, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_rectangular(length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.Rectangular)
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindRectangleDesignSingleUTube"
        outputs = ghe.collect_outputs("Project Name", "Notes", "Author", "Iteration Name", output_file_directory)
        u_tube_height = outputs['ghe_system']['active_borehole_length']
        self.assertAlmostEqual(123.26, u_tube_height, delta=0.01)
        # TODO: This was being checked, but I don't see this in the output structure, need to mine it out
        # selected_coordinates = outputs['ghe_system']['selected_coordinates']
        self.assertEqual(180, len(ghe._search.selected_coordinates))

    def test_double_u_tube(self):
        ghe = GHEManager()
        ghe.set_double_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.013335, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(length=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_rectangular(length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.Rectangular)
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindRectangleDesignDoubleUTube"
        outputs = ghe.collect_outputs("Project Name", "Notes", "Author", "Iteration Name", output_file_directory)
        u_tube_height = outputs['ghe_system']['active_borehole_length']
        self.assertAlmostEqual(127.66, u_tube_height, delta=0.01)
        # TODO: This was being checked, but I don't see this in the output structure, need to mine it out
        # selected_coordinates = outputs['ghe_system']['selected_coordinates']
        self.assertEqual(144, len(ghe._search.selected_coordinates))

    def test_coaxial_pipe(self):
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
        ghe.set_geometry_constraints_rectangular(length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.Rectangular)
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestFindRectangleDesignCoaxialUTube"
        outputs = ghe.collect_outputs("Project Name", "Notes", "Author", "Iteration Name", output_file_directory)
        u_tube_height = outputs['ghe_system']['active_borehole_length']
        self.assertAlmostEqual(132.58, u_tube_height, delta=0.01)
        # TODO: This was being checked, but I don't see this in the output structure, need to mine it out
        # selected_coordinates = outputs['ghe_system']['selected_coordinates']
        self.assertEqual(144, len(ghe._search.selected_coordinates))
