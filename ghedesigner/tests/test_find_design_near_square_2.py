from ghedesigner.manager import GHEManager
from ghedesigner.tests.ghe_base_case import GHEBaseTest


class TestNearSquare(GHEBaseTest):

    def test_design_selection_system(self):
        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.013335, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0
        )
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "system" flow?
        ghe.set_design(flow_rate=31.2, flow_type="system", design_method_geo=ghe.DesignGeomType.NearSquare)
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestDesignSelectionSystem"
        outputs = ghe.collect_outputs("Project Name", "Notes", "Author", "Iteration Name", output_file_directory)
        u_tube_height = outputs['ghe_system']['active_borehole_length']
        self.assertAlmostEqual(130.27, u_tube_height, delta=0.01)
        # TODO: This was being checked, but I don't see this in the output structure, need to mine it out
        # selected_coordinates = outputs['ghe_system']['selected_coordinates']
        self.assertEqual(156, len(ghe._search.selected_coordinates))

    def test_design_selection_borehole(self):
        ghe = GHEManager()
        ghe.set_single_u_tube_pipe(
            inner_radius=0.0108, outer_radius=0.013335, shank_spacing=0.0323,
            roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0
        )
        ghe.set_soil(conductivity=2.0, rho_cp=2343493.0, undisturbed_temp=18.3)
        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=96.0, buried_depth=2.0, radius=0.075)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=135, min_height=60)
        ghe.set_ground_loads_from_hourly_list(self.get_atlanta_loads())
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "borehole" flow?
        ghe.set_design(flow_rate=0.2, flow_type="borehole", design_method_geo=ghe.DesignGeomType.NearSquare)
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestDesignSelectionBorehole"
        outputs = ghe.collect_outputs("Project Name", "Notes", "Author", "Iteration Name", output_file_directory)
        u_tube_height = outputs['ghe_system']['active_borehole_length']
        self.assertAlmostEqual(130.27, u_tube_height, delta=0.01)
        # TODO: This was being checked, but I don't see this in the output structure, need to mine it out
        # selected_coordinates = outputs['ghe_system']['selected_coordinates']
        self.assertEqual(156, len(ghe._search.selected_coordinates))
