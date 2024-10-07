from ghedesigner.manager import GroundHeatExchanger
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestNearSquare(GHEBaseTest):
    def test_design_selection_system(self):
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
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "system" flow?
        ghe.set_design(flow_rate=31.2, flow_type_str="system")
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestDesignSelectionSystem"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(133.9, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(145, len(nbh))

    def test_design_selection_borehole(self):
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
        ghe.set_geometry_constraints_near_square(b=5.0, length=155)  # borehole spacing and field side length
        # perform a design search assuming "borehole" flow?
        ghe.set_design(flow_rate=0.5, flow_type_str="borehole")
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestDesignSelectionBorehole"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(127.9, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(157, len(nbh))
