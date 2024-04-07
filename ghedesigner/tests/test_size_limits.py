from ghedesigner.manager import GHEManager
from ghedesigner.tests.ghe_base_case import GHEBaseTest


class TestFindNearSquareDesign(GHEBaseTest):

    def test_small_loads(self):
        ghe = GHEManager()

        # 1-1/4" in DR-11 HDPE
        ghe.set_single_u_tube_pipe(inner_diameter=0.03404, outer_diameter=0.04216, shank_spacing=0.01856,
                                   roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=3.493, rho_cp=2.5797E06, undisturbed_temp=10.0)

        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=152.4, buried_depth=2.0, diameter=0.152)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=213, min_height=60,
                                      continue_if_design_unmet=True)

        ghe.set_ground_loads_from_hourly_list([1.0e2] * 8760)

        ghe.set_geometry_constraints_near_square(b=6.096, length=20)
        ghe.set_design(flow_rate=1.0, flow_type_str="borehole")
        ghe.find_design()

        project_name = ""
        note = ""
        author = ""
        iteration_name = ""
        output_file_directory = self.test_outputs_directory / "TestSmallLoads"
        ghe.prepare_results(project_name, note, author, iteration_name)
        ghe.write_output_files(output_file_directory, "")
        # can grab data off the outputs dict
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(u_tube_height, 60, delta=1e-2)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(1 + 1, len(nbh))

    def test_big_loads(self):
        ghe = GHEManager()

        # 1-1/4" in DR-11 HDPE
        ghe.set_single_u_tube_pipe(inner_diameter=0.03404, outer_diameter=0.04216, shank_spacing=0.01856,
                                   roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=3.493, rho_cp=2.5797E06, undisturbed_temp=10.0)

        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=152.4, buried_depth=2.0, diameter=0.152)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=213, min_height=60,
                                      continue_if_design_unmet=True)

        ghe.set_ground_loads_from_hourly_list([1.0e6] * 8760)

        ghe.set_geometry_constraints_near_square(b=6.096, length=20)
        ghe.set_design(flow_rate=1.0, flow_type_str="borehole")
        ghe.find_design()

        project_name = ""
        note = ""
        author = ""
        iteration_name = ""
        output_file_directory = self.test_outputs_directory / "TestBigLoads"
        ghe.prepare_results(project_name, note, author, iteration_name)
        ghe.write_output_files(output_file_directory, "")
        # can grab data off the outputs dict
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(u_tube_height, 213, delta=1e-2)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(20 + 1, len(nbh))

    def test_big_loads_with_max_boreholes(self):
        ghe = GHEManager()

        # 1-1/4" in DR-11 HDPE
        ghe.set_single_u_tube_pipe(inner_diameter=0.03404, outer_diameter=0.04216, shank_spacing=0.01856,
                                   roughness=1.0e-6, conductivity=0.4, rho_cp=1542000.0)
        ghe.set_soil(conductivity=3.493, rho_cp=2.5797E06, undisturbed_temp=10.0)

        ghe.set_grout(conductivity=1.0, rho_cp=3901000.0)
        ghe.set_fluid()
        ghe.set_borehole(height=152.4, buried_depth=2.0, diameter=0.152)
        ghe.set_simulation_parameters(num_months=240, max_eft=35, min_eft=5, max_height=213, min_height=60,
                                      max_boreholes=100, continue_if_design_unmet=True)

        ghe.set_ground_loads_from_hourly_list([1.0e6] * 8760)

        ghe.set_geometry_constraints_near_square(b=6.096, length=100)
        ghe.set_design(flow_rate=1.0, flow_type_str="borehole")
        ghe.find_design()

        project_name = ""
        note = ""
        author = ""
        iteration_name = ""
        output_file_directory = self.test_outputs_directory / "TestBigLoadsWithMaxBoreholes"
        ghe.prepare_results(project_name, note, author, iteration_name)
        ghe.write_output_files(output_file_directory, "")
        # can grab data off the outputs dict
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(u_tube_height, 213, delta=1e-2)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(90 + 1, len(nbh))
