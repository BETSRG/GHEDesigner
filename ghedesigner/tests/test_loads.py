from ghedesigner.manager import GroundHeatExchanger
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestLoads(GHEBaseTest):
    def test_balanced_loads(self):
        num_hr_in_month = 730
        load = 20000
        jan = feb = mar = octo = nov = dec = [load] * num_hr_in_month
        apr = may = jun = jul = aug = sept = [-load] * num_hr_in_month
        all_months = [jan, feb, mar, apr, may, jun, jul, aug, sept, octo, nov, dec]
        hourly_loads = [val for month in all_months for val in month]

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
        ghe.set_borehole(buried_depth=2.0, diameter=0.140)
        ghe.set_simulation_parameters(num_months=240)
        ghe.set_ground_loads_from_hourly_list(hourly_loads)
        ghe.set_geometry_constraints_rectangle(max_height=135, min_height=60, length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        ghe.set_design(flow_rate=0.5, flow_type_str="borehole", max_eft=35, min_eft=5)
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestBalancedLoads"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(118.53, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(7, len(nbh))

    def test_imbalanced_heating_loads(self):
        num_hr_in_month = 730
        load = 20000
        jan = feb = mar = octo = nov = dec = [load] * num_hr_in_month
        apr = may = jun = jul = aug = sept = [load] * num_hr_in_month
        all_months = [jan, feb, mar, apr, may, jun, jul, aug, sept, octo, nov, dec]
        hourly_loads = [val for month in all_months for val in month]

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
        ghe.set_borehole(buried_depth=2.0, diameter=0.140)
        ghe.set_simulation_parameters(num_months=240)
        ghe.set_ground_loads_from_hourly_list(hourly_loads)
        ghe.set_geometry_constraints_rectangle(max_height=135, min_height=60, length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        ghe.set_design(flow_rate=0.5, flow_type_str="borehole", max_eft=35, min_eft=5)
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestImbalancedHeatingLoads"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(130.40, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(11, len(nbh))

    def test_imbalanced_cooling_loads(self):
        num_hr_in_month = 730
        load = 20000
        jan = feb = mar = octo = nov = dec = [-load] * num_hr_in_month
        apr = may = jun = jul = aug = sept = [-load] * num_hr_in_month
        all_months = [jan, feb, mar, apr, may, jun, jul, aug, sept, octo, nov, dec]
        hourly_loads = [val for month in all_months for val in month]

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
        ghe.set_borehole(buried_depth=2.0, diameter=0.140)
        ghe.set_simulation_parameters(num_months=240)
        ghe.set_ground_loads_from_hourly_list(hourly_loads)
        ghe.set_geometry_constraints_rectangle(max_height=135, min_height=60, length=85.0, width=36.5, b_min=3.0, b_max=10.0)
        ghe.set_design(flow_rate=0.5, flow_type_str="borehole", max_eft=35, min_eft=5)
        ghe.find_design()
        output_file_directory = self.test_outputs_directory / "TestImbalancedCoolingLoads"
        ghe.prepare_results("Project Name", "Notes", "Author", "Iteration Name")
        ghe.write_output_files(output_file_directory, "")
        u_tube_height = ghe.results.output_dict['ghe_system']['active_borehole_length']['value']
        self.assertAlmostEqual(126.99, u_tube_height, delta=0.1)
        nbh = ghe.results.borehole_location_data_rows  # includes a header row
        self.assertEqual(9, len(nbh))
