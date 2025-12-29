from ghedesigner.district_system import GHEHPSystem
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestDistrictSys(GHEBaseTest):
    def test_simulate_1_pipe_3_ghe_6_bldg_district(self):
        f_path_json = self.demos_path / "simulate_1_pipe_3_ghe_6_bldg_district.json"
        system = GHEHPSystem(f_path_json)
        system.solve_system()

        # don't put in the timestamped directory for now
        # system.create_output(self.test_outputs_directory / "test_district_sys" / "output_simulate_3_bldg_3_ghe.csv")
        system.create_output(
            self.tests_directory / self.test_data_directory / "simulate_1_pipe_3_ghe_6_bldg_district.csv"
        )

    def test_simulate_1_pipe_1_ghe_1_bldg_district(self):
        f_path_json = self.demos_path / "simulate_1_pipe_1_ghe_1_bldg_district.json"
        system = GHEHPSystem(f_path_json)
        system.solve_system()

        # don't put in the timestamped directory for now
        # system.create_output(self.test_outputs_directory / "test_simple_district" / "output_simple_district.csv")
        system.create_output(
            self.tests_directory / self.test_data_directory / "simulate_1_pipe_1_ghe_1_bldg_district.csv"
        )
