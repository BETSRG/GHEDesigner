import json

from ghedesigner.district_system import GHEHPSystem
from ghedesigner.district_system_again import GHEHPSystem as GHEHPSystemAgain
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestDistrictSys(GHEBaseTest):
    def test_district_sys(self):
        f_path_json = self.demos_path / "simulate_3_bldg_3_ghe.json"
        System = GHEHPSystem(f_path_json)
        System.solve_system()

        # don't put in the timestamped directory for now
        # System.create_output(self.test_outputs_directory / "test_district_sys" / "output_simulate_3_bldg_3_ghe.csv")
        System.create_output(self.tests_directory / "test_outputs" / "output_3_bldg_3_ghe_district.csv")

    def test_simple_district(self):
        f_path_json = self.demos_path / "simple_district.json"
        System = GHEHPSystem(f_path_json)
        System.solve_system()

        # don't put in the timestamped directory for now
        # System.create_output(self.test_outputs_directory / "test_simple_district" / "output_simple_district.csv")
        System.create_output(self.tests_directory / "test_outputs" / "output_simple_district.csv")


class TestDistrictSysAgain(GHEBaseTest):

    def test_updated_1_pipe_3_ghe_6_bldg_without_hx(self):
        f1 = self.test_data_directory / "1-pipe_3ghe-6hp_system_wo_ISHX_input.txt"
        f2 = self.demos_path / "find_design_bi_rectangle_single_u_tube.json"

        json_data = json.loads(f2.read_text())

        f_path_json = self.demos_path / "simulate_1_pipe_3_bldg_3_ghe_without_hx.json"
        System = GHEHPSystemAgain(f_path_json)
        System.read_GHEHPSystem_data(f1)
        System.read_data_from_json_file(json_data)

        fluid, pipe, grout, soil, borehole = System.read_data_from_json_file(json_data)
        System.solveSystem(fluid, pipe, grout, soil, borehole)
        System.write_state_outputs(self.test_data_directory / f"{f1.stem.strip('_input_')}_output.csv")
        System.write_energy_outputs(self.test_data_directory / f"{f1.stem.strip('_input_')}_energy.csv")

    def test_updated_1_pipe_3_ghe_6_bldg_with_hx(self):
        f1 = self.test_data_directory / "1-pipe_3ghe-6hp_system_w_ISHX_input.txt"
        f2 = self.demos_path / "find_design_bi_rectangle_single_u_tube.json"

        json_data = json.loads(f2.read_text())

        f_path_json = self.demos_path / "simulate_1_pipe_3_bldg_3_ghe_with_hx.json"
        System = GHEHPSystemAgain(f_path_json)
        System.read_GHEHPSystem_data(f1)
        System.read_data_from_json_file(json_data)

        fluid, pipe, grout, soil, borehole = System.read_data_from_json_file(json_data)
        System.solveSystem(fluid, pipe, grout, soil, borehole)
        System.write_state_outputs(self.test_data_directory / f"{f1.stem.strip('_input_')}_output.csv")
        System.write_energy_outputs(self.test_data_directory / f"{f1.stem.strip('_input_')}_energy.csv")

    def test_updated_2_pipe_3_ghe_6_bldg_without_hx(self):
        f1 = self.test_data_directory / "2-pipe_3ghe-6hp_system_wo_ISHX_input.txt"
        f2 = self.demos_path / "find_design_bi_rectangle_single_u_tube.json"

        old_json = json.loads(f2.read_text())

        f_path_json = self.demos_path / "simulate_2_pipe_3_bldg_3_ghe_without_hx.json"
        System = GHEHPSystemAgain(f_path_json)
        System.read_GHEHPSystem_data(f1)
        System.read_data_from_json_file(old_json)

        fluid, pipe, grout, soil, borehole = System.read_data_from_json_file(old_json)
        System.solveSystem(fluid, pipe, grout, soil, borehole)
        System.write_state_outputs(self.test_data_directory / f"{f1.stem.strip('_input_')}_output.csv")
        System.write_energy_outputs(self.test_data_directory / f"{f1.stem.strip('_input_')}_energy.csv")
