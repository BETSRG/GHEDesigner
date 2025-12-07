import json
from pathlib import Path

from ghedesigner.district_system import GHEHPSystem
from ghedesigner.district_system_again import GHEHPSystem
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestDistrictSys(GHEBaseTest):
    def test_district_sys(self):
        f_path_json = Path(__file__).resolve().parent.parent.parent / "demos" / "simulate_3_bldg_3_ghe.json"
        System = GHEHPSystem(f_path_json)
        System.solve_system()

        # don't put in the timestamped directory for now
        # System.create_output(self.test_outputs_directory / "test_district_sys" / "output_simulate_3_bldg_3_ghe.csv")
        System.create_output(self.tests_directory / "test_outputs" / "output_3_bldg_3_ghe_district.csv")

    def test_simple_district(self):
        f_path_json = Path(__file__).resolve().parent.parent.parent / "demos" / "simple_district.json"
        System = GHEHPSystem(f_path_json)
        System.solve_system()

        # don't put in the timestamped directory for now
        # System.create_output(self.test_outputs_directory / "test_simple_district" / "output_simple_district.csv")
        System.create_output(self.tests_directory / "test_outputs" / "output_simple_district.csv")

class TestDistrictSysAgain(GHEBaseTest):

    def test_updated_1_pipe_3_ghe_6_bldg_without_hx(self):
        f1 = Path("ghedesigner/tests/test_data/1-pipe_3ghe-6hp_system_wo_ISHX_input.txt")
        f2 = Path("demos/find_design_bi_rectangle_single_u_tube.json")

        json_data = json.loads(f2.read_text())

        System = GHEHPSystem()
        System.read_GHEHPSystem_data(f1)
        System.read_data_from_json_file(json_data)

        fluid, pipe, grout, soil, borehole = System.read_data_from_json_file(json_data)
        System.solveSystem(fluid, pipe, grout, soil, borehole)
        System.createOutput(Path(f"ghedesigner/tests/test_data/{f1.stem.strip('_input_')}_output.csv"))
        System.output_file_energy_consumption(Path(f"ghedesigner/tests/test_data/{f1.stem.strip('_input_')}_energy.csv"))

    def test_updated_1_pipe_3_ghe_6_bldg_with_hx(self):
        f1 = Path("ghedesigner/tests/test_data/1-pipe_3ghe-6hp_system_w_ISHX_input.txt")
        f2 = Path("demos/find_design_bi_rectangle_single_u_tube.json")

        json_data = json.loads(f2.read_text())

        System = GHEHPSystem()
        System.read_GHEHPSystem_data(f1)
        System.read_data_from_json_file(json_data)

        fluid, pipe, grout, soil, borehole = System.read_data_from_json_file(json_data)
        System.solveSystem(fluid, pipe, grout, soil, borehole)
        System.createOutput(Path(f"ghedesigner/tests/test_data/{f1.stem.strip('_input_')}_output.csv"))
        System.output_file_energy_consumption(Path(f"ghedesigner/tests/test_data/{f1.stem.strip('_input_')}_energy.csv"))

    def test_updated_2_pipe_3_ghe_6_bldg_without_hx(self):
        f1 = Path("ghedesigner/tests/test_data/2-pipe_3ghe-6hp_system_wo_ISHX_input.txt")
        f2 = Path("demos/find_design_bi_rectangle_single_u_tube.json")

        json_data = json.loads(f2.read_text())

        System = GHEHPSystem()
        System.read_GHEHPSystem_data(f1)
        System.read_data_from_json_file(json_data)

        fluid, pipe, grout, soil, borehole = System.read_data_from_json_file(json_data)
        System.solveSystem(fluid, pipe, grout, soil, borehole)
        System.createOutput(Path(f"ghedesigner/tests/test_data/{f1.stem.strip('_input_')}_output.csv"))
        System.output_file_energy_consumption(Path(f"ghedesigner/tests/test_data/{f1.stem.strip('_input_')}_energy.csv"))