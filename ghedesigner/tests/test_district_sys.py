from pathlib import Path

from ghedesigner.district_system import GHEHPSystem
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
