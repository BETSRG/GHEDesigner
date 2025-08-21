from pathlib import Path

from ghedesigner.district_system import GHEHPSystem
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestDistrictSys(GHEBaseTest):
    def test_district_sys(self):
        f_path_json = Path(__file__).resolve().parent.parent.parent / "demos" / "simulate_3_bldg_3_ghe.json"
        System = GHEHPSystem(f_path_json)
        System.solve_system()
        System.create_output(self.tests_directory / "output_results.csv")

    def test_simple_district(self):
        f_path_json = Path(__file__).resolve().parent.parent.parent / "demos" / "simple_district.json"
        System = GHEHPSystem(f_path_json)
        System.solve_system()
        System.create_output(self.test_outputs_directory / "simple_district.csv")
