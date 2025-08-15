from pathlib import Path

from ghedesigner.district_system import GHEHPSystem
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestDistrictSys(GHEBaseTest):
    def test_district_sys(self):
        f_path_txt = Path(__file__).resolve().parent / "test_data" / "3ghe-6hp_layout_input file.txt"
        f_path_json = Path(__file__).resolve().parent.parent.parent / "demos" / "simulate_3_bldg_3_ghe.json"

        System = GHEHPSystem(f_path_txt, f_path_json, f_path_txt.parent)
        System.solve_system()
        System.create_output(self.tests_directory)
