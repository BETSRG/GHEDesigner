from pathlib import Path

from ghedesigner.district_system import GHEHPSystem
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import GHEFluid, Grout, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest


class TestDistrictSys(GHEBaseTest):
    def test_district_sys(self):
        f_path_txt = Path(__file__).resolve().parent / "test_data" / "3ghe-6hp_layout_input file.txt"
        f_path_json = Path(__file__).resolve().parent.parent.parent / "demos" / "simulate_3_bldg_3_ghe.json"

        System = GHEHPSystem()
        System.read_ghe_hp_system_data(f_path_txt, f_path_json, f_path_txt.parent)

        pipe = Pipe.init_single_u_tube(
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            conductivity=0.4,
            rho_cp=1542000.0,
        )

        soil = Soil(k=2.0, rho_cp=2343493.0, ugt=6.1)
        grout = Grout(k=1.0, rho_cp=3901000.0)
        fluid = GHEFluid(fluid_str="PropyleneGlycol", percent=30.0, temperature=20.0)
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.07)

        System.solve_system(fluid, pipe, grout, soil, borehole)
        System.create_output(self.test_outputs_directory)
