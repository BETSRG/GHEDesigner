import unittest

from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.boreholes.single_u_borehole import SingleUTube
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import GHEFluid, Grout, Soil


class TestRadialNumericalBorehole(unittest.TestCase):
    def test_calc_sts_g_functions(self):
        fluid = GHEFluid(fluid_str="WATER", percent=0)
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.075)
        grout = Grout(k=2.0, rho_cp=2000000.0)

        r_out = 0.04216 / 2.0
        r_in = 0.03404 / 2.0
        shank_spacing = 0.02
        roughness = 1e-6
        k_pipe = 0.4
        rho_cp_pipe = 1542000
        pipe = Pipe.init_single_u_tube(k_pipe, rho_cp_pipe, r_in * 2, r_out * 2, shank_spacing, roughness, 1)
        soil = Soil(k=2.0, rho_cp=3901000, ugt=20)
        m_dot_bh = 0.5
        bh = SingleUTube(m_dot_bh, fluid, borehole, pipe, grout, soil)
        bh.calc_sts_g_functions()

        self.assertAlmostEqual(-1.182, bh.g[0], delta=0.001)
        self.assertAlmostEqual(2.217, bh.g[-1], delta=0.001)
        self.assertAlmostEqual(0.0, bh.g_bhw[0], delta=0.001)
        self.assertAlmostEqual(2.094, bh.g_bhw[-1], delta=0.001)
