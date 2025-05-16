from unittest import TestCase

from ghedesigner.enums import BHPipeType
from ghedesigner.ghe.manager import GroundHeatExchanger


class TestPreDesignedGHE(TestCase):
    def test_pre_designed_ghe_single_u(self):
        k_g = 1.0
        rho_cp_g = 3901000.0
        k_s = 2.0
        rho_cp_s = 2343493.0
        ugt = 10
        burial_depth = 2.0
        bh_radius = 0.07
        pipe_params = {
            "conductivity": 0.4,
            "rho_cp": 1500000.0,
            "inner_diameter": 0.03404,
            "outer_diameter": 0.04216,
            "shank_spacing": 0.01856,
            "roughness": 1e-6,
        }

        ghe = GroundHeatExchanger(
            grout_conductivity=k_g,
            grout_rho_cp=rho_cp_g,
            soil_conductivity=k_s,
            soil_rho_cp=rho_cp_s,
            soil_undisturbed_temperature=ugt,
            borehole_buried_depth=burial_depth,
            borehole_radius=bh_radius,
            pipe_arrangement_type=BHPipeType.SINGLEUTUBE,
            pipe_parameters=pipe_params,
            fluid_name="water",
            fluid_concentration_percent=0,
            fluid_temperature=20,
        )

        get_g_func_inputs = {
            "ground-heat-exchanger": {
                "ghe1": {"flow_rate": 0.5, "flow_type": "BOREHOLE", "pre_designed": {"H": 150, "x": [0.0], "y": [0.0]}}
            }
        }

        log_time_vals, g_vals, g_bhw_vals = ghe.get_g_function(get_g_func_inputs, "ghe1")

        self.assertAlmostEqual(-48.623, float(log_time_vals[0]), delta=0.001)
        self.assertAlmostEqual(3.003, float(log_time_vals[-1]), delta=0.001)
        self.assertAlmostEqual(-1.730, float(g_vals[0]), delta=0.001)
        self.assertAlmostEqual(6.6406, float(g_vals[-1]), delta=0.001)
