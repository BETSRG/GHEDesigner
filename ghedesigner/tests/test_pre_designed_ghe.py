from unittest import TestCase

from ghedesigner.enums import PipeType
from ghedesigner.ghe.manager import GroundHeatExchanger


class TestPreDesignedGHE(TestCase):
    def test_pre_designed_ghe_single_u(self):
        k_g = 1.0
        rho_cp_g = 3901000.0
        k_s = 2.0
        rho_cp_s = 2000000
        ugt = 10
        burial_depth = 2.0
        bh_radius = 0.08
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
            pipe_arrangement_type=PipeType.SINGLEUTUBE,
            pipe_parameters=pipe_params,
            fluid_parameters={
                "fluid_name": "water",
                "concentration_percent": 0,
                "temperature": 20,
            }
        )

        # testing using UBWT for comparisons against the GDR g-function library
        # https://gdr.openei.org/submissions/1325

        # case 1: "g-function_library_1.0/rectangle_5m_v1.0.json, 1_1, 5._192._0.08"
        get_g_func_inputs = {
            "flow_rate": 0.5,
            "flow_type": "BOREHOLE",
            "pre_designed": {"arrangement": "MANUAL", "H": 192, "x": [0.0], "y": [0.0]},
        }
        log_time_vals, g_vals, g_bhw_vals = ghe.get_g_function(get_g_func_inputs, boundary_condition="UBWT")
        self.assertAlmostEqual(-49.769, float(log_time_vals[0]), delta=0.001)
        self.assertAlmostEqual(3.003, float(log_time_vals[-1]), delta=0.001)
        self.assertAlmostEqual(2.8351, float(g_vals[30]), delta=0.001)
        self.assertAlmostEqual(6.7569, float(g_vals[-1]), delta=0.1)

        # case 2: "g-function_library_1.0/rectangle_5m_v1.0.json, 2_2, 5._192._0.08"
        get_g_func_inputs = {
            "flow_rate": 0.5,
            "flow_type": "BOREHOLE",
            "pre_designed": {
                "arrangement": "MANUAL",
                "H": 192,
                "x": [0.0, 0.0, 5.0, 5.0],
                "y": [0.0, 5.0, 0.0, 5.0],
            },
        }
        log_time_vals, g_vals, g_bhw_vals = ghe.get_g_function(get_g_func_inputs, boundary_condition="UBWT")
        self.assertAlmostEqual(2.8351, float(g_vals[30]), delta=0.001)
        self.assertAlmostEqual(14.0908, float(g_vals[-1]), delta=0.15)

        # case 3: "g-function_library_1.0/rectangle_5m_v1.0.json, 4_4, 5._192._0.08"
        get_g_func_inputs = {
            "flow_rate": 0.5,
            "flow_type": "BOREHOLE",
            "pre_designed": {
                "arrangement": "MANUAL",
                "H": 192,
                "x": [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 10.0, 15.0, 15.0, 15.0, 15.0],
                "y": [0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0, 0.0, 5.0, 10.0, 15.0],
            },
        }
        log_time_vals, g_vals, g_bhw_vals = ghe.get_g_function(get_g_func_inputs, boundary_condition="UBWT")
        self.assertAlmostEqual(2.8352, float(g_vals[30]), delta=0.15)
        self.assertAlmostEqual(33.5639, float(g_vals[-1]), delta=1.0)

        # TODO: should be investigated further - test values are not as close as I had hoped, but
        #  MIFT results are consistent with what was happening previously
