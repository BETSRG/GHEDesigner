import numpy as np

from ghedesigner.constants import SEC_IN_HR
from ghedesigner.enums import PipeType, TimestepType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.coordinates import rectangle
from ghedesigner.ghe.gfunction import calc_g_func_for_multiple_lengths
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.tests.test_base_case import GHEBaseTest
from ghedesigner.utilities import eskilson_log_times


class TestLoadAgg(GHEBaseTest):
    def setUp(self):
        # Borehole dimensions
        self.H = 100.0  # Borehole length (m)
        self.D = 2.0    # Borehole buried depth (m)
        self.dia = 0.140  # Borehole diameter (m)
        self.B = 5.0    # Borehole spacing (m)

        # Pipe dimensions
        d_out = 0.04216  # Pipe outer diameter (m)
        d_in = 0.03404   # Pipe inner diameter (m)
        s = 0.01856      # Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)

        # Thermal conductivities
        k_p = 0.4  # Pipe (W/m.K)
        k_s = 2.0  # Ground (W/m.K)
        k_g = 1.0  # Grout (W/m.K)

        # Volumetric heat capacities
        rho_cp_p = 1542000.0  # Pipe (J/K.m3)
        rho_cp_s = 2343493.0  # Soil (J/K.m3)
        rho_cp_g = 3901000.0  # Grout (J/K.m3)

        self.pipe_s = Pipe.init_single_u_tube(
            conductivity=k_p, rho_cp=rho_cp_p,
            inner_diameter=d_in, outer_diameter=d_out,
            shank_spacing=s, roughness=epsilon,
        )

        ugt = 18.3  # Undisturbed ground temperature (C)
        self.soil = Soil(k_s, rho_cp_s, ugt)
        self.grout = Grout(k_g, rho_cp_g)

        nx, ny = 12, 13
        self.coordinates = rectangle(nx, ny, self.B, self.B)

        self.log_time = eskilson_log_times()
        self.H_values = [24.0, 48.0, 96.0, 192.0, 384.0]

        v_flow_borehole = 0.5  # (L/s)
        self.fluid = Fluid(fluid_name="Water", percent=0.0)
        self.v_flow_system = v_flow_borehole * float(nx * ny)
        self.m_flow_borehole = v_flow_borehole / 1000.0 * self.fluid.rho

        self.hourly_extraction_ground_loads = self.get_atlanta_loads()

    def _make_ghe(self, num_months):
        """Helper: construct a single-U-tube GHE for a given simulation duration."""
        borehole = Borehole(
            borehole_height=self.H, burial_depth=self.D,
            borehole_radius=self.dia / 2.0,
        )
        g_function = calc_g_func_for_multiple_lengths(
            self.B,
            self.H_values,
            self.dia / 2.0,
            self.D,
            self.m_flow_borehole,
            PipeType.SINGLEUTUBE,
            self.log_time,
            self.coordinates,
            self.fluid,
            self.pipe_s,
            self.grout,
            self.soil,
        )
        return GHE(
            self.v_flow_system, self.B, PipeType.SINGLEUTUBE, self.fluid,
            borehole, self.pipe_s, self.grout, self.soil, g_function,
            1, num_months, self.hourly_extraction_ground_loads,
        )

    def test_init_agg_bins_structure(self):
        """Verify _init_agg_bins returns correctly structured bin arrays."""
        ghe = self._make_ghe(num_months=12)
        time_values = np.arange(1, 8761)  # 1 year = 8760 hours

        energy_bins, dts = ghe._init_agg_bins(time_values)

        # Arrays must be the same length
        self.assertEqual(len(energy_bins), len(dts))
        # All bins start empty
        self.assertTrue(np.all(energy_bins == 0.0))
        # Newest bin (index 0) is exactly 1 hour
        self.assertAlmostEqual(dts[0], SEC_IN_HR)
        # Bins must collectively cover at least the full simulation runtime
        self.assertGreaterEqual(np.sum(dts), len(time_values) * SEC_IN_HR)

    def test_hourly_with_load_agg_matches_hourly(self):
        """Verify HOURLYWITHLOADAGG results are within 0.5°C of HOURLY for 1 year."""
        # Limit to 1 year so the HOURLY method completes in reasonable time
        num_months_1yr = 12

        ghe_hourly = self._make_ghe(num_months=num_months_1yr)
        ghe_agg = self._make_ghe(num_months=num_months_1yr)

        max_eft_hourly, min_eft_hourly = ghe_hourly.simulate(method=TimestepType.HOURLY)
        max_eft_agg, min_eft_agg = ghe_agg.simulate(method=TimestepType.HOURLYWITHLOADAGG)

        self.assertAlmostEqual(max_eft_hourly, max_eft_agg, delta=0.5)
        self.assertAlmostEqual(min_eft_hourly, min_eft_agg, delta=0.5)