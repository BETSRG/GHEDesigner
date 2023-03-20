from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import SingleUTube, MultipleUTube, CoaxialPipe
from ghedesigner.coordinates import rectangle
from ghedesigner.enums import BHPipeType, TimestepType
from ghedesigner.gfunction import calc_g_func_for_multiple_lengths
from ghedesigner.ground_heat_exchangers import GHE
from ghedesigner.media import Pipe, Soil, Grout, GHEFluid
from ghedesigner.simulation import SimulationParameters
from ghedesigner.tests.ghe_base_case import GHEBaseTest
from ghedesigner.utilities import eskilson_log_times


class TestGHE(GHEBaseTest):
    def setUp(self):
        super().setUp()
        # Borehole dimensions
        # -------------------
        self.H = 100.0  # Borehole length (m)
        self.D = 2.0  # Borehole buried depth (m)
        self.dia = 0.150  # Borehole diameter
        self.B = 5.0  # Borehole spacing (m)

        # Pipe dimensions
        # ---------------
        # U-tubes
        d_out = 0.02667  # Pipe outer diameter (m)
        d_in = 0.0216  # Pipe inner diameter (m)
        r_out = d_out / 2.0
        r_in = d_in / 2.0
        s = 0.0323  # Inner-tube to inner-tube Shank spacing (m)
        # Coaxial
        # Inner pipe radii
        d_in_in = 44.2 / 1000.0
        d_in_out = 50.0 / 1000.0
        # Outer pipe radii
        d_out_in = 97.4 / 1000.0
        d_out_out = 110.0 / 1000.0
        # Pipe radii
        # Note: This convention is different from pygfunction
        r_inner = [d_in_in / 2.0, d_in_out / 2.0]  # The radii of the inner pipe from in to out
        r_outer = [d_out_in / 2.0, d_out_out / 2.0]  # The radii of the outer pipe from in to out

        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos_s = Pipe.place_pipes(s, r_out, 1)
        # Double U-tube
        pos_d = Pipe.place_pipes(s, r_out, 2)
        # Coaxial
        pos_c = (0, 0)

        # Thermal conductivities
        # ----------------------
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)
        # Pipe thermal conductivity list for coaxial
        k_p_c = [0.4, 0.4]  # Inner and outer pipe thermal conductivity (W/m.K)

        # Volumetric heat capacities
        # --------------------------
        rho_cp_p = 1542000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343493.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # ------------------
        # Pipe
        self.pipe_s = Pipe(pos_s, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        self.pipe_d = Pipe(pos_d, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        self.pipe_c = Pipe(pos_c, r_inner, r_outer, s, epsilon, k_p_c, rho_cp_p)

        # Single U-tube BHE object
        self.SingleUTube = SingleUTube
        # Double U-tube bhe object
        self.DoubleUTube = MultipleUTube
        # Coaxial tube bhe object
        self.CoaxialTube = CoaxialPipe

        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        self.soil = Soil(k_s, rho_cp_s, ugt)
        # Grout
        self.grout = Grout(k_g, rho_cp_g)

        # Coordinates
        nx = 12
        ny = 13
        self.coordinates = rectangle(nx, ny, self.B, self.B)

        # Compute a range of g-functions for interpolation
        self.log_time = eskilson_log_times()
        self.H_values = [24.0, 48.0, 96.0, 192.0, 384.0]
        self.bh_depth = 2.0

        # Inputs related to fluid
        # -----------------------
        v_flow_borehole = 0.2  # System volumetric flow rate (L/s)

        # -----------------------
        # Fluid properties
        self.fluid = GHEFluid(fluid_str="Water", percent=0.0)
        # System volumetric flow rate (L/s)
        self.V_flow_system = v_flow_borehole * float(nx * ny)
        # Total fluid mass flow rate per borehole (kg/s)
        self.m_flow_borehole = v_flow_borehole / 1000.0 * self.fluid.rho

        # Simulation start month and end month
        # --------------------------------
        # Simulation start month and end month
        start_month = 1
        n_years = 20
        end_month = n_years * 12
        # Maximum and minimum allowable fluid temperatures
        max_eft_allowable = 35  # degrees Celsius
        min_eft_allowable = 5  # degrees Celsius
        # Maximum and minimum allowable heights
        max_height = 384  # in meters
        min_height = 24  # in meters
        self.sim_params = SimulationParameters(
            start_month,
            end_month,
            max_eft_allowable,
            min_eft_allowable,
            max_height,
            min_height,
        )

        # Process loads from file
        self.hourly_extraction_ground_loads = self.get_atlanta_loads()

    def test_single_u_tube(self):
        # Define a borehole
        borehole = GHEBorehole(self.H, self.D, self.dia / 2.0, x=0.0, y=0.0)

        # Initialize GHE object
        g_function = calc_g_func_for_multiple_lengths(
            self.B,
            self.H_values,
            self.dia / 2.0,
            self.bh_depth,
            self.m_flow_borehole,
            BHPipeType.SINGLEUTUBE,
            self.log_time,
            self.coordinates,
            self.fluid,
            self.pipe_s,
            self.grout,
            self.soil,
        )

        # Initialize the GHE object
        ghe = GHE(
            self.V_flow_system,
            self.B,
            BHPipeType.SINGLEUTUBE,
            self.fluid,
            borehole,
            self.pipe_s,
            self.grout,
            self.soil,
            g_function,
            self.sim_params,
            self.hourly_extraction_ground_loads,
        )

        max_hp_eft, min_hp_eft = ghe.simulate(method=TimestepType.HYBRID)

        self.assertAlmostEqual(39.07, max_hp_eft, delta=0.01)
        self.assertAlmostEqual(16.66, min_hp_eft, delta=0.01)

        ghe.size(method=TimestepType.HYBRID)

        self.assertAlmostEqual(ghe.bhe.b.H, 130.24, delta=0.01)

    def test_double_u_tube(self):
        # Define a borehole
        borehole = GHEBorehole(self.H, self.D, self.dia / 2.0, x=0.0, y=0.0)

        # Initialize GHE object
        g_function = calc_g_func_for_multiple_lengths(
            self.B,
            self.H_values,
            self.dia / 2.0,
            self.bh_depth,
            self.m_flow_borehole,
            BHPipeType.DOUBLEUTUBEPARALLEL,
            self.log_time,
            self.coordinates,
            self.fluid,
            self.pipe_d,
            self.grout,
            self.soil,
        )

        # Initialize the GHE object
        ghe = GHE(
            self.V_flow_system,
            self.B,
            BHPipeType.DOUBLEUTUBEPARALLEL,
            self.fluid,
            borehole,
            self.pipe_d,
            self.grout,
            self.soil,
            g_function,
            self.sim_params,
            self.hourly_extraction_ground_loads,
        )

        max_hp_eft, min_hp_eft = ghe.simulate(method=TimestepType.HYBRID)

        self.assertAlmostEqual(37.97, max_hp_eft, delta=0.01)
        self.assertAlmostEqual(16.95, min_hp_eft, delta=0.01)

        ghe.size(method=TimestepType.HYBRID)

        self.assertAlmostEqual(ghe.bhe.b.H, 122.10, delta=0.01)

    def test_coaxial_tube(self):
        # Define a borehole
        borehole = GHEBorehole(self.H, self.D, self.dia / 2.0, x=0.0, y=0.0)

        # Initialize GHE object
        g_function = calc_g_func_for_multiple_lengths(
            self.B,
            self.H_values,
            self.dia / 2.0,
            self.bh_depth,
            self.m_flow_borehole,
            BHPipeType.COAXIAL,
            self.log_time,
            self.coordinates,
            self.fluid,
            self.pipe_c,
            self.grout,
            self.soil,
        )

        # Re-Initialize the GHE object
        ghe = GHE(
            self.V_flow_system,
            self.B,
            BHPipeType.COAXIAL,
            self.fluid,
            borehole,
            self.pipe_c,
            self.grout,
            self.soil,
            g_function,
            self.sim_params,
            self.hourly_extraction_ground_loads,
        )

        max_hp_eft, min_hp_eft = ghe.simulate(method=TimestepType.HYBRID)

        self.assertAlmostEqual(38.04, max_hp_eft, delta=0.01)
        self.assertAlmostEqual(17.18, min_hp_eft, delta=0.01)

        ghe.size(method=TimestepType.HYBRID)

        self.assertAlmostEqual(ghe.bhe.b.H, 124.79, delta=0.01)
