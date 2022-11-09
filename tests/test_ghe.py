from .ghe_base_case import GHEBaseTest
from ghedt import utilities, coordinates, ground_heat_exchangers, gfunction
from ghedt.peak_load_analysis_tool import borehole_heat_exchangers, media
import pygfunction as gt


class TestGHE(GHEBaseTest):
    def setUp(self):
        super().setUp()
        # Borehole dimensions
        # -------------------
        self.H = 100.0  # Borehole length (m)
        self.D = 2.0  # Borehole buried depth (m)
        self.r_b = 0.075  # Borehole radius]
        self.B = 5.0  # Borehole spacing (m)

        # Pipe dimensions
        # ---------------
        # U-tubes
        r_out = 26.67 / 1000.0 / 2.0  # Pipe outer radius (m)
        r_in = 21.6 / 1000.0 / 2.0  # Pipe inner radius (m)
        s = 32.3 / 1000.0  # Inner-tube to inner-tube Shank spacing (m)
        # Coaxial
        # Inner pipe radii
        r_in_in = 44.2 / 1000.0 / 2.0
        r_in_out = 50.0 / 1000.0 / 2.0
        # Outer pipe radii
        r_out_in = 97.4 / 1000.0 / 2.0
        r_out_out = 110.0 / 1000.0 / 2.0
        # Pipe radii
        # Note: This convention is different from pygfunction
        r_inner = [r_in_in, r_in_out]  # The radii of the inner pipe from in to out
        r_outer = [r_out_in, r_out_out]  # The radii of the outer pipe from in to out

        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos_s = media.Pipe.place_pipes(s, r_out, 1)
        # Double U-tube
        pos_d = media.Pipe.place_pipes(s, r_out, 2)
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
        rho_cp_p = 1542.0 * 1000.0  # Pipe volumetric heat capacity (J/K.m3)
        rho_cp_s = 2343.493 * 1000.0  # Soil volumetric heat capacity (J/K.m3)
        rho_cp_g = 3901.0 * 1000.0  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # ------------------
        # Pipe
        self.pipe_s = media.Pipe(pos_s, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        self.pipe_d = media.Pipe(pos_d, r_in, r_out, s, epsilon, k_p, rho_cp_p)
        self.pipe_c = media.Pipe(
            pos_c, r_inner, r_outer, s, epsilon, k_p_c, rho_cp_p
        )

        # Single U-tube BHE object
        self.SingleUTube = borehole_heat_exchangers.SingleUTube
        # Double U-tube bhe object
        self.DoubleUTube = borehole_heat_exchangers.MultipleUTube
        # Coaxial tube bhe object
        self.CoaxialTube = borehole_heat_exchangers.CoaxialPipe

        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        self.soil = media.Soil(k_s, rho_cp_s, ugt)
        # Grout
        self.grout = media.Grout(k_g, rho_cp_g)

        # Coordinates
        nx = 12
        ny = 13
        self.coordinates = coordinates.rectangle(nx, ny, self.B, self.B)

        # Compute a range of g-functions for interpolation
        self.log_time = utilities.eskilson_log_times()
        self.H_values = [24.0, 48.0, 96.0, 192.0, 384.0]
        self.r_b_values = [self.r_b] * len(self.H_values)
        self.D_values = [2.0] * len(self.H_values)

        # Inputs related to fluid
        # -----------------------
        v_flow_borehole = 0.2  # System volumetric flow rate (L/s)

        # -----------------------
        # Fluid properties
        self.fluid = gt.media.Fluid(fluid_str="Water", percent=0.0)
        self.V_flow_system = v_flow_borehole * float(
            nx * ny
        )  # System volumetric flow rate (L/s)
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
        self.sim_params = media.SimulationParameters(
            start_month,
            end_month,
            max_eft_allowable,
            min_eft_allowable,
            max_height,
            min_height,
        )

        # Process loads from file
        # -----------------------
        # read in the csv file and convert the loads to a list of length 8760
        csv_file = self.test_data_directory / 'Atlanta_Office_Building_Loads.csv'
        raw_lines = csv_file.read_text().split('\n')
        self.hourly_extraction_ground_loads = [float(x) for x in raw_lines[1:] if x.strip() != '']

    def test_single_u_tube(self):

        # Define a borehole
        borehole = gt.boreholes.Borehole(self.H, self.D, self.r_b, x=0.0, y=0.0)

        # Initialize GHE object
        g_function = gfunction.compute_live_g_function(
            self.B,
            self.H_values,
            self.r_b_values,
            self.D_values,
            self.m_flow_borehole,
            self.SingleUTube,
            self.log_time,
            self.coordinates,
            self.fluid,
            self.pipe_s,
            self.grout,
            self.soil,
        )

        # Initialize the GHE object
        ghe = ground_heat_exchangers.GHE(
            self.V_flow_system,
            self.B,
            self.SingleUTube,
            self.fluid,
            borehole,
            self.pipe_s,
            self.grout,
            self.soil,
            g_function,
            self.sim_params,
            self.hourly_extraction_ground_loads,
        )

        max_hp_eft, min_hp_eft = ghe.simulate(method="hybrid")

        self.assertAlmostEqual(39.09, max_hp_eft, delta=0.01)
        self.assertAlmostEqual(16.66, min_hp_eft, delta=0.01)

        ghe.size(method="hybrid")

        self.assertAlmostEqual(ghe.bhe.b.H, 130.22, places=2)

    def test_double_u_tube(self):

        # Define a borehole
        borehole = gt.boreholes.Borehole(self.H, self.D, self.r_b, x=0.0, y=0.0)

        # Initialize GHE object
        g_function = gfunction.compute_live_g_function(
            self.B,
            self.H_values,
            self.r_b_values,
            self.D_values,
            self.m_flow_borehole,
            self.DoubleUTube,
            self.log_time,
            self.coordinates,
            self.fluid,
            self.pipe_d,
            self.grout,
            self.soil,
        )

        # Initialize the GHE object
        ghe = ground_heat_exchangers.GHE(
            self.V_flow_system,
            self.B,
            self.DoubleUTube,
            self.fluid,
            borehole,
            self.pipe_d,
            self.grout,
            self.soil,
            g_function,
            self.sim_params,
            self.hourly_extraction_ground_loads,
        )

        max_hp_eft, min_hp_eft = ghe.simulate(method="hybrid")

        self.assertAlmostEqual(37.76, max_hp_eft, delta=0.01)
        self.assertAlmostEqual(17.09, min_hp_eft, delta=0.01)

        ghe.size(method="hybrid")

        self.assertAlmostEqual(ghe.bhe.b.H, 120.40, places=2)

    def test_coaxial_tube(self):

        # Define a borehole
        borehole = gt.boreholes.Borehole(self.H, self.D, self.r_b, x=0.0, y=0.0)

        # Initialize GHE object
        g_function = gfunction.compute_live_g_function(
            self.B,
            self.H_values,
            self.r_b_values,
            self.D_values,
            self.m_flow_borehole,
            self.CoaxialTube,
            self.log_time,
            self.coordinates,
            self.fluid,
            self.pipe_c,
            self.grout,
            self.soil,
        )

        # Re-Initialize the GHE object
        ghe = ground_heat_exchangers.GHE(
            self.V_flow_system,
            self.B,
            self.CoaxialTube,
            self.fluid,
            borehole,
            self.pipe_c,
            self.grout,
            self.soil,
            g_function,
            self.sim_params,
            self.hourly_extraction_ground_loads,
        )

        max_hp_eft, min_hp_eft = ghe.simulate(method="hybrid")

        self.assertAlmostEqual(38.06, max_hp_eft, delta=0.01)
        self.assertAlmostEqual(17.22, min_hp_eft, delta=0.01)

        ghe.size(method="hybrid")

        self.assertAlmostEqual(ghe.bhe.b.H, 124.84, delta=0.01)
