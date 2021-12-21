# Jack C. Cook
# Monday, October 11, 2021

import unittest

import ghedt.pygfunction as gt
import pandas as pd
import ghedt


class TestGHE(unittest.TestCase):

    def setUp(self) -> None:
        import ghedt.peak_load_analysis_tool as plat
        # Borehole dimensions
        # -------------------
        self.H = 100.  # Borehole length (m)
        self.D = 2.  # Borehole buried depth (m)
        self.r_b = 0.075  # Borehole radius]
        self.B = 5.  # Borehole spacing (m)

        # Pipe dimensions
        # ---------------
        # U-tubes
        r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
        r_in = 21.6 / 1000. / 2.  # Pipe inner radius (m)
        s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
        # Coaxial
        # Inner pipe radii
        r_in_in = 44.2 / 1000. / 2.
        r_in_out = 50. / 1000. / 2.
        # Outer pipe radii
        r_out_in = 97.4 / 1000. / 2.
        r_out_out = 110. / 1000. / 2.
        # Pipe radii
        # Note: This convention is different from pygfunction
        r_inner = [r_in_in,
                   r_in_out]  # The radii of the inner pipe from in to out
        r_outer = [r_out_in,
                   r_out_out]  # The radii of the outer pipe from in to out

        epsilon = 1.0e-6  # Pipe roughness (m)

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos_s = plat.media.Pipe.place_pipes(s, r_out, 1)
        # Double U-tube
        pos_d = plat.media.Pipe.place_pipes(s, r_out, 2)
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
        rhoCp_p = 1542. * 1000.  # Pipe volumetric heat capacity (J/K.m3)
        rhoCp_s = 2343.493 * 1000.  # Soil volumetric heat capacity (J/K.m3)
        rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # ------------------
        # Pipe
        self.pipe_s = \
            plat.media.Pipe(pos_s, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        self.pipe_d = \
            plat.media.Pipe(pos_d, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        self.pipe_c = \
            plat.media.Pipe(pos_c, r_inner, r_outer, s, epsilon, k_p_c, rhoCp_p)

        # Single U-tube BHE object
        self.SingleUTube = plat.borehole_heat_exchangers.SingleUTube
        # Double U-tube bhe object
        self.DoubleUTube = plat.borehole_heat_exchangers.MultipleUTube
        # Coaxial tube bhe object
        self.CoaxialTube = plat.borehole_heat_exchangers.CoaxialPipe

        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        self.soil = plat.media.Soil(k_s, rhoCp_s, ugt)
        # Grout
        self.grout = plat.media.ThermalProperty(k_g, rhoCp_g)

        # Coordinates
        Nx = 12
        Ny = 13
        self.coordinates = ghedt.coordinates.rectangle(Nx, Ny, self.B, self.B)

        # Compute a range of g-functions for interpolation
        self.log_time = ghedt.utilities.Eskilson_log_times()
        self.H_values = [24., 48., 96., 192., 384.]
        self.r_b_values = [self.r_b] * len(self.H_values)
        self.D_values = [2.] * len(self.H_values)

        # Inputs related to fluid
        # -----------------------
        V_flow_borehole = 0.2  # System volumetric flow rate (L/s)
        mixer = 'MEG'  # Ethylene glycol mixed with water
        percent = 0.  # Percentage of ethylene glycol added in

        # -----------------------
        # Fluid properties
        self.fluid = gt.media.Fluid(mixer=mixer, percent=percent)
        self.V_flow_system = V_flow_borehole * float(
            Nx * Ny)  # System volumetric flow rate (L/s)
        # Total fluid mass flow rate per borehole (kg/s)
        self.m_flow_borehole = V_flow_borehole / 1000. * self.fluid.rho

        # Simulation start month and end month
        # --------------------------------
        # Simulation start month and end month
        start_month = 1
        n_years = 20
        end_month = n_years * 12
        # Maximum and minimum allowable fluid temperatures
        max_EFT_allowable = 35  # degrees Celsius
        min_EFT_allowable = 5  # degrees Celsius
        # Maximum and minimum allowable heights
        max_Height = 384  # in meters
        min_Height = 24  # in meters
        self.sim_params = plat.media.SimulationParameters(
            start_month, end_month, max_EFT_allowable, min_EFT_allowable,
            max_Height, min_Height)

        # Process loads from file
        # -----------------------
        # read in the csv file and convert the loads to a list of length 8760
        hourly_extraction: dict = \
            pd.read_csv('Atlanta_Office_Building_Loads.csv').to_dict('list')
        # Take only the first column in the dictionary
        self.hourly_extraction_ground_loads: list = \
            hourly_extraction[list(hourly_extraction.keys())[0]]

    def test_single_u_tube(self):

        # Define a borehole
        borehole = gt.boreholes.Borehole(self.H, self.D, self.r_b, x=0., y=0.)

        # Initialize GHE object
        g_function = ghedt.gfunction.compute_live_g_function(
            self.B, self.H_values, self.r_b_values, self.D_values,
            self.m_flow_borehole, self.SingleUTube,
            self.log_time, self.coordinates, self.fluid, self.pipe_s,
            self.grout, self.soil)

        # Initialize the GHE object
        ghe = ghedt.ground_heat_exchangers.GHE(
            self.V_flow_system, self.B, self.SingleUTube, self.fluid,
            borehole, self.pipe_s, self.grout, self.soil,
            g_function, self.sim_params, self.hourly_extraction_ground_loads)

        max_HP_EFT, min_HP_EFT = ghe.simulate(method='hybrid')

        self.assertEqual(39.084419566119934, max_HP_EFT)
        self.assertEqual(16.660966674440232, min_HP_EFT)

        ghe.size(method='hybrid')

        self.assertAlmostEqual(ghe.bhe.b.H, 130.13510780396268, places=2)

    def test_double_u_tube(self):

        # Define a borehole
        borehole = gt.boreholes.Borehole(self.H, self.D, self.r_b, x=0., y=0.)

        # Initialize GHE object
        g_function = ghedt.gfunction.compute_live_g_function(
            self.B, self.H_values, self.r_b_values, self.D_values,
            self.m_flow_borehole, self.DoubleUTube,
            self.log_time, self.coordinates, self.fluid, self.pipe_d,
            self.grout, self.soil)

        # Initialize the GHE object
        ghe = ghedt.ground_heat_exchangers.GHE(
            self.V_flow_system, self.B, self.DoubleUTube, self.fluid,
            borehole, self.pipe_d, self.grout, self.soil,
            g_function, self.sim_params, self.hourly_extraction_ground_loads)

        max_HP_EFT, min_HP_EFT = ghe.simulate(method='hybrid')

        self.assertEqual(37.97229212228275, max_HP_EFT)
        self.assertEqual(16.989189768401793, min_HP_EFT)

        ghe.size(method='hybrid')

        self.assertAlmostEqual(ghe.bhe.b.H, 121.85659366084005, places=2)

    def test_coaxial_tube(self):

        # Define a borehole
        borehole = gt.boreholes.Borehole(self.H, self.D, self.r_b, x=0., y=0.)

        # Initialize GHE object
        g_function = ghedt.gfunction.compute_live_g_function(
            self.B, self.H_values, self.r_b_values, self.D_values,
            self.m_flow_borehole, self.CoaxialTube,
            self.log_time, self.coordinates, self.fluid, self.pipe_c,
            self.grout, self.soil)

        # Re-Initialize the GHE object
        ghe = ghedt.ground_heat_exchangers.GHE(
            self.V_flow_system, self.B, self.CoaxialTube, self.fluid,
            borehole, self.pipe_c, self.grout, self.soil,
            g_function, self.sim_params, self.hourly_extraction_ground_loads)

        max_HP_EFT, min_HP_EFT = ghe.simulate(method='hybrid')

        self.assertAlmostEqual(37.73503652828782, max_HP_EFT)
        self.assertAlmostEqual(17.61422171583412, min_HP_EFT)

        ghe.size(method='hybrid')

        self.assertAlmostEqual(ghe.bhe.b.H, 120.89971616555863, places=2)
