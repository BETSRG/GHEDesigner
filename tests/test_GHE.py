# Jack C. Cook
# Monday, October 11, 2021

import unittest

import gFunctionDatabase as gfdb
import pygfunction as gt
import pandas as pd


class TestGHE(unittest.TestCase):

    def setUp(self) -> None:
        from GHEDT import PLAT
        # Borehole dimensions
        # -------------------
        self.H = 100.  # Borehole length (m)
        self.D = 2.  # Borehole buried depth (m)
        self.r_b = 150. / 1000. / 2.  # Borehole radius
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
        pos_s = PLAT.media.Pipe.place_pipes(s, r_out, 1)
        # Double U-tube
        pos_d = PLAT.media.Pipe.place_pipes(s, r_out, 2)
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
        self.pipe_s = PLAT.media.Pipe(pos_s, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        self.pipe_d = PLAT.media.Pipe(pos_d, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        self.pipe_c = \
            PLAT.media.Pipe(pos_c, r_inner, r_outer, s, epsilon, k_p_c, rhoCp_p)

        # Single U-tube BHE object
        self.SingleUTube = PLAT.borehole_heat_exchangers.SingleUTube
        # Double U-tube bhe object
        self.DoubleUTube = PLAT.borehole_heat_exchangers.MultipleUTube
        # Coaxial tube bhe object
        self.CoaxialTube = PLAT.borehole_heat_exchangers.CoaxialPipe

        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        self.soil = PLAT.media.Soil(k_s, rhoCp_s, ugt)
        # Grout
        self.grout = PLAT.media.ThermalProperty(k_g, rhoCp_g)

        # Number in the x and y
        # ---------------------
        N = 12
        M = 13
        configuration = 'rectangle'
        nbh = N * M

        # GFunction
        # ---------
        # Access the database for specified configuration
        r = gfdb.Management.retrieval.Retrieve(configuration)
        # There is just one value returned in the unimodal domain for rectangles
        r_unimodal = r.retrieve(N, M)
        key = list(r_unimodal.keys())[0]
        r_data = r_unimodal[key]

        # Configure the database data for input to the goethermal GFunction
        # object
        geothermal_g_input = gfdb.Management. \
            application.GFunction.configure_database_file_for_usage(r_data)

        # Initialize the GFunction object
        self.GFunction = \
            gfdb.Management.application.GFunction(**geothermal_g_input)

        # Inputs related to fluid
        # -----------------------
        self.V_flow_system = 31.2 # System volumetric flow rate (L/s)
        mixer = 'MEG'  # Ethylene glycol mixed with water
        percent = 0.  # Percentage of ethylene glycol added in

        # -----------------------
        # Fluid properties
        self.fluid = gt.media.Fluid(mixer=mixer, percent=percent)

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
        self.sim_params = PLAT.media.SimulationParameters(
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
        from GHEDT.ground_heat_exchangers import GHE

        # Define a borehole
        borehole = gt.boreholes.Borehole(self.H, self.D, self.r_b, x=0., y=0.)

        # Initialize Hybrid GLHE object
        GHE = GHE(
            self.V_flow_system, self.B, self.SingleUTube, self.fluid, borehole,
            self.pipe_s, self.grout, self.soil, self.GFunction, self.sim_params,
            self.hourly_extraction_ground_loads)

        max_HP_EFT, min_HP_EFT = GHE.simulate(method='Hybrid')

        self.assertEqual(38.7730129823955, max_HP_EFT)
        self.assertEqual(16.660969488710354, min_HP_EFT)

        GHE.size()

        self.assertAlmostEqual(GHE.bhe.b.H, 126.85677082230617, places=4)

    def test_double_u_tube(self):
        from GHEDT.ground_heat_exchangers import GHE

        # Define a borehole
        borehole = gt.boreholes.Borehole(self.H, self.D, self.r_b, x=0., y=0.)

        # Initialize Hybrid GLHE object
        GHE = GHE(
            self.V_flow_system, self.B, self.DoubleUTube, self.fluid, borehole,
            self.pipe_d, self.grout, self.soil, self.GFunction, self.sim_params,
            self.hourly_extraction_ground_loads)

        GHE.size()

        self.assertAlmostEqual(GHE.bhe.b.H, 120.37362456661276, places=4)

    def test_coaxial_tube(self):
        from GHEDT.ground_heat_exchangers import GHE

        # Define a borehole
        borehole = gt.boreholes.Borehole(self.H, self.D, self.r_b, x=0., y=0.)

        # Initialize Hybrid GLHE object
        GHE = GHE(
            self.V_flow_system, self.B, self.CoaxialTube, self.fluid, borehole,
            self.pipe_c, self.grout, self.soil, self.GFunction, self.sim_params,
            self.hourly_extraction_ground_loads)

        GHE.size()

        self.assertAlmostEqual(GHE.bhe.b.H, 119.57422578754375, places=4)
