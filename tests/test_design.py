import copy
import unittest
import os

import ghedt as dt
import ghedt.peak_load_analysis_tool as plat
import pygfunction as gt

import pandas as pd


TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__),
                                 'Atlanta_Office_Building_Loads.csv')


class DesignBase:
    def __init__(self):
        # Borehole dimensions
        # -------------------
        H = 96.  # Borehole length (m)
        D = 2.  # Borehole buried depth (m)
        r_b = 0.075  # Borehole radius (m)

        # Pipe dimensions
        # ---------------
        # Single and Multiple U-tubes
        r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
        r_in = 21.6 / 1000. / 2.  # Pipe inner radius (m)
        s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
        epsilon = 1.0e-6  # Pipe roughness (m)
        # Coaxial tube
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

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos_single = plat.media.Pipe.place_pipes(s, r_out, 1)
        # Single U-tube BHE object
        self.single_u_tube = plat.borehole_heat_exchangers.SingleUTube
        # Double U-tube
        pos_double = plat.media.Pipe.place_pipes(s, r_out, 2)
        double_u_tube = plat.borehole_heat_exchangers.MultipleUTube
        # Coaxial tube
        pos_coaxial = (0, 0)
        coaxial_tube = plat.borehole_heat_exchangers.CoaxialPipe

        # Thermal conductivities
        # ----------------------
        k_p = 0.4  # Pipe thermal conductivity (W/m.K)
        k_p_coax = [0.4, 0.4]  # Pipes thermal conductivity (W/m.K)
        k_s = 2.0  # Ground thermal conductivity (W/m.K)
        k_g = 1.0  # Grout thermal conductivity (W/m.K)

        # Volumetric heat capacities
        # --------------------------
        rhoCp_p = 1542. * 1000.  # Pipe volumetric heat capacity (J/K.m3)
        rhoCp_s = 2343.493 * 1000.  # Soil volumetric heat capacity (J/K.m3)
        rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)

        # Thermal properties
        # ------------------
        # Pipe
        self.pipe_single = \
            plat.media.Pipe(pos_single, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        pipe_double = \
            plat.media.Pipe(pos_double, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        pipe_coaxial = \
            plat.media.Pipe(pos_coaxial, r_inner, r_outer, 0, epsilon, k_p_coax,
                            rhoCp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        self.soil = plat.media.Soil(k_s, rhoCp_s, ugt)
        # Grout
        self.grout = plat.media.Grout(k_g, rhoCp_g)

        # Inputs related to fluid
        # -----------------------
        # Fluid properties
        mixer = 'MEG'  # Ethylene glycol mixed with water
        percent = 0.  # Percentage of ethylene glycol added in
        self.fluid = gt.media.Fluid(mixer=mixer, percent=percent)

        # Fluid flow rate
        V_flow = 0.2  # Borehole volumetric flow rate (L/s)
        self.V_flow_borehole = copy.deepcopy(V_flow)

        # Define a borehole
        self.borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

        # Simulation parameters
        # ---------------------
        # Simulation start month and end month
        start_month = 1
        n_years = 20
        end_month = n_years * 12
        # Maximum and minimum allowable fluid temperatures
        max_EFT_allowable = 35  # degrees Celsius
        min_EFT_allowable = 5  # degrees Celsius
        # Maximum and minimum allowable heights
        max_Height = 135.  # in meters
        min_Height = 60  # in meters
        self.sim_params = plat.media.SimulationParameters(
            start_month, end_month, max_EFT_allowable, min_EFT_allowable,
            max_Height, min_Height)


class TestNearSquare(unittest.TestCase, DesignBase):

    def setUp(self) -> None:

        DesignBase.__init__(self)
        # Note: Based on these inputs, the resulting near-square test will
        # determine a system with 156 boreholes.
        self.V_flow_system = self.V_flow_borehole * 156.

        # Process loads from file
        # -----------------------
        # read in the csv file and convert the loads to a list of length 8760
        hourly_extraction: dict = \
            pd.read_csv(TESTDATA_FILENAME).to_dict('list')
        # Take only the first column in the dictionary
        self.hourly_extraction_ground_loads: list = \
            hourly_extraction[list(hourly_extraction.keys())[0]]

        # Geometric constraints for the `near-square` routine
        # Required geometric constraints for the uniform rectangle design: B
        B = 5.  # Borehole spacing (m)
        number_of_boreholes = 32
        length = dt.utilities.length_of_side(number_of_boreholes, B)
        self.geometric_constraints = \
            dt.media.GeometricConstraints(B=B, length=length)

    def test_design_selection(self):
        # Single U-tube
        # -------------
        # Design a single U-tube with a system volumetric flow rate
        design_single_u_tube_a = dt.design.Design(
            self.V_flow_system, self.borehole, self.single_u_tube, self.fluid,
            self.pipe_single, self.grout, self.soil, self.sim_params,
            self.geometric_constraints, self.hourly_extraction_ground_loads,
            flow='system', routine='near-square')
        # Find the near-square design for a single U-tube and size it.
        bisection_search = design_single_u_tube_a.find_design()
        bisection_search.ghe.compute_g_functions()
        bisection_search.ghe.size(method='hybrid')
        H_single_u_tube_a = bisection_search.ghe.bhe.b.H

        design_single_u_tube_b = dt.design.Design(
            self.V_flow_borehole, self.borehole, self.single_u_tube, self.fluid,
            self.pipe_single, self.grout, self.soil, self.sim_params,
            self.geometric_constraints, self.hourly_extraction_ground_loads,
            flow='borehole', routine='near-square')
        # Find the near-square design for a single U-tube and size it.
        bisection_search = design_single_u_tube_b.find_design()
        bisection_search.ghe.compute_g_functions()
        bisection_search.ghe.size(method='hybrid')
        H_single_u_tube_b = bisection_search.ghe.bhe.b.H

        # Verify that the `flow` toggle is properly working
        self.assertAlmostEqual(H_single_u_tube_a, H_single_u_tube_b, places=8)
        # Verify that the proper height as been found
        # Note: This reference was calculated on MacOS. It seems that on Linux
        # the values are not equal starting around the 9th decimal place.
        H_reference = 130.18183587536208
        self.assertAlmostEqual(H_reference, H_single_u_tube_a, places=8)
