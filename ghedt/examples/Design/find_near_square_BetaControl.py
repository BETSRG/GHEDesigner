# Jack C. Cook
# Sunday, December 26, 2021

# Purpose: Design a square or near-square field using the common design
# interface with a single U-tube, multiple U-tube and coaxial tube.

# This search is described in section 4.3.2 of Cook (2021) from pages 123-129.

import ghedt as dt
import ghedt.peak_load_analysis_tool as plat
import pygfunction as gt
import pandas as pd
from time import time as clock
from ghedt import Output
import csv



def main():
    #heights = [133.64,154.159,165.215,174.941,182.765,189.053,194.846,199.787,203.887,207.207,209.92,212.246,214.288]
    heights = [165]
    outputArray = []
    outputArray.append(["AvgH", "fieldSpecifier","Total Drilling"])
    for H in heights:
        pN = "Atlanta Office Building"
        notes = "Why?"
        author = "Jeremy Johnson"
        mN = "V1"
        # Borehole dimensions
        # -------------------
        D = 2.  # Borehole buried depth (m)
        r_b = 0.075  # Borehole radius (m)
        B = 5.  # Borehole spacing (m)

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
        r_inner = [r_in_in, r_in_out]  # The radii of the inner pipe from in to out
        r_outer = [r_out_in,
                   r_out_out]  # The radii of the outer pipe from in to out

        # Pipe positions
        # --------------
        # Single U-tube [(x_in, y_in), (x_out, y_out)]
        pos_single = plat.media.Pipe.place_pipes(s, r_out, 1)
        # Single U-tube BHE object
        single_u_tube = plat.borehole_heat_exchangers.SingleUTube
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
        pipe_single = \
            plat.media.Pipe(pos_single, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        pipe_double = \
            plat.media.Pipe(pos_double, r_in, r_out, s, epsilon, k_p, rhoCp_p)
        pipe_coaxial = \
            plat.media.Pipe(pos_coaxial, r_inner, r_outer, 0, epsilon, k_p_coax,
                            rhoCp_p)
        # Soil
        ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
        soil = plat.media.Soil(k_s, rhoCp_s, ugt)
        # Grout
        grout = plat.media.Grout(k_g, rhoCp_g)

        # Inputs related to fluid
        # -----------------------
        # Fluid properties
        mixer = 'MEG'  # Ethylene glycol mixed with water
        percent = 0.  # Percentage of ethylene glycol added in
        fluid = gt.media.Fluid(mixer=mixer, percent=percent)

        # Fluid properties
        V_flow = 0.2  # Volumetric flow rate (L/s)
        # Note: The flow parameter can be borehole or system.
        flow = 'borehole'

        # Define a borehole
        borehole = gt.boreholes.Borehole(96.0, D, r_b, x=0., y=0.)

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
        max_Height = H  # in meters
        min_Height = 24  # in meters
        sim_params = plat.media.SimulationParameters(
            start_month, end_month, max_EFT_allowable, min_EFT_allowable,
            max_Height, min_Height)

        # Process loads from file
        # -----------------------
        # read in the csv file and convert the loads to a list of length 8760
        hourly_extraction: dict = \
            pd.read_csv('../Atlanta_Office_Building_Loads.csv').to_dict('list')
        # Take only the first column in the dictionary
        hourly_extraction_ground_loads: list = \
            hourly_extraction[list(hourly_extraction.keys())[0]]

        """ Geometric constraints for the `near-square` routine.
        Required geometric constraints for the uniform rectangle design:
          - B
          - length
        """
        # B is already defined above
        #number_of_boreholes = 32
        #length = dt.utilities.length_of_side(number_of_boreholes, B)
        #geometric_constraints = dt.media.GeometricConstraints(B=B, length=length)
        length = 12*5
        width = 11*5
        B_min = 1
        B_max = 20
        geometric_constraints = dt.media.GeometricConstraints(length=length,width = width, B_min = B_min, B_max_x=B_max)
        # Single U-tube
        # -------------
        #design_single_u_tube = dt.design.Design(
            #V_flow, borehole, single_u_tube, fluid, pipe_single, grout,
            #soil, sim_params, geometric_constraints, hourly_extraction_ground_loads,
            #method='hybrid', flow=flow, routine='near-square')
        design_single_u_tube = dt.design.Design(
            V_flow, borehole, single_u_tube, fluid, pipe_single, grout,
            soil, sim_params, geometric_constraints, hourly_extraction_ground_loads,
            flow=flow, routine='rectangle')
        # Find the near-square design for a single U-tube and size it.
        tic = clock()
        bisection_search = design_single_u_tube.find_design(disp=True)
        bisection_search.ghe.compute_g_functions()
        bisection_search.ghe.size(method='hybrid')
        toc = clock()
        ghe = bisection_search.ghe
        gfunction_obj = bisection_search.ghe.GFunction
        totalDrilling = ghe.bhe.b.H*ghe.nbh
        outputArray.append(
            [ghe.averageHeight(), ghe.fieldSpecifier, totalDrilling])
        gfOA = []
        gfROne = ["ln(t/ts)"]
        for key in gfunction_obj.g_lts:
            gfROne.append(str(key))
        gfOA.append(gfROne)
        logTime = gfunction_obj.log_time
        for i in range(len(logTime)):
            gfRow = [logTime[i]]
            for key in gfunction_obj.g_lts:
                gfRow.append(gfunction_obj.g_lts[key][i])
            gfRow.append(gfunction_obj.g_function_interpolation(ghe.B_spacing / ghe.averageHeight())[0][i])
            gfOA.append(gfRow)
        with open("Gfunctions_H" + str(H) + ".csv", "w", newline="") as OutputFile:
            cW = csv.writer(OutputFile)
            cW.writerows(gfOA)
    with open("BetaStudyResults.csv", "w", newline="") as OutputFile:
        cW = csv.writer(OutputFile)
        cW.writerows(outputArray)


if __name__ == '__main__':
    main()
