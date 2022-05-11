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
from ghedt import ground_heat_exchangers
from ghedt import gfunction
import csv
import numpy as np
from math import pi


def main():
    pN = "Atlanta Office Building"
    notes = "Why?"
    author = "Jeremy Johnson"
    mN = "V1"
    # Borehole dimensions
    # -------------------
    BetaStart = 1
    BetaStop = 1.2
    nBeta = 1
    cB = BetaStart
    BetaVals = []
    if nBeta == 1:
        BetaVals.append(BetaStart)
    else:
        BetaStep = (BetaStop - BetaStart) / nBeta
        while cB <= BetaStop:
            BetaVals.append(cB)
            cB += BetaStep

    TiltStart = 0
    TiltStop = 60*(pi/180.)
    nTilts = 4
    tiltVals = []
    if nTilts == 1:
        tiltVals.append(TiltStart)
    else:
        tiltStep = (TiltStop-TiltStart)/nTilts
        cT = TiltStart
        while cT <= TiltStop:
            tiltVals.append(cT)
            cT += tiltStep
    outputArray = []
    outputArray.append(["AvgH","1/Beta","Beta","Tilt","fieldSpecifier","h1","h2","Total Drilling","Computation Time"])
    for BetaVal in BetaVals:
        for TiltVal in tiltVals:
            avgH = 96.  # Borehole length (m)
            D = 2.  # Borehole buried depth (m)
            r_b = 0.075  # Borehole radius (m)
            B = 5.  # Borehole spacing (m)
            Betas = [1,1,1,1,1,1,1,1,1,1/BetaVal]
            m=6
            n=6

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
            #pos_double = plat.media.Pipe.place_pipes(s, r_out, 2)
            #double_u_tube = plat.borehole_heat_exchangers.MultipleUTube
            # Coaxial tube
            #pos_coaxial = (0, 0)
            #coaxial_tube = plat.borehole_heat_exchangers.CoaxialPipe

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
            #pipe_double = \
                #plat.media.Pipe(pos_double, r_in, r_out, s, epsilon, k_p, rhoCp_p)
            #pipe_coaxial = \
               # plat.media.Pipe(pos_coaxial, r_inner, r_outer, 0, epsilon, k_p_coax,
                #                rhoCp_p)
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
            #borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)
            print("Tilt Value: ",TiltVal)
            coords,indices,boreholes = ground_heat_exchangers.multiDepthSquare(avgH,Betas,m,n,D,r_b,spacing=B,tilt=TiltVal)

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
            max_Height = 384.  # in meters
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
            hourly_extraction_ground_loads = np.array(hourly_extraction_ground_loads)
            hourly_extraction_ground_loads = .1*hourly_extraction_ground_loads
            hourly_extraction_ground_loads = list(hourly_extraction_ground_loads)

            """ Geometric constraints for the `near-square` routine.
            Required geometric constraints for the uniform rectangle design:
              - B
              - length
            """
            tic = clock()
            mflow = V_flow / 1000. * fluid.rho
            logTime = [-8.5, -7.8, -7.2, -6.5, -5.9, -5.2, -4.5, -3.963, -3.27, -2.864, -2.577, -2.171, -1.884
                , -1.191, -0.497, -0.274, -0.051, 0.196, 0.419, 0.642, 0.873, 1.112, 1.335, 1.679, 2.028, 2.275, 3.003]
            print("Starting Gfunction Calculation")
            gfunction_obj = gfunction.compute_live_g_function_dH(B,[24,48,96,192,384],indices,Betas
                                                                 ,[.075,.075,.075,.08,.0875],
                                                                 [2,2,2,2,2],mflow,single_u_tube,logTime,coords,fluid,pipe_single,grout,soil,boreholes,solver='similarities')
            print("Finished Gfunction Calculation")
            # B is already defined above
            #number_of_boreholes = 32
            #length = dt.utilities.length_of_side(number_of_boreholes, B)
            #geometric_constraints = dt.media.GeometricConstraints(B=B, length=length)

            # Single U-tube
            # -------------
            #design_single_u_tube = dt.design.Design(
             #   [V_flow,V_flow], boreholes,Betas, single_u_tube, fluid, pipe_single, grout,
              #  soil, sim_params, geometric_constraints, hourly_extraction_ground_loads,
               # method='hybrid', flow=flow, routine='near-square')
            #bisection_search = design_single_u_tube.find_design(disp=True)
            #bisection_search.ghe.compute_g_functions()
            #bisection_search.ghe.size(method="hybrid")
            # Find the near-square design for a single U-tube and size it.

            print("Instantiating Ground Heat Exchanger...")
            ghe = ground_heat_exchangers.GHE([V_flow]*len(Betas), B, single_u_tube
                                             , fluid, boreholes, indices, pipe_single, grout, soil
                                             , gfunction_obj, sim_params, hourly_extraction_ground_loads)
            print("Sizing Ground Heat Exchanger...")
            ghe.size()
            print("Calculations Done...")
            #bisection_search = design_single_u_tube.find_design(disp=True)
            #bisection_search.ghe.compute_g_functions()
            #bisection_search.ghe.size(method='hybrid')
            toc = clock()
            #subtitle = '* Single U-tube'
            #print(subtitle + '\n' + len(subtitle) * '-')
            #print('Calculation time: {0:.2f} seconds'.format(toc - tic))
            #print('Height: {0:.4f} meters'.format(bisection_search.ghe.bhe.b.H))
            #nbh = len(bisection_search.ghe.GFunction.bore_locations)
            #print('Number of boreholes: {}'.format(nbh))
            #print('Total Drilling: {0:.1f} meters\n'.
                  #format(bisection_search.ghe.bhe.b.H * nbh))

            # Plot the selected borehole coordinates for the single U-tube
            #ghe = bisection_search.ghe
            #coordinates = ghe.GFunction.bore_locations
            #fig, ax = dt.gfunction.GFunction.visualize_area_and_constraints(
                #[], coordinates, no_go=None)
            #fig.savefig('near-square.png', bbox_inches='tight', pad_inches=0.1)
            #Output.OutputDesignDetails(bisection_search,toc-tic,pN,notes,author,mN)
            print("Height: ",ghe.averageHeight())
            H = ghe.averageHeight()
            n1 = 2*(m)+2*(n-2)
            n2 = (m*n)-(2*m+2*(n-2))
            nSum = n1+n2
            nBSum = np.sum(ghe.templateIndexer(Betas,ghe.templateIndices))
            h1 = (H*nSum)/nBSum
            h2 = (H*nSum*(1.0/BetaVal))/nBSum
            totalDrilling = n1*h1 +n2*h2
            outputArray.append([ghe.averageHeight(), 1.0/BetaVal, BetaVal,TiltVal,h1,h2, totalDrilling,tic-toc])
            #gfOA = [["ln(t/ts)","24m","48m","96m","192m","384m",str(ghe.averageHeight())+"m"]]
            #for i in range(len(logTime)):
            #    gfRow = [logTime[i]]
            #    for key in gfunction_obj.g_lts:
            #        gfRow.append(gfunction_obj.g_lts[key][i])
            #    gfRow.append(gfunction_obj.g_function_interpolation(5.0/ghe.averageHeight())[0][i])
            #    gfOA.append(gfRow)
            #with open("Gfunctions_B"+str(BetaVal)+".csv","w",newline="") as OutputFile:
             #   cW = csv.writer(OutputFile)
             #   cW.writerows(gfOA)
    with open("BetaStudyResults.csv","w",newline="") as OutputFile:
        cW = csv.writer(OutputFile)
        cW.writerows(outputArray)
if __name__ == '__main__':
    main()
