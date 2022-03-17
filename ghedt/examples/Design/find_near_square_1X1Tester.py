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
import numpy as np
import csv


def main():
    pN = "Atlanta Office Building"
    notes = "Why?"
    author = "Jeremy Johnson"
    mN = "V1"
    # Borehole dimensions
    # -------------------
    H = 96.  # Borehole length (m)
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
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

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

    """ Geometric constraints for the `near-square` routine.
    Required geometric constraints for the uniform rectangle design:
      - B
      - length
    """
    # B is already defined above
    number_of_boreholes = 32
    length = dt.utilities.length_of_side(number_of_boreholes, B)
    geometric_constraints = dt.media.GeometricConstraints(B=B, length=length)
    loadIterStart = .005
    loadIterStop = .1
    nPoints = 100
    loadIterStep = (loadIterStop-loadIterStart)/nPoints
    currentLoad = loadIterStart
    loadVals = []
    while currentLoad < loadIterStop:
        loadVals.append(currentLoad)
        currentLoad += loadIterStep
    print("Number of Designs: ",len(loadVals))
    outputArray = [["Load Multiplier","Field","Field Depth","Excess Temperature","FieldLength"]]
    unloadedLoadings = np.array(hourly_extraction_ground_loads)
    for loadMult in loadVals:
        loads = unloadedLoadings*loadMult
        # Single U-tube
        # -------------
        design_single_u_tube = dt.design.Design(
            V_flow, borehole, single_u_tube, fluid, pipe_single, grout,
            soil, sim_params, geometric_constraints, loads.tolist(),
            method='hybrid', flow=flow, routine='near-square')

        # Find the near-square design for a single U-tube and size it.
        tic = clock()
        bisection_search = design_single_u_tube.find_design(disp=True)
        bisection_search.ghe.compute_g_functions()
        bisection_search.ghe.size(method='hybrid')
        toc = clock()
        '''
        subtitle = '* Single U-tube'
        print(subtitle + '\n' + len(subtitle) * '-')
        print('Calculation time: {0:.2f} seconds'.format(toc - tic))
        print('Height: {0:.4f} meters'.format(bisection_search.ghe.bhe.b.H))
        nbh = len(bisection_search.ghe.GFunction.bore_locations)
        print('Number of boreholes: {}'.format(nbh))
        print('Total Drilling: {0:.1f} meters\n'.
              format(bisection_search.ghe.bhe.b.H * nbh))
        '''

        ghe = bisection_search.ghe
        #print(ghe.GFunction.bore_locations)
        outputArray.append([loadMult,ghe.fieldSpecifier,ghe.bhe.b.H,ghe.cost(np.max(ghe.HPEFT),np.min(ghe.HPEFT)),len(ghe.GFunction.bore_locations)])

        with open("Output.csv","w",newline="") as outputFile:
            cW = csv.writer(outputFile)
            cW.writerows(outputArray)




if __name__ == '__main__':
    main()
