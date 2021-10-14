# Jack C. Cook
# Friday, September 17, 2021

import PLAT
import matplotlib.pyplot as plt
import pandas as pd
import PLAT.pygfunction as gt
import gFunctionDatabase as gfdb
import GLHEDT


def main():
    # --------------------------------------------------------------------------

    # Borehole dimensions
    # -------------------
    H = 100.  # Borehole length (m)
    D = 2.  # Borehole buried depth (m)
    r_b = 150. / 1000. / 2.  # Borehole radius
    B = 3.65  # m

    # Pipe dimensions
    # ---------------
    r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
    r_in = 21.6 / 1000. / 2.  # Pipe inner radius (m)
    s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # --------------
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos = PLAT.media.Pipe.place_pipes(s, r_out, 1)

    # Thermal conductivities
    # ----------------------
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)

    # Volumetric heat capacities
    # --------------------------
    rhoCp_p = 1542. * 1000.  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # ------------------
    pipe = PLAT.media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp=rhoCp_p)
    soil = PLAT.media.ThermalProperty(k=k_s, rhoCp=rhoCp_s)
    grout = PLAT.media.ThermalProperty(k=k_g, rhoCp=rhoCp_g)

    # Number in the x and y
    # ---------------------
    N = 11
    M = 19
    configuration = 'zoned'
    nbh = N * M

    # Inputs related to fluid
    # -----------------------
    V_flow_system = 15.  # System volumetric flow rate (L/s)
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in

    # Simulation start month and end month
    # ------------------------------------
    start_month = 1
    n_years = 20
    end_month = n_years * 12
    # Initial ground temperature
    ugt = 18.3  # undisturbed ground temperature in Celsius
    # Maximum and minimum allowable fluid temperatures
    max_EFT_allowable = 35  # degrees Celsius
    min_EFT_allowable = 5  # degrees Celsius
    # Maximum and minimum allowable heights
    max_Height = 200  # in meters
    min_Height = 50  # in meters

    # Process loads from file
    # -----------------------
    # read in the csv file and convert the loads to a list of length 8760
    hourly_extraction: dict = \
        pd.read_csv('Atlanta_Office_Building_Loads.csv').to_dict('list')
    # Take only the first column in the dictionary
    hourly_extraction_loads: list = \
        hourly_extraction[list(hourly_extraction.keys())[0]]

    # --------------------------------------------------------------------------

    # Borehole heat exchanger
    # -----------------------
    # Fluid properties
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)
    # Volumetric flow rate per borehole (L/s)
    V_flow_borehole = V_flow_system / nbh
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = V_flow_borehole / 1000. * fluid.rho

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    single_u_tube = PLAT.borehole_heat_exchangers.SingleUTube(
        m_flow_borehole, fluid, borehole, soil, grout, pipe)

    # Radial Numerical short time step g-function
    # -------------------------------------------
    # Compute short time step now that the BHE is defined
    # Compute short time step
    radial_numerical = \
        PLAT.radial_numerical_borehole.RadialNumericalBH(single_u_tube)

    radial_numerical.calc_sts_g_functions(single_u_tube)

    # Hybrid load
    # -----------
    # Split the extraction loads into heating and cooling for input to the
    # HybridLoad object
    hourly_rejection_loads, hourly_extraction_loads = \
        PLAT.ground_loads.HybridLoad.split_heat_and_cool(
            hourly_extraction_loads)

    hybrid_load = PLAT.ground_loads.HybridLoad(
        hourly_rejection_loads, hourly_extraction_loads, single_u_tube,
        radial_numerical, start_month, end_month)

    # GFunction
    # ---------
    # Access the database for specified configuration
    r = gfdb.Management.retrieval.Retrieve(configuration)
    # There is just one value returned in the unimodal domain for rectangles
    r_unimodal = r.retrieve(N, M)
    key = list(r_unimodal.keys())[-14]
    print('The key value: {}'.format(key))
    r_data = r_unimodal[key]

    # Configure the database data for input to the goethermal GFunction object
    geothermal_g_input = \
        GLHEDT.geothermal.GFunction.configure_database_file_for_usage(r_data)

    # Initialize the GFunction object
    GFunction = GLHEDT.geothermal.GFunction(**geothermal_g_input)

    # Hybrid GLHE
    # -----------
    # Initialize a HybridGLHE
    HybridGLHE = GLHEDT.ground_heat_exchangers.HybridGLHE(single_u_tube,
                                                          radial_numerical,
                                                          hybrid_load,
                                                          GFunction)
    HybridGLHE.size(
        max_Height, min_Height, max_EFT_allowable, min_EFT_allowable, B=B)

    print('The sized height: {}'.format(single_u_tube.b.H))
    print('Number of boreholes: {}'.format(len(GFunction.bore_locations)))
    print('Total depth: {}'.format(single_u_tube.b.H * len(GFunction.bore_locations)))

    # Plot go and no-go zone with corrected borefield
    # -----------------------------------------------
    new_coordinates = GFunction.correct_coordinates(B)

    perimeter = [[0., 0.], [70.104, 0.], [70.104, 80.772], [0., 80.772]]
    no_go = [[9.997, 36.51], [9.997, 69.79], [59.92, 69.79], [59.92, 36.51]]

    fig, ax = GFunction.visualize_area_and_constraints(perimeter,
                                                       new_coordinates,
                                                       no_go=no_go)

    fig.savefig('zoned_rectangle_case_1.png')

    # Plot long time step g-functions
    # -------------------------------
    fig, ax = GFunction.visualize_g_functions()

    B_over_H = B / single_u_tube.b.H
    # interpolate for the Long time step g-function
    g_function, rb_value, D_value, H_eq = \
        GFunction.g_function_interpolation(B_over_H)
    # correct the long time step for borehole radius
    g_function_corrected = \
        GFunction.borehole_radius_correction(g_function,
                                             rb_value,
                                             single_u_tube.b.r_b)
    ax.plot(GFunction.log_time, g_function_corrected, '--')

    fig.savefig('zoned_rectangle_case_1_gFunctions.png')


if __name__ == '__main__':
    main()

