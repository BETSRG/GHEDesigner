# Jack C. Cook
# Saturday, October 9, 2021

import GLHEDT.PLAT as PLAT
import matplotlib.pyplot as plt
import pandas as pd
import GLHEDT.PLAT.pygfunction as gt
import gFunctionDatabase as gfdb
import GLHEDT


def main():
    # --------------------------------------------------------------------------

    # Borehole dimensions
    # -------------------
    H = 100.  # Borehole length (m)
    D = 2.  # Borehole buried depth (m)
    r_b = 150. / 1000. / 2.  # Borehole radius]
    B = 5.  # Borehole spacing (m)

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
    # Pipe
    pipe = PLAT.media.Pipe(pos, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = PLAT.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = PLAT.media.ThermalProperty(k_g, rhoCp_g)

    # Number in the x and y
    # ---------------------
    N = 12
    M = 13
    configuration = 'rectangle'
    nbh = N * M

    # Inputs related to fluid
    # -----------------------
    V_flow_system = 31.2  # System volumetric flow rate (L/s)
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in

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
    max_Height = 100  # in meters
    min_Height = 60  # in meters
    sim_params = PLAT.media.SimulationParameters(
        start_month, end_month, max_EFT_allowable, min_EFT_allowable,
        max_Height, min_Height)

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
        m_flow_borehole, fluid, borehole, pipe, grout, soil)

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
        radial_numerical, sim_params)

    # GFunction
    # ---------
    # Access the database for specified configuration
    r = gfdb.Management.retrieval.Retrieve(configuration)
    # There is just one value returned in the unimodal domain for rectangles
    r_unimodal = r.retrieve(N, M)
    key = list(r_unimodal.keys())[0]
    print('The key value: {}'.format(key))
    r_data = r_unimodal[key]

    # Configure the database data for input to the goethermal GFunction object
    geothermal_g_input = gfdb.Management.\
        application.GFunction.configure_database_file_for_usage(r_data)

    # Initialize the GFunction object
    GFunction = gfdb.Management.application.GFunction(**geothermal_g_input)

    # Hybrid GLHE
    # -----------
    # Initialize a HybridGLHE
    # Initialize a HybridGLHE
    HybridGLHE = GLHEDT.ground_heat_exchangers.HybridGLHE(
        single_u_tube, radial_numerical, hybrid_load, GFunction, sim_params)

    # --------------------------------------------------------------------------

    # Range through height values and return the T_excess value
    height_values = [48. + float(12 * i) for i in range(0, 16)]

    T_excess_values = []

    for height in height_values:
        HybridGLHE.bhe.b.H = height
        max_HP_EFT, min_HP_EFT = HybridGLHE.simulate(B)
        T_excess = HybridGLHE.cost(max_HP_EFT, min_HP_EFT)
        T_excess_values.append(T_excess)

    # Plot excess values
    # ------------------
    fig, ax = plt.subplots()

    ax.plot(height_values, T_excess_values, marker='o', ls='--')

    ax.set_xlabel('Borehole height (m)')
    ax.set_ylabel('Excess fluid temperature ($\degree$C)')

    ax.grid()
    ax.set_axisbelow(True)

    fig.tight_layout()

    fig.savefig('range_excess.png')


if __name__ == '__main__':
    main()
