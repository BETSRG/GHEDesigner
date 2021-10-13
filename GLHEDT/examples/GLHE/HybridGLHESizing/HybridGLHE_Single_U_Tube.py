# Jack C. Cook
# Wednesday, October 13, 2021


import GLHEDT.PLAT as PLAT
import matplotlib.pyplot as plt
import pandas as pd
import GLHEDT.PLAT.pygfunction as gt
import gFunctionDatabase as gfdb
import GLHEDT
from time import time as clock


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
    # Single U-tube BHE object
    bhe_object = PLAT.borehole_heat_exchangers.SingleUTube

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
    geothermal_g_input = gfdb.Management. \
        application.GFunction.configure_database_file_for_usage(r_data)

    # Initialize the GFunction object
    GFunction = gfdb.Management.application.GFunction(**geothermal_g_input)

    # Inputs related to fluid
    # -----------------------
    V_flow_system = 31.2  # System volumetric flow rate (L/s)
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in
    # Fluid properties
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

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
    max_Height = 150  # in meters
    min_Height = 60  # in meters
    sim_params = PLAT.media.SimulationParameters(
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

    # --------------------------------------------------------------------------

    # Initialize Hybrid GLHE object
    HybridGLHE = GLHEDT.ground_heat_exchangers._HybridGLHE(
        V_flow_system, B, bhe_object, fluid, borehole, pipe, grout, soil,
        GFunction, sim_params, hourly_extraction_ground_loads)

    HybridGLHE.size()

    print(HybridGLHE.bhe.b.H)

    GLHE_info = HybridGLHE.__repr__()

    file = open('SingleUTube-HybridGLHE-info.txt', 'w+')
    file.write(GLHE_info)
    file.close()

    # Borehole heat exchanger
    # -----------------------

    # # Volumetric flow rate per borehole (L/s)
    # V_flow_borehole = V_flow_system / nbh
    # # Total fluid mass flow rate per borehole (kg/s)
    # m_flow_borehole = V_flow_borehole / 1000. * fluid.rho
    #
    #
    #
    # single_u_tube = PLAT.borehole_heat_exchangers.SingleUTube(
    #     m_flow_borehole, fluid, borehole, pipe, grout, soil)
    #
    # # Radial Numerical short time step g-function
    # # -------------------------------------------
    # # Compute short time step now that the BHE is defined
    # # Compute short time step
    # radial_numerical = \
    #     PLAT.radial_numerical_borehole.RadialNumericalBH(single_u_tube)
    #
    # radial_numerical.calc_sts_g_functions(single_u_tube)
    #
    # # Hybrid load
    # # -----------
    # # Split the extraction loads into heating and cooling for input to the
    # # HybridLoad object
    # hourly_rejection_loads, hourly_extraction_loads = \
    #     PLAT.ground_loads.HybridLoad.split_heat_and_cool(
    #         hourly_extraction_loads)
    #
    # hybrid_load = PLAT.ground_loads.HybridLoad(
    #     hourly_rejection_loads, hourly_extraction_loads, single_u_tube,
    #     radial_numerical, sim_params)
    #
    #
    #
    #
    #
    # # Hybrid GLHE
    # # -----------
    # # Initialize a HybridGLHE
    # HybridGLHE = GLHEDT.ground_heat_exchangers.HybridGLHE(
    #     single_u_tube, radial_numerical, hybrid_load, GFunction, sim_params)

    # --------------------------------------------------------------------------

    # Plot the simulation results
    # # ---------------------------
    # fig, ax = plt.subplots()
    #
    # heat_pump_EFT = HybridGLHE.HPEFT[2:]
    # months = range(1, len(heat_pump_EFT) + 1)
    #
    # min_HP_EFT_idx = HybridGLHE.HPEFT.index(min_HP_EFT) - 1
    # max_HP_EFT_idx = HybridGLHE.HPEFT.index(max_HP_EFT) - 1
    #
    # ax.plot(months, heat_pump_EFT, 'k')
    # ax.scatter(min_HP_EFT_idx, min_HP_EFT, color='b', marker='X', s=200,
    #            label='Minimum Temperature')
    # ax.scatter(max_HP_EFT_idx, max_HP_EFT, color='r',  marker='P', s=200,
    #            label='Maximum Temperature')
    #
    # ax.set_xlabel('Month number')
    # ax.set_ylabel('Heat pump entering fluid temperature ($\degree$C)')
    #
    # ax.grid()
    # ax.set_axisbelow(True)
    #
    # fig.legend(bbox_to_anchor=(.5, .95))
    #
    # fig.tight_layout()
    #
    # fig.savefig('hybrid_monthly_simulation.png')
    #
    # # Plot the hourly load profile
    # # ---------------------
    # fig = HybridGLHE.hybrid_load.visualize_hourly_heat_extraction()
    #
    # fig.savefig('Atlanta_Office_Building_extraction_loads.png')
    #
    # # Plot the hybrid load representation
    # # -----------------------------------
    # fig, ax = plt.subplots()
    #
    # ax.plot(HybridGLHE.hybrid_load.load)
    #
    # ax.set_xlabel('Month number')
    # ax.set_ylabel('Monthly load (kW)')
    #
    # fig.tight_layout()
    #
    # fig.savefig('monthly_load_representation.png')


if __name__ == '__main__':
    main()
