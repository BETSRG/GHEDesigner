# Jack C. Cook
# Wednesday, October 13, 2021


import ghedt.PLAT as PLAT
import matplotlib.pyplot as plt
import pandas as pd
import ghedt.PLAT.pygfunction as gt
import gFunctionDatabase as gfdb
import ghedt
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

    # Inputs related to fluid
    # -----------------------
    V_flow_system = 31.2  # System volumetric flow rate (L/s)
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in
    # Fluid properties
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)
    m_flow_borehole = V_flow_system / 1000. * fluid.rho / (12. * 13.)

    height_values = [24., 48., 96., 192., 384.]
    r_b_values = [r_b] * len(height_values)
    D_values = [2.] * len(height_values)

    log_time = ghedt.utilities.Eskilson_log_times()

    N = 12
    M = 13
    coordinates = ghedt.coordinates.rectangle(N, M, B, B)

    GFunction = ghedt.gfunction.compute_live_g_function(
        B, height_values, r_b_values, D_values, m_flow_borehole, bhe_object,
        log_time, coordinates, fluid, pipe, grout, soil)

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

    # Constant hourly rejection ground loads
    # Process loads from file
    # -----------------------
    # read in the csv file and convert the loads to a list of length 8760
    hourly_extraction: dict = \
        pd.read_csv('../Atlanta_Office_Building_Loads.csv').to_dict('list')
    # Take only the first column in the dictionary
    hourly_extraction_ground_loads: list = \
        hourly_extraction[list(hourly_extraction.keys())[0]]

    # --------------------------------------------------------------------------

    # Initialize Hourly GLHE object
    GHE = ghedt.ground_heat_exchangers.GHE(
        V_flow_system, B, bhe_object, fluid, borehole, pipe, grout, soil,
        GFunction, sim_params, hourly_extraction_ground_loads)

    max_HP_EFT, min_HP_EFT = GHE.simulate(method='hybrid')

    print('Min EFT: {}\nMax EFT: {}'.format(min_HP_EFT, max_HP_EFT))

    # Plot the simulation results
    # ---------------------------
    fig = gt.gfunction._initialize_figure()
    ax = fig.add_subplot(111)
    gt.utilities._format_axes(ax)

    min_HP_EFT_idx = GHE.HPEFT.index(min_HP_EFT) - 1
    max_HP_EFT_idx = GHE.HPEFT.index(max_HP_EFT) - 1

    hours = GHE.hybrid_load.hour[2:].tolist()
    print('Number of points in load: {}'.format(len(hours)))
    years = [hours[i] / 8760 for i in range(len(hours))]

    ax.plot(years, GHE.HPEFT, 'k', zorder=1)

    ax.scatter(years[min_HP_EFT_idx], min_HP_EFT, color='b', marker='X', s=200,
               label='Minimum Temperature')
    ax.scatter(years[max_HP_EFT_idx], max_HP_EFT, color='r', marker='P', s=200,
               label='Maximum Temperature')

    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('Heat pump entering fluid temperature ($\degree$C)')

    ax.grid()
    ax.set_axisbelow(True)

    fig.legend(bbox_to_anchor=(.5, .95))

    fig.tight_layout()

    fig.savefig('hybrid_monthly_simulation.png')

    # Plot the hourly load profile
    # ---------------------
    fig = GHE.hybrid_load.visualize_hourly_heat_extraction()

    fig.savefig('Atlanta_Office_Building_extraction_loads.png')

    # Plot the hybrid load representation
    # -----------------------------------
    fig = gt.gfunction._initialize_figure()
    ax = fig.add_subplot(111)

    ax.plot(years, GHE.hybrid_load.load[2:])

    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Ground rejection load (kW)')

    fig.tight_layout()

    fig.savefig('monthly_load_representation.png')

    # ----------------------------------------------------------
    # Now simulate with a sized height
    GHE.size(method='hybrid')

    print(GHE.bhe.b.H)

    max_HP_EFT, min_HP_EFT = GHE.simulate(method='hybrid')

    print('Min EFT: {}\nMax EFT: {}'.format(min_HP_EFT, max_HP_EFT))

    # Plot the simulation results
    # ---------------------------
    fig = gt.gfunction._initialize_figure()
    ax = fig.add_subplot(111)
    gt.utilities._format_axes(ax)

    min_HP_EFT_idx = GHE.HPEFT.index(min_HP_EFT) - 1
    max_HP_EFT_idx = GHE.HPEFT.index(max_HP_EFT) - 1

    hours = GHE.hybrid_load.hour[2:].tolist()
    years = [hours[i] / 8760 for i in range(len(hours))]

    ax.plot(years, GHE.HPEFT, 'k', zorder=1, label='Heat Pump Entering Fluid')

    # ax.scatter(years[min_HP_EFT_idx], min_HP_EFT, color='b', marker='X', s=200,
    #            label='Minimum Temperature')
    # ax.scatter(years[max_HP_EFT_idx], max_HP_EFT, color='r', marker='P', s=200,
    #            label='Maximum Temperature')

    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('Temperature ($\degree$C)')

    ax.hlines(y=35, xmin=-4, xmax=25, color='r', linestyle='--',
              label='Max EFT Allowable')

    ax.grid()
    ax.set_axisbelow(True)

    fig.legend(bbox_to_anchor=(.48, .90))

    ax.set_xlim([-2, 22])

    fig.tight_layout()

    fig.savefig('hybrid_monthly_simulation_sized.png')

    # Plot borefield
    Nx = 12
    Ny = 13
    Bx = 5
    By = 5
    coordinates = ghedt.coordinates.rectangle(Nx, Ny, Bx, By)
    fig, ax = ghedt.coordinates.visualize_coordinates(coordinates)

    fig.savefig('12x13_visualized.png', bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()
