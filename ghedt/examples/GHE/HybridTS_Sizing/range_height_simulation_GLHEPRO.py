# Jack C. Cook
# Saturday, October 9, 2021
import copy

import ghedt.PLAT as PLAT
import matplotlib.pyplot as plt
import pandas as pd
import ghedt.PLAT.pygfunction as gt
import gFunctionDatabase as gfdb
import ghedt


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

    # Read in g-functions from GLHEPro
    file = '../../1DInterpolation/GLHEPRO_gFunctions_12x13.json'
    r_data, _ = gfdb.fileio.read_file(file)

    # Configure the database data for input to the goethermal GFunction object
    geothermal_g_input = gfdb.Management. \
        application.GFunction.configure_database_file_for_usage(r_data)

    # Initialize the GFunction object
    GFunction = gfdb.Management.application.GFunction(**geothermal_g_input)

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
    max_Height = 100  # in meters
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

    # Ground heat exchanger objects
    # -----------
    # Initialize a GHE
    GHE = ghedt.ground_heat_exchangers.GHE(
        V_flow_system, B, bhe_object, fluid, borehole, pipe, grout, soil,
        GFunction, sim_params, hourly_extraction_ground_loads)

    # --------------------------------------------------------------------------

    # Range through height values and return the T_excess value
    height_values = [48. + float(12 * i) for i in range(0, 14)]

    Excess_temperatures = {'Hourly': [], 'Hybrid': []}

    min_HP_EFTs = []
    max_HP_EFTs = []

    for height in height_values:
        print(height)
        GHE.bhe.b.H = height
        max_HP_EFT, min_HP_EFT = GHE.simulate()
        min_HP_EFTs.append(min_HP_EFT)
        max_HP_EFTs.append(max_HP_EFT)
        T_excess_ = GHE.cost(max_HP_EFT, min_HP_EFT)
        Excess_temperatures['Hybrid'].append(T_excess_)

    # Plot excess values
    # ------------------
    fig = gt.gfunction._initialize_figure()
    ax = fig.add_subplot(111)
    gt.utilities._format_axes(ax)

    from scipy import interpolate
    f = interpolate.interp1d(Excess_temperatures['Hybrid'], height_values)
    print(f(0))
    ax.vlines(x=f(0), ymin=-10, ymax=0, color='k', zorder=0, linestyles='--')
    ax.scatter(f(0), 0, label='Root', color='r', marker='x', s=50, zorder=2)

    ax.plot(height_values, Excess_temperatures['Hybrid'],
            marker='s', ls='--', label='Temperature', zorder=1)

    ax.set_xlabel('Borehole height (m)')
    ax.set_ylabel('Excess fluid temperature ($\degree$C)')

    ax.grid()
    ax.set_axisbelow(True)

    ax.set_ylim([-7, 20])

    fig.legend()

    fig.tight_layout()

    fig.savefig('range_excess_GLHEPRO.png')
    plt.close(fig)

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    gt.utilities._format_axes(ax)

    ax.scatter(height_values, min_HP_EFTs, color='b', label='min(EFT$_{HP}$)')
    ax.scatter(height_values, max_HP_EFTs, color='r', label='max(EFT$_{HP}$',
               zorder=2)

    ax.set_ylabel('Heat Pump Entering fluid temperature ($\degree$C)')
    ax.set_xlabel('Borehole height (m)')

    ax.hlines(y=35, xmin=0, xmax=210, color='k', linestyle='--', zorder=1)
    ax.hlines(y=5, xmin=0, xmax=210, color='k', linestyle='--')

    ax.grid()
    ax.set_axisbelow(True)

    fig.legend()

    fig.tight_layout()

    ax.set_xlim([50, 210])

    fig.savefig('min_and_max_HPEFTs_GLHEPRO.png')

    # Now Size for the height


if __name__ == '__main__':
    main()
