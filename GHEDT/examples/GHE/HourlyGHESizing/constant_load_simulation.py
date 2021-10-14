# Jack C. Cook
# Wednesday, October 13, 2021


import GHEDT.PLAT as PLAT
import matplotlib.pyplot as plt
import pandas as pd
import GHEDT.PLAT.pygfunction as gt
import gFunctionDatabase as gfdb
import GHEDT
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
    n_years = 1
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
    import numpy as np
    two_pi_k = 2. * np.pi * soil.k
    hourly_extraction_ground_loads = [-(two_pi_k) * nbh * H] * 8760

    # --------------------------------------------------------------------------

    # Initialize Hybrid GLHE object
    HourlyGHE = GHEDT.ground_heat_exchangers.HourlyGHE(
        V_flow_system, B, bhe_object, fluid, borehole, pipe, grout, soil,
        GFunction, sim_params, hourly_extraction_ground_loads)

    max_HP_EFT, min_HP_EFT = HourlyGHE.simulate()

    print('Min EFT: {}\nMax EFT: {}'.format(min_HP_EFT, max_HP_EFT))

    # Plot change in fluid temperature
    # --------------------------------
    fig, ax = plt.subplots()

    time_values = list(range(1, 8760 * n_years))
    lntts = [np.log(time_values[i] * 3600 / HourlyGHE.radial_numerical.t_s)
             for i in range(len(time_values))]

    B_over_H = B / HourlyGHE.bhe.b.H
    g = HourlyGHE.grab_g_function(B_over_H)

    ax.plot(g.x, g.y)
    ax.plot(lntts, HourlyGHE.delta_Tb, '--')

    ax.set_ylabel('g = $\Delta T_b$')
    ax.set_xlabel('ln(t/ts)')

    fig.tight_layout()

    fig.savefig('hourly_simulation_Tb.png')


if __name__ == '__main__':
    main()
