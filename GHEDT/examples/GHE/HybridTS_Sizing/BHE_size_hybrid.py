# Jack C. Cook
# Saturday, October 9, 2021

"""
Compare the size of GHEs over a range of flow rates with differing BHEs to
GLHEPro.
"""

import copy

import GHEDT.PLAT as PLAT
import matplotlib.pyplot as plt
import pandas as pd
import GHEDT.PLAT.pygfunction as gt
import gFunctionDatabase as gfdb
import GHEDT


def main():
    # read in GLHEPro results
    file_name = '../BHE_size_variation_GLHEPRO.xlsx'

    xlsx = pd.ExcelFile(file_name)
    sheet_name = xlsx.sheet_names[0]
    d = pd.read_excel(xlsx, sheet_name=sheet_name).to_dict('list')

    # --------------------------------------------------------------------------

    # Borehole dimensions
    # -------------------
    H = 100.  # Borehole length (m)
    D = 2.  # Borehole buried depth (m)
    r_b = 150. / 1000. / 2.  # Borehole radius
    B = 5.  # Borehole spacing (m)

    # Pipe dimensions
    # ---------------
    # U-tubes
    r_out = 26.67 / 1000. / 2.  # Pipe outer radius (m)
    r_in = 21.6 / 1000. / 2.  # Pipe inner radius (m)
    s = 32.3 / 1000.  # Inner-tube to inner-tube Shank spacing (m)
    # Coaxial
    # Inner pipe radii
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

    epsilon = 1.0e-6  # Pipe roughness (m)

    # Pipe positions
    # --------------
    # Single U-tube [(x_in, y_in), (x_out, y_out)]
    pos_s = PLAT.media.Pipe.place_pipes(s, r_out, 1)
    # Double U-tube
    pos_d = PLAT.media.Pipe.place_pipes(s, r_out, 2)
    # Coaxial
    pos_c = (0, 0)

    # Thermal conductivities
    # ----------------------
    k_p = 0.4  # Pipe thermal conductivity (W/m.K)
    k_s = 2.0  # Ground thermal conductivity (W/m.K)
    k_g = 1.0  # Grout thermal conductivity (W/m.K)
    # Pipe thermal conductivity list for coaxial
    k_p_c = [0.4, 0.4]  # Inner and outer pipe thermal conductivity (W/m.K)

    # Volumetric heat capacities
    # --------------------------
    rhoCp_p = 1542. * 1000.  # Pipe volumetric heat capacity (J/K.m3)
    rhoCp_s = 2343.493 * 1000.  # Soil volumetric heat capacity (J/K.m3)
    rhoCp_g = 3901. * 1000.  # Grout volumetric heat capacity (J/K.m3)

    # Thermal properties
    # ------------------
    # Pipe
    pipe_s = PLAT.media.Pipe(pos_s, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    pipe_d = PLAT.media.Pipe(pos_d, r_in, r_out, s, epsilon, k_p, rhoCp_p)
    pipe_c = \
        PLAT.media.Pipe(pos_c, r_inner, r_outer, s, epsilon, k_p_c, rhoCp_p)
    # Soil
    ugt = 18.3  # Undisturbed ground temperature (degrees Celsius)
    soil = PLAT.media.Soil(k_s, rhoCp_s, ugt)
    # Grout
    grout = PLAT.media.ThermalProperty(k_g, rhoCp_g)

    # BHE object pointers
    # -------------------
    single_u_tube_object = PLAT.borehole_heat_exchangers.SingleUTube
    double_u_tube_object = PLAT.borehole_heat_exchangers.MultipleUTube
    coaxial_tube_object = PLAT.borehole_heat_exchangers.CoaxialPipe

    # Read in g-functions from GLHEPro
    file = '../../1DInterpolation/GLHEPRO_gFunctions_12x13.json'
    r_data, _ = gfdb.fileio.read_file(file)

    # Configure the database data for input to the goethermal GFunction object
    geothermal_g_input = gfdb.Management. \
        application.GFunction.configure_database_file_for_usage(r_data)

    # Initialize the GFunction object
    GFunction = gfdb.Management.application.GFunction(**geothermal_g_input)

    # Inputs related to fluid
    # -----------------------
    V_flow_system_rates = [7.8, 15.6, 31.2, 46.8, 62.4, 78.]
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
    max_Height = 384  # in meters
    min_Height = 24  # in meters
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

    print('Volumetric Flow Rate per Borehole (L/s)\tSingle U-tube'
          '\tDouble U-tube\tCoaxial')

    V_flow_borehole_rates = []

    # Borehole heat exchanger
    # -----------------------
    # Fluid properties
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)

    sized_height_dictionary = {'Single U-tube': [],
                               'Double U-tube': [],
                               'Coaxial': []}

    for i in range(0, len(V_flow_system_rates)):
        V_flow_system = V_flow_system_rates[i]

        # Define a borehole
        borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)
        borehole_d = copy.deepcopy(borehole)
        borehole_c = copy.deepcopy(borehole)

        # Size Hybrid GLHEs
        # Single U-tube GHE
        GHE_s = GHEDT.ground_heat_exchangers.GHE(
            V_flow_system, B, single_u_tube_object, fluid, borehole,
            pipe_s, grout, soil, GFunction, sim_params,
            hourly_extraction_ground_loads)
        V_flow_borehole_rates.append(GHE_s.V_flow_borehole)
        # Double u-Tube GHE
        GHE_d = GHEDT.ground_heat_exchangers.GHE(
            V_flow_system, B, double_u_tube_object, fluid, borehole_d,
            pipe_d, grout, soil, GFunction, sim_params,
            hourly_extraction_ground_loads)
        # Coaxial tube GHE
        GHE_c = GHEDT.ground_heat_exchangers.GHE(
            V_flow_system, B, coaxial_tube_object, fluid, borehole_c,
            pipe_c, grout, soil, GFunction, sim_params,
            hourly_extraction_ground_loads)

        # Size each GHE using a hybrid simulation
        GHE_s.size(method='hybrid')
        GHE_d.size(method='hybrid')
        GHE_c.size(method='hybrid')

        sized_height_dictionary['Single U-tube'].append(GHE_s.bhe.b.H)
        sized_height_dictionary['Double U-tube'].append(GHE_d.bhe.b.H)
        sized_height_dictionary['Coaxial'].append(GHE_c.bhe.b.H)

        # Effective borehole thermal resistances
        Rb_star_s = \
            gt.pipes.borehole_thermal_resistance(GHE_s.bhe,
                                                 GHE_s.bhe.m_flow_borehole,
                                                 GHE_s.bhe.fluid.cp)
        Rb_star_d = \
            gt.pipes.borehole_thermal_resistance(GHE_d.bhe,
                                                 GHE_d.bhe.m_flow_borehole,
                                                 GHE_d.bhe.fluid.cp)
        Rb_star_c = \
            gt.pipes.borehole_thermal_resistance(GHE_c.bhe,
                                                 GHE_c.bhe.m_flow_borehole,
                                                 GHE_c.bhe.fluid.cp)
        # Local borehole thermal resistances
        import numpy as np
        Rb_s = 1 / np.trace(1 / GHE_s.bhe._Rd)
        Rb_d = 1 / np.trace(1 / GHE_d.bhe._Rd)
        Rb_c = 1 / np.trace(1 / GHE_c.bhe._Rd)

        # Reynolds numbers
        Re_s = PLAT.borehole_heat_exchangers.compute_Reynolds(
            GHE_s.bhe.m_flow_pipe, GHE_s.bhe.r_in, GHE_s.bhe.pipe.eps,
            GHE_s.bhe.fluid)
        Re_d = PLAT.borehole_heat_exchangers.compute_Reynolds(
            GHE_d.bhe.m_flow_pipe, GHE_d.bhe.r_in, GHE_d.bhe.pipe.eps,
            GHE_d.bhe.fluid)
        Re_c = PLAT.borehole_heat_exchangers.compute_Reynolds_concentric(
            GHE_c.bhe.m_flow_pipe, GHE_c.bhe.r_in_out, GHE_c.bhe.r_out_in,
            GHE_c.bhe.fluid)

        print('{0:.8f}\t{1:.8f}\t{2:.8f}\t{3:.8f}'.format(
            GHE_s.V_flow_borehole, GHE_s.bhe.b.H,
            GHE_d.bhe.b.H, GHE_c.bhe.b.H))
        # print('{0:.8f}\t{1:.8f}\t{2:.8f}\t{3:.8f}'.format(
        #     GHE_s.V_flow_borehole, Rb_star_s,
        #     Rb_star_d, Rb_star_c))
        # print('{0:.8f}\t{1:.8f}\t{2:.8f}\t{3:.8f}'.format(
        #     GHE_s.V_flow_borehole, Rb_s,
        #     Rb_d, Rb_c))
        # print('{0:.8f}\t{1:.8f}\t{2:.8f}\t{3:.8f}'.format(
        #     GHE_s.V_flow_borehole, Re_s, Re_d, Re_c))

    # Create plot of V_flow_borehole vs. Sized Height
    # -----------------------------------------------
    fig = gt.gfunction._initialize_figure()
    ax = fig.add_subplot(111)

    ax.plot(V_flow_borehole_rates, sized_height_dictionary['Single U-tube'],
            label='Single U-tube', marker='o', ls='--')
    ax.plot(V_flow_borehole_rates, sized_height_dictionary['Double U-tube'],
            label='Double U-tube', marker='s', ls='--')
    ax.plot(V_flow_borehole_rates, sized_height_dictionary['Coaxial'],
            label='Coaxial', marker='*', ls='--')

    ax.set_xlabel('Volumetric flow rate per borehole (L/s)')
    ax.set_ylabel('Height of boreholes (m)')

    ax.grid()
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.legend(bbox_to_anchor=(.4, .95))

    fig.savefig('BHE_HYTS_size_variation.png')

    # Create a plot comparing GLHEPro results
    # ---------------------------------------
    fig = gt.gfunction._initialize_figure()
    ax = fig.subplots(3, sharex=True, sharey=True)
    # fig, ax = plt.subplots(3, sharex=True, sharey=True)
    ax[0].plot(d['V_flow_borehole'], d['Single U-tube'],
               label='Single U-tube (GLHEPRO)')
    ax[1].plot(d['V_flow_borehole'], d['Double U-tube'],
               label='Double U-tube (GLHEPRO)')
    ax[2].plot(d['V_flow_borehole'], d['Coaxial'], label='Coaxial (GLHEPRO)')
    ax[0].plot(V_flow_borehole_rates, sized_height_dictionary['Single U-tube'],
               '--', label='Single U-tube (GHEDT)')
    ax[1].plot(V_flow_borehole_rates, sized_height_dictionary['Double U-tube'],
               '--', label='Double U-tube (GHEDT)')
    ax[2].plot(V_flow_borehole_rates, sized_height_dictionary['Coaxial'],
               '--', label='Coaxial (GHEDT)')
    for i in range(3):
        ax[i].grid()
        ax[i].set_axisbelow(True)
        ax[i].legend()
    ax[2].set_xlabel('Volumetric flow rate per borehole (L/s)')
    ax[1].set_ylabel(r'Height of boreholes (m)')
    fig.tight_layout()
    fig.savefig('bhe_sizing_comparison.png')


if __name__ == '__main__':
    main()
