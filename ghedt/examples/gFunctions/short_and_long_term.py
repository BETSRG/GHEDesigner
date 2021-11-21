# Jack C. Cook
# Saturday, November 13, 2021

import ghedt
import ghedt.PLAT as PLAT
import numpy as np
import gFunctionDatabase as gfdb
import ghedt.PLAT.pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    # -------------------
    H = 96.  # Borehole length (m)
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

    alpha = soil.k / soil.rhoCp

    # Eskilson's original ln(t/ts) values
    ts = H ** 2 / (9. * alpha)  # Bore field characteristic time
    log_time_lts = np.array(gfdb.utilities.Eskilson_log_times())
    time_values = np.exp(log_time_lts) * ts

    # Inputs related to fluid
    # -----------------------
    # Fluid properties
    mixer = 'MEG'  # Ethylene glycol mixed with water
    percent = 0.  # Percentage of ethylene glycol added in
    fluid = gt.media.Fluid(mixer=mixer, percent=percent)

    V_flow_borehole = 0.2  # System volumetric flow rate (L/s)
    # Total fluid mass flow rate per borehole (kg/s)
    m_flow_borehole = V_flow_borehole / 1000. * fluid.rho

    # Define a borehole
    borehole = gt.boreholes.Borehole(H, D, r_b, x=0., y=0.)

    # Single U-tube borehole heat exchanger
    single_u_tube = \
        bhe_object(m_flow_borehole, fluid, borehole, pipe, grout, soil)

    radial_numerical = \
        PLAT.radial_numerical_borehole.RadialNumericalBH(single_u_tube)

    radial_numerical.calc_sts_g_functions(single_u_tube)

    log_time_sts = radial_numerical.lntts.tolist()[1:]
    g_sts = radial_numerical.g.tolist()[1:]

    # Calculate g-functions
    # g-Function calculation options
    disp = True

    # Calculate a uniform inlet fluid temperature g-function with 12 equal
    # segments using the similarities solver
    nSegments = 8
    unequal = 'unequal'
    equal = 'equal'
    boundary = 'MIFT'
    equivalent = 'equivalent'
    similarities = 'similarities'

    equivalent_errors = []
    similar_errors = []

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    gt.utilities._format_axes(ax)

    line_styles = ['-', '--', '-.']

    for i in range(1, 4):

        # Field of 12x13 (n=156) boreholes
        N_1 = i
        N_2 = i
        coordinates = gfdb.coordinates.rectangle(N_1, N_2, B, B)

        gfunc_equal_Tf_in_uneq = ghedt.gfunction.calculate_g_function(
            m_flow_borehole, bhe_object, time_values, coordinates, borehole,
            fluid, pipe, grout, soil, nSegments=nSegments, segments=unequal,
            solver=equivalent, boundary=boundary, disp=disp)

        g_lts = gfunc_equal_Tf_in_uneq.gFunc.tolist()

        # make sure the short time step doesn't overlap with the long time step
        max_log_time_sts = max(log_time_sts)
        min_log_time_lts = min(log_time_lts)

        if max_log_time_sts < min_log_time_lts:
            log_time = log_time_sts + log_time_lts.tolist()
            g = g_sts + g_lts
        else:
            # find where to stop in sts
            i = 0
            value = log_time_sts[i]
            while value <= min_log_time_lts:
                i += 1
                value = log_time_sts[i]
            log_time = log_time_sts[0:i] + log_time_lts
            g = g_sts[0:i] + g_lts

        ax.plot(log_time, g, label='{} boreholes'.format(i*i),
                linestyle=line_styles[i-1])

    ax.axvline(x=-8.5, zorder=0, color='k', linestyle='--')

    ax.text(-12.5, 11, 'Short-time step')

    ax.annotate('', xy=(-8.5, 10), xytext=(-13.5, 10),
                arrowprops=dict(arrowstyle= '<|-',
                                color='black',
                                lw=1,
                                ls='-'))

    ax.text(-8, 11, 'Long-time step')

    ax.annotate('', xy=(-8.5, 10), xytext=(-3.5, 10),
                arrowprops=dict(arrowstyle='<|-',
                                color='black',
                                lw=1,
                                ls='-'))

    ax.set_ylabel('g-function')
    ax.set_xlabel('ln(t/t$_s$)')

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())

    ticklabels = []
    for x in ax.get_xticks():
        if x < -8.5:
            ticklabels.append(round(np.exp(x) * ts / 3600., 3))
        else:
            ticklabels.append(round(np.exp(x) * ts / 3600. / 8760., 3))

    # ax2.set_xticklabels([round(np.exp(x) * ts / 3600. / 8760., 3)
    #                      for x in ax.get_xticks()])
    ax2.set_xticklabels(ticklabels)
    top_xlabel = 'Time (hours)' + 70 * ' ' + 'Time (years)'
    top_xlabel = top_xlabel.ljust(30, ' ')
    ax2.set_xlabel(top_xlabel)
    gt.utilities._format_axes(ax2)

    ax.grid()
    ax.set_axisbelow(True)

    fig.tight_layout()

    fig.legend(bbox_to_anchor=(0.3, 0.9))
    fig.savefig('short_and_long_time_step.png')


if __name__ == '__main__':
    main()
