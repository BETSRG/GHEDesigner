# Jack C. Cook
# Monday, October 25, 2021

import ghedt
import ghedt.PLAT as PLAT
import numpy as np
import gFunctionDatabase as gfdb
import pygfunction as gt


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
    log_time = np.array(gfdb.utilities.Eskilson_log_times())
    time_values = np.exp(log_time) * ts

    # Field of 12x13 (n=156) boreholes
    N_1 = 12
    N_2 = 13
    coordinates = gfdb.coordinates.rectangle(N_1, N_2, B, B)

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

    # Calculate g-functions
    # g-Function calculation options
    disp = True

    # Calculate a uniform inlet fluid temperature g-function with 12 equal
    # segments using the similarities solver
    nSegments = 12
    segments = 'equal'
    boundary = 'MIFT'
    solver = 'similarities'

    gfunc_equal_Tf_in = ghedt.gfunction.calculate_g_function(
        m_flow_borehole, bhe_object, time_values, coordinates, borehole,
        nSegments, fluid, pipe, grout, soil, segments=segments,
        solver=solver, boundary=boundary, disp=disp)

    # Calculate a uniform inlet fluid temperature g-function with 12 equal
    # segments using the equivalent solver

    solver = 'equivalent'
    gfunc_equal_Tf_in_eq = ghedt.gfunction.calculate_g_function(
        m_flow_borehole, bhe_object, time_values, coordinates, borehole,
        nSegments, fluid, pipe, grout, soil, segments=segments,
        solver=solver, boundary=boundary, disp=disp)

    segments = 'unequal'

    gfunc_equal_Tf_in_uneq = ghedt.gfunction.calculate_g_function(
        m_flow_borehole, bhe_object, time_values, coordinates, borehole,
        nSegments, fluid, pipe, grout, soil, segments=segments,
        solver=solver, boundary=boundary, disp=disp)

    # Calculate a uniform borehole wall temperature g-function with 12 equal
    # segments using the similarities solver
    boundary = 'UBWT'

    gfunc_uniform_T = ghedt.gfunction.calculate_g_function(
        m_flow_borehole, bhe_object, time_values, coordinates, borehole,
        nSegments, fluid, pipe, grout, soil, segments=segments,
        solver=solver, boundary=boundary, disp=disp
    )

    # GLHEPro g-function
    file_path = '12x13_Calculated_g_Functions/GLHEPRO_gFunctions_12x13.json'
    data, file_name = gfdb.fileio.read_file(file_path)

    geothermal_g_input = gfdb.Management.application.GFunction. \
        configure_database_file_for_usage(data)

    # Initialize the GFunction object
    GFunction = gfdb.Management.application.GFunction(**geothermal_g_input)
    B_over_H = B / H
    rb = r_b
    # interpolate for the Long time step g-function
    g_function, rb_value, D_value, H_eq = \
        GFunction.g_function_interpolation(B_over_H)
    # correct the long time step for borehole radius
    g_function_corrected_GLHEPro = \
        GFunction.borehole_radius_correction(g_function,
                                             rb_value,
                                             rb)
    # Reference UIFT g-function with 96 equal segments
    file_path = '12x13_Calculated_g_Functions/' \
                '96_Equal_Segments_Similarities_UIFT.json'
    data, file_name = gfdb.fileio.read_file(file_path)

    geothermal_g_input = gfdb.Management.application.GFunction. \
        configure_database_file_for_usage(data)

    # Initialize the GFunction object
    GFunction = gfdb.Management.application.GFunction(**geothermal_g_input)
    B_over_H = B / H
    rb = r_b
    # interpolate for the Long time step g-function
    g_function, rb_value, D_value, H_eq = \
        GFunction.g_function_interpolation(B_over_H)
    # correct the long time step for borehole radius
    g_function_corrected_UIFT_ref = \
        GFunction.borehole_radius_correction(g_function,
                                             rb_value,
                                             rb)

    # Plot the g-functions
    fig = gt.gfunction._initialize_figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'ln$(t/t_s)$')
    ax.set_ylabel(r'$g$-function')
    gt.gfunction._format_axes(ax)

    ax.plot(log_time, gfunc_equal_Tf_in.gFunc, marker=None, linestyle='-',
            label='12 Equal Similarities UIFT')
    ax.plot(log_time, gfunc_equal_Tf_in_eq.gFunc, marker=None, linestyle='-',
            label='12 Equal Equivalent UIFT')
    ax.plot(log_time, gfunc_equal_Tf_in_uneq.gFunc, marker=None, linestyle='-',
            label='8 Unequal Equivalent UIFT')
    ax.plot(log_time, gfunc_uniform_T.gFunc,
            label='12 Equal Similarities UBWT')
    ax.plot(log_time, g_function_corrected_UIFT_ref, marker='x',
            linestyle='', label='96 Equal Similarities UIFT')
    ax.plot(log_time, g_function_corrected_GLHEPro, marker=None, linestyle='--',
            label=r'GLHEPro (UBWT, $D\approx5$m)')

    fig.legend(bbox_to_anchor=(0.5, 0.88))

    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels([round(np.exp(x)*ts / 3600. / 8760., 3)
                         for x in ax.get_xticks()])
    ax2.set_xlabel('Time (seconds)')
    gt.utilities._format_axes(ax2)

    fig.tight_layout()

    fig.savefig('g_function_comparison_plot.png')

    # Verify the MPEs make sense
    mpe_UBWT = compute_mpe(g_function_corrected_UIFT_ref, gfunc_uniform_T.gFunc)
    print(mpe_UBWT)


def compute_mpe(actual: list, predicted: list) -> float:
    """
    The following mean percentage error formula is used:
    .. math::
        MPE = \dfrac{100\%}{n}\sum_{i=0}^{n-1}\dfrac{a_t-p_t}{a_t}
    Parameters
    ----------
    actual: list
        The actual computed g-function values
    predicted: list
        The predicted g-function values
    Returns
    -------
    **mean_percent_error: float**
        The mean percentage error in percent
    """
    # the lengths of the two lists should be the same
    assert len(actual) == len(predicted)
    # create a summation variable
    summation: float = 0.
    for i in range(len(actual)):
        summation += (predicted[i] - actual[i]) / actual[i]
    mean_percent_error = summation * 100 / len(actual)
    return mean_percent_error


if __name__ == '__main__':
    main()
