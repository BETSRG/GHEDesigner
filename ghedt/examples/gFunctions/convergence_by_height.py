# Jack C. Cook
# Friday, November 12, 2021

import ghedt
import ghedt.PLAT as PLAT
import numpy as np
import gFunctionDatabase as gfdb
import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    mpe_errors = {}
    nSegments = 50

    # Borehole dimensions
    # -------------------
    H_values = [24., 48., 96., 192., 384.]  # Borehole length (m)

    for j in range(len(H_values)):

        H = H_values[j]
        mpe_errors[H] = []
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
            GFunction.borehole_radius_correction(g_function, rb_value, rb)

        # Calculate g-functions
        # g-Function calculation options
        disp = True

        # Calculate a uniform inlet fluid temperature g-function with 12 equal
        # segments using the similarities solver
        unequal = 'unequal'
        equal = 'equal'
        boundary = 'MIFT'
        equivalent = 'equivalent'
        similarities = 'similarities'

        equivalent_errors = []
        similar_errors = []

        for i in range(3, nSegments):

            gfunc_equal_Tf_in_eq = ghedt.gfunction.calculate_g_function(
                m_flow_borehole, bhe_object, time_values, coordinates, borehole,
                fluid, pipe, grout, soil, nSegments=i, segments=equal,
                solver=equivalent, boundary=boundary, disp=disp)

            mpe = ghedt.utilities.compute_mpe(g_function_corrected_UIFT_ref,
                                              gfunc_equal_Tf_in_eq.gFunc)
            similar_errors.append(mpe)

            mpe_errors[H].append(mpe)

    # Plot the g-functions
    fig = gt.gfunction._initialize_figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'Number of segments, $n_q$')
    ax.set_ylabel('Mean Percent Error = '
                  r'$\dfrac{\mathbf{p} - \mathbf{r}}{\mathbf{r}} \;\; '
                  r'\dfrac{100\%}{n} $')
    gt.gfunction._format_axes(ax)

    segments = list(range(3, nSegments))

    marker_styles = ['o', 'v', '^', '<', '>']

    for j in range(len(H_values)):
        H = H_values[j]
        ax.scatter(segments, mpe_errors[H],
                   label='H = {} m'.format(H), marker=marker_styles[j])

    ax.grid()
    ax.set_axisbelow(True)

    fig.legend(ncol=1)

    fig.tight_layout()

    fig.savefig('g_function_convergence_height.png')


if __name__ == '__main__':
    main()
