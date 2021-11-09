# Jack C. Cook
# Sunday, November 7, 2021

import numpy as np
import ghedt
import ghedt.PLAT as PLAT
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
    boundary = 'MIFT'
    equivalent = 'equivalent'

    x = np.arange(5, 25, 1, dtype=int)  # segments
    y = np.arange(0.01, 0.06, 0.01, dtype=float)  # alpha

    xv, yv = np.meshgrid(x, y)

    z = np.zeros_like(yv)

    for i in range((len(z))):
        for j in range(len(z[i])):
            nSegments = xv[i][j].tolist()
            alpha = yv[i][j].tolist()
            gfunc = ghedt.gfunction.calculate_g_function(
                m_flow_borehole, bhe_object, time_values, coordinates, borehole,
                fluid, pipe, grout, soil, nSegments=nSegments,
                end_length_ratio=alpha, segments=unequal, solver=equivalent,
                boundary=boundary, disp=disp)

            mpe = ghedt.utilities.compute_mpe(g_function_corrected_UIFT_ref,
                                              gfunc.gFunc)

            z[i][j] = mpe
    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    gt.utilities._format_axes(ax)

    import matplotlib.pyplot as plt
    import matplotlib
    # ax.contourf(x, y, z)
    # sc = ax.scatter(xv.ravel(), yv.ravel(), c=z.ravel(), vmin=0, vmax=0.3)
    print(z.min())
    print(z.max())
    # sc = ax.scatter(xv.ravel(), yv.ravel(), c=z.ravel(),
    #                 norm=matplotlib.colors.LogNorm(z.min(), z.max()),
    #                 cmap='magma')
    sc = ax.scatter(xv.ravel(), yv.ravel(), c=z.ravel(), vmin=z.min(),
                    vmax=z.max(), cmap='magma')
    # sc = ax.pcolor(x, y, z,
    #                 norm=matplotlib.colors.LogNorm(0.000001, 10))
    cbar = fig.colorbar(sc, ax=ax, extend='max')
    cbar.ax.set_ylabel('Mean Percent Error = '
                      r'$\dfrac{\mathbf{p} - \mathbf{r}}{\mathbf{r}} \;\; '
                      r'\dfrac{100\%}{n} $')

    ax.set_xlabel('Number of sources, $n_q$')
    ax.set_ylabel(r'End segment ratio, $\alpha$')

    fig.tight_layout()
    fig.savefig('vertical_meshgrid_analysis.png')


if __name__ == '__main__':
    main()
