# Jack C. Cook
# Wednesday, October 20, 2021

import gFunctionDatabase as gfdb
import matplotlib.pyplot as plt


def main():
    # Number in the x and y
    # ---------------------
    N = 12
    M = 13
    configuration = 'rectangle'
    nbh = N * M

    file_path = 'GLHEPRO_gFunctions_12x13.json'

    data, file_name = gfdb.fileio.read_file(file_path)

    geothermal_g_input = gfdb.Management.application.GFunction.\
        configure_database_file_for_usage(data)

    # Initialize the GFunction object
    GFunction = gfdb.Management.application.GFunction(**geothermal_g_input)

    B = 5.
    H = 96.
    rb = 0.075
    B_over_H = B / H

    # interpolate for the Long time step g-function
    g_function, rb_value, D_value, H_eq = \
        GFunction.g_function_interpolation(B_over_H)
    # correct the long time step for borehole radius
    g_function_corrected = \
        GFunction.borehole_radius_correction(g_function, rb_value, rb)

    figure, axs = plt.subplots()

    axs.plot(GFunction.log_time, g_function_corrected, label='GLHEPro')

    fig, ax = GFunction.visualize_g_functions()

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

    for h in GFunction.g_lts:
        rb = 0.075
        g = GFunction.borehole_radius_correction(GFunction.g_lts[h],
                                                 GFunction.r_b_values[h],
                                                 rb)
        ax.plot(GFunction.log_time, g, '--')

    line_1 = fig.gca().get_lines()[0]
    line_m1 = fig.gca().get_lines()[-1]

    legend = plt.legend([line_1, line_m1], ['GLHEPro', 'GHEDT (GFDB)'])

    fig.gca().add_artist(legend)

    fig.savefig('12x13_gFunction_comparison.png')

    # interpolate for the Long time step g-function
    g_function, rb_value, D_value, H_eq = \
        GFunction.g_function_interpolation(B_over_H)
    # correct the long time step for borehole radius
    g_function_corrected = \
        GFunction.borehole_radius_correction(g_function,
                                             rb_value,
                                             rb)

    axs.plot(GFunction.log_time, g_function_corrected, '--',
             label='GHEDT (GFDB)')

    axs.set_ylabel('g')
    axs.set_xlabel(r'ln(t/t$_s$)')

    axs.grid()

    axs.set_axisbelow(True)

    figure.legend(bbox_to_anchor=(0.36, 0.95))

    figure.tight_layout()

    figure.savefig('g_function_96m_comparison.png')


if __name__ == '__main__':
    main()
