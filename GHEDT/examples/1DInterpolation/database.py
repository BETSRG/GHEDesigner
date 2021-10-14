# Jack C. Cook
# Thursday, September 16, 2021

"""
Purpose:
    - Pull a configuration from the g-function database
    - load the GLHEDT.geothermal.GFunction from the database configuration
    - perform interpolation
    - do correction for rb value
"""


import PLAT.pygfunction as gt
import PLAT
import gFunctionDatabase as gfdb
import GLHEDT


def main():
    # Borehole dimensions
    H = 100.  # Borehole length (m)
    D = 2.  # Borehole buried depth (m)
    r_b = 150. / 1000. / 2.  # Borehole radius

    # Borefield spacing
    B = 7.5  # Borefield spacing (m)

    # Number in the x and y
    N = 7
    M = 10

    # configuration
    r_configuration = 'rectangle'
    r = gfdb.Management.retrieval.Retrieve(r_configuration)
    # There is just one value returned in the unimodal domain for rectangles
    r_unimodal = r.retrieve(N, M)
    key = list(r_unimodal.keys())[0]
    print('The key value: {}'.format(key))
    r_data = r_unimodal[key]

    # Configure the database data for input to the goethermal GFunction object
    geothermal_g_input = \
        GLHEDT.geothermal.GFunction.configure_database_file_for_usage(r_data)

    # Initialize the GFunction object
    GFunction = GLHEDT.geothermal.GFunction(**geothermal_g_input)

    # Perform interpolation
    B_over_H = B / H
    print('Interpolate for a B/H = {}'.format(B/H))
    g_function, rb_value, D_value, H_eq = \
        GFunction.g_function_interpolation(B_over_H)
    print('rb: {}\tD: {}\tH_eq: {}'.format(rb_value, D_value, H_eq))

    # We need to correct the g-function for the desired borehole radius
    g_function_corrected = \
        GFunction.borehole_radius_correction(g_function, rb_value, r_b)

    # Plot the results
    fig, ax = GFunction.visualize_g_functions()

    ax.plot(GFunction.log_time, g_function, '--')
    ax.plot(GFunction.log_time, g_function_corrected, '-.')

    fig.show()


if __name__ == '__main__':
    main()
