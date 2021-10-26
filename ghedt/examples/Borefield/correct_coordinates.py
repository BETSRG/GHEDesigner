# Jack C. Cook
# Thursday, September 16, 2021

import gFunctionDatabase as gfdb


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
        gfdb.Management.application.GFunction.configure_database_file_for_usage(
            r_data)

    # Initialize the GFunction object
    GFunction = gfdb.Management.application.GFunction(**geothermal_g_input)

    new_coordinates = GFunction.correct_coordinates(B)

    perimeter = [[0., 0.], [50., 0.], [50., 50.], [0., 50.]]

    fig, ax = GFunction.visualize_area_and_constraints(perimeter,
                                                       new_coordinates)

    fig.show()


if __name__ == '__main__':
    main()
