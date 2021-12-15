# Jack C. Cook
# Monday, December 13, 2021

import ghedt as dt
from time import time as clock


def main():
    # Single U-tube
    path_to_file = 'ghedt_input.obj'

    design = dt.design.read_input_file(path_to_file)

    tic = clock()
    bisection_search = design.find_design()
    toc = clock()
    print('Time to perform bisection search: {0:.2f} seconds'.format(toc - tic))

    print('Number of boreholes: {}'.
          format(len(bisection_search.selected_coordinates)))

    # Perform sizing in between the min and max bounds
    tic = clock()
    ghe = bisection_search.ghe
    ghe.compute_g_functions()

    ghe.size(method='hybrid')
    toc = clock()
    print('Time to compute g-functions and size: {0:.2f} '
          'seconds'.format(toc - tic))

    print('Sized height of boreholes: {0:.2f} m'.format(ghe.bhe.b.H))

    dt.design.oak_ridge_export(bisection_search, )


if __name__ == '__main__':
    main()
