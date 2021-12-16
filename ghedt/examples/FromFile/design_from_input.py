# Jack C. Cook
# Monday, December 13, 2021

import ghedt as dt


def main():
    # Note: The create_near_square_input_file.py needs to be run to create an
    # input file.

    # Enter the path to the input file named `ghedt_input.obj`.
    path_to_file = 'ghedt_input.obj'
    # Initialize a Design object that is based on the content of the
    # path_to_file variable.
    design = dt.design.read_input_file(path_to_file)
    # Find the design based on the inputs.
    bisection_search = design.find_design()
    # Perform sizing in between the min and max bounds.
    ghe = bisection_search.ghe
    ghe.compute_g_functions()
    ghe.size(method='hybrid')
    # Export the g-function to a file named `ghedt_output`. A json file will be
    # created.
    dt.design.oak_ridge_export(bisection_search, file_name='ghedt_output')


if __name__ == '__main__':
    main()
