from ghedt import utilities, search_routines
from pathlib import Path
from unittest import TestCase


class TestDesignFromInput(TestCase):

    def test_design_from_input(self):
        # Note: The test_create_near_square_input_file.py needs to be run to create an
        # input file.

        # Enter the path to the input file named `ghedt_input.obj`.
        test_data_dir = Path(__file__).resolve().parent / 'test_data'
        path_to_file = test_data_dir / 'ghedt_input.obj'
        # Initialize a Design object that is based on the content of the
        # path_to_file variable.
        design = utilities.read_input_file(path_to_file)
        # Find the design based on the inputs.
        bisection_search = design.find_design()
        # Perform sizing in between the min and max bounds.
        ghe = bisection_search.ghe
        ghe.compute_g_functions()
        ghe.size(method="hybrid")
        # Export the g-function to a file named `ghedt_output`. A json file will be
        # created.
        search_routines.oak_ridge_export(bisection_search, file_name="ghedt_output")
