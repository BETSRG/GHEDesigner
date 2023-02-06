from unittest import TestCase


class TestFindDesignFromInput(TestCase):

    def test_find_design_from_input(self):
        # This used to be a standalone test, but it required the test_create_near_square_input_file test to be run
        # first in order to create the input file, then this test would pick up the resulting input file.  This
        # also required manually moving the file around.  Now the input file is still created there, and then
        # immediately read in and exercised, doing what this example used to do.  To find more information, check out
        # the bottom of that unit test.
        pass
