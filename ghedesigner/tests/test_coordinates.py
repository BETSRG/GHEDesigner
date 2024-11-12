import unittest

from ghedesigner.ghe.geometry.coordinates import open_rectangle, rectangle


class TestCoordinates(unittest.TestCase):
    def test_rectangle(self):
        coords = rectangle(4, 4, 1, 1)
        assert len(coords) == 16
        assert coords[0][0] == 0
        assert coords[0][1] == 0
        assert coords[-1][0] == 3
        assert coords[-1][1] == 3

    def test_open_rectangle(self):
        coords = open_rectangle(4, 4, 1, 1)
        assert len(coords) == 12
        assert coords[0][0] == 0
        assert coords[0][1] == 0
        assert coords[-1][0] == 3
        assert coords[-1][1] == 3

    # def test_c_shape(self):
    #     coords = c_shape(6, 6, 1, 1, 6)
