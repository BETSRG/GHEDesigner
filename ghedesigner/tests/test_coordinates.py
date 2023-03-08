import unittest

from ghedesigner.coordinates import open_rectangle, rectangle


class TestCoordinates(unittest.TestCase):
    def test_rectangle(self):
        coords = rectangle(4, 4, 1, 1)
        self.assertEqual(len(coords), 16)
        self.assertEqual(coords[0][0], 0)
        self.assertEqual(coords[0][1], 0)
        self.assertEqual(coords[-1][0], 3)
        self.assertEqual(coords[-1][1], 3)

    def test_open_rectangle(self):
        coords = open_rectangle(4, 4, 1, 1)
        self.assertEqual(len(coords), 12)
        self.assertEqual(coords[0][0], 0)
        self.assertEqual(coords[0][1], 0)
        self.assertEqual(coords[-1][0], 3)
        self.assertEqual(coords[-1][1], 3)

    # def test_c_shape(self):
    #     coords = c_shape(6, 6, 1, 1, 6)
