import os
import unittest

import matplotlib.pyplot as plt
from ghedt.coordinates import c_shape, open_rectangle, rectangle

if "PLOT_COORDINATES" in os.environ:
    show_plots = True
else:
    show_plots = False


def plot_coordinates(coords, title):
    plt.scatter(*zip(*coords))
    plt.title(title)
    plt.grid()
    plt.show()


class TestCoordinates(unittest.TestCase):
    def test_rectangle(self):
        coords = rectangle(4, 4, 1, 1)
        self.assertEqual(len(coords), 16)
        self.assertEqual(coords[0][0], 0)
        self.assertEqual(coords[0][1], 0)
        self.assertEqual(coords[-1][0], 3)
        self.assertEqual(coords[-1][1], 3)

        if show_plots:
            plot_coordinates(coords, "Rectangle")

    def test_open_rectangle(self):
        coords = open_rectangle(4, 4, 1, 1)
        self.assertEqual(len(coords), 12)
        self.assertEqual(coords[0][0], 0)
        self.assertEqual(coords[0][1], 0)
        self.assertEqual(coords[-1][0], 3)
        self.assertEqual(coords[-1][1], 3)

        if show_plots:
            plot_coordinates(coords, "Open Rectangle")

    def test_c_shape(self):
        coords = c_shape(6, 6, 1, 1, 6)

        if show_plots:
            plot_coordinates(coords, "C-Shape")
