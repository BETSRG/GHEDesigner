from math import sqrt
from unittest import TestCase

from ghedesigner.ghe.geometry.shape import point_polygon_check


class TestShapes(TestCase):
    def test_point_polygon_check(self):
        # return constants for convenience
        INSIDE = 1
        OUTSIDE = -1
        ON_EDGE = 0

        # convex hexagon test shape
        hexagon = [(2, 0), (1, sqrt(3)), (-1, sqrt(3)), (-2, 0), (-1, -sqrt(3)), (1, -sqrt(3))]
        #  (-1, sqrt(3))   _______  (1, sqrt(3))
        #                /         \
        #               /           \
        #      (-2, 0) /             \
        #              \             /  (2, 0)
        #               \           /
        # (-1, -sqrt(3)) \ _______ /  (1, -sqrt(3))

        # center, should be inside
        point = (0, 0)
        assert point_polygon_check(hexagon, point) == INSIDE

        # on top edge
        point = (0, sqrt(3))
        assert point_polygon_check(hexagon, point) == ON_EDGE

        # above top, outside
        point = (0, 4)
        assert point_polygon_check(hexagon, point) == OUTSIDE

        # non-convex L-shape
        l_shape = [(0, 0), (4, 0), (4, 4), (3, 4), (3, 1), (0, 1)]
        #                 (3, 4)  ----  (4, 4)
        #                        |    |
        #                        |    |
        #                        |    |
        #                        |    |
        # (0, 1)          (3, 1) |    |
        #  ----------------------     |
        # |                           |
        #  ---------------------------
        # (0, 0)                     (4, 0)

        # inside
        point = (0.5, 0.5)
        assert point_polygon_check(l_shape, point) == INSIDE

        # on edge
        point = (2, 0)
        assert point_polygon_check(l_shape, point) == ON_EDGE

        # above
        point = (2, 2)
        assert point_polygon_check(l_shape, point) == OUTSIDE

        # below
        point = (-2, 2)
        assert point_polygon_check(l_shape, point) == OUTSIDE
