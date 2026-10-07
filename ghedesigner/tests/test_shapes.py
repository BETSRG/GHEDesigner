from math import sqrt
from unittest import TestCase

import numpy as np

from ghedesigner.ghe.shape import Shapes, clip_two_shapes, merge_two_shapes, point_polygon_check


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

    def test_polygon_merging(self):

        # Shared Inputs
        comparison_tolerance = 1e-8
        shape_1 = Shapes([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])

        # Example 1
        shape_2 = Shapes([[7.5, 7.5], [12.5, 7.5], [12.5, 12.5], [7.5, 12.5]])
        merged_shape_reference = [
            Shapes(
                [[10.0, 7.5], [12.5, 7.5], [12.5, 12.5], [7.5, 12.5], [7.5, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]]
            )
        ]
        merged_hole_reference = []
        merged_shapes, holes = merge_two_shapes(shape_1, shape_2)
        assert polygon_list_comparison(merged_shape_reference, merged_shapes, comparison_tolerance)
        assert polygon_list_comparison(merged_hole_reference, holes, comparison_tolerance)

        # Example 2
        shape_2 = Shapes([[5.0, 15.0], [2.0, 0.0], [5.0, 12.5], [8.5, 4.0]])
        merged_shape_reference = [
            Shapes(
                [
                    [2.0, 0.0],
                    [10.0, 0.0],
                    [10.0, 10.0],
                    [6.59090909090909, 10.0],
                    [5.0, 15.0],
                    [4.0, 10.0],
                    [0.0, 10.0],
                    [0.0, 0.0],
                ]
            )
        ]
        merged_hole_reference = [Shapes([[6.02941176470588, 10.0], [5.0, 12.5], [4.4, 10.0]])]
        merged_shapes, holes = merge_two_shapes(shape_1, shape_2)
        assert polygon_list_comparison(merged_shape_reference, merged_shapes, comparison_tolerance)
        assert polygon_list_comparison(merged_hole_reference, holes, comparison_tolerance)

        # Example 3
        shape_2 = Shapes(
            [[0.0, 10.0], [5.0, 5.0], [11.0, 11.0], [6.0, 10.0], [4.0, 11.0], [3.0, 10.0], [5.0, 6.0], [2.0, 10.0]]
        )
        merged_shape_reference = [
            Shapes(
                [
                    [10.0, 10.0],
                    [11.0, 11.0],
                    [6.0, 10.0],
                    [4.0, 11.0],
                    [3.0, 10.0],
                    [2.0, 10.0],
                    [0.0, 10.0],
                    [0.0, 0.0],
                    [10.0, 0.0],
                ]
            )
        ]
        merged_hole_reference = []
        merged_shapes, holes = merge_two_shapes(shape_1, shape_2)
        assert polygon_list_comparison(merged_shape_reference, merged_shapes, comparison_tolerance)
        assert polygon_list_comparison(merged_hole_reference, holes, comparison_tolerance)

        # Example 4
        shape_2 = Shapes([[10.0, 0.0], [10.0, 4.0], [11.0, 12.0], [7.0, 12.0], [7.0, 10.0], [8.0, 11.0], [9.0, 10.0]])
        merged_shape_reference = [
            Shapes([[10.0, 0.0], [10.0, 4.0], [11.0, 12.0], [7.0, 12.0], [7.0, 10.0], [0.0, 10.0], [0.0, 0.0]])
        ]
        merged_hole_reference = [Shapes([[9.0, 10.0], [8.0, 11.0], [7.0, 10.0]])]
        merged_shapes, holes = merge_two_shapes(shape_1, shape_2)
        assert polygon_list_comparison(merged_shape_reference, merged_shapes, comparison_tolerance)
        assert polygon_list_comparison(merged_hole_reference, holes, comparison_tolerance)

        # Example 5
        shape_2 = Shapes([[10.0, 2.0], [11.0, 5.0], [11.0, 8.0], [10.0, 9.0]])
        merged_shape_reference = [
            Shapes(
                [[10.0, 2.0], [11.0, 5.0], [11.0, 8.0], [10.0, 9.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]]
            )
        ]
        merged_hole_reference = []
        merged_shapes, holes = merge_two_shapes(shape_1, shape_2)
        assert polygon_list_comparison(merged_shape_reference, merged_shapes, comparison_tolerance)
        assert polygon_list_comparison(merged_hole_reference, holes, comparison_tolerance)

    def test_polygon_clipping(self):

        # Shared Inputs
        comparison_tolerance = 1e-8
        shape_1 = Shapes([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])

        # Example 1
        shape_2 = Shapes([[7.5, 7.5], [12.5, 7.5], [12.5, 12.5], [7.5, 12.5]])
        clipped_shape_reference = [Shapes([[10.0, 7.5], [7.5, 7.5], [7.5, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]])]
        clipped_shapes = clip_two_shapes(shape_1, shape_2)
        assert polygon_list_comparison(clipped_shape_reference, clipped_shapes, comparison_tolerance)

        # Example 2
        shape_2 = Shapes([[5.0, 15.0], [2.0, 0.0], [5.0, 12.5], [8.5, 4.0]])
        clipped_shape_reference = [
            Shapes([[2.0, 0.0], [4.0, 10.0], [0.0, 10.0], [0.0, 0.0]]),
            Shapes(
                [
                    [4.4, 10.0],
                    [2.0, 0.0],
                    [10.0, 0.0],
                    [10.0, 10.0],
                    [6.59090909090909, 10.0],
                    [8.5, 4.0],
                    [6.02941176470588, 10.0],
                ]
            ),
        ]
        clipped_shapes = clip_two_shapes(shape_1, shape_2)
        assert polygon_list_comparison(clipped_shape_reference, clipped_shapes, comparison_tolerance)

        # Example 3
        shape_2 = Shapes(
            [[0.0, 10.0], [5.0, 5.0], [11.0, 11.0], [6.0, 10.0], [4.0, 11.0], [3.0, 10.0], [5.0, 6.0], [2.0, 10.0]]
        )
        clipped_shape_reference = [
            Shapes([[10.0, 10.0], [5.0, 5.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]]),
            Shapes([[3.0, 10.0], [2.0, 10.0], [5.0, 6.0]]),
        ]
        clipped_shapes = clip_two_shapes(shape_1, shape_2)
        assert polygon_list_comparison(clipped_shape_reference, clipped_shapes, comparison_tolerance)

        # Example 4
        shape_2 = Shapes([[10.0, 0.0], [10.0, 4.0], [11.0, 12.0], [7.0, 12.0], [7.0, 10.0], [8.0, 11.0], [9.0, 10.0]])
        clipped_shape_reference = [Shapes([[10.0, 0.0], [9.0, 10.0], [7.0, 10.0], [0.0, 10.0], [0.0, 0.0]])]
        clipped_shapes = clip_two_shapes(shape_1, shape_2)
        assert polygon_list_comparison(clipped_shape_reference, clipped_shapes, comparison_tolerance)

        # Example 5
        shape_2 = Shapes([[10.0, 2.0], [11.0, 5.0], [11.0, 8.0], [10.0, 9.0]])
        clipped_shape_reference = [
            Shapes([[10.0, 2.0], [10.0, 9.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]])
        ]
        clipped_shapes = clip_two_shapes(shape_1, shape_2)
        assert polygon_list_comparison(clipped_shape_reference, clipped_shapes, comparison_tolerance)


def polygon_list_comparison(poly_list_1, poly_list_2, tolerance):
    for i in range(len(poly_list_1)):
        poly_1 = poly_list_1[i].c
        poly_2 = poly_list_2[i].c
        return np.all([point_comparison(poly_1[j], poly_2[j], tolerance) for j in range(len(poly_1))])
    return len(poly_list_2) == 0


def point_comparison(p1, p2, tolerance):
    return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance
