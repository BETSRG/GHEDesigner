from math import sqrt
from unittest import TestCase

import numpy as np

from ghedesigner.ghe.shape import (
    Shapes,
    clip_shapes,
    get_area,
    merge_shapes,
    offset_polygon,
    point_polygon_check,
    trim_vertex_intersection,
    vector_intersect,
)

CLIPPING_SHAPES = [
    Shapes([[7.5, 7.5], [12.5, 7.5], [12.5, 12.5], [7.5, 12.5]]),
    Shapes([[5.0, 15.0], [2.0, 0.0], [5.0, 12.5], [8.5, 4.0]]),
    Shapes([[0.0, 10.0], [5.0, 5.0], [11.0, 11.0], [6.0, 10.0], [4.0, 11.0], [3.0, 10.0], [5.0, 6.0], [2.0, 10.0]]),
    Shapes([[10.0, 0.0], [10.0, 4.0], [11.0, 12.0], [7.0, 12.0], [7.0, 10.0], [8.0, 11.0], [9.0, 10.0]]),
    Shapes([[10.0, 2.0], [11.0, 5.0], [11.0, 8.0], [10.0, 9.0]]),
]
CLIPPING_TOLERANCE = 1e-6
SUBJECT_SHAPE = Shapes([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


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
        comparison_tolerance = CLIPPING_TOLERANCE

        # Example 1
        merged_shape_reference = [
            Shapes(
                [[10.0, 7.5], [12.5, 7.5], [12.5, 12.5], [7.5, 12.5], [7.5, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]]
            )
        ]
        merged_hole_reference = []
        merged_shapes, holes = merge_shapes([SUBJECT_SHAPE, CLIPPING_SHAPES[0]])
        assert polygon_list_comparison(merged_shape_reference, merged_shapes, comparison_tolerance)
        assert polygon_list_comparison(merged_hole_reference, holes, comparison_tolerance)

        # Example 2
        merged_shape_reference = [
            Shapes(
                [
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
        merged_shapes, holes = merge_shapes([SUBJECT_SHAPE, CLIPPING_SHAPES[1]])
        assert polygon_list_comparison(merged_shape_reference, merged_shapes, comparison_tolerance)
        assert polygon_list_comparison(merged_hole_reference, holes, comparison_tolerance)

        # Example 3
        merged_shape_reference = [
            Shapes(
                [
                    [10.0, 10.0],
                    [11.0, 11.0],
                    [6.0, 10.0],
                    [4.0, 11.0],
                    [3.0, 10.0],
                    [0.0, 10.0],
                    [0.0, 0.0],
                    [10.0, 0.0],
                ]
            )
        ]
        merged_hole_reference = []
        merged_shapes, holes = merge_shapes([SUBJECT_SHAPE, CLIPPING_SHAPES[2]])
        assert polygon_list_comparison(merged_shape_reference, merged_shapes, comparison_tolerance)
        assert polygon_list_comparison(merged_hole_reference, holes, comparison_tolerance)

        # Example 4
        merged_shape_reference = [
            Shapes(
                [
                    [0.0, 0.0],
                    [10.0, 0.0],
                    [10.0, 4.0],
                    [11.0, 12.0],
                    [7.0, 12.0],
                    [7.0, 10.0],
                    [8.0, 11.0],
                    [9.0, 10.0],
                    [0.0, 10.0],
                ]
            )
        ]
        merged_hole_reference = []
        merged_shapes, holes = merge_shapes([SUBJECT_SHAPE, CLIPPING_SHAPES[3]])
        assert polygon_list_comparison(merged_shape_reference, merged_shapes, comparison_tolerance)
        assert polygon_list_comparison(merged_hole_reference, holes, comparison_tolerance)

        # Example 5
        merged_shape_reference = [
            Shapes(
                [[10.0, 2.0], [11.0, 5.0], [11.0, 8.0], [10.0, 9.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]]
            )
        ]
        merged_hole_reference = []
        merged_shapes, holes = merge_shapes([SUBJECT_SHAPE, CLIPPING_SHAPES[4]])
        assert polygon_list_comparison(merged_shape_reference, merged_shapes, comparison_tolerance)
        assert polygon_list_comparison(merged_hole_reference, holes, comparison_tolerance)

    def test_polygon_clipping(self):

        # Shared Inputs
        comparison_tolerance = CLIPPING_TOLERANCE

        # Example 1
        clipped_shape_reference = [Shapes([[10.0, 7.5], [7.5, 7.5], [7.5, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]])]
        clipped_shapes, _holes = clip_shapes([SUBJECT_SHAPE], [CLIPPING_SHAPES[0]])
        assert polygon_list_comparison(clipped_shape_reference, clipped_shapes, comparison_tolerance)

        # Example 2
        clipped_shape_reference = [
            Shapes(
                [
                    [10.0, 10.0],
                    [6.59090909, 10.0],
                    [8.5, 4.0],
                    [6.02941176, 10.0],
                    [4.4, 10.0],
                    [2.0, 0.0],
                    [4.0, 10.0],
                    [0.0, 10.0],
                    [0.0, 0.0],
                    [10.0, 0.0],
                ]
            )
        ]
        clipped_shapes, _holes = clip_shapes([SUBJECT_SHAPE], [CLIPPING_SHAPES[1]])
        assert polygon_list_comparison(clipped_shape_reference, clipped_shapes, comparison_tolerance)

        # Example 3
        clipped_shape_reference = [
            Shapes([[10.0, 10.0], [5.0, 5.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]]),
            Shapes([[3.0, 10.0], [2.0, 10.0], [5.0, 6.0]]),
        ]
        clipped_shapes, _holes = clip_shapes([SUBJECT_SHAPE], [CLIPPING_SHAPES[2]])
        assert polygon_list_comparison(clipped_shape_reference, clipped_shapes, comparison_tolerance)

        # Example 4
        clipped_shape_reference = [Shapes([[10.0, 0.0], [9.0, 10.0], [0.0, 10.0], [0.0, 0.0]])]
        clipped_shapes, _holes = clip_shapes([SUBJECT_SHAPE], [CLIPPING_SHAPES[3]])
        assert polygon_list_comparison(clipped_shape_reference, clipped_shapes, comparison_tolerance)

        # Example 5
        clipped_shape_reference = [Shapes([[10.0, 10.0], [0.0, 10.0], [0.0, 0.0], [10.0, 0.0]])]
        clipped_shapes, _holes = clip_shapes([SUBJECT_SHAPE], [CLIPPING_SHAPES[4]])
        assert polygon_list_comparison(clipped_shape_reference, clipped_shapes, comparison_tolerance)

    def test_polygon_offsetting(self):

        # Shared Inputs
        comparison_tolerance = CLIPPING_TOLERANCE

        # Example 1
        offset = 5.0
        offset_shapes = [
            Shapes(
                [
                    [19.46202532, 108.8860759],
                    [19.67827004, 94.46835443],
                    [24.65189873, 75.3164557],
                    [37.19409283, 56.59493671],
                    [51.68248945, 45.83544304],
                    [84.33544304, 38.94936709],
                    [112.0147679, 38.94936709],
                    [131.0443038, 35.50632911],
                    [147.2626582, 28.83544304],
                    [160.8860759, 18.07594937],
                    [171.6983122, 18.29113924],
                    [167.157173, 72.94936709],
                    [169.1033755, 80.48101266],
                    [177.3206751, 99.63291139],
                    [182.2943038, 115.7721519],
                    [182.2943038, 121.3670886],
                    [155.0474684, 118.5696203],
                    [104.0137, 115.4427416],
                    [108.5548523, 81.34177215],
                    [129.0981013, 80.91139241],
                    [128.8818565, 78.5443038],
                    [128.8818565, 63.26582278],
                    [121.5295359, 62.18987342],
                    [120.0158228, 53.15189873],
                    [93.85021097, 52.50632911],
                    [73.0907173, 53.36708861],
                    [74.38818565, 80.69620253],
                    [95.7964135, 81.7721519],
                    [95.58016878, 114.9260125],
                    [53.19620253, 112.3291139],
                ]
            )
        ]
        offset_holes = []
        offset_reference = [
            Shapes(
                [
                    [166.2745591, 23.1841835],
                    [162.1395087, 72.95463028],
                    [162.210932, 73.79298042],
                    [164.3100794, 81.91651028],
                    [164.4332964, 82.27731506],
                    [172.6208106, 101.3597934],
                    [177.0722818, 115.8046524],
                    [155.5070092, 113.5905176],
                    [155.4045526, 113.5821226],
                    [109.6787047, 110.7804644],
                    [112.945255, 86.25089063],
                    [131.3769209, 85.86474842],
                    [134.2752003, 82.62206507],
                    [133.8818565, 78.31639145],
                    [133.8818565, 61.40449283],
                    [131.4475659, 58.58803651],
                    [125.8589532, 57.77019075],
                    [124.6502833, 50.55354688],
                    [121.9357252, 48.19774582],
                    [93.89091938, 47.50581189],
                    [93.72564235, 47.50719796],
                    [70.80550313, 48.4575452],
                    [67.99771093, 51.5266793],
                    [69.48531023, 82.86060022],
                    [72.21018413, 85.59304989],
                    [90.76530293, 86.52560576],
                    [90.61472025, 109.6123991],
                    [53.6031005, 107.3446684],
                    [24.53021153, 104.3773791],
                    [24.66870104, 95.1438441],
                    [29.27810704, 77.39445378],
                    [40.86783266, 60.09466823],
                    [53.78497911, 50.50202944],
                    [84.85692504, 43.94936709],
                    [112.2386609, 43.94936709],
                    [112.6846548, 43.90934488],
                    [132.1945666, 40.37939187],
                    [132.7018681, 40.23097996],
                    [149.4868264, 33.32704007],
                    [150.0882224, 32.975196],
                    [162.5785125, 23.11062315],
                ]
            )
        ]
        offset_polygons, holes = offset_polygon(offset_shapes, offset_holes, offset)
        assert polygon_list_comparison(offset_polygons + holes, offset_reference, comparison_tolerance)

        # Example 2
        offset = 5.0
        offset_shapes = [
            Shapes([[0.0, 0.0], [50.0, 0.0], [50.0, 50.0], [0.0, 50.0]]),
            Shapes([[60.0, 0.0], [80.0, 0.0], [80.0, 20.0], [60.0, 20.0]]),
        ]
        offset_holes = [Shapes([[20.0, 20.0], [30.0, 20.0], [30.0, 30.0], [20.0, 30.0]])]
        offset_reference = [
            Shapes([[5.0, 5.0], [45.0, 5.0], [45.0, 45.0], [5.0, 45.0]]),
            Shapes([[65.0, 5.0], [75.0, 5.0], [75.0, 15.0], [65.0, 15.0]]),
        ]
        hole_reference = [
            Shapes(
                [
                    [32.07106781, 15.0],
                    [35.0, 17.92893219],
                    [35.0, 32.07106781],
                    [32.07106781, 35.0],
                    [17.92893219, 35.0],
                    [15.0, 32.07106781],
                    [15.0, 17.92893219],
                    [17.92893219, 15.0],
                ]
            )
        ]
        offset_polygons, holes = offset_polygon(offset_shapes, offset_holes, offset)
        assert polygon_list_comparison(offset_polygons, offset_reference, comparison_tolerance)
        assert polygon_list_comparison(holes, hole_reference, comparison_tolerance)

    def test_vector_intersect_vertical_touching(self):
        # Collinear vertical segments that share exactly one endpoint.
        result = vector_intersect(
            [0.0, 0.0, 0.0, 5.0], [0.0, 5.0, 0.0, 10.0], 1e-8, advanced=True, mark_intersections=True
        )
        assert len(result) == 1
        assert point_comparison(result[0][0], (0.0, 5.0), 1e-8)

    def test_vector_intersect_nearly_parallel(self):
        # Slopes differ by less than the tolerance, so the segments are treated as parallel.
        assert vector_intersect([0.0, 0.0, 10.0, 0.0], [0.0, 1.0, 10.0, 1.0 + 1e-10], 1e-8) == []

    def test_trim_vertex_intersection_preserves_order(self):
        intersections = [
            {"location": (0, 0), "coincident_vertices": [(0, 1)]},
            {"location": (0, 0), "coincident_vertices": [(0, 1)]},
            {"location": (1, 0), "coincident_vertices": [(1, 2)]},
            {"location": (1, 0), "coincident_vertices": [(1, 2)]},
            {"location": (2, 0), "coincident_vertices": []},
            {"location": (3, 0), "coincident_vertices": []},
        ]
        trimmed = trim_vertex_intersection(intersections)
        assert [t["location"] for t in trimmed] == [(0, 0), (1, 0), (2, 0), (3, 0)]

    def test_merge_overlapping_clip_shapes(self):
        # The two clip shapes overlap each other outside the subject; the overlap must not become a hole.
        clip_a = Shapes([[12.0, 0.0], [20.0, 0.0], [20.0, 10.0], [12.0, 10.0]])
        clip_b = Shapes([[15.0, 0.0], [25.0, 0.0], [25.0, 10.0], [15.0, 10.0]])
        merged_shapes, holes = merge_shapes([SUBJECT_SHAPE, clip_a, clip_b])
        assert len(holes) == 0
        assert sum(get_area(shape.c) for shape in merged_shapes) == 100.0 + 130.0

    def test_shape_is_contained_within(self):
        inner = Shapes([[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]])
        assert inner.shape_is_contained_within(SUBJECT_SHAPE)
        assert not SUBJECT_SHAPE.shape_is_contained_within(inner)


def polygon_list_comparison(poly_list_1, poly_list_2, tolerance):

    if len(poly_list_1) != len(poly_list_2):
        return False

    poly_list_1 = sort_polygons(poly_list_1)
    poly_list_2 = sort_polygons(poly_list_2)

    for shape_1, shape_2 in zip(poly_list_1, poly_list_2):
        poly_1 = shape_1.c
        poly_2 = shape_2.c
        if len(poly_1) != len(poly_2):
            return False
        if not np.all([point_comparison(poly_1[j], poly_2[j], tolerance) for j in range(len(poly_1))]):
            return False
    return True


def sort_polygons(polygon_list):
    oriented_list = []
    for polygon in polygon_list:
        oriented_list.append(orient_polygon(polygon))

    # Sort by area, then by the first (minimum) vertex to break ties between equal-area polygons.
    return sorted(oriented_list, key=lambda polygon: (get_area(polygon.c), *polygon.c[0]))


def orient_polygon(polygon):
    min_index = None
    polygon = polygon.c.tolist()
    for i, point in enumerate(polygon):
        if min_index is None or point_less_than(point, polygon[min_index]):
            min_index = i
    return Shapes(polygon[min_index:] + polygon[0:min_index])


def point_less_than(p1, p2):
    if p1[0] < p2[0]:
        return True
    elif p1[0] == p2[0]:
        return p1[1] < p2[1]
    else:
        return False


def point_comparison(p1, p2, tolerance):
    return abs(p1[0] - p2[0]) < tolerance and abs(p1[1] - p2[1]) < tolerance
