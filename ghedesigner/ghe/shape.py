from math import atan, inf, pi, sin, sqrt

import numpy as np
from pyclipper import (
    CT_DIFFERENCE,
    CT_UNION,
    ET_CLOSEDPOLYGON,
    JT_SQUARE,
    PFT_NONZERO,
    PT_CLIP,
    PT_SUBJECT,
    Pyclipper,
    PyclipperOffset,
    scale_from_clipper,
    scale_to_clipper,
)

from ghedesigner.constants import PI_OVER_2, TWO_PI

POSSIBLE_NUMBER_OF_VERTEX_INTERSECTIONS = (2, 4)


class Shapes:
    def __init__(self, c) -> None:
        """
        constructs a shape object
        """

        # Make sure the vertices are counter-clockwise.
        if is_clockwise(c):
            self.reversed_c = np.array(c, dtype=float)
            self.c = np.array(reverse_points(c), dtype=float)
        else:
            self.c = np.array(c, dtype=float)
            self.reversed_c = np.array(reverse_points(c), dtype=float)

        self.max_x = np.max(self.c[:, 0])
        self.min_x = np.min(self.c[:, 0])
        self.max_y = np.max(self.c[:, 1])
        self.min_y = np.min(self.c[:, 1])
        self.area = self.get_area()
        self.arc_lengths = self._calc_arc_lengths()
        self.total_arc_length = sum(self.arc_lengths)
        self.piecewise_maxes = self._prep_arc_length_piecewise_function()
        self.number_of_edges = len(self.arc_lengths)

    def _calc_arc_lengths(self):
        lengths = []
        for i in range(len(self.c)):
            if i == len(self.c) - 1:
                c1 = self.c[i]
                c2 = self.c[0]
            else:
                c1 = self.c[i]
                c2 = self.c[i + 1]
            lengths.append(distance(c1, c2))
        return lengths

    def get_polygon_vertices(self, clock_wise=False):
        return self.reversed_c if clock_wise else self.c

    def _prep_arc_length_piecewise_function(self):
        piecewise_maxes = [0.0]
        current_length = 0
        maximum_length = self.total_arc_length
        for arc_length in self.arc_lengths:
            current_length += arc_length
            piecewise_maxes.append(current_length / maximum_length)
        return piecewise_maxes

    def _get_point_along_arc(self, ratio, arc_index):
        if arc_index == len(self.c) - 1:
            c1 = self.c[arc_index]
            c2 = self.c[0]
        else:
            c1 = self.c[arc_index]
            c2 = self.c[arc_index + 1]
        return ratio * (c2[0] - c1[0]) + c1[0], ratio * (c2[1] - c1[1]) + c1[1]

    def get_edge_vertex_indices(self, edge_index):
        if edge_index == self.number_of_edges - 1:
            return [self.number_of_edges - 1, 0]
        else:
            return [edge_index, edge_index + 1]

    def get_edge_line_segment(self, edge_index):
        return [self.c[edge_index], self.c[(edge_index + 1) % self.number_of_edges]]

    def get_shape_intersections(self, other_shape, intersection_tolerance=1e-8):
        intersections = []
        for edx_1 in range(self.number_of_edges):
            for edx_2 in range(other_shape.number_of_edges):
                edge_intersections = vector_intersect(
                    [val for point in self.get_edge_line_segment(edx_1) for val in point],
                    [val for point in other_shape.get_edge_line_segment(edx_2) for val in point],
                    intersection_tolerance,
                    advanced=True,
                    mark_intersections=True,
                )
                shape_1_vertex_indices = self.get_edge_vertex_indices(edx_1)
                shape_2_vertex_indices = other_shape.get_edge_vertex_indices(edx_2)
                coincident_vertices = []
                for edge_intersection in edge_intersections:
                    if len(edge_intersection[1]) > 0:
                        coincident_vertices.append(
                            [
                                (e[0], shape_1_vertex_indices[e[1]] if e[0] == 0 else shape_2_vertex_indices[e[1]])
                                for e in edge_intersection[1]
                            ]
                        )
                    else:
                        coincident_vertices.append([])

                edge_intersections = [edge_intersection[0] for edge_intersection in edge_intersections]
                if len(edge_intersections) == 1:
                    intersections.append(
                        {
                            "location": edge_intersections[0],
                            "coincident_vertices": coincident_vertices[0],
                            "shape_1_vertex_indices": shape_1_vertex_indices,
                            "shape_2_vertex_indices": shape_2_vertex_indices,
                        }
                    )
                elif len(edge_intersections) > 1:
                    # Make sure the two intersections are counter-clockwise w/ respect to self.
                    intersection_distances = (
                        distance(self.c[shape_1_vertex_indices[0]], edge_intersections[0]),
                        distance(self.c[shape_1_vertex_indices[0]], edge_intersections[1]),
                    )
                    if intersection_distances[0] > intersection_distances[1]:
                        edge_intersections = (edge_intersections[1], edge_intersections[0])
                        coincident_vertices = (coincident_vertices[1], coincident_vertices[0])
                    intersections.extend(
                        [
                            {
                                "location": edge_intersections[0],
                                "coincident_vertices": coincident_vertices[0],
                                "shape_1_vertex_indices": shape_1_vertex_indices,
                                "shape_2_vertex_indices": shape_2_vertex_indices,
                            },
                            {
                                "location": edge_intersections[1],
                                "coincident_vertices": coincident_vertices[1],
                                "shape_1_vertex_indices": shape_1_vertex_indices,
                                "shape_2_vertex_indices": shape_2_vertex_indices,
                            },
                        ]
                    )

        return trim_vertex_intersection(intersections)

    def shape_is_contained_within(self, other_shape):
        return np.all([point_polygon_check(other_shape.c, p) == 1 for p in self.c])

    def get_point_along_curve(self, relative_distance_along_curve):
        larger_index = next(i for i, x in enumerate(self.piecewise_maxes) if x >= relative_distance_along_curve)
        arc_ratio = (relative_distance_along_curve - self.piecewise_maxes[larger_index - 1]) / (
            self.piecewise_maxes[larger_index] - self.piecewise_maxes[larger_index - 1]
        )
        return self._get_point_along_arc(arc_ratio, larger_index - 1)

    def line_intersect(self, xy, rotate=0, intersection_tolerance=1e-6):
        """
        returns the intersections between a line segment and the shape

        Parameters
        -----------
        :param xy: [float,float,float,float]
            the x,y values of both endpoints of the line segment
        :param rotate:
        :param intersection_tolerance:

        :return: [[float]]
            the x,y values of the intersections
        """
        x1, y1, x2, y2 = xy
        # minx, maxx = (x1, x2) if x1 < x2 else (x2, x1)
        # miny, maxy = (y1, y2) if y1 < y2 else (y2, y1)

        r_a = []

        # Check if they fall within the same rectangular area - note: calling functions seem to assume
        # that this is an infinite line intersection check (rather than line segment).
        # if maxx < self.min_x or minx > self.max_x or maxy < self.min_y or miny > self.max_y:
        # return r_a

        for i in range(len(self.c)):
            if i == len(self.c) - 1:
                c1 = self.c[len(self.c) - 1]
                c2 = self.c[0]
                r = vector_intersect(
                    [c1[0], c1[1], c2[0], c2[1]],
                    [x1, y1, x2, y2],
                    intersection_tolerance,
                )

                if len(r) == 1:
                    r = r[0]
                    if (
                        (r[0] - max(c2[0], c1[0])) > intersection_tolerance
                        or (r[0] - min(c2[0], c1[0])) < -1 * intersection_tolerance
                        or (r[1] - max(c2[1], c1[1])) > intersection_tolerance
                        or (r[1] - min(c2[1], c1[1])) < -1 * intersection_tolerance
                    ):
                        continue
                    r_a.append(r)
            else:
                c1 = self.c[i]
                c2 = self.c[i + 1]
                r = vector_intersect(
                    [c1[0], c1[1], c2[0], c2[1]],
                    [x1, y1, x2, y2],
                    intersection_tolerance,
                )
                if len(r) == 1:
                    r = r[0]
                    if (
                        (r[0] - max(c2[0], c1[0])) > intersection_tolerance
                        or (r[0] - min(c2[0], c1[0])) < -1 * intersection_tolerance
                        or (r[1] - max(c2[1], c1[1])) > intersection_tolerance
                        or (r[1] - min(c2[1], c1[1])) < -1 * intersection_tolerance
                    ):
                        continue
                    r_a.append(r)

        r_a = sort_intersections(r_a, rotate)
        return r_a

    def point_intersect(self, xy, tol=1e-4):
        """
        returns whether the given point is inside the rectangle

        Parameters
        -----------
        :param xy: [float,float]
            x,y value of point
        :param tol: float
            The tolerance for

        :return: boolean
            true if inside, false if not
        """
        x, y = xy
        if (x > self.max_x + tol or x + tol < self.min_x) or (y > self.max_y + tol or y + tol < self.min_y):
            # print("Returning False b/c outside of box")
            return False
        far_x = self.min_x - 10
        inters = self.line_intersect([far_x, y, far_x + 1, y], intersection_tolerance=tol)
        inters = [inter for inter in inters if inter[0] <= x]
        if len(inters) == 1:
            return True
        i = 0
        while i < len(inters):
            for vert in self.c:
                if inters[i][0] == vert[0] and inters[i][1] == vert[1]:
                    inters.pop(i)
                    i -= 1
                    break
            i += 1

        # False if even, True if odd
        return len(inters) % 2 != 0

    def get_area(self):
        """
        returns area of shape

        :return: float
            area of shape
        """
        area_sum = 0
        for i in range(len(self.c)):
            if i == len(self.c) - 1:
                area_sum += self.c[len(self.c) - 1][0] * self.c[0][1] - (self.c[len(self.c) - 1][1] * self.c[0][0])
                continue
            area_sum += self.c[i][0] * self.c[i + 1][1] - (self.c[i][1] * self.c[i + 1][0])
        return 0.5 * area_sum


def sort_intersections(r_a, rotate):
    if len(r_a) == 0:
        return r_a
    vals = [0] * len(r_a)
    for i, inter in enumerate(r_a):
        phi = PI_OVER_2 if inter[0] == 0 else atan(inter[1] / inter[0])
        dist_inter = sqrt(inter[1] ** 2 + inter[0] ** 2)
        ref_ang = PI_OVER_2 - phi
        # sign = 1
        if phi > PI_OVER_2:
            if phi > pi:  # noqa: SIM108
                ref_ang = TWO_PI - phi if phi > 3 * PI_OVER_2 else 3.0 * PI_OVER_2 - phi
            else:
                ref_ang = pi - phi
        # if phi > pi/2 + rotate and phi < 3*pi/2 + rotate:
        # sign = -1
        vals[i] = dist_inter * sin(ref_ang + rotate)
    zipped = sorted(zip(vals, r_a))
    r_a = [row for _, row in zipped]
    return r_a


def trim_vertex_intersection(intersections):

    # Mark where duplicate vertex intersections are.
    vertex_dictionary = {}
    for i, intersection in enumerate(intersections):
        coincident_vertices = intersection["coincident_vertices"]
        if len(coincident_vertices) > 0:
            cv_key = "_".join([f"({cv[0]}_{cv[1]})" for cv in coincident_vertices])
            if cv_key in vertex_dictionary:
                vertex_dictionary[cv_key].append(i)
            else:
                vertex_dictionary[cv_key] = [i]
        else:
            intersection["coincident_vertices"] = None

    # If vertex intersection logic is working properly, intersections should appear in groups of 2 or 4.
    if not np.all(
        [(len(vertex_dictionary[key]) in POSSIBLE_NUMBER_OF_VERTEX_INTERSECTIONS) for key in vertex_dictionary]
    ):
        raise ValueError(
            "Vertex intersections found in a group size of neither 2 or 4."
            " This suggests an issue with the polygon intersection logic."
        )

    # Keep one copy of each vertex intersection, preserving the original ordering.
    kept_indices = {min(index_list) for index_list in vertex_dictionary.values()}
    return [
        intersection
        for i, intersection in enumerate(intersections)
        if intersection["coincident_vertices"] is None or i in kept_indices
    ]


def mark_endpoint_intersections(perform_marking, l1, l2, intersection_points, intersection_tolerance):
    if perform_marking:
        reference_points = [(l1[0], l1[1]), (l1[2], l1[3]), (l2[0], l2[1]), (l2[2], l2[3])]
        points_to_return = []
        for ip in intersection_points:
            distances = np.array([distance(ip, rp) for rp in reference_points], dtype=float)
            if np.all(distances > intersection_tolerance):
                points_to_return.append([ip, []])
            else:
                coincident_vertices = []
                for i, d in enumerate(distances):
                    if d <= intersection_tolerance:
                        coincident_vertices.append((int(i // 2), int(i % 2)))
                points_to_return.append([ip, coincident_vertices])
        return points_to_return
    else:
        return intersection_points


def tolerance_comparison(val_1, val_2, tolerance, conservative=False):
    if conservative:
        return (val_2 - val_1) > (tolerance)
    else:
        return (val_2 - val_1) > (-tolerance)


def vector_intersect(l1, l2, intersection_tolerance, advanced=False, mark_intersections=False):
    """
     Gives the intersection between two line segments. Returns a list of intersections. Two are returned
     if the line segments are intersecting, collinear, and advanced intersections are turned on.

    Parameters
    -----------
    :param l1: [[float]]
        endpoints of first line segment
    :param l2: [[float]]
        endpoints of the second line segment
    :param intersection_tolerance: float
        Used to determine if two vertical segments occupy the same x value, and if two intersections are treated
        as a single intersection.
    :param advanced: bool
        Determines if intersecting collinear line segments are treated as intersections. "True" includes them. Default
        is False.
    :param mark_intersections: bool
        Determines if intersections at an endpoint of either linesegment will be considered as an intersection.
    :return: [(float,float)]
        x,y values of intersection (returns None if there is none)
    """
    x11, y11, x12, y12 = l1
    x21, y21, x22, y22 = l2

    # Check they can feasibly intersect.
    l1_min_x_val, l1_max_x_val = sorted((x11, x12))
    l1_min_y_val, l1_max_y_val = sorted((y11, y12))
    l2_min_x_val, l2_max_x_val = sorted((x21, x22))
    l2_min_y_val, l2_max_y_val = sorted((y21, y22))
    if advanced and (
        tolerance_comparison(l2_max_x_val, l1_min_x_val, intersection_tolerance, conservative=True)
        or tolerance_comparison(l1_max_x_val, l2_min_x_val, intersection_tolerance, conservative=True)
        or tolerance_comparison(l2_max_y_val, l1_min_y_val, intersection_tolerance, conservative=True)
        or tolerance_comparison(l1_max_y_val, l2_min_y_val, intersection_tolerance, conservative=True)
    ):
        return []
    if x12 - x11 == 0:
        a1 = inf
    else:
        a1 = (y12 - y11) / (x12 - x11)
        c1 = y11 - x11 * a1
    if x22 - x21 == 0:
        a2 = inf
    else:
        a2 = (y22 - y21) / (x22 - x21)
        c2 = y21 - x21 * a2
    # Are either of the line segments vertical?
    if inf in (a1, a2):
        # Are both line segments vertical?
        if a1 == inf and a2 == inf:
            # Do they occupy the same x value?
            if abs(x11 - x21) < intersection_tolerance:
                if not advanced:
                    return mark_endpoint_intersections(
                        mark_intersections, l1, l2, [[x11, y11], [x12, y12]], intersection_tolerance
                    )
                else:  # Lines might have collinear intersection.
                    l1_min_y, l1_max_y = sorted((y11, y12))
                    l2_min_y, l2_max_y = sorted((y21, y22))

                    # Is there an overlapping portion of the lines?
                    if l1_min_y > l2_max_y or l1_max_y < l2_min_y:
                        return []
                    else:
                        x_val = 0.25 * (x11 + x12 + x21 + x22)
                        lower_y_val = max(l1_min_y, l2_min_y)
                        upper_y_val = min(l1_max_y, l2_max_y)
                        # Are lines intersecting at exactly one point?
                        if abs(lower_y_val - upper_y_val) <= intersection_tolerance:
                            return mark_endpoint_intersections(
                                mark_intersections,
                                l1,
                                l2,
                                [[x_val, 0.5 * (lower_y_val + upper_y_val)]],
                                intersection_tolerance,
                            )
                        # Return the overlapping portion of the line segments
                        return mark_endpoint_intersections(
                            mark_intersections,
                            l1,
                            l2,
                            [[x_val, lower_y_val], [x_val, upper_y_val]],
                            intersection_tolerance,
                        )
            else:
                return []
        elif a1 == inf:  # Is l1 vertical?
            if not advanced:
                return mark_endpoint_intersections(
                    mark_intersections, l1, l2, [[x11, a2 * x11 + c2]], intersection_tolerance
                )
            x_a = 0.5 * (x11 + x12)
            y_p = a2 * x_a + c2
            x2_min, x2_max = sorted((x21, x22))
            y1_min, y1_max = sorted((y11, y12))
            if (
                tolerance_comparison(y1_min, y_p, intersection_tolerance)
                and tolerance_comparison(y_p, y1_max, intersection_tolerance)
                and tolerance_comparison(x2_min, x_a, intersection_tolerance)
                and tolerance_comparison(x_a, x2_max, intersection_tolerance)
            ):
                return mark_endpoint_intersections(mark_intersections, l1, l2, [[x_a, y_p]], intersection_tolerance)
            else:
                return []
        else:  # l2 is vertical
            if not advanced:
                return mark_endpoint_intersections(
                    mark_intersections, l1, l2, [[x21, a1 * x21 + c1]], intersection_tolerance
                )
            x_a = 0.5 * (x21 + x22)
            y_p = a1 * x_a + c1
            x1_min, x1_max = sorted((x11, x12))
            y2_min, y2_max = sorted((y21, y22))
            if (
                tolerance_comparison(y2_min, y_p, intersection_tolerance)
                and tolerance_comparison(y_p, y2_max, intersection_tolerance)
                and tolerance_comparison(x1_min, x_a, intersection_tolerance)
                and tolerance_comparison(x_a, x1_max, intersection_tolerance)
            ):
                return mark_endpoint_intersections(mark_intersections, l1, l2, [[x_a, y_p]], intersection_tolerance)
            else:
                return []
    if abs(a1 - a2) <= intersection_tolerance:  # Are line segments parallel?
        if not advanced:
            return []
        elif abs(y22 - (a1 * x22 + c1)) <= intersection_tolerance:  # Are line segments collinear?
            (l1_min_x_val, l1_min_y_val), (l1_max_x_val, l1_max_y_val) = sorted(((x11, y11), (x12, y12)))
            (l2_min_x_val, l2_min_y_val), (l2_max_x_val, l2_max_y_val) = sorted(((x21, y21), (x22, y22)))
            if tolerance_comparison(
                l2_max_x_val, l1_min_x_val, intersection_tolerance, conservative=True
            ) or tolerance_comparison(
                l1_max_x_val, l2_min_x_val, intersection_tolerance, conservative=True
            ):  # Do the line segments overlap?
                return []
            else:
                lower_intersection_point = max((l1_min_x_val, l1_min_y_val), (l2_min_x_val, l2_min_y_val))
                upper_intersection_point = min((l1_max_x_val, l1_max_y_val), (l2_max_x_val, l2_max_y_val))
                # Do the lines intersect at only one point?
                if distance(lower_intersection_point, upper_intersection_point) <= intersection_tolerance:
                    return mark_endpoint_intersections(
                        mark_intersections,
                        l1,
                        l2,
                        [point_average((lower_intersection_point, upper_intersection_point))],
                        intersection_tolerance,
                    )
                else:
                    return mark_endpoint_intersections(
                        mark_intersections,
                        l1,
                        l2,
                        [lower_intersection_point, upper_intersection_point],
                        intersection_tolerance,
                    )
        else:
            return []
    rx = (c2 - c1) / (a1 - a2)
    ry = a1 * (c2 - c1) / (a1 - a2) + c1
    if advanced and (
        tolerance_comparison(rx, l1_min_x_val, intersection_tolerance, conservative=True)
        or tolerance_comparison(rx, l2_min_x_val, intersection_tolerance, conservative=True)
        or tolerance_comparison(l1_max_x_val, rx, intersection_tolerance, conservative=True)
        or tolerance_comparison(l2_max_x_val, rx, intersection_tolerance, conservative=True)
        or tolerance_comparison(ry, l1_min_y_val, intersection_tolerance, conservative=True)
        or tolerance_comparison(ry, l2_min_y_val, intersection_tolerance, conservative=True)
        or tolerance_comparison(l1_max_y_val, ry, intersection_tolerance, conservative=True)
        or tolerance_comparison(l2_max_y_val, ry, intersection_tolerance, conservative=True)
    ):
        return []
    return mark_endpoint_intersections(mark_intersections, l1, l2, [[rx, ry]], intersection_tolerance)


def point_average(points):
    number_of_points = len(points)
    x_average = np.sum([p[0] for p in points]) / number_of_points
    y_average = np.sum([p[1] for p in points]) / number_of_points
    return x_average, y_average


def distance(pt_1, pt_2) -> float:
    return sqrt((pt_1[0] - pt_2[0]) ** 2 + (pt_1[1] - pt_2[1]) ** 2)


def point_polygon_check(contour, point, on_edge_tolerance=1e-6):
    """
    Mimics pointPolygonTest from OpenCV-Python

    Adapted from the methods outlined in the links below.
    https://stackoverflow.com/a/63436180/5965685
    https://stackoverflow.com/a/17693146/5965685

    :param contour: list of tuples containing (x, y) contour boundary points
    :param point: tuple containing the (x, y) point to test
    :param on_edge_tolerance: float value representing the distance tolerance
                               to determine whether the test point is on the edge of the polygon.
                               Points within this distance from the edge are considered "on the edge".
                               Defaults to 0.001.


    :returns: -1 if outside, 0 if on edge, 1 if inside
    :rtype: int
    """

    # check if on edge
    # use Pythagoras to check whether the distance between the test point
    # and the line vertices add to the distance between the line vertices
    # if they are within tolerance, the point is co-linear
    # if not, it is off the line

    for idx, vertex in enumerate(contour):
        v1 = contour[idx - 1]
        v2 = vertex
        test_dist = distance(v1, point) + distance(v2, point)
        v12_dist = distance(v1, v2)

        if abs(test_dist - v12_dist) < on_edge_tolerance:
            return 0

    # if made it to here, not on edge and check if inside/outside
    def between(p, a, b) -> bool:
        return ((p >= a) and (p <= b)) or ((p <= a) and (p >= b))

    inside = True
    px = point[0]
    py = point[1]

    for idx, vertex in enumerate(contour):
        v1 = contour[idx - 1]
        v2 = vertex
        v1x = v1[0]
        v1y = v1[1]
        v2x = v2[0]
        v2y = v2[1]

        if between(py, v1y, v2y):  # points inside vertical range
            if ((py == v1y) and (v2y >= v1y)) or ((py == v2y) and (v1y >= v2y)):
                continue

            # calc cross product `PA X PB`, P lies on left side of AB if c > 0
            c = (v1x - px) * (v2y - py) - (v2x - px) * (v1y - py)

            if c == 0:
                return 0

            if (v1y < v2y) == (c > 0):
                inside = not inside

    return -1 if inside else 1


def get_area(c):
    """
    returns area of shape

    :return: float
        area of shape
    """
    area_sum = 0
    for i in range(len(c)):
        if i == len(c) - 1:
            area_sum += c[len(c) - 1][0] * c[0][1] - (c[len(c) - 1][1] * c[0][0])
            continue
        area_sum += c[i][0] * c[i + 1][1] - (c[i][1] * c[i + 1][0])
    return 0.5 * area_sum


def is_clockwise(points):
    return get_area(points) < 0


def reverse_points(points):
    return list(points)[::-1]


def merge_shapes(shapes):

    # You get different behavior with PyClipper merging depending on if the polygons are encoded as a PT_SUBJECT
    # or a PT_CLIP. Both solutions are correct, but with testing, the behavior from merging two polygons at a time
    # w/ one PT_CLIP and one PT_SUBJECT seems to be most appropriate in our use case.

    # It should be noted that if one polygon is completely contained within another, it is deleted. If two polygons
    # do not intersect, then the individual polygons are returned.

    # Call PyClipper
    merger = Pyclipper()
    for i, shape in enumerate(shapes):
        if i == 0:
            merger.AddPath(scale_to_clipper(shape.c), PT_SUBJECT, True)  # True for enclosed polygon.
        else:
            merger.AddPath(scale_to_clipper(shape.c), PT_CLIP, True)  # True for enclosed polygon.
    merge_solution = merger.Execute(CT_UNION, PFT_NONZERO, PFT_NONZERO)

    # Split solution into outer polygons and holes.
    outer_polygons = []
    holes = []
    for path in merge_solution:
        if is_clockwise(path):
            holes.append(Shapes(scale_from_clipper(path)))
        else:
            outer_polygons.append(Shapes(scale_from_clipper(path)))
    return [outer_polygons, holes]


def clip_shapes(outer_shapes, clipping_shapes):

    # Call PyClipper
    clipper = Pyclipper()
    for shape in outer_shapes:
        clipper.AddPath(scale_to_clipper(shape.c), PT_SUBJECT, True)  # True for enclosed polygon.
    for shape in clipping_shapes:
        clipper.AddPath(scale_to_clipper(shape.c), PT_CLIP, True)  # True for enclosed polygon.
    clip_solution = clipper.Execute(CT_DIFFERENCE, PFT_NONZERO, PFT_NONZERO)

    # Split solution into outer polygons and holes.
    outer_polygons = []
    holes = []
    for path in clip_solution:
        if is_clockwise(path):
            holes.append(Shapes(scale_from_clipper(path)))
        else:
            outer_polygons.append(Shapes(scale_from_clipper(path)))

    return [outer_polygons, holes]


def offset_polygon(outer_polygons, holes, offset):

    # Call PyClipper
    offsetter = PyclipperOffset()
    offsetter.AddPaths([scale_to_clipper(polygon.c) for polygon in outer_polygons], JT_SQUARE, ET_CLOSEDPOLYGON)
    if len(holes) > 0:
        offsetter.AddPaths([scale_to_clipper(polygon.reversed_c) for polygon in holes], JT_SQUARE, ET_CLOSEDPOLYGON)
    offset_solution = offsetter.Execute(-scale_to_clipper(offset))

    # Split solution into outer polygons and holes
    outer_polygons = []
    holes = []
    for path in offset_solution:
        if is_clockwise(path):
            holes.append(Shapes(scale_from_clipper(path)))
        else:
            outer_polygons.append(Shapes(scale_from_clipper(path)))

    return [outer_polygons, holes]
