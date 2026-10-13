from math import atan, inf, pi, sin, sqrt

import numpy as np

from ghedesigner.constants import PI_OVER_2, TWO_PI

POSSIBLE_NUMBER_OF_VERTEX_INTERSECTIONS = (2, 4)
MINIMUM_NUMBER_OF_VERTICES_FOR_POLYGON = 3


class Shapes:
    def __init__(self, c) -> None:
        """
        constructs a shape object
        """

        # Make sure the vertices are counter-clockwise.
        if is_clockwise(c):
            self.c = np.array(reverse_points(c), dtype=float)
        else:
            self.c = np.array(c, dtype=float)

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
        if edge_index == self.number_of_edges - 1:
            return [self.c[-1], self.c[0]]
        else:
            return [self.c[edge_index], self.c[edge_index + 1]]

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
        return np.all([point_polygon_check(self.c, p) == 1 for p in other_shape.c])

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
    new_intersection_list = []
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
            new_intersection_list.append(intersection)

    # If vertex intersection logic is working properly, intersections should appear in groups of 2 or 4.
    if not np.all(
        [(len(vertex_dictionary[key]) in POSSIBLE_NUMBER_OF_VERTEX_INTERSECTIONS) for key in vertex_dictionary]
    ):
        raise ValueError(
            "Vertex intersections found in a group size of neither 2 or 4."
            " This suggests an issue with the polygon intersection logic."
        )

    # Add in one copy of the vertex intersection back into the intersection list.
    for _, index_list in vertex_dictionary.items():
        new_index = min(index_list)
        new_intersection_list.insert(new_index, intersections[new_index])

    return new_intersection_list


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
                        x_val = 0.25 * (x12 + x22 + x21 + x22)
                        lower_y_val = max(l1_min_y, l2_min_y)
                        upper_y_val = min(l1_max_y, l2_max_y)
                        # Are lines intersecting at exactly one point?
                        if abs(lower_y_val - upper_y_val) <= intersection_tolerance:
                            return mark_endpoint_intersections(
                                mark_intersections, l1, l2, [[x_val, 0.5 * (lower_y_val + upper_y_val)]]
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
    if abs(a1 - a2) <= 0:  # Are line segments parallel?
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
    summation = 0
    for i in range(len(points)):
        p2 = points[i]
        p1 = points[i - 1]
        summation += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return summation > 0


def reverse_points(points):
    return_list = []
    for i in range(len(points)):
        return_list.append(points[-1 - i])
    return return_list


def circular_range(start, end, n):
    if start <= end:
        return list(range(start, end + 1))
    return list(range(start, n)) + list(range(end + 1))


class PolygonIntersectionNode:
    def __init__(self, location, index, coincident_vertices):
        self.type = "intersection"
        self.index = index
        self.location = location
        self.predecessor_1 = None
        self.successor_1 = None
        self.predecessor_2 = None
        self.successor_2 = None
        self.coincident_vertex_1 = None
        self.coincident_vertex_2 = None
        if coincident_vertices is not None:
            for coincident_vertex in coincident_vertices:
                if coincident_vertex[0] == 0:
                    self.coincident_vertex_1 = coincident_vertex[1]
                else:
                    self.coincident_vertex_2 = coincident_vertex[1]

    def update_predecessor_1(self, new_predecessor):
        self.predecessor_1 = new_predecessor

    def update_predecessor_2(self, new_predecessor):
        self.predecessor_2 = new_predecessor

    def update_successor_1(self, new_successor):
        self.successor_1 = new_successor

    def update_successor_2(self, new_successor):
        self.successor_2 = new_successor

    def get_identifier(self):
        return self.type, self.index

    def update_predecessor(self, new_predecessor, shape_1=True):
        if shape_1:
            self.update_predecessor_1(new_predecessor)
        else:
            self.update_predecessor_2(new_predecessor)

    def update_successor(self, new_successor, shape_1=True):
        if shape_1:
            self.update_successor_1(new_successor)
        else:
            self.update_successor_2(new_successor)

    def next(self, reverse=False, shape_1=True):
        if shape_1:
            return self.next_1(reverse=reverse)
        else:
            return self.next_2(reverse=reverse)

    def next_1(self, reverse=False):
        if reverse:
            return self.predecessor_1
        else:
            return self.successor_1

    def next_2(self, reverse=False):
        if reverse:
            return self.predecessor_2
        else:
            return self.successor_2


class PolygonVertexNode:
    def __init__(self, location, index, shape_number):
        self.type = "vertex"
        self.index = index
        self.location = location
        self.successor = None
        self.predecessor = None
        self.shape_number = shape_number

    def update_predecessor(self, new_predecessor, shape_1=True):  # noqa: ARG002
        self.predecessor = new_predecessor

    def update_successor(self, new_successor, shape_1=True):  # noqa: ARG002
        self.successor = new_successor

    def get_identifier(self):
        return self.type, self.index

    def next(self, reverse=False, shape_1=True):  # noqa: ARG002
        if reverse:
            return self.predecessor
        else:
            return self.successor


def insert_intersection_node_into_graph(intersection_node, shape_node_indices, shape_vertex_list, shape_1=True):
    # Note: we are assuming that shape_1_reference node is a vertex-type node.

    # Search where to place this intersection (this covers the case that there is already an intersection placed
    # along this edge of shape.
    shape_reference_node_index, shape_final_node_index = shape_node_indices
    shape_reference_node = shape_vertex_list[shape_reference_node_index]
    shape_final_node = shape_vertex_list[shape_final_node_index]
    shape_predecessor_node = shape_reference_node
    shape_successor_node = shape_predecessor_node.next(shape_1=shape_1)
    insertion_location_found = False

    if (
        shape_1 and intersection_node.coincident_vertex_1 is not None
    ):  # Replace one of the two vertices in Shape 1's line segment.
        if intersection_node.coincident_vertex_1 == shape_reference_node_index:
            intersection_node.update_successor(shape_reference_node.next(shape_1=shape_1), shape_1=shape_1)
            intersection_node.update_predecessor(
                shape_reference_node.next(shape_1=shape_1, reverse=True), shape_1=shape_1
            )
            (intersection_node.next(shape_1=shape_1)).update_predecessor(intersection_node, shape_1=shape_1)
            (intersection_node.next(shape_1=shape_1, reverse=True)).update_successor(intersection_node, shape_1=shape_1)
            shape_reference_node = intersection_node
        elif intersection_node.coincident_vertex_1 == shape_final_node_index:
            intersection_node.update_successor(shape_final_node.next(shape_1=shape_1), shape_1=shape_1)
            intersection_node.update_predecessor(shape_final_node.next(shape_1=shape_1, reverse=True), shape_1=shape_1)
            (intersection_node.next(shape_1=shape_1)).update_predecessor(intersection_node, shape_1=shape_1)
            (intersection_node.next(shape_1=shape_1, reverse=True)).update_successor(intersection_node, shape_1=shape_1)
            shape_final_node = intersection_node
        else:
            raise ValueError(
                "Coincident vertex on shape 1 found from intersection does not coincide with one of the corresponding"
                " line segment's endpoints. This suggests the presence of a bug in the polygon intersection algorithm."
            )
    elif (
        not shape_1 and intersection_node.coincident_vertex_2 is not None
    ):  # Replace one of the two vertices in Shape 2's line segment.
        if intersection_node.coincident_vertex_2 == shape_reference_node_index:
            intersection_node.update_successor(shape_reference_node.next(shape_1=shape_1), shape_1=shape_1)
            intersection_node.update_predecessor(
                shape_reference_node.next(shape_1=shape_1, reverse=True), shape_1=shape_1
            )
            (intersection_node.next(shape_1=shape_1)).update_predecessor(intersection_node, shape_1=shape_1)
            (intersection_node.next(shape_1=shape_1, reverse=True)).update_successor(intersection_node, shape_1=shape_1)
            shape_reference_node = intersection_node
        elif intersection_node.coincident_vertex_2 == shape_final_node_index:
            intersection_node.update_successor(shape_final_node.next(shape_1=shape_1), shape_1=shape_1)
            intersection_node.update_predecessor(shape_final_node.next(shape_1=shape_1, reverse=True), shape_1=shape_1)
            (intersection_node.next(shape_1=shape_1)).update_predecessor(intersection_node, shape_1=shape_1)
            (intersection_node.next(shape_1=shape_1, reverse=True)).update_successor(intersection_node, shape_1=shape_1)
            shape_final_node = intersection_node
        else:
            raise ValueError(
                "Coincident vertex on shape 2 found from intersection does not coincide with one of the corresponding"
                " line segment's endpoints. This suggests the presence of a bug in the polygon intersection algorithm."
            )
    else:
        # The vertices of shape 1 are listed counter-clockwise. If intersection(s) have been inserted into this
        # line-segment of shape 1 previously, we can use its distance from the preceding vertex to tell where
        # it should be placed.

        distance_from_reference = distance(intersection_node.location, shape_reference_node.location)
        while not insertion_location_found:
            successor_distance_from_reference = distance(shape_successor_node.location, shape_reference_node.location)
            if successor_distance_from_reference > distance_from_reference:
                insertion_location_found = True
            else:
                shape_predecessor_node = shape_successor_node
                shape_successor_node = shape_predecessor_node.next(shape_1=shape_1)

        # Insert intersection
        intersection_node.update_predecessor(shape_predecessor_node, shape_1=shape_1)
        intersection_node.update_successor(shape_successor_node, shape_1=shape_1)
        shape_predecessor_node.update_successor(intersection_node, shape_1=shape_1)
        shape_successor_node.update_predecessor(intersection_node, shape_1=shape_1)

    return shape_reference_node, shape_final_node


class PolygonIntersectionGraph:
    def __init__(self, p1_vertices, p2_vertices, intersections):

        # Instantiate Shape 1
        self.shape_1_nodes = []
        self.shape_1_size = len(p1_vertices)
        for i, vertex in enumerate(p1_vertices):
            self.shape_1_nodes.append(PolygonVertexNode(vertex, i, 0))
        for vdx in range(self.shape_1_size):
            self.shape_1_nodes[vdx].update_predecessor(self.shape_1_nodes[(vdx - 1) % self.shape_1_size])
            self.shape_1_nodes[vdx].update_successor(self.shape_1_nodes[(vdx + 1) % self.shape_1_size])

        # Instantiate Shape 2
        self.shape_2_nodes = []
        self.shape_2_size = len(p2_vertices)
        index_offset = self.shape_1_size
        for i, vertex in enumerate(p2_vertices):
            self.shape_2_nodes.append(PolygonVertexNode(vertex, i + index_offset, 1))
        for vdx in range(self.shape_2_size):
            self.shape_2_nodes[vdx].update_predecessor(self.shape_2_nodes[(vdx - 1) % self.shape_2_size])
            self.shape_2_nodes[vdx].update_successor(self.shape_2_nodes[(vdx + 1) % self.shape_2_size])

        # Add intersection connections.
        self.intersection_nodes = []
        for i, intersection in enumerate(intersections):
            new_intersection_node = PolygonIntersectionNode(
                intersection["location"], i, intersection["coincident_vertices"]
            )
            self.intersection_nodes.append(new_intersection_node)
            shape_1_vertex_indices = intersection["shape_1_vertex_indices"]
            shape_2_vertex_indices = intersection["shape_2_vertex_indices"]

            # Insert into shape 1
            self.shape_1_nodes[shape_1_vertex_indices[0]], self.shape_1_nodes[shape_1_vertex_indices[1]] = (
                insert_intersection_node_into_graph(
                    new_intersection_node, shape_1_vertex_indices, self.shape_1_nodes, shape_1=True
                )
            )

            # Insert into shape 2
            self.shape_2_nodes[shape_2_vertex_indices[0]], self.shape_2_nodes[shape_2_vertex_indices[1]] = (
                insert_intersection_node_into_graph(
                    new_intersection_node, shape_2_vertex_indices, self.shape_2_nodes, shape_1=False
                )
            )


def merge_two_shapes(shape_1, shape_2, omit_contained_shapes=True, print_intersections=False):
    # This is an implementation of the Weiler-Atherton algorithm for merging.
    # Returns intersections sorted counter-clockwise.
    shape_intersections = shape_1.get_shape_intersections(shape_2)
    if print_intersections:
        print("Polygon Intersections")
        for inter in shape_intersections:
            coord = inter["location"]
            print(f"{coord[0]}\t{coord[1]}")
    number_of_intersections = len(shape_intersections)
    # Shapes are either completely disparate, or one is completely contained within another.
    if number_of_intersections == 0:
        if omit_contained_shapes and shape_1.shape_is_contained_within(shape_2):
            return [[shape_1], []]
        elif omit_contained_shapes and shape_2.shape_is_contained_within(shape_1):
            return [[shape_2], []]
        else:
            return [[shape_1, shape_2], []]
    elif number_of_intersections == 1:  # Shapes touch at a single vertex. In this case, we will consider them separate.
        return [[shape_1, shape_2], []]
    else:  # The shapes meaningfully intersect
        int_graph = PolygonIntersectionGraph(shape_1.c, shape_2.c, shape_intersections)
        included_intersections = [False for _ in range(number_of_intersections)]
        contours = []
        branches = {}
        while not np.all(included_intersections):
            current_node = int_graph.intersection_nodes[included_intersections.index(False)]
            first_run = True
            current_walk_nodes = []
            current_walk_ids = []
            while (
                cdx := current_node.get_identifier()
            ) not in current_walk_ids or first_run:  # We terminate once we reach the node we started at.
                current_walk_nodes.append(current_node)
                current_walk_ids.append(cdx)
                if first_run:
                    first_run = False
                if current_node.type == "vertex":
                    current_node = current_node.next()
                else:
                    # We need to determine if traveling along shape 1 or shape 2 is appropriate at intersections.
                    candidate_point_1 = current_node.next_1()
                    candidate_point_2 = current_node.next_2()
                    shape_1_edge_in_shape_2 = point_polygon_check(
                        shape_2.c, point_average((current_node.location, candidate_point_1.location))
                    )
                    shape_2_edge_in_shape_1 = point_polygon_check(
                        shape_1.c, point_average((current_node.location, candidate_point_2.location))
                    )

                    # We want to choose the edge (from the two) that does not lead inside one of the polygons.
                    if shape_1_edge_in_shape_2 == 1 and shape_2_edge_in_shape_1 == 1:
                        raise ValueError("Both exiting edges from intersection in merge shapes lead inside a shape.")
                    elif shape_1_edge_in_shape_2 == -1 and shape_2_edge_in_shape_1 == -1:
                        # Both branches are valid. This happens when a intersection node is part of two loops.
                        # We can take either branch, but we need to remember to take the other branch in future loops.
                        if cdx in branches:
                            current_node = candidate_point_1
                        else:
                            current_node = candidate_point_2
                            branches[cdx] = True
                    elif shape_1_edge_in_shape_2 == -1:
                        current_node = candidate_point_1
                    elif shape_2_edge_in_shape_1 == -1:
                        current_node = candidate_point_2
                    elif (
                        shape_1_edge_in_shape_2 == 0
                    ):  # Neither edges lead outside, but one might lead along existing edge.
                        current_node = candidate_point_1
                    else:
                        current_node = candidate_point_2

            # There is a loop in the walk we performed, so we need to trim out the non-loop part.
            new_contour_nodes = current_walk_nodes[current_walk_ids.index(cdx) :]
            new_contour = [node.location for node in new_contour_nodes]
            for node in new_contour_nodes:
                if node.type == "intersection":
                    included_intersections[node.index] = True
            contours.append(Shapes(new_contour))
        contour_areas = [contour.get_area() for contour in contours]
        max_area_index = contour_areas.index(max(contour_areas))
        return [[contours[max_area_index]], contours[:max_area_index] + contours[max_area_index + 1 :]]


def clip_two_shapes(shape_1, shape_2, print_intersections=False):
    # This is an implementation of the Weiler-Atherton algorithm for clipping.
    # Returns intersections sorted counter-clockwise.
    shape_intersections = shape_1.get_shape_intersections(shape_2)
    if print_intersections:
        print("Polygon Intersections")
        for inter in shape_intersections:
            coord = inter["location"]
            print(f"{coord[0]}\t{coord[1]}")
    number_of_intersections = len(shape_intersections)
    # Shapes are either completely disparate, or one is completely contained within another.
    if number_of_intersections == 0:
        if shape_2.shape_is_contained_within(shape_1):
            return []
        else:
            return [shape_1]
    elif number_of_intersections == 1:  # Shapes touch at a single vertex. In this case, we will consider them separate.
        return [shape_1]
    else:  # The shapes meaningfully intersect
        int_graph = PolygonIntersectionGraph(shape_1.c, shape_2.c, shape_intersections)
        included_intersections = [False for _ in range(number_of_intersections)]
        contours = []
        branches = {}
        while not np.all(included_intersections):
            current_node = int_graph.intersection_nodes[included_intersections.index(False)]
            first_run = True
            current_walk_nodes = []
            current_walk_ids = []
            while (
                cdx := current_node.get_identifier()
            ) not in current_walk_ids or first_run:  # We terminate once we reach the node we started at.
                current_walk_nodes.append(current_node)
                current_walk_ids.append(cdx)
                if first_run:
                    first_run = False
                if current_node.type == "vertex":
                    current_node = (
                        current_node.next() if current_node.shape_number == 0 else current_node.next(reverse=True)
                    )
                else:
                    # We need to determine if traveling along shape 1 or shape 2 is appropriate at intersections.
                    candidate_point_1 = current_node.next_1()
                    candidate_point_2 = current_node.next_2(reverse=True)  # We trace clipping polygon clockwise.
                    shape_1_edge_in_shape_2 = point_polygon_check(
                        shape_2.c, point_average((current_node.location, candidate_point_1.location))
                    )
                    shape_2_edge_in_shape_1 = point_polygon_check(
                        shape_1.c, point_average((current_node.location, candidate_point_2.location))
                    )

                    # The edge selection is more complicated than for polygon merging.
                    # At each intersection, we would like to follow the edge of shape 2 that goes inside of shape 1.
                    # If that does not exist, we want to make sure that we do not follow an edge of shape 1 that goes
                    # inside of shape 2.
                    if shape_1_edge_in_shape_2 == 1 and shape_2_edge_in_shape_1 == -1:
                        if (
                            len(current_walk_ids) == 1
                        ):  # When clipping, there are some intersections that we start at that are not part of any
                            # loop. This helps us still mark them off.
                            break
                        else:
                            raise ValueError("Both exiting edges from intersection in clip shapes are not valid.")
                    elif shape_1_edge_in_shape_2 == -1 and shape_2_edge_in_shape_1 == 1:
                        # Both branches are valid. This happens when a intersection node is part of two loops.
                        # We can take either branch, but we need to remember to take the other branch in future loops.
                        if cdx in branches:
                            current_node = candidate_point_1
                        else:
                            current_node = candidate_point_2
                            branches[cdx] = True
                    elif shape_2_edge_in_shape_1 == 1:  # Follow shape 2 edge into shape 1
                        current_node = candidate_point_2
                    elif shape_1_edge_in_shape_2 == -1:  # Follow shape 1 edge outside of shape 2
                        current_node = candidate_point_1
                    elif (
                        shape_2_edge_in_shape_1 == 0
                    ):  # Neither edges lead outside, but one might lead along existing edge.
                        current_node = candidate_point_2
                    else:
                        current_node = candidate_point_1

            # There is a loop in the walk we performed, so we need to trim out the non-loop part.
            new_contour_nodes = current_walk_nodes[current_walk_ids.index(cdx) :]
            new_contour = [node.location for node in new_contour_nodes]
            for node in new_contour_nodes:
                if node.type == "intersection":
                    included_intersections[node.index] = True
            if len(new_contour) > 1:
                # There are instances where some intersections will only lead to existing contours.
                # We need to make sure to mark these off to prevent looping forever.
                if len(contours) == 0:
                    include_new_contour = True
                else:
                    include_new_contour = False
                    for existing_contour in contours:
                        if len(existing_contour.c) == len(new_contour):
                            for i in range(len(new_contour)):
                                if new_contour[i] not in existing_contour.c:
                                    include_new_contour = True
                        else:
                            include_new_contour = True
                if include_new_contour:
                    contours.append(Shapes(new_contour))
                else:
                    for node in current_walk_nodes:
                        if node.type == "intersection":
                            included_intersections[node.index] = True

        return contours

def offset_line(l1, offset_distance):
    p1, p2 = l1
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0:
        slope = inf
    else:
        slope = dy / dx

    line_segment_magnitude = distance(p1, p2)
    offset_vector = [-offset_distance * dy / line_segment_magnitude, offset_distance * dx / line_segment_magnitude]
    p_avg = point_average([p1, p2])
    pass_through_point = [p_avg[0] + offset_vector[0], p_avg[1] + offset_vector[1]]
    if slope == inf:
        return slope, pass_through_point[0]
    else:
        intercept = -slope * pass_through_point[0] + pass_through_point[1]
        return slope, intercept

def line_intersect(l1, l2, tolerance=1e-8):
    m1, b1 = l1
    m2, b2 = l2

    if m1 == m2: # Lines are parallel.
        if abs(b1 - b2) < tolerance: # lines overlap
            return inf
        else:
            return None
    else:
        if m1 == inf:
            return b1, m2 * b1 + b2
        elif m2 == inf:
            return b2, m1 * b2 + b1
        else:
            x_int = (b2 - b1) / (m1 - m2)
            y_int = 0.5 * ((m1 * x_int + b1) + (m2 * x_int + b2))
            return x_int, y_int

def zero_length_edge_and_spike_removal(new_polygon_vertices, intersection_tolerance):
    # Next we would like to move zero-length edges and spikes (adjacent edges w/ coincident angles equal to zero).
    # Removing one case can cause the other to appear, so we have to keep processing edges/vertices until we make
    # a lap without detecting zero-length edges or spikes.
    number_of_vertices = len(new_polygon_vertices)
    rv_idx = (0, False)
    v_idx = 0
    first_run = True
    processing_edge = False
    while (((v_idx, processing_edge) != rv_idx or first_run) and
           number_of_vertices >= MINIMUM_NUMBER_OF_VERTICES_FOR_POLYGON):
        if first_run:
            first_run = False
        if processing_edge:  # Check for zero-length edge
            p1 = new_polygon_vertices[v_idx]
            v2_idx = (v_idx + 1) % number_of_vertices
            p2 = new_polygon_vertices[v2_idx]
            edge_length = distance(p1, p2)
            if edge_length < intersection_tolerance:
                # Remove edge and check next one
                new_vertex = point_average([p1, p2])
                new_polygon_vertices[v_idx] = new_vertex
                del new_polygon_vertices[v2_idx]
                number_of_vertices -= 1
                rv_idx = (v_idx, True)
                first_run = True  # Delay checking for loop exit for one iteration.
            else:  # Edge is fine move on to next vertex
                v_idx = (v_idx + 1) % number_of_vertices
                processing_edge = False
        else:  # Check for spike
            # It should be noted that this will also detect collinear edges. Which are not inherently a problem,
            # but deleting them will not cause an issue either, so they are removed since a simple cross product
            # check is an easy way to detect spikes.
            v1_idx = (v_idx + 1) % number_of_vertices
            v2_idx = (v_idx - 1) % number_of_vertices
            p1 = new_polygon_vertices[v1_idx]
            pr = new_polygon_vertices[v_idx]
            p2 = new_polygon_vertices[v2_idx]
            e1_vector = (p1[0] - pr[0], p1[1] - pr[1])
            e2_vector = (p2[0] - pr[0], p2[1] - pr[1])
            cross_product = e1_vector[0] * e2_vector[1] - e1_vector[1] * e2_vector[0]
            if abs(cross_product) < intersection_tolerance:
                del new_polygon_vertices[v_idx]
                number_of_vertices -= 1
                rv_idx = (v_idx, False)
                first_run = True  # Delay checking for loop exit for one iteration.
            else:  # Edges are not collinear or a spike. Move on to check edge length.
                processing_edge = True
    return new_polygon_vertices

def offset_polygon(polygon, offset_distance, intersection_tolerance=1e-8):

    number_of_edges = polygon.number_of_edges

    # Generate lines that are offset from polygon's current line segments.
    new_line_list = []
    for edx in range(number_of_edges):
        new_line_list.append(offset_line(polygon.get_edge_line_segment(edx), offset_distance))

    # Generate new polygon from lines.
    new_polygon_vertices = []
    for edx in range(number_of_edges):
        int_point = line_intersect(new_line_list[edx % number_of_edges], new_line_list[(edx + 1) % number_of_edges],
                                   intersection_tolerance=intersection_tolerance)
        if int_point is None:
            raise ValueError("Adjacent offset lines do not intersect.")
        elif int_point == inf:
            pass # In the case of two parallel lines, we can just treat them as one moving forward.
        else:
            new_polygon_vertices.append(int_point)


    new_polygon_vertices = zero_length_edge_and_spike_removal(new_polygon_vertices, intersection_tolerance)
    number_of_vertices = len(new_polygon_vertices)


    # In many cases this new polygon will still contain self-intersecting portions. The next step is to build
    # a directed graph from intersections between non-adjacent line-segments.
    intersections = []
    for i in range(number_of_edges):
        ldx_1 = i % number_of_edges
        rdx_1 = (i + 1) % number_of_edges
        for j in range(number_of_edges):
            if abs(j - i) <= 1:  # Don't check adjacent lines.
                continue
            ldx_2 = j % number_of_edges
            rdx_2 = (j + 1) % number_of_edges
            intersections = vector_intersect(new_polygon_vertices[ldx_1] + new_polygon_vertices[rdx_1],
                                             new_polygon_vertices[ldx_2] + new_polygon_vertices[rdx_2],
                                             intersection_tolerance, advanced=True, mark_intersections=True)




