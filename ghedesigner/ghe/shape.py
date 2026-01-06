from math import atan, inf, pi, sin, sqrt

import numpy as np

from ghedesigner.constants import PI_OVER_2, TWO_PI


class Shapes:
    def __init__(self, c) -> None:
        """
        constructs a shape object
        """
        self.c = np.array(c)
        # print(c)
        self.max_x = np.max(self.c[:, 0])
        self.min_x = np.min(self.c[:, 0])
        self.max_y = np.max(self.c[:, 1])
        self.min_y = np.min(self.c[:, 1])
        self.area = self.get_area()
        self.arc_lengths = self._calc_arc_lengths()
        self.total_arc_length = sum(self.arc_lengths)
        self.piecewise_maxes = self._prep_arc_length_piecewise_function()

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
        # print("x value: %f, r values:"%x1)
        # print(r_a)

        r_a = sort_intersections(r_a, rotate)
        # print(r_a)
        return r_a

    def point_intersect(self, xy):
        """
        returns whether the given point is inside the rectangle

        Parameters
        -----------
        :param xy: [float,float]
            x,y value of point

        :return: boolean
            true if inside, false if not
        """
        x, y = xy
        if (x > self.max_x or x < self.min_x) or (y > self.max_y or y < self.min_y):
            # print("Returning False b/c outside of box")
            return False
        far_x = self.min_x - 10
        inters = self.line_intersect([far_x, y, far_x + 1, y])
        # print(inters)
        inters = [inter for inter in inters if inter[0] <= x]
        # print("x: %f"%x,inters)
        if len(inters) == 1:
            # print("Returning True")
            return True
        i = 0
        while i < len(inters):
            for vert in self.c:
                if inters[i][0] == vert[0] and inters[i][1] == vert[1]:
                    inters.pop(i)
                    i -= 1
                    break
            i += 1
        return len(inters) % 2 != 0
        # False if even, True if odd

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


def vector_intersect(l1, l2, intersection_tolerance):
    """
     gives the intersection between two line segments

    Parameters
    -----------
    :param l1: [[float]]
        endpoints of first line segment
    :param l2: [[float]]
        endpoints of the second line segment
    :param intersection_tolerance:

    :return: [float,float]
        x,y values of intersection (returns None if there is none)
    """
    x11, y11, x12, y12 = l1
    x21, y21, x22, y22 = l2
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
    if inf in (a1, a2):
        if a1 == inf and a2 == inf:
            if abs(x11 - x21) < intersection_tolerance:
                return [[x11, y11], [x12, y12]]
            else:
                return []
        elif a1 == inf:
            return [[x11, a2 * x11 + c2]]
        else:
            return [[x21, a1 * x21 + c1]]
    if abs(a1 - a2) <= intersection_tolerance:
        if abs(y22 - (a1 * x22 + c1)) <= intersection_tolerance:
            # return [[x11,y11],[x12,y12]]
            return []
        else:
            return []
    rx = (c2 - c1) / (a1 - a2)
    ry = a1 * (c2 - c1) / (a1 - a2) + c1
    return [[rx, ry]]


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
