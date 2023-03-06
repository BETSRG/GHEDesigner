from math import atan, pi, sin, sqrt

import numpy as np

from ghedesigner.constants import PI_OVER_2, TWO_PI


class Shapes:

    def __init__(self, c):
        """
         constructs a shape object
        """
        self.c = np.array(c)
        # print(c)
        xs = [0] * len(self.c)
        ys = [0] * len(self.c)
        for i in range(len(self.c)):
            xs[i] = self.c[i][0]
            ys[i] = self.c[i][1]
        self.max_x = max(xs)
        self.min_x = min(xs)
        self.max_y = max(ys)
        self.min_y = min(ys)

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
        r_a = []
        for i in range(len(self.c)):
            if i == len(self.c) - 1:
                c1 = self.c[len(self.c) - 1]
                c2 = self.c[0]
                r = vector_intersect(
                    [c1[0], c1[1], c2[0], c2[1]],
                    [x1, y1, x2, y2],
                    intersection_tolerance,
                )
                # print(r)
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
        if len(inters) % 2 == 0:
            # print(len(inters))
            return False
        else:
            # print("returning True")
            return True

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
    i = 0
    for inter in r_a:
        if inter[0] == 0:
            phi = PI_OVER_2
        else:
            phi = atan(inter[1] / inter[0])
        dist_inter = sqrt(inter[1] ** 2 + inter[0] ** 2)
        ref_ang = PI_OVER_2 - phi
        # sign = 1
        if phi > PI_OVER_2:
            if phi > pi:
                if phi > 3 * PI_OVER_2:
                    ref_ang = TWO_PI - phi
                else:
                    ref_ang = 3.0 * PI_OVER_2 - phi
            else:
                ref_ang = pi - phi
        # if phi > pi/2 + rotate and phi < 3*pi/2 + rotate:
        # sign = -1
        vals[i] = dist_inter * sin(ref_ang + rotate)
        i += 1
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
        a1 = float("inf")
    else:
        a1 = (y12 - y11) / (x12 - x11)
        c1 = y11 - x11 * a1
    if x22 - x21 == 0:
        a2 = float("inf")
    else:
        a2 = (y22 - y21) / (x22 - x21)
        c2 = y21 - x21 * a2
    if a1 == float("inf") or a2 == float("inf"):
        if a1 == float("inf") and a2 == float("inf"):
            if abs(x11 - x21) < intersection_tolerance:
                return [[x11, y11], [x12, y12]]
            else:
                return []
        elif a1 == float("inf"):
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
