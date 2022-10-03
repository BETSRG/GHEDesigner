from math import atan, pi, sin, sqrt

import numpy as np


class Shapes:
    """
    a class to represent nogo zones

    Attributes
    ----------
    cx : float
        the x value of the centroid
    cy : float
        the y value of the centroid
    xw: float
        the xwidth
    yw : float
        the ywidth
    theta: float
        the rotation of the shape
    c00: [float,float]
        x,y location of 1 vertex
    c01: [float,float]
        x,y location of 2nd vertex
    c10: [float,float]
        x,y location of 3rd vertex
    c11: [float,float]
        x,y location of 4th vertex
    c: [[float]]
        array containing x,y locations of vertices
    maxy : float
        maximum y value of shape
    miny: float
        minimum y value of shape
    maxx: float
        maximum x value of shape
    minx: float
        minimum x value of shape
    Methods
    -------
    lineintersect(xy)
        determines the intersection of this shape and the given line segment
    pointintersect(xy)
        determines whether the given point is inside of the rectangle
    """

    def __init__(self, c):
        """
         contructs a shape object

        Parameters
        ----------
        :param cx: float
            the x location of the centroid
        :param cy: float
            the y location of the centroid
        :param xw: float
            the width in the x dir
        :param yw: float
            the width in the y dir
        :param theta: float
            the amount of rotation in radians
        :param sh: string
            string specifying the desired shape supports:
            B,S,L,U,T,BL
        """
        self.c = np.array(c)
        # print(c)
        xs = [0] * len(self.c)
        ys = [0] * len(self.c)
        for i in range(len(self.c)):
            xs[i] = self.c[i][0]
            ys[i] = self.c[i][1]
        self.maxx = max(xs)
        self.minx = min(xs)
        self.maxy = max(ys)
        self.miny = min(ys)

    def lineintersect(self, xy, rotate=0, intersection_tolerance=1e-6):
        """
        returns the intersections between a line segment and the shape

        Parameters
        -----------
        :param xy: [float,float,float,float]
            the x,y values of both endpoints of the line segment
        :return: [[float]]
            the x,y values of the intersections
        """
        x1, y1, x2, y2 = xy
        rA = []
        for i in range(len(self.c)):
            if i == len(self.c) - 1:
                c1 = self.c[len(self.c) - 1]
                c2 = self.c[0]
                r = vectorintersect(
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
                    rA.append(r)
            else:
                c1 = self.c[i]
                c2 = self.c[i + 1]
                r = vectorintersect(
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
                    rA.append(r)
        # print("x value: %f, r values:"%x1)
        # print(rA)

        rA = sortIntersections(rA, rotate)
        # print(rA)
        return rA

    def pointintersect(self, xy):
        """
        returns whether the given point is inside of the rectangle

        Parameters
        -----------
        :param xy: [float,float]
            x,y value of point
        :return: boolean
            true if inside, false if not
        """
        x, y = xy
        if (x > self.maxx or x < self.minx) or (y > self.maxy or y < self.miny):
            # print("Returning False b/c outside of box")
            return False
        farX = self.minx - 10
        inters = self.lineintersect([farX, y, farX + 1, y])
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

    def getArea(self):
        """
        returns area of shape
        :return: float
            area of shape
        """
        sum = 0
        for i in range(len(self.c)):
            if i == len(self.c) - 1:
                sum += self.c[len(self.c) - 1][0] * self.c[0][1] - (
                    self.c[len(self.c) - 1][1] * self.c[0][0]
                )
                continue
            sum += self.c[i][0] * self.c[i + 1][1] - (self.c[i][1] * self.c[i + 1][0])
        return 0.5 * sum


def sortIntersections(rA, rotate):
    if len(rA) == 0:
        return rA
    vals = [0] * len(rA)
    i = 0
    for inter in rA:
        phi = 0
        if inter[0] == 0:
            phi = pi / 2
        else:
            phi = atan(inter[1] / inter[0])
        R = sqrt(inter[1] * inter[1] + inter[0] * inter[0])
        refang = pi / 2 - phi
        # sign = 1
        if phi > pi / 2:
            if phi > pi:
                if phi > 3 * pi / 2.0:
                    refang = 2 * pi - phi
                else:
                    refang = 3.0 * pi / 2.0 - phi
            else:
                refang = pi - phi
        # if phi > pi/2 + rotate and phi < 3*pi/2 + rotate:
        # sign = -1
        vals[i] = R * sin(refang + rotate)
        i += 1
    zipped = zip(vals, rA)
    zipped = sorted(zipped)
    rA = [row for _, row in zipped]
    return rA


def vectorintersect(l1, l2, intersection_tolerance):
    """
     gives the intersection between two line segments

    Parameters
    -----------
    :param l1: [[float]]
        endpoints of first line segment
    :param l2: [[float]]
        endpoints of the second line segment
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
