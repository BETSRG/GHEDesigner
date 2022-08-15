'''
creates and handles rectangular nogo zones

Classes:
    rec
Functions:

    angleBetween(float,float,float) - > float
    vectorintersect()->[[float]]

'''
from math import acos
from math import cos
from math import pi
from math import sin

import matplotlib.pyplot as plt
import numpy as np


class rec:
    '''
    a class to represent rectangular nogo zones

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
        the rotation of the rectangle
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
    '''

    def __init__(self, cx, cy, xw, yw, theta=0):
        '''
        contructs a rectangle object

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
        '''
        self.cx = cx
        self.cy = cy
        self.xw = xw
        self.yw = yw
        self.theta = theta
        hxwc = .5 * xw * cos(theta)
        hywc = .5 * yw * cos(theta)
        hxws = .5 * xw * sin(theta)
        hyws = .5 * yw * sin(theta)
        self.c00 = [cx - hxwc + hyws, cy - hxws - hywc]
        self.c10 = [cx + hxwc + hyws, cy + hxws - hywc]
        self.c01 = [cx - hxwc - hyws, cy - hxws + hywc]
        self.c11 = [cx + hxwc - hyws, cy + hxws + hywc]
        self.c = [self.c00, self.c10, self.c11, self.c01]
        self.maxy = max(self.c00[1], self.c10[1], self.c01[1], self.c11[1])
        self.miny = min(self.c00[1], self.c10[1], self.c01[1], self.c11[1])
        self.maxx = max(self.c00[0], self.c10[0], self.c01[0], self.c11[0])
        self.minx = min(self.c00[0], self.c10[0], self.c01[0], self.c11[0])

    def lineintersect(self, xy):
        '''
        returns the intersections between a line segment and the rectanlge

        Parameters
        -----------
        :param xy: [float,float,float,float]
            the x,y values of both endpoints of the line segment
        :return: [[float]]
            the x,y values of the intersections
        '''
        x1, y1, x2, y2 = xy
        if (y1 > self.maxy and y2 > self.maxy) or (y1 < self.miny and y2 < self.miny):
            return []
        rA = []
        for i in range(4):
            if i == 3:
                c1 = self.c[3]
                c2 = self.c[0]
                r = vectorintersect([c1[0], c1[1], c2[0], c2[1]], [x1, y1, x2, y2])
                if r != None:
                    rA.append(rA, r, )
            else:
                c1 = self.c[i]
                c2 = self.c[i + 1]
                r = vectorintersect([c1[0], c1[1], c2[0], c2[1]], [x1, y1, x2, y2])
                if r != None:
                    rA.append(rA, r, )
        return rA

    def pointintersect(self, xy):
        '''
        returns whether the given point is inside of the rectangle

        Parameters
        -----------
        :param xy: [float,float]
            x,y value of point
        :return: boolean
            true if inside, false if not
        '''
        x, y = xy
        if (x > self.maxx or x < self.minx) or (y > self.maxy or y < self.miny):
            return False
        farX = self.minx
        inters = self.lineintersect([farX, y, farX + 1, y])
        if len(inters) == 1:
            return True
        for inter in inters:
            for vert in self.c:
                if inter[0] == vert[0] and inter[1] == vert[1]:
                    inters.remove(inter)
        if len(inters) % 2 == 0:
            return False
        else:
            True


def angleBetween(a, b, c):
    '''
     gives the angle opposite of side a

    Parameters
    -----------
    :param a: float
        length of side a
    :param b: float
        length of side b
    :param c: float
        length of side c
    :return: double
        angle across from a in triangle
    '''
    if a == 0 or b == 0 or c == 0:
        return 0
    r = (a * a + b * b - c * c) / (2 * a * b)
    if r > 1 or r < -1:
        return 0
    else:
        return acos(r)


def vectorintersect(l1, l2):
    '''
     gives the intersection between two line segments

    Parameters
    -----------
    :param l1: [[float]]
        endpoints of first line segment
    :param l2: [[float]]
        endpoints of the second line segment
    :return: [float,float]
        x,y values of intersection (returns None if there is none)
    '''
    x11, y11, x12, y12 = l1
    x21, y21, x22, y22 = l2

    p = np.array([x11, y11])
    q = np.array([x21, y21])
    r = np.array([x12 - x11, y12 - y11])
    s = np.array([x22 - x21, y22 - y21])
    rcs = np.cross(r, s)
    if rcs != 0:
        qpr = np.cross(np.subtract(q, p), r)
        qps = np.cross(np.subtract(q, p), s)
        u = qpr / rcs
        t = qps / rcs
        if 0 <= t <= 1 and 0 <= u <= 1:
            return [p[0] + t * r[0], p[1] + t * r[1]]
    return None


def main():
    '''
   tests rectangle class

    Parameters
    -----------
    :return: none
    '''
    rect = rec(3.0, 4.0, 7.0, 9.0, pi / 5.0)
    print("Should be True: ", rect.pointintersect([4.0, 5.0]))
    print("Should be False: ", rect.pointintersect([-3.0, 9.0]))
    print("Should be False: ", rect.pointintersect([4.0, -1.0]))
    print("Should be True: ", rect.pointintersect([-2.0, 5.5]))
    xy = [-3, -2, 9, 7]
    points = rect.lineintersect(xy)
    plt.plot([rect.c00[0], rect.c01[0]], [rect.c00[1], rect.c01[1]], "r")
    plt.plot([rect.c01[0], rect.c11[0]], [rect.c01[1], rect.c11[1]], "r")
    plt.plot([rect.c11[0], rect.c10[0]], [rect.c11[1], rect.c10[1]], "r")
    plt.plot([rect.c10[0], rect.c00[0]], [rect.c10[1], rect.c00[1]], "r")
    plt.plot([xy[0], xy[2]], [xy[1], xy[3]], "b")
    X = np.zeros(len(points))
    Y = np.zeros(len(points))
    for i in range(len(points)):
        X[i] = points[i][0]
        Y[i] = points[i][1]
    plt.plot(X, Y, "go")
    plt.show()


if __name__ == "__main__":
    main()
