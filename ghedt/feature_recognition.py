# Jack C. Cook
# Wednesday, January 15, 2020

"""
History:
Wednesday, January 15, 2020
    Original field fit for work related to Stanford paper, and first attempt at Linear Regression
Sunday, August 30, 2020
    Becomes feature recognition, will be able to handle three field types
        - Square
        - Rectangle
        - Bi-Uniform
Wednesday, September 30, 2020
    Added functions
        - distance
        - uniform_features
        - determine_origin
        - find_line
    Can now determine features associated uniform fields containing rectangular convexes
Saturday, February 20, 2021
    Finds a home in the gFunctionDatabase
"""

from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math

import numpy as np
from scipy.spatial.qhull import _Qhull


# https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/

# Define Infinite (Using INT_MAX
# caused overflow problems)
INT_MAX = 10000
# Given three collinear points p, q, r,
# the function checks if point q lies
# on line segment 'pr'
def onSegment(p: tuple, q: tuple, r: tuple) -> bool:
    if ((q[0] <= max(p[0], r[0])) &
            (q[0] >= min(p[0], r[0])) &
            (q[1] <= max(p[1], r[1])) &
            (q[1] >= min(p[1], r[1]))):
        return True

    return False


# To find orientation of ordered triplet (p, q, r).
# The function returns following values
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p: tuple, q: tuple, r: tuple) -> int:
    val = (((q[1] - p[1]) *
            (r[0] - q[0])) -
           ((q[0] - p[0]) *
            (r[1] - q[1])))

    if val == 0:
        return 0
    if val > 0:
        return 1  # Collinear
    else:
        return 2  # Clock or counterclock


def doIntersect(p1, q1, p2, q2):
    # Find the four orientations needed for
    # general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases
    # p1, q1 and p2 are collinear and
    # p2 lies on segment p1q1
    if (o1 == 0) and (onSegment(p1, p2, q1)):
        return True

    # p1, q1 and p2 are collinear and
    # q2 lies on segment p1q1
    if (o2 == 0) and (onSegment(p1, q2, q1)):
        return True

    # p2, q2 and p1 are collinear and
    # p1 lies on segment p2q2
    if (o3 == 0) and (onSegment(p2, p1, q2)):
        return True

    # p2, q2 and q1 are collinear and
    # q1 lies on segment p2q2
    if (o4 == 0) and (onSegment(p2, q1, q2)):
        return True

    return False


# Returns true if the point p lies
# inside the polygon[] with n vertices
def is_inside_polygon(points: list, p: tuple) -> bool:
    n = len(points)

    # There must be at least 3 vertices
    # in polygon
    if n < 3:
        return False

    # Create a point for line segment
    # from p to infinite
    extreme = (INT_MAX, p[1])
    count = i = 0

    while True:
        next = (i + 1) % n

        # Check if the line segment from 'p' to
        # 'extreme' intersects with the line
        # segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(points[i],
                        points[next],
                        p, extreme)):

            # If the point 'p' is collinear with line
            # segment 'i-next', then check if it lies
            # on segment. If it lies, return true, otherwise false
            if orientation(points[i], p,
                           points[next]) == 0:
                return onSegment(points[i], p,
                                 points[next])

            count += 1

        i = next

        if (i == 0):
            break

    # Return true if count is odd, false otherwise
    return (count % 2 == 1)


def cross_product(a, b):
    return a[0] * b[1] - a[1] * b[0]


def cross(o, a, b):
    """ 2D cross product of OA and OB vectors,
     i.e. z-component of their 3D cross product.
    :param o: point O
    :param a: point A
    :param b: point B
    :return cross product of vectors OA and OB (OA x OB),
     positive if OAB makes a counter-clockwise turn,
     negative for clockwise turn, and zero
     if the points are colinear.
    """

    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def in_hull(points, queries):
    hull = _Qhull(b"i", points,
                  options=b"",
                  furthest_site=False,
                  incremental=False,
                  interior_point=None)
    equations = hull.get_simplex_facet_array()[2].T
    return np.all(queries @ equations[:-1] < - equations[-1], axis=1)


# http://alienryderflex.com/polygon/
# def pointInPolygon(point, polygon):
#     x, y = point
#
#     j = len(polygon) - 1
#
#     for i in range(len(polygon)):
#         if ((polygon[i][1] < y and polygon[j][0] >=y))
#
#
#   for (i=0; i<polyCorners; i++) {
#     if ((polyY[i]< y && polyY[j]>=y
#     ||   polyY[j]< y && polyY[i]>=y)
#     &&  (polyX[i]<=x || polyX[j]<=x)) {
#       oddNodes^=(polyX[i]+(y-polyY[i])/(polyY[j]-polyY[i])*(polyX[j]-polyX[i])<x); }
#     j=i; }
#
#   return oddNodes; }


# def point_in_hull(point, hull, tolerance=1e-12):
#     return all(
#         (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
#         for eq in hull.equations)

def point_in_polygon(point, polygon):
    # Points in polygon must be specified in a counter clockwise direction
    inside = True
    for i in range(1, len(polygon)):
        res = cross(polygon[i - 1], polygon[i], point)
        if res < 0.0:
            inside = False
    return inside


def points_in_polygon(polygon, pts):
    # https://stackoverflow.com/a/67460792/11637415
    pts = np.asarray(pts,dtype='float32')
    polygon = np.asarray(polygon,dtype='float32')
    contour2 = np.vstack((polygon[1:], polygon[:1]))
    test_diff = contour2-polygon
    mask1 = (pts[:,None] == polygon).all(-1).any(-1)
    m1 = (polygon[:,1] > pts[:,None,1]) != (contour2[:,1] > pts[:,None,1])
    slope = ((pts[:,None,0]-polygon[:,0])*test_diff[:,1])-(test_diff[:,0]*(pts[:,None,1]-polygon[:,1]))
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)
    m3 = (slope < 0) != (contour2[:,1] < polygon[:,1])
    m4 = m1 & m3
    count = np.count_nonzero(m4,axis=-1)
    mask3 = ~(count%2==0)
    mask = mask1 | mask2 | mask3
    return mask


def remove_cutout(coordinates, no_go=None):
    if no_go is None:
        no_go = []

    coordinate = coordinates[0]

    _a = cross_product(coordinate, no_go[0])
    _b = cross_product(coordinate, no_go[1])

    m = points_in_polygon(no_go, coordinates)
    b = np.argwhere(m == True)

    b = b.tolist()
    indices = []
    for i in range(len(b)):
        idx = b[i][0]
        indices.append(idx)

    new_coordinates = []
    for i in range(len(coordinates)):
        if i in indices:
            pass
        else:
            new_coordinates.append(coordinates[i])

    # for i in range(len(coordinates)):
    #     if not is_inside_polygon(points=no_go, p = coordinates[i]):
    #         new_coordiates.append(coordinates[i])

    return new_coordinates

    # coordinates = np.array(coordinates)
    # no_go = np.array(no_go)
    #
    # hull = ConvexHull(no_go)
    # hull_vertices = no_go[hull.vertices]
    # coordinate = coordinates[0]
    # for i in range(1, len(hull_vertices)):
    #     res = cross(hull_vertices[i - 1], hull_vertices[i], coordinate)
    #
    # values = []
    # for i in range(len(hull_vertices)):
    #     x,y = hull_vertices[i].tolist()
    #     values.append((x,y))
    #
    # x,y = list(zip(*values))
    #
    # fig, ax = plt.subplots()
    # ax.scatter(x, y)
    #
    # plt.show()
    #
    # a = in_hull(no_go, coordinates)
    # b = np.argwhere(a==True)
    #
    # b = b.tolist()
    # indices = []
    # for i in range(len(b)):
    #     idx = b[i][0]
    #     indices.append(idx)
    #
    # coordinates = coordinates.tolist()
    # new_coordinates = []
    # for i in range(len(coordinates)):
    #     if i in indices:
    #         pass
    #     else:
    #         new_coordinates.append(coordinates[i])
    #
    # return new_coordinates


class FeatureRecognition:
    def __init__(self, x: list, y: list):
        self.x = x
        self.y = y
        corners, interior, perimeter = self.field_convex(x, y)
        self.corners = corners
        self.interior = interior
        self.perimeter = perimeter
        lx, nx, ly, ny = self.uniform_features(corners, perimeter)
        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny

    @staticmethod
    def field_convex(x: list, y: list) -> tuple:
        """
        Given a list of x and y coordinates, determine the corner, interior and
        perimeter points of the field

        Parameters
        ----------
        x: list
            a list of the x points
        y:  list
            a list of the y points
        Returns
        -------
        corners, interior, perimeter: tuple
            The corner, interior and perimeter (non-corner) points in a tuple
        """
        # if there's not 3 points then its a single borehole or a line,
        # so there's only "corner" points
        if len(x) < 3:
            corners = list(zip(x, y))
            interior = list()
            perimeter = list()
            return corners, interior, perimeter
        # make a numpy array of x, y points
        points = np.array(list(zip(x, y)))
        # plt.plot(points[:, 0], points[:, 1], 'o')
        try:
            hull = ConvexHull(points)
        except:
            line = list(zip(x, y))
            corners = [line[0]] + [line[-1]]
            interior = list()
            perimeter = line[1:len(line)-1]
            return corners, interior, perimeter

        corners = [(points[hull.vertices[i], 0], points[hull.vertices[i], 1])
                   for i in range(len(hull.vertices))]

        xlst = []
        ylst = []
        for i in range(len(corners)):
            xlst.append(corners[i][0])
            ylst.append(corners[i][1])

        interior = []
        for i in range(len(points)):
            tmppt = (points[i][0], points[i][1])
            tmpx = points[i][0]
            tmpy = points[i][1]
            if tmpx in xlst or tmpy in ylst:
                continue
            else:
                interior.append(tmppt)
        interior = list(set(interior))

        pts = points.tolist()
        allpts = [(pts[i][0], pts[i][1]) for i in range(len(pts))]

        perimeter = list(set(allpts) - set(interior) - set(corners))

        return corners, interior, perimeter

    @staticmethod
    def distance(point_1, point_2):
        x_1, y_1 = point_1
        x_2, y_2 = point_2
        dist = math.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
        return dist

    def uniform_features(self, corners, perimeter):
        """
        Determine features which apply to uniform fields
        # TODO: make this function work for 1 borehole fields
        Parameters
        ----------
        corners: list
            the corner boreholes as returned from `:func:field_convex`
        perimeter: list
            the perimeter boreholes as returned from `:func:field_convex`
        Returns
        -------
        Lx, Nx, Ly, Ny: tuple
            A tuple of the length and number in the x-direction, and the length and number in the y-direction
        """
        tol = 1e-06

        def determine_origin(origin=(0, 0)):
            """
            The origin is typically at (0, 0), but may not always be, determine the origin
            :param origin: an optional argument for setting the desired origin
            :return: the point which is the origin
            """
            distance_min = 99999999999
            location = 0
            for j in range(len(corners)):
                dist = self.distance(origin, corners[j])
                if dist < distance_min:
                    distance_min = deepcopy(dist)
                    location = deepcopy(j)
                if distance_min < tol:
                    return location
            return location

        index = determine_origin()

        origin = corners[index]
        x_origin, y_origin = origin

        def find_line(points):
            x_points = []
            y_points = []
            for i in range(len(points)):
                x, y = points[i]
                diff_x = x_origin - x
                diff_y = y_origin - y
                if abs(diff_x) < tol:
                    y_points.append(points[i])
                elif abs(diff_y) < tol:
                    x_points.append(points[i])
            return x_points, y_points

        x_perimeter, y_perimeter = find_line(perimeter)
        corners_no_origin = [corners[k] for k in range(len(corners))
                             if corners[k] != origin]

        x_corner, y_corner = find_line(corners_no_origin)
        if len(x_corner) != 0:
            lx = self.distance(origin, x_corner[0])
            nx = len(x_perimeter) + 2
        else:
            lx = 1
            nx = 1
        if len(y_corner) != 0:
            ly = self.distance(origin, y_corner[0])
            ny = len(y_perimeter) + 2
        else:
            ly = 1
            ny = 1

        return lx, nx, ly, ny

    def plot_field_convex(self):
        """
        Plot the field convex points given what is returned from the function
        `:func:field_convex`

        Returns
        -------
        fig, ax: tuple
            the figure and the axis
        """
        corners: list = self.corners
        interior: list = self.interior
        perimeter: list = self.perimeter

        fig, ax = plt.subplots()

        ax.scatter(*zip(*corners), c='red', label='Corner', marker='*')
        if len(interior) != 0:
            ax.scatter(*zip(*interior), c='blue', label='Interior')
        if len(perimeter) != 0:
            ax.scatter(*zip(*perimeter), c='black', label='Perimeter',
                       marker='^')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        ax.set_ylabel('y (m)')
        ax.set_xlabel('x (m)')

        # Put a legend below current axis
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.1),
                  fancybox=True, shadow=True, ncol=3)

        return fig, ax
