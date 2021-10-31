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


def in_hull(points, queries):
    hull = _Qhull(b"i", points,
                  options=b"",
                  furthest_site=False,
                  incremental=False,
                  interior_point=None)
    equations = hull.get_simplex_facet_array()[2].T
    return np.all(queries @ equations[:-1] < - equations[-1], axis=1)


def remove_cutout(coordinates, no_go=None):
    if no_go is None:
        no_go = []

    coordinates = np.array(coordinates)
    no_go = np.array(no_go)
    a = in_hull(no_go, coordinates)
    b = np.argwhere(a==True)

    b = b.tolist()
    indices = []
    for i in range(len(b)):
        idx = b[i][0]
        indices.append(idx)

    coordinates = coordinates.tolist()
    new_coordinates = []
    for i in range(len(coordinates)):
        if i in indices:
            pass
        else:
            new_coordinates.append(coordinates[i])

    return new_coordinates


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
