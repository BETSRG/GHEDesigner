# Jack C. Cook
# Tuesday, October 26, 2021

# utilities.py - this module contains general utility functions. The functions
# are categorically grouped by their purpose. If any one of these categories
# becomes substantial, then a module can be created that is dedicated to
# functions of that category. Some of these functions may be more applicable
# to a different module. If that is the case, the function needs to contain a
# deprecation warning until the next major release.

import copy
import numpy as np
import json
from matplotlib.ticker import Locator
import pickle
import warnings


# Time functions
# --------------
def Eskilson_log_times():
    # Return a list of Eskilson's original 27 dimensionless points in time
    log_time = [-8.5, -7.8, -7.2, -6.5, -5.9, -5.2, -4.5, -3.963, -3.27,
                -2.864, -2.577, -2.171, -1.884,
                -1.191, -0.497, -0.274, -0.051, 0.196, 0.419,
                0.642, 0.873, 1.112, 1.335, 1.679, 2.028, 2.275, 3.003]
    return log_time


# Spatial functions
# -----------------
def borehole_spacing(borehole, coordinates):
    # Use the distance between the first pair of coordinates as the B-spacing
    x_0, y_0 = coordinates[0]
    if len(coordinates) == 1:
        # Set the spacing to be the borehole radius if there's just one borehole
        B = copy.deepcopy(borehole.r_b)
    elif len(coordinates) > 1:
        x_1, y_1 = coordinates[1]
        B = max(borehole.r_b,
                np.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2))
    else:
        raise ValueError('The coordinates_domain needs to contain a positive'
                         'number of (x, y) pairs.')
    return B


def polygonal_area(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

# TODO: Add `set_shank` functionality to utilities.py
# def set_shank(configuration: str, rb: float, r_in: float, r_out: float):
#     raise ValueError('This function is incomplete.')
#     if configuration == 'A':
#         a = 1
#     elif configuration == 'B':
#         a = 1
#     elif configuration == 'C':
#         a = 1
#     else:
#         raise ValueError('Only configurations A, B, or C are valid.')


def make_rectangle_perimeter(length_x, length_y, origin=(0, 0)):
    # Create an outer rectangular perimeter given an origin and side lengths.
    origin_x = origin[0]
    origin_y = origin[1]
    rectangle_perimeter = \
        [[origin_x, origin_y], [origin_x + length_x, origin_y],
         [origin_x + length_x, origin_y + length_y],
         [origin_x, origin_y + length_y]]
    return rectangle_perimeter


def number_of_boreholes(length, B, func=np.ceil):
    N = func(length / B) + 1
    return int(N)


def length_of_side(N, B):
    L = (N - 1) * B
    return L


def spacing_along_length(L, N):
    B = L / (N - 1)
    return B


# Design oriented functions
# -------------------------
def sign(x: float) -> int:
    """
    Determine the sign of a value, pronounced "sig-na"
    :param x: the input value
    :type x: float
    :return: a 1 or a -1
    """
    return int(abs(x) / x)


def check_bracket(sign_xL, sign_xR, disp=None) -> bool:
    if disp is not None:
        warnings.warn('The disp option in check_bracket will be removed in '
                      'the ghedt 0.2 release.')
    if sign_xL < 0 < sign_xR:
        # Bracketed the root
        return True
    elif sign_xR < 0 < sign_xL:
        # Bracketed the root
        return True
    else:
        # The root has not been bracketed, this method will return false.
        return False


# File input/output or file path handling functions.
# --------------------------------------------------
def js_dump(file_name, d, indent=4):
    with open(file_name + '.json', 'w') as fp:
        json.dump(d, fp, indent=indent)


def js_load(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def create_if_not(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_input_file(self, file_name='ghedt_input'):
    # Store an object in a file using pickle.
    file_handler = open(file_name + '.obj', 'wb')
    pickle.dump(self, file_handler)
    file_handler.close()

    return


def read_input_file(path_to_file):
    # Open a .obj file and return the ghedt object.
    file = open(path_to_file, 'rb')
    object_file = pickle.load(file)
    file.close()

    return object_file


# Functions related to computing statistics.
# ------------------------------------------
def compute_mpe(actual: list, predicted: list) -> float:
    """
    The following mean percentage error formula is used:
    .. math::
        MPE = \dfrac{100\%}{n}\sum_{i=0}^{n-1}\dfrac{a_t-p_t}{a_t}
    Parameters
    ----------
    actual: list
        The actual computed g-function values
    predicted: list
        The predicted g-function values
    Returns
    -------
    **mean_percent_error: float**
        The mean percentage error in percent
    """
    # the lengths of the two lists should be the same
    assert len(actual) == len(predicted)
    # create a summation variable
    summation: float = 0.
    for i in range(len(actual)):
        summation += (predicted[i] - actual[i]) / actual[i]
    mean_percent_error = summation * 100 / len(actual)
    return mean_percent_error


# Functions related to plotting.
# ------------------------------
class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh, nints=10):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically. nints gives the number of
        intervals that will be bounded by the minor ticks.
        """
        self.linthresh = linthresh
        self.nintervals = nints

    def __call__(self):
        # Return the locations of the ticks
        majorlocs = self.axis.get_majorticklocs()

        if len(majorlocs) == 1:
            return self.raise_if_exceeds(np.array([]))

        # add temporary major tick locs at either end of the current range
        # to fill in minor tick gaps
        # major tick difference at lower end
        dmlower = majorlocs[1] - majorlocs[0]
        # major tick difference at upper end
        dmupper = majorlocs[-1] - majorlocs[-2]

        # add temporary major tick location at the lower end
        if majorlocs[0] != 0. and \
                ((majorlocs[0] != self.linthresh and
                  dmlower > self.linthresh) or
                 (dmlower == self.linthresh and majorlocs[0] < 0)):
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]*10.)
        else:
            majorlocs = np.insert(majorlocs, 0, majorlocs[0]-self.linthresh)

        # add temporary major tick location at the upper end
        if majorlocs[-1] != 0. and \
                ((np.abs(majorlocs[-1]) != self.linthresh and
                  dmupper > self.linthresh) or
                 (dmupper == self.linthresh and majorlocs[-1] > 0)):
            majorlocs = np.append(majorlocs, majorlocs[-1]*10.)
        else:
            majorlocs = np.append(majorlocs, majorlocs[-1]+self.linthresh)

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = self.nintervals
            else:
                ndivs = self.nintervals - 1.

            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))

