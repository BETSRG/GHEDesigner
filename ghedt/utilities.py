# Jack C. Cook
# Tuesday, October 26, 2021

import copy
import numpy as np
import json
from matplotlib.ticker import Locator


def Eskilson_log_times():
    log_time = [-8.5, -7.8, -7.2, -6.5, -5.9, -5.2, -4.5, -3.963, -3.27,
                -2.864, -2.577, -2.171, -1.884,
                -1.191, -0.497, -0.274, -0.051, 0.196, 0.419,
                0.642, 0.873, 1.112, 1.335, 1.679, 2.028, 2.275, 3.003]
    return log_time


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


def sign(x: float) -> int:
    """
    Determine the sign of a value, pronounced "sig-na"
    :param x: the input value
    :type x: float
    :return: a 1 or a -1
    """
    return int(abs(x) / x)


def check_bracket(sign_xL, sign_xR, disp=False) -> bool:
    if sign_xL < 0 < sign_xR:
        if disp:
            print('Bracketed the root')
        return True
    elif sign_xR < 0 < sign_xL:
        if disp:
            print('Bracketed the root')
        return True
    else:
        if disp:
            print('The root has not been bracketed, '
                  'this method will return false.')
        return False


def js_dump(file_name, d, indent=4):
    with open(file_name + '.json', 'w') as fp:
        json.dump(d, fp, indent=indent)


def js_load(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def verify_excess(domain):
    unimodal = True
    delta_T_values = []
    for i in range(1, len(domain)):
        delta_T = domain[i] - domain[i-1]
        delta_T_values.append(delta_T)
        if delta_T > 0:
            unimodal = False

    return delta_T_values, unimodal


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


def polygonal_area(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


def reorder_domain(domain):
    # TODO: move this to the domains.py module
    # Reorder the domain so that the number of boreholes successively grow
    numbers = {}
    for i in range(len(domain)):
        numbers[i] = len(domain[i])

    sorted_values = sorted(numbers.values())

    reordered_domain = []

    for i in sorted_values:
        for j in numbers.keys():
            if numbers[j] == i:
                reordered_domain.append(domain[j])
                break

    return reordered_domain


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
