# utilities.py - this module contains general utility functions. The functions
# are categorically grouped by their purpose. If any one of these categories
# becomes substantial, then a module can be created that is dedicated to
# functions of that category. Some of these functions may be more applicable
# to a different module. If that is the case, the function needs to contain a
# deprecation warning until the next major release.

import pickle
from enum import auto, Enum
from pathlib import Path

from math import sqrt
from scipy.optimize import brentq


# TODO: Move this class to a centralized place with other enumerations
class DesignMethod(Enum):
    Hybrid = auto()
    Hourly = auto()


# Time functions
# --------------
def eskilson_log_times():
    # Return a list of Eskilson's original 27 dimensionless points in time
    log_time = [
        -8.5,
        -7.8,
        -7.2,
        -6.5,
        -5.9,
        -5.2,
        -4.5,
        -3.963,
        -3.27,
        -2.864,
        -2.577,
        -2.171,
        -1.884,
        -1.191,
        -0.497,
        -0.274,
        -0.051,
        0.196,
        0.419,
        0.642,
        0.873,
        1.112,
        1.335,
        1.679,
        2.028,
        2.275,
        3.003,
    ]
    return log_time


# Spatial functions
# -----------------
def borehole_spacing(borehole, coordinates):
    # Use the distance between the first pair of coordinates as the B-spacing
    x_0, y_0 = coordinates[0]
    if len(coordinates) == 1:
        # Set the spacing to be the borehole radius if there's just one borehole
        return borehole.r_b
    elif len(coordinates) > 1:
        x_1, y_1 = coordinates[1]
        return max(borehole.r_b, sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2))
    else:
        raise ValueError("The coordinates_domain needs to contain a positive number of (x, y) pairs.")


# TODO: Add `set_shank` functionality to utilities.py


def length_of_side(n, b):
    return (n - 1) * b


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


def check_bracket(sign_x_l, sign_x_r) -> bool:
    if sign_x_l < 0 < sign_x_r:
        # Bracketed the root
        return True
    elif sign_x_r < 0 < sign_x_l:
        # Bracketed the root
        return True
    else:
        # The root has not been bracketed, this method will return false.
        return False


def solve_root(x, objective_function, lower=None, upper=None, abs_tol=1.0e-6, rel_tol=1.0e-6, max_iter=50):
    # Vary flow rate to match the convective resistance

    # Use Brent Quadratic to find the root
    # Define a lower and upper for thermal conductivities
    if lower is None:
        lower = x / 100.0
    else:
        lower = lower
    if upper is None:
        upper = x * 10.0
    else:
        upper = upper
    # Check objective function upper and lower bounds to make sure the root is
    # bracketed
    minus = objective_function(lower)
    plus = objective_function(upper)
    # get signs of upper and lower thermal conductivity bounds
    kg_minus_sign = int(minus / abs(minus))
    kg_plus_sign = int(plus / abs(plus))

    # Solve the root if we can, if not, take the higher value
    if kg_plus_sign != kg_minus_sign:
        x = brentq(objective_function, lower, upper, xtol=abs_tol, rtol=rel_tol, maxiter=max_iter)
    elif kg_plus_sign == -1 and kg_minus_sign == -1:
        x = lower
    elif kg_plus_sign == 1 and kg_minus_sign == 1:
        x = upper

    return x


# File input/output or file path handling functions.
# --------------------------------------------------

def create_input_file(self, file_path_obj: Path):
    # Store an object in a file using pickle.
    with open(str(file_path_obj), "wb") as file_handler:
        pickle.dump(self, file_handler)
    return


def read_input_file(path_file_obj: Path):
    # Open a .obj file and return the ghedesigner object.
    with open(path_file_obj, "rb") as file:
        object_file = pickle.load(file)
    return object_file


def dummy_entry_point():
    print("Hello, GHE world!")
