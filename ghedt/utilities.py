# utilities.py - this module contains general utility functions. The functions
# are categorically grouped by their purpose. If any one of these categories
# becomes substantial, then a module can be created that is dedicated to
# functions of that category. Some of these functions may be more applicable
# to a different module. If that is the case, the function needs to contain a
# deprecation warning until the next major release.

import copy
import json
import pickle
import warnings

import numpy as np


# Time functions
# --------------
def Eskilson_log_times():
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
        B = copy.deepcopy(borehole.r_b)
    elif len(coordinates) > 1:
        x_1, y_1 = coordinates[1]
        B = max(borehole.r_b, np.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2))
    else:
        raise ValueError(
            "The coordinates_domain needs to contain a positive"
            "number of (x, y) pairs."
        )
    return B


# TODO: Add `set_shank` functionality to utilities.py


def number_of_boreholes(length, B, func=np.ceil):
    N = func(length / B) + 1
    return int(N)


def length_of_side(N, B):
    L = (N - 1) * B
    return L


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
        warnings.warn(
            "The disp option in check_bracket will be removed in "
            "the ghedt 0.2 release."
        )
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
    with open(file_name + ".json", "w") as fp:
        json.dump(d, fp, indent=indent)


def js_load(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def create_input_file(self, file_name="ghedt_input"):
    # Store an object in a file using pickle.
    file_handler = open(file_name + ".obj", "wb")
    pickle.dump(self, file_handler)
    file_handler.close()

    return


def read_input_file(path_to_file):
    # Open a .obj file and return the ghedt object.
    file = open(path_to_file, "rb")
    object_file = pickle.load(file)
    file.close()

    return object_file

