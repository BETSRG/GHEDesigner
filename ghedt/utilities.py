# Jack C. Cook
# Tuesday, October 26, 2021

import copy
import numpy as np


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
        B = copy.deepcopy(borehole.rb)
    elif len(coordinates) > 1:
        x_1, y_1 = coordinates[1]
        B = max(borehole.rb,
                np.sqrt((x_1 - x_0) ** 2 + (y_1 - y_0) ** 2))
    else:
        raise ValueError('The coordinates_domain needs to contain a positive'
                         'number of (x, y) pairs.')
    return B
