# Jack C. Cook
# Wednesdday, October 27, 2021
import copy

import ghedt
import numpy as np
import ghedt.PLAT.pygfunction as gt
import matplotlib.pyplot as plt


def square_and_near_square(lower: int,
                           upper: int,
                           B: float):
    if lower or upper <= 0:
        raise ValueError('The lower and upper arguments must be positive'
                         'integer values.')
    if upper < lower:
        raise ValueError('The lower argument should be less than or equal to'
                         'the upper.')

    coordinates_domain = []

    for i in range(lower, upper+1):
        for j in range(2):
            coordinates = \
                ghedt.coordinates.rectangle(i, i+j, B, B)

            coordinates_domain.append(coordinates)

    return coordinates_domain


def rectangular(length_x, length_y, B_min, B_max):
    # Make this work for the transpose
    if length_x >= length_y:
        length_1 = length_x
        length_2 = length_y
        transpose = False
    else:
        length_1 = length_y
        length_2 = length_x
        transpose = True

    rectangle_domain = []
    # find the maximum number of boreholes as a float
    n_1_max = (length_1 / B_min) + 1
    n_1_min = (length_1 / B_max) + 1

    N_min = int(np.ceil(n_1_min).tolist())
    N_max = int(np.floor(n_1_max).tolist())

    iter = 0
    for N in range(N_min, N_max+1):
        # Check to see if we bracket
        B = length_1 / (N - 1)
        n_2 = int(np.floor((length_2 / B) + 1))

        if iter == 0:
            for i in range(1, N_min):
                r = ghedt.coordinates.rectangle(i, 1, B, B)
                if transpose:
                    r = ghedt.coordinates.transpose_coordinates(r)
                rectangle_domain.append(r)
            for j in range(1, n_2):
                r = ghedt.coordinates.rectangle(N_min, j, B, B)
                if transpose:
                    r = ghedt.coordinates.transpose_coordinates(r)
                rectangle_domain.append(r)

            iter += 1
        r = ghedt.coordinates.rectangle(N, n_2, B, B)
        if transpose:
            r = ghedt.coordinates.transpose_coordinates(r)
        rectangle_domain.append(r)

        N += 1

    return rectangle_domain


def bi_rectangular(length_x, length_y, B_min, B_max_x, B_max_y):
    # Make this work for the transpose
    if length_x >= length_y:
        length_1 = length_x
        length_2 = length_y
        B_max_1 = B_max_x
        B_max_2 = B_max_y
    else:
        length_1 = length_y
        length_2 = length_x
        B_max_1 = B_max_y
        B_max_2 = B_max_x

    def func(B, length, n):
        _n = (length / B) + 1
        return n - _n

    bi_rectangle_domain = []
    # find the maximum number of boreholes as a float
    n_1_max = (length_1 / B_min) + 1
    n_1_min = (length_1 / B_max_1) + 1

    # if it is the first case in the domain, we want to step up from one
    # borehole, to a line, to adding the rows
    iter = 0

    N_min = int(np.ceil(n_1_min).tolist())
    N_max = int(np.floor(n_1_max).tolist())
    for n_1 in range(N_min, N_max + 1):
        # Check to see if we bracket
        a = func(n_1, length_1, B_min)
        b = func(n_1, length_1, B_max_1)
        if ghedt.utilities.sign(a) != ghedt.utilities.sign(b):

            n_2 = int(np.ceil((length_2 / B_max_2) + 1))
            b_2 = length_2 / (n_2 - 1)

            b_1 = length_1 / (n_1 - 1)

            if iter == 0:
                for i in range(1, n_1):
                    bi_rectangle_domain.append(
                        ghedt.coordinates.rectangle(i, 1, b_1, b_2))
                for j in range(1, n_2):
                    bi_rectangle_domain.append(
                        ghedt.coordinates.rectangle(n_1, j, b_1, b_2))

                iter += 1

            bi_rectangle_domain.append(
                ghedt.coordinates.rectangle(n_1, n_2, b_1, b_2))
        else:
            raise ValueError('The solution was not bracketed, and this function'
                             'is always supposed to bracket')

        n_1 += 1

    return bi_rectangle_domain


def bi_rectangle_nested(length_x, length_y, B_min, B_max_x, B_max_y):
    # Make this work for the transpose
    if length_x >= length_y:
        length_1 = length_x
        length_2 = length_y
        B_max_1 = B_max_x
        B_max_2 = B_max_y
    else:
        length_1 = length_y
        length_2 = length_x
        B_max_1 = B_max_y
        B_max_2 = B_max_x

    # find the maximum number of boreholes as a float
    n_2_max = (length_2 / B_min) + 1
    n_2_min = (length_2 / B_max_2) + 1

    N_min = int(np.ceil(n_2_min).tolist())
    N_max = int(np.floor(n_2_max).tolist())

    bi_rectangle_nested_domain = []

    for n_2 in range(N_min, N_max + 1):
        b_2 = length_2 / (n_2 - 1)
        bi_rectangle_domain = ghedt.domains.bi_rectangular(
            length_1, length_2, B_min, B_max_1, b_2)
        bi_rectangle_nested_domain.append(bi_rectangle_domain)

    return bi_rectangle_nested_domain


def visualize_domain(domain, output_folder_name):
    import os
    if not os.path.exists(output_folder_name):
        os.makedirs(output_folder_name)

    for i in range(len(domain)):
        fig = gt.utilities._initialize_figure()
        ax = fig.add_subplot(111)

        coordinates = domain[i]
        x, y = list(zip(*coordinates))

        ax.scatter(x, y)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        fig.tight_layout()

        name = output_folder_name + '/' + str(i).zfill(3)

        fig.savefig(name)
        plt.close(fig)
