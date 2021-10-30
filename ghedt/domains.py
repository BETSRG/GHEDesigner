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
    else:
        length_1 = length_y
        length_2 = length_x

    def func(B, length, n):
        _n = (length / B) + 1
        return n - _n

    rectangle_domain = []
    # find the maximum number of boreholes as a float
    n_1_max = (length_1 / B_min) + 1
    n_1_min = (length_1 / B_max) + 1

    N_min = int(np.ceil(n_1_min).tolist())
    N_max = int(np.floor(n_1_max).tolist())
    for N in range(N_min, N_max+1):
        # Check to see if we bracket
        a = func(N, length_1, B_min)
        b = func(N, length_1, B_max)
        if ghedt.utilities.sign(a) != ghedt.utilities.sign(b):
            B = length_1 / (N - 1)

            n_2 = int(np.floor((length_2 / B) + 1))
            rectangle_domain.append(ghedt.coordinates.rectangle(N, n_2, B, B))
        else:
            raise ValueError('The solution was not bracketed, and this function'
                             'is always supposed to bracket')

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


def zoned_rectangle_domain(length_x, length_y, n_x, n_y):
    # Make this work for the transpose
    if length_x >= length_y:
        length_1 = length_x
        length_2 = length_y
        n_1 = n_x
        n_2 = n_y
    else:
        length_1 = length_y
        length_2 = length_x
        n_1 = n_y
        n_2 = n_x

    b_1 = length_1 / (n_1 - 1)
    b_2 = length_2 / (n_2 - 1)

    zoned_rectangle_domain = []

    # for i in range(1, n_1-1):
    #     z = ghedt.coordinates.zoned_rectangle(n_1, n_2, b_1, b_2, i, 1)
    #     zoned_rectangle_domain.append(z)
    # for i in range(1, n_2-1):
    #     z = ghedt.coordinates.zoned_rectangle(n_1, n_2, b_1, b_2, n_1 - 2, i)
    #     zoned_rectangle_domain.append(z)

    n_i1 = 2
    n_i2 = 2

    for _ in range(n_1-2):

        # general case where we can reduce in either direction
        # current x spacing
        x_spacing_c = (n_1 - 1) * b_1 / (n_i1 + 1)
        # current y spacing
        y_spacing_c = (n_2 - 1) * b_2 / (n_i2 + 1)
        # x spacing if we reduce one column
        x_spacing_t = (n_1 - 1) * b_1 / n_i1
        # y spacing if we reduce one row
        y_spacing_t = (n_2 - 1) * b_2 / n_i2

        # possible outcomes
        # ratio (fraction) if we reduce one column
        f_x = x_spacing_t / y_spacing_c
        if f_x < 1:
            f_x = 1 / f_x
        # ratio (fraction) if we reduce one row
        f_y = x_spacing_c / y_spacing_t
        if f_y < 1:
            f_y = 1 / f_y
        d_f_x = f_x - 1  # distance of ratio from 1 if we reduce one column
        d_f_y = f_y - 1  # distance of ratio from 1 if wer reduce one row
        if d_f_x > d_f_y:
            # Niy = Niym1
            n_i2 += 1
        else:
            # Nix = Nixm1
            n_i1 += 1
        z = ghedt.coordinates.zoned_rectangle(n_1, n_2, b_1, b_2, n_i1, n_i2)
        zoned_rectangle_domain.append(z)

    return zoned_rectangle_domain


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
