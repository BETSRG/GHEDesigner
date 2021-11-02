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


def bi_rectangular(length_x, length_y, B_min, B_max_x, B_max_y, transpose=False):
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
                    coordinates = ghedt.coordinates.rectangle(i, 1, b_1, b_2)
                    if transpose:
                        coordinates = \
                            ghedt.coordinates.transpose_coordinates(coordinates)
                    bi_rectangle_domain.append(coordinates)
                for j in range(1, n_2):
                    coordinates = ghedt.coordinates.rectangle(n_1, j, b_1, b_2)
                    if transpose:
                        coordinates = \
                            ghedt.coordinates.transpose_coordinates(coordinates)
                    bi_rectangle_domain.append(coordinates)

                iter += 1

            coordinates = ghedt.coordinates.rectangle(n_1, n_2, b_1, b_2)
            if transpose:
                coordinates = \
                    ghedt.coordinates.transpose_coordinates(coordinates)
            bi_rectangle_domain.append(coordinates)

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
        transpose = False
    else:
        length_1 = length_y
        length_2 = length_x
        B_max_1 = B_max_y
        B_max_2 = B_max_x
        transpose = True

    # find the maximum number of boreholes as a float
    n_2_max = (length_2 / B_min) + 1
    n_2_min = (length_2 / B_max_2) + 1

    N_min = int(np.ceil(n_2_min).tolist())
    N_max = int(np.floor(n_2_max).tolist())

    bi_rectangle_nested_domain = []

    for n_2 in range(N_min, N_max + 1):
        b_2 = length_2 / (n_2 - 1)
        bi_rectangle_domain = ghedt.domains.bi_rectangular(
            length_1, length_2, B_min, B_max_1, b_2, transpose=transpose)
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

    n_i1 = 1
    n_i2 = 1

    z = ghedt.coordinates.zoned_rectangle(n_1, n_2, b_1, b_2, n_i1, n_i2)
    zoned_rectangle_domain.append(z)

    while n_i1 < (n_1 - 2) or n_i2 < (n_2 - 2):

        ratio = b_1 / b_2

        # general case where we can reduce in either direction
        # inner rectangular spacing
        bi_1 = (n_1 - 1) * b_1 / (n_i1 + 1)
        bi_2 = (n_2 - 1) * b_2 / (n_i2 + 1)
        # inner spacings for increasing each row
        bi_1_p1 = (n_1 - 1) * b_1 / (n_i1 + 2)
        bi_2_p1 = (n_2 - 1) * b_2 / (n_i2 + 2)

        ratio_1 = bi_1 / bi_2_p1
        ratio_2 = bi_2 / bi_1_p1

        # we only want to increase one at a time, and we want to increase
        # the one that will keep the inner rectangle furthest from the perimeter

        if ratio_1 > ratio:
            n_i1 += 1
        elif ratio_1 <= ratio:
            n_i2 += 1
        else:
            raise ValueError('This function should not have ever made it to '
                             'this point, there may be a problem with the '
                             'inputs.')
        z = ghedt.coordinates.zoned_rectangle(n_1, n_2, b_1, b_2, n_i1, n_i2)
        zoned_rectangle_domain.append(z)

    return zoned_rectangle_domain


def _bi_rectangle_zoned_nested(length_x, length_y, B_min, B_max_x, B_max_y):
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
    n_1_max = (length_1 / B_min) + 1
    n_1_min = (length_2 / B_max_1) + 1

    n_2_max = (length_2 / B_min) + 1
    n_2_min = (length_2 / B_max_2) + 1

    N_min_1 = int(np.ceil(n_1_min).tolist())
    N_max_1 = int(np.floor(n_1_max).tolist())

    N_min_2 = int(np.ceil(n_2_min).tolist())
    N_max_2 = int(np.floor(n_2_max).tolist())

    bi_rectangle_zoned_nested_domain = []

    n_1_values = list(range(N_min_1, N_max_1+1))
    n_2_values = list(range(N_min_2, N_max_2+1))

    j = 0  # pertains to n_1_values
    k = 0  # pertains to n_2_values

    for i in range(len(n_1_values) + len(n_2_values)-1):
        if i % 2 == 0:
            bi_rectangle_zoned_domain = \
                zoned_rectangle_domain(length_1, length_2, n_1_values[j],
                                       n_2_values[k])
            print('{}x{}'.format(n_1_values[j], n_2_values[k]))
            if j < len(n_1_values)-1:
                j += 1
            else:
                k += 1
        else:
            bi_rectangle_zoned_domain = \
                zoned_rectangle_domain(length_1, length_2, n_1_values[j],
                                       n_2_values[k])
            print('{}x{}'.format(n_1_values[j], n_2_values[k]))
            if k < len(n_2_values)-1:
                k += 1
            else:
                j += 1

        bi_rectangle_zoned_nested_domain.append(bi_rectangle_zoned_domain)

    return bi_rectangle_zoned_nested_domain


def bi_zoned_domain_restructured(
        length_x, length_y, B_min, B_max_x, B_max_y):
    bi_rectangle_zoned_nested_domain = \
        _bi_rectangle_zoned_nested(length_x, length_y, B_min, B_max_x, B_max_y)

    bi_rectangle_zoned_nested_domain_restructured = []

    k = 0

    for j in range(len(bi_rectangle_zoned_nested_domain[-1])):
        domain = []
        for i in range(len(bi_rectangle_zoned_nested_domain)):

            if k == 0:
                n_1_min = (length_x / B_max_x) + 1
                n_2_min = (length_y / B_max_y) + 1

                N_min_1 = int(np.ceil(n_1_min).tolist())
                N_min_2 = int(np.ceil(n_2_min).tolist())

                b_x = length_x / (N_min_1 - 1)
                b_y = length_y / (N_min_2 - 1)

                # go from one borehole to a line
                for l in range(1, N_min_1 + 1):
                    r = ghedt.coordinates.rectangle(l, 1, b_x, b_y)
                    domain.append(r)

                # go from a line to an L
                for l in range(2, N_min_2 + 1):
                    L = ghedt.coordinates.L_shape(N_min_1, l, b_x, b_y)
                    domain.append(L)

                # go from an L to a U
                for l in range(2, N_min_2 + 1):
                    lop_u = \
                        ghedt.coordinates.lop_U(N_min_1, N_min_2, b_x, b_y, l)
                    domain.append(lop_u)

                # go from a U to an open
                for l in range(1, N_min_1-1):
                    c = ghedt.coordinates.C_shape(N_min_1, N_min_2, b_x, b_y, l)
                    domain.append(c)

                k += 1

            if j >= len(bi_rectangle_zoned_nested_domain[i]):
                coordinates = bi_rectangle_zoned_nested_domain[i][-1]
            else:
                coordinates = bi_rectangle_zoned_nested_domain[i][j]
            domain.append(coordinates)

        bi_rectangle_zoned_nested_domain_restructured.append(domain)

    return bi_rectangle_zoned_nested_domain_restructured


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
