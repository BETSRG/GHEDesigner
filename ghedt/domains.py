# Jack C. Cook
# Wednesday, October 27, 2021
import copy

import ghedt as dt
import numpy as np
import pygfunction as gt
import matplotlib.pyplot as plt


def square_and_near_square(lower: int,
                           upper: int,
                           B: float):
    if lower < 1 or upper < 1:
        raise ValueError('The lower and upper arguments must be positive'
                         'integer values.')
    if upper < lower:
        raise ValueError('The lower argument should be less than or equal to'
                         'the upper.')

    coordinates_domain = []

    for i in range(lower, upper+1):
        for j in range(2):
            coordinates = \
                dt.coordinates.rectangle(i, i+j, B, B)

            coordinates_domain.append(coordinates)

    return coordinates_domain


def rectangular(length_x, length_y, B_min, B_max, disp=False):
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

    n_2_old = 1

    if disp:
        print(50 * '-')
        print('Rectangular Domain\nNx\tNy\tBx\tBy')

    iter = 0
    for N in range(N_min, N_max+1):
        # Check to see if we bracket
        B = length_1 / (N - 1)
        n_2 = int(np.floor((length_2 / B) + 1))

        if iter == 0:
            for i in range(1, N_min):
                r = dt.coordinates.rectangle(i, 1, B, B)
                if transpose:
                    r = dt.coordinates.transpose_coordinates(r)
                rectangle_domain.append(r)
            for j in range(1, n_2):
                r = dt.coordinates.rectangle(N_min, j, B, B)
                if transpose:
                    r = dt.coordinates.transpose_coordinates(r)
                rectangle_domain.append(r)

            iter += 1
        if n_2_old == n_2:
            pass
        else:
            r = dt.coordinates.rectangle(N, n_2, B, B)
            if disp:
                print('{}\t{}\t{}\t{}'.format(N, n_2, B, B))
            if transpose:
                r = dt.coordinates.transpose_coordinates(r)
            rectangle_domain.append(r)
            n_2_old = copy.deepcopy(n_2)

        N += 1

    return rectangle_domain


def bi_rectangular(
        length_x, length_y, B_min, B_max_x, B_max_y, transpose=False,
        disp=False):
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

        n_2 = int(np.ceil((length_2 / B_max_2) + 1))
        b_2 = length_2 / (n_2 - 1)

        b_1 = length_1 / (n_1 - 1)

        if iter == 0:
            for i in range(1, n_1):
                coordinates = dt.coordinates.rectangle(i, 1, b_1, b_2)
                if transpose:
                    coordinates = \
                        dt.coordinates.transpose_coordinates(coordinates)
                bi_rectangle_domain.append(coordinates)
            for j in range(1, n_2):
                coordinates = dt.coordinates.rectangle(n_1, j, b_1, b_2)
                if transpose:
                    coordinates = \
                        dt.coordinates.transpose_coordinates(coordinates)
                bi_rectangle_domain.append(coordinates)

            iter += 1

        if disp:
            print('{0}x{1} with {2:.1f}x{3:.1f}'.format(n_1, n_2, b_1, b_2))

        coordinates = dt.coordinates.rectangle(n_1, n_2, b_1, b_2)
        if transpose:
            coordinates = \
                dt.coordinates.transpose_coordinates(coordinates)
        bi_rectangle_domain.append(coordinates)

        n_1 += 1

    return bi_rectangle_domain


def bi_rectangle_nested(length_x, length_y, B_min, B_max_x, B_max_y,
                        disp=False):
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
        bi_rectangle_domain = dt.domains.bi_rectangular(
            length_1, length_2, B_min, B_max_1, b_2, transpose=transpose,
            disp=disp)
        bi_rectangle_nested_domain.append(bi_rectangle_domain)

    return bi_rectangle_nested_domain


def zoned_rectangle_domain(length_x, length_y, n_x, n_y, transpose=False):
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

    z = dt.coordinates.zoned_rectangle(n_1, n_2, b_1, b_2, n_i1, n_i2)
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
        z = dt.coordinates.zoned_rectangle(n_1, n_2, b_1, b_2, n_i1, n_i2)
        if transpose:
            z = dt.coordinates.transpose_coordinates(z)
        zoned_rectangle_domain.append(z)

    return zoned_rectangle_domain


def bi_rectangle_zoned_nested(length_x, length_y, B_min, B_max_x, B_max_y):
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
    n_1_max = (length_1 / B_min) + 1
    n_1_min = (length_1 / B_max_1) + 1

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
    l = 0

    for i in range(len(n_1_values) + len(n_2_values)-1):
        domain = []
        if l == 0:
            b_x = length_x / (N_min_1 - 1)
            b_y = length_y / (N_min_2 - 1)

            # go from one borehole to a line
            for l in range(1, N_min_1 + 1):
                r = dt.coordinates.rectangle(l, 1, b_x, b_y)
                if transpose:
                    r = dt.coordinates.transpose_coordinates(r)
                domain.append(r)

            # go from a line to an L
            for l in range(2, N_min_2 + 1):
                L = dt.coordinates.L_shape(N_min_1, l, b_x, b_y)
                if transpose:
                    L = dt.coordinates.transpose_coordinates(L)
                domain.append(L)

            # go from an L to a U
            for l in range(2, N_min_2 + 1):
                lop_u = \
                    dt.coordinates.lop_U(N_min_1, N_min_2, b_x, b_y, l)
                if transpose:
                    lop_u = dt.coordinates.transpose_coordinates(lop_u)
                domain.append(lop_u)

            # go from a U to an open
            for l in range(1, N_min_1 - 1):
                c = dt.coordinates.C_shape(N_min_1, N_min_2, b_x, b_y, l)
                if transpose:
                    c = dt.coordinates.transpose_coordinates(c)
                domain.append(c)

            l += 1

        if i % 2 == 0:
            bi_rectangle_zoned_domain = \
                zoned_rectangle_domain(length_1, length_2, n_1_values[j],
                                       n_2_values[k], transpose=transpose)
            domain.extend(bi_rectangle_zoned_domain)
            if j < len(n_1_values)-1:
                j += 1
            else:
                k += 1
        else:
            bi_rectangle_zoned_domain = \
                zoned_rectangle_domain(length_1, length_2, n_1_values[j],
                                       n_2_values[k], transpose=transpose)
            domain.extend(bi_rectangle_zoned_domain)
            if k < len(n_2_values)-1:
                k += 1
            else:
                j += 1

        bi_rectangle_zoned_nested_domain.append(domain)

    return bi_rectangle_zoned_nested_domain


def polygonal_land_constraint(property_boundary, B_min, B_max_x, B_max_y,
                              building_description=None):
    if building_description is None:
        building_description = []

    outer_rectangle = \
        dt.feature_recognition.determine_largest_rectangle(property_boundary)

    x, y = list(zip(*outer_rectangle))
    length = max(x)
    width = max(y)
    coordinates_domain_nested = \
            dt.domains.bi_rectangle_nested(length, width, B_min, B_max_x,
                                              B_max_y)

    coordinates_domain_nested_cutout = []

    for i in range(len(coordinates_domain_nested)):
        new_coordinates_domain = []
        for j in range(len(coordinates_domain_nested[i])):
            coordinates = coordinates_domain_nested[i][j]
            # Remove boreholes outside of property
            new_coordinates = dt.feature_recognition.remove_cutout(
                coordinates, boundary=property_boundary, remove_inside=False)
            # Remove boreholes inside of building
            if len(new_coordinates) == 0:
                continue
            new_coordinates = dt.feature_recognition.remove_cutout(
                new_coordinates, boundary=building_description,
                remove_inside=True, keep_contour=False)
            new_coordinates_domain.append(new_coordinates)
        coordinates_domain_nested_cutout.append(new_coordinates_domain)

    coordinates_domain_nested_cutout_reordered = []
    for i in range(len(coordinates_domain_nested_cutout)):
        domain = coordinates_domain_nested_cutout[i]
        domain_reordered = reorder_domain(domain)
        coordinates_domain_nested_cutout_reordered.append(domain_reordered)

    return coordinates_domain_nested_cutout_reordered


# The following functions are utility functions specific to domains.py
# ------------------------------------------------------------------------------
def verify_excess(domain):
    # Verify that the domain is unimodal
    unimodal = True
    delta_T_values = []
    for i in range(1, len(domain)):
        delta_T = domain[i] - domain[i-1]
        delta_T_values.append(delta_T)
        if delta_T > 0:
            unimodal = False

    return delta_T_values, unimodal


def reorder_domain(domain):
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
