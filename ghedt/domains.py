import copy

import numpy as np

from ghedt.coordinates import (
    rectangle,
    transpose_coordinates,
    zoned_rectangle,
    l_shape,
    c_shape,
    lop_u,
)
from ghedt.feature_recognition import remove_cutout, determine_largest_rectangle


def square_and_near_square(lower: int, upper: int, B: float):
    if lower < 1 or upper < 1:
        raise ValueError(
            "The lower and upper arguments must be positive" "integer values."
        )
    if upper < lower:
        raise ValueError(
            "The lower argument should be less than or equal to" "the upper."
        )

    fieldDescriptors = []
    coordinates_domain = []
    coordinates_domain.append([[0, 0]])
    coordinates_domain.append([[0, 0], [0, B]])
    coordinates_domain.append([[0, 0], [0, B], [0, 2 * B]])
    fieldDescriptors.append("1X1")
    fieldDescriptors.append("1X2")
    fieldDescriptors.append("1X3")

    for i in range(lower, upper + 1):
        for j in range(2):
            coordinates = rectangle(i, i + j, B, B)

            coordinates_domain.append(coordinates)
            fieldDescriptors.append("{}X{}".format(i, i + j))

    return coordinates_domain, fieldDescriptors


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
    fieldDescriptorFormatString = "{}X{}_B{:.2f}"
    fieldDescriptors = []
    # find the maximum number of boreholes as a float
    n_1_max = (length_1 / B_min) + 1
    n_1_min = (length_1 / B_max) + 1

    N_min = int(np.ceil(n_1_min).tolist())
    N_max = int(np.floor(n_1_max).tolist())

    n_2_old = 1

    if disp:
        print(50 * "-")
        print("Rectangular Domain\nNx\tNy\tBx\tBy")

    iter = 0

    for N in range(N_min, N_max + 1):
        # Check to see if we bracket
        B = length_1 / (N - 1)
        n_2 = int(np.floor((length_2 / B) + 1))

        if iter == 0:
            for i in range(1, N_min):
                r = rectangle(i, 1, B, B)
                if transpose:
                    r = transpose_coordinates(r)
                rectangle_domain.append(r)
                fieldDescriptors.append(fieldDescriptorFormatString.format(i, 1, B))
            for j in range(1, n_2):
                r = rectangle(N_min, j, B, B)
                if transpose:
                    r = transpose_coordinates(r)
                rectangle_domain.append(r)
                fieldDescriptors.append(fieldDescriptorFormatString.format(N_min, j, B))

            iter += 1
        if n_2_old == n_2:
            pass
        else:
            r = rectangle(N, n_2, B, B)
            if disp:
                print("{}\t{}\t{}\t{}".format(N, n_2, B, B))
            if transpose:
                r = transpose_coordinates(r)
            rectangle_domain.append(r)
            fieldDescriptors.append(fieldDescriptorFormatString.format(N, n_2, B))
            n_2_old = copy.deepcopy(n_2)

        N += 1

    return rectangle_domain, fieldDescriptors


def bi_rectangular(
    length_x, length_y, B_min, B_max_x, B_max_y, transpose=False, disp=False
):
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
    fieldDescriptorFormatString = "{}X{}_B1{:.2f}_B2{:.2f}"
    fieldDescriptors = []
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
                coordinates = rectangle(i, 1, b_1, b_2)
                if transpose:
                    coordinates = transpose_coordinates(coordinates)
                bi_rectangle_domain.append(coordinates)
                fieldDescriptors.append(
                    fieldDescriptorFormatString.format(i, 1, b_1, b_2)
                )
            for j in range(1, n_2):
                coordinates = rectangle(n_1, j, b_1, b_2)
                if transpose:
                    coordinates = transpose_coordinates(coordinates)
                bi_rectangle_domain.append(coordinates)
                fieldDescriptors.append(
                    fieldDescriptorFormatString.format(n_1, j, b_1, b_2)
                )

            iter += 1

        if disp:
            print("{0}x{1} with {2:.1f}x{3:.1f}".format(n_1, n_2, b_1, b_2))

        coordinates = rectangle(n_1, n_2, b_1, b_2)
        if transpose:
            coordinates = transpose_coordinates(coordinates)
        bi_rectangle_domain.append(coordinates)
        fieldDescriptors.append(fieldDescriptorFormatString.format(n_1, n_2, b_1, b_2))

        n_1 += 1

    return bi_rectangle_domain, fieldDescriptors


def bi_rectangle_nested(length_x, length_y, B_min, B_max_x, B_max_y, disp=False):
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
    fieldDescriptors = []

    for n_2 in range(N_min, N_max + 1):
        b_2 = length_2 / (n_2 - 1)
        bi_rectangle_domain, fD = bi_rectangular(
            length_1, length_2, B_min, B_max_1, b_2, transpose=transpose, disp=disp
        )
        # print("Bi-Rectangular: ",bi_rectangle_domain)
        bi_rectangle_nested_domain.append(bi_rectangle_domain)
        # fieldDescriptors.append(str(length_1) + "X" + str(length_2) + "_" + str(B_min) + "_" + str(B_max_1)+"_"+str(b_2))
        fieldDescriptors.append(fD)

    return bi_rectangle_nested_domain, fieldDescriptors


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
    fieldDescriptorFormatString = "{}X{}_{}X{}_B1{:.2f}_B2{:.2f}"
    fieldDescriptors = []

    n_i1 = 1
    n_i2 = 1

    z = zoned_rectangle(n_1, n_2, b_1, b_2, n_i1, n_i2)
    zoned_rectangle_domain.append(z)
    fieldDescriptors.append(
        fieldDescriptorFormatString.format(n_1, n_2, n_i1, n_i2, b_1, b_2)
    )

    while n_i1 < (n_1 - 2) or n_i2 < (n_2 - 2):

        ratio = b_1 / b_2

        # general case where we can reduce in either direction
        # inner rectangular spacing
        bi_1 = (n_1 - 1) * b_1 / (n_i1 + 1)
        # bi_2 = (n_2 - 1) * b_2 / (n_i2 + 1)
        # inner spacings for increasing each row
        # bi_1_p1 = (n_1 - 1) * b_1 / (n_i1 + 2)
        bi_2_p1 = (n_2 - 1) * b_2 / (n_i2 + 2)

        ratio_1 = bi_1 / bi_2_p1
        # ratio_2 = bi_2 / bi_1_p1

        # we only want to increase one at a time, and we want to increase
        # the one that will keep the inner rectangle furthest from the perimeter

        if ratio_1 > ratio:
            n_i1 += 1
        elif ratio_1 <= ratio:
            n_i2 += 1
        else:
            raise ValueError(
                "This function should not have ever made it to "
                "this point, there may be a problem with the "
                "inputs."
            )
        z = zoned_rectangle(n_1, n_2, b_1, b_2, n_i1, n_i2)
        if transpose:
            z = transpose_coordinates(z)
        zoned_rectangle_domain.append(z)
        fieldDescriptors.append(
            fieldDescriptorFormatString.format(n_1, n_2, n_i1, n_i2, b_1, b_2)
        )

    return zoned_rectangle_domain, fieldDescriptors


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
    fieldDescriptorFormatString = "{}X{}_{:.2f}X{:.2f}"
    fieldDescriptors = []

    n_1_values = list(range(N_min_1, N_max_1 + 1))
    n_2_values = list(range(N_min_2, N_max_2 + 1))

    j = 0  # pertains to n_1_values
    k = 0  # pertains to n_2_values
    l = 0

    for i in range(len(n_1_values) + len(n_2_values) - 1):
        domain = []
        fD = []
        if l == 0:
            b_x = length_x / (N_min_1 - 1)
            b_y = length_y / (N_min_2 - 1)

            # go from one borehole to a line
            for l in range(1, N_min_1 + 1):
                r = rectangle(l, 1, b_x, b_y)
                if transpose:
                    r = transpose_coordinates(r)
                domain.append(r)
                fD.append(fieldDescriptorFormatString.format(l, 1, b_x, b_y))

            # go from a line to an L
            for l in range(2, N_min_2 + 1):
                L = l_shape(N_min_1, l, b_x, b_y)
                if transpose:
                    L = transpose_coordinates(L)
                domain.append(L)
                fD.append(fieldDescriptorFormatString.format(N_min_1, l, b_x, b_y))

            # go from an L to a U
            for l in range(2, N_min_2 + 1):
                lop_u_field = lop_u(N_min_1, N_min_2, b_x, b_y, l)
                if transpose:
                    lop_u_field = transpose_coordinates(lop_u_field)
                domain.append(lop_u_field)
                fD.append(
                    fieldDescriptorFormatString.format(N_min_1, N_min_2, b_x, b_y)
                )

            # go from a U to an open
            for l in range(1, N_min_1 - 1):
                c = c_shape(N_min_1, N_min_2, b_x, b_y, l)
                if transpose:
                    c = transpose_coordinates(c)
                domain.append(c)
                fD.append(
                    fieldDescriptorFormatString.format(N_min_1, N_min_2, b_x, b_y)
                )

            l += 1

        if i % 2 == 0:
            bi_rectangle_zoned_domain, fDs = zoned_rectangle_domain(
                length_1, length_2, n_1_values[j], n_2_values[k], transpose=transpose
            )
            domain.extend(bi_rectangle_zoned_domain)
            fD.extend(fDs)
            if j < len(n_1_values) - 1:
                j += 1
            else:
                k += 1
        else:
            bi_rectangle_zoned_domain, fDs = zoned_rectangle_domain(
                length_1, length_2, n_1_values[j], n_2_values[k], transpose=transpose
            )
            domain.extend(bi_rectangle_zoned_domain)
            fD.extend(fDs)
            if k < len(n_2_values) - 1:
                k += 1
            else:
                j += 1

    bi_rectangle_zoned_nested_domain.append(domain)
    fieldDescriptors.append(fD)

    return bi_rectangle_zoned_nested_domain, fieldDescriptors


def polygonal_land_constraint(
    property_boundary, B_min, B_max_x, B_max_y, building_descriptions=None
):
    if building_descriptions is None:
        building_descriptions = []

    outer_rectangle = determine_largest_rectangle(property_boundary)

    x, y = list(zip(*outer_rectangle))
    length = max(x)
    width = max(y)
    coordinates_domain_nested, fieldDescriptors = bi_rectangle_nested(
        length, width, B_min, B_max_x, B_max_y
    )

    coordinates_domain_nested_cutout = []

    for i in range(len(coordinates_domain_nested)):
        new_coordinates_domain = []
        for j in range(len(coordinates_domain_nested[i])):
            coordinates = coordinates_domain_nested[i][j]
            # Remove boreholes outside of property
            new_coordinates = remove_cutout(
                coordinates, boundary=property_boundary, remove_inside=False
            )
            # Remove boreholes inside of building
            if len(new_coordinates) == 0:
                continue
            for building_description in building_descriptions:
                new_coordinates = remove_cutout(
                    new_coordinates,
                    boundary=building_description,
                    remove_inside=True,
                    keep_contour=False,
                )
            new_coordinates_domain.append(new_coordinates)
        coordinates_domain_nested_cutout.append(new_coordinates_domain)

    coordinates_domain_nested_cutout_reordered = []
    fieldDescriptors_reordered = []
    for i in range(len(coordinates_domain_nested_cutout)):
        domain = coordinates_domain_nested_cutout[i]
        domain_reordered, fD_reordered = reorder_domain(domain, fieldDescriptors[i])
        coordinates_domain_nested_cutout_reordered.append(domain_reordered)
        fieldDescriptors_reordered.append(fD_reordered)

    return coordinates_domain_nested_cutout_reordered, fieldDescriptors_reordered


def reorder_domain(domain, descriptors):
    # Reorder the domain so that the number of boreholes successively grow
    numbers = {}
    for i in range(len(domain)):
        numbers[i] = len(domain[i])

    sorted_values = sorted(numbers.values())

    reordered_domain = []
    reordered_descriptors = []

    for i in sorted_values:
        for j in numbers.keys():
            if numbers[j] == i:
                reordered_domain.append(domain[j])
                reordered_descriptors.append(descriptors[j])
                break

    return reordered_domain, reordered_descriptors
