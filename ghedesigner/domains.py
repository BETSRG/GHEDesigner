from math import ceil, floor

from ghedesigner.coordinates import rectangle, transpose_coordinates, zoned_rectangle, l_shape, c_shape, lop_u
from ghedesigner.feature_recognition import remove_cutout, determine_largest_rectangle


def square_and_near_square(lower: int, upper: int, b: float):
    if lower < 1 or upper < 1:
        raise ValueError("The lower and upper arguments must be positive" "integer values.")
    if upper < lower:
        raise ValueError("The lower argument should be less than or equal to" "the upper.")

    field_descriptors = ["1X1", "1X2", "1X3"]
    coordinates_domain = [
        [[0, 0]],
        [[0, 0], [0, b]],
        [[0, 0], [0, b], [0, 2 * b]]
    ]

    for i in range(lower, upper + 1):
        for j in range(2):
            coordinates = rectangle(i, i + j, b, b)

            coordinates_domain.append(coordinates)
            field_descriptors.append(f"{i}X{i + j}")

    return coordinates_domain, field_descriptors


def rectangular(length_x: float, length_y: float, b_min: float, b_max: float, disp: bool = False):
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
    field_descriptors = []
    # find the maximum number of boreholes as a float
    n_1_max = (length_1 / b_min) + 1
    n_1_min = (length_1 / b_max) + 1

    n_min = ceil(n_1_min)
    n_max = floor(n_1_max)

    n_2_old = 1

    if disp:
        print(50 * "-")
        print("Rectangular Domain\nNx\tNy\tBx\tBy")

    _iter = 0

    for N in range(n_min, n_max + 1):
        # Check to see if we bracket
        b = length_1 / (N - 1)
        n_2 = floor((length_2 / b) + 1)

        if _iter == 0:
            for i in range(1, n_min):
                r = rectangle(i, 1, b, b)
                if transpose:
                    r = transpose_coordinates(r)
                rectangle_domain.append(r)
                field_descriptors.append(f"{i}X{1}_B{b:0.2f}")
            for j in range(1, n_2):
                r = rectangle(n_min, j, b, b)
                if transpose:
                    r = transpose_coordinates(r)
                rectangle_domain.append(r)
                field_descriptors.append(f"{n_min}X{j}_B{b:0.2f}")

            _iter += 1
        if n_2_old == n_2:
            pass
        else:
            r = rectangle(N, n_2, b, b)
            if disp:
                print(f"{N}\t{n_2}\t{b}\t{b}")
            if transpose:
                r = transpose_coordinates(r)
            rectangle_domain.append(r)
            field_descriptors.append(f"{N}X{n_2}_B{b:0.2f}")
            n_2_old = n_2

        N += 1

    return rectangle_domain, field_descriptors


def bi_rectangular(length_x, length_y, b_min, b_max_x, b_max_y, transpose=False, disp=False):
    # Make this work for the transpose
    if length_x >= length_y:
        length_1 = length_x
        length_2 = length_y
        b_max_1 = b_max_x
        b_max_2 = b_max_y
    else:
        length_1 = length_y
        length_2 = length_x
        b_max_1 = b_max_y
        b_max_2 = b_max_x

    bi_rectangle_domain = []
    field_descriptors = []
    # find the maximum number of boreholes as a float
    n_1_max = (length_1 / b_min) + 1
    n_1_min = (length_1 / b_max_1) + 1

    # if it is the first case in the domain, we want to step up from one
    # borehole, to a line, to adding the rows
    _iter = 0

    n_min = ceil(n_1_min)
    n_max = floor(n_1_max)
    for n_1 in range(n_min, n_max + 1):

        n_2 = ceil((length_2 / b_max_2) + 1)
        b_2 = length_2 / (n_2 - 1)

        b_1 = length_1 / (n_1 - 1)

        if _iter == 0:
            for i in range(1, n_1):
                coordinates = rectangle(i, 1, b_1, b_2)
                if transpose:
                    coordinates = transpose_coordinates(coordinates)
                bi_rectangle_domain.append(coordinates)
                field_descriptors.append(f"{i}X{1}_B1{b_1:0.2f}_B2{b_2:0.2f}")
            for j in range(1, n_2):
                coordinates = rectangle(n_1, j, b_1, b_2)
                if transpose:
                    coordinates = transpose_coordinates(coordinates)
                bi_rectangle_domain.append(coordinates)
                field_descriptors.append(f"{n_1}X{j}_B1{b_1:0.2f}_B2{b_2:0.2f}")

            _iter += 1

        if disp:
            print(f"{n_1}x{n_2} with {b_1:0.1f}x{b_2:0.1f}")

        coordinates = rectangle(n_1, n_2, b_1, b_2)
        if transpose:
            coordinates = transpose_coordinates(coordinates)
        bi_rectangle_domain.append(coordinates)
        field_descriptors.append(f"{n_1}X{n_2}_B1{b_1:0.2f}_B2{b_2:0.2f}")

        n_1 += 1

    return bi_rectangle_domain, field_descriptors


def bi_rectangle_nested(length_x, length_y, b_min, b_max_x, b_max_y, disp=False):
    # Make this work for the transpose
    if length_x >= length_y:
        length_1 = length_x
        length_2 = length_y
        b_max_1 = b_max_x
        b_max_2 = b_max_y
        transpose = False
    else:
        length_1 = length_y
        length_2 = length_x
        b_max_1 = b_max_y
        b_max_2 = b_max_x
        transpose = True

    # find the maximum number of boreholes as a float
    n_2_max = (length_2 / b_min) + 1
    n_2_min = (length_2 / b_max_2) + 1

    n_min = ceil(n_2_min)
    n_max = floor(n_2_max)

    bi_rectangle_nested_domain = []
    field_descriptors = []

    for n_2 in range(n_min, n_max + 1):
        b_2 = length_2 / (n_2 - 1)
        bi_rectangle_domain, f_d = bi_rectangular(length_1, length_2, b_min, b_max_1,
                                                  b_2, transpose=transpose, disp=disp)
        # print("Bi-Rectangular: ",bi_rectangle_domain)
        bi_rectangle_nested_domain.append(bi_rectangle_domain)
        # fieldDescriptors.append(
        #     str(length_1) + "X" + str(length_2) + "_" + str(B_min) + "_" + str(B_max_1)+"_"+str(b_2)
        # )
        field_descriptors.append(f_d)

    return bi_rectangle_nested_domain, field_descriptors


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

    _zoned_rectangle_domain = []
    field_descriptors = []

    n_i1 = 1
    n_i2 = 1

    z = zoned_rectangle(n_1, n_2, b_1, b_2, n_i1, n_i2)
    _zoned_rectangle_domain.append(z)
    field_descriptors.append(f"{n_1}X{n_2}_{n_i1}X{n_i2}_B1{b_1:0.2f}_B2{b_2:0.2f}")

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
        _zoned_rectangle_domain.append(z)
        field_descriptors.append(f"{n_1}X{n_2}_{n_i1}X{n_i2}_B1{b_1:0.2f}_B2{b_2:0.2f}")

    return _zoned_rectangle_domain, field_descriptors


def bi_rectangle_zoned_nested(length_x, length_y, b_min, b_max_x, b_max_y):
    # Make this work for the transpose
    if length_x >= length_y:
        length_1 = length_x
        length_2 = length_y
        b_max_1 = b_max_x
        b_max_2 = b_max_y
        transpose = False
    else:
        length_1 = length_y
        length_2 = length_x
        b_max_1 = b_max_y
        b_max_2 = b_max_x
        transpose = True

    # find the maximum number of boreholes as a float
    n_1_max = (length_1 / b_min) + 1
    n_1_min = (length_1 / b_max_1) + 1

    n_2_max = (length_2 / b_min) + 1
    n_2_min = (length_2 / b_max_2) + 1

    n_min_1 = ceil(n_1_min)
    n_max_1 = floor(n_1_max)

    n_min_2 = ceil(n_2_min)
    n_max_2 = floor(n_2_max)

    bi_rectangle_zoned_nested_domain = []
    field_descriptors = []

    n_1_values = list(range(n_min_1, n_max_1 + 1))
    n_2_values = list(range(n_min_2, n_max_2 + 1))

    j = 0  # pertains to n_1_values
    k = 0  # pertains to n_2_values
    index_l = 0

    domain = []
    f_d = []
    for i in range(len(n_1_values) + len(n_2_values) - 1):
        if index_l == 0:
            b_x = length_x / (n_min_1 - 1)
            b_y = length_y / (n_min_2 - 1)

            # go from one borehole to a line
            for index_l in range(1, n_min_1 + 1):
                r = rectangle(index_l, 1, b_x, b_y)
                if transpose:
                    r = transpose_coordinates(r)
                domain.append(r)
                f_d.append(f"{index_l}X{1}_{b_x:0.2f}X{b_y:0.2f}")

            # go from a line to an L
            for index_l in range(2, n_min_2 + 1):
                l_shape_object = l_shape(n_min_1, index_l, b_x, b_y)
                if transpose:
                    l_shape_object = transpose_coordinates(l_shape_object)
                domain.append(l_shape_object)
                f_d.append(f"{n_min_1}X{index_l}_{b_x:0.2f}X{b_y:0.2f}")

            # go from an L to a U
            for index_l in range(2, n_min_2 + 1):
                lop_u_field = lop_u(n_min_1, n_min_2, b_x, b_y, index_l)
                if transpose:
                    lop_u_field = transpose_coordinates(lop_u_field)
                domain.append(lop_u_field)
                f_d.append(f"{n_min_1}X{n_min_2}_{b_x:0.2f}X{b_y:0.2f}")

            # go from a U to an open
            for index_l in range(1, n_min_1 - 1):
                c = c_shape(n_min_1, n_min_2, b_x, b_y, index_l)
                if transpose:
                    c = transpose_coordinates(c)
                domain.append(c)
                f_d.append(f"{n_min_1}X{n_min_2}_{b_x:0.2f}X{b_y:0.2f}")

            index_l += 1

        if i % 2 == 0:
            bi_rectangle_zoned_domain, f_ds = zoned_rectangle_domain(length_1, length_2, n_1_values[j], n_2_values[k],
                                                                     transpose=transpose)
            domain.extend(bi_rectangle_zoned_domain)
            f_d.extend(f_ds)
            if j < len(n_1_values) - 1:
                j += 1
            else:
                k += 1
        else:
            bi_rectangle_zoned_domain, f_ds = zoned_rectangle_domain(length_1, length_2, n_1_values[j], n_2_values[k],
                                                                     transpose=transpose)
            domain.extend(bi_rectangle_zoned_domain)
            f_d.extend(f_ds)
            if k < len(n_2_values) - 1:
                k += 1
            else:
                j += 1

    bi_rectangle_zoned_nested_domain.append(domain)
    field_descriptors.append(f_d)

    return bi_rectangle_zoned_nested_domain, field_descriptors


def polygonal_land_constraint(b_min, b_max_x, b_max_y, property_boundary, no_go_boundaries=None):
    if no_go_boundaries is None:
        no_go_boundaries = []

    outer_rectangle = determine_largest_rectangle(property_boundary)

    x, y = list(zip(*outer_rectangle))
    length = max(x)
    width = max(y)
    coordinates_domain_nested, field_descriptors = bi_rectangle_nested(length, width, b_min, b_max_x, b_max_y)

    coordinates_domain_nested_cutout = []

    for domain in coordinates_domain_nested:
        new_coordinates_domain = []
        for coordinates in domain:
            # Remove boreholes outside of property
            new_coordinates = remove_cutout(coordinates, boundary=property_boundary, remove_inside=False)
            # Remove boreholes inside of building
            if len(new_coordinates) == 0:
                continue
            for no_go_zone in no_go_boundaries:
                new_coordinates = remove_cutout(new_coordinates, boundary=no_go_zone, remove_inside=True,
                                                keep_contour=False)
            new_coordinates_domain.append(new_coordinates)
        coordinates_domain_nested_cutout.append(new_coordinates_domain)

    coordinates_domain_nested_cutout_reordered = []
    field_descriptors_reordered = []
    for idx, domain in enumerate(coordinates_domain_nested_cutout):
        domain_reordered, f_d_reordered = reorder_domain(domain, field_descriptors[idx])
        coordinates_domain_nested_cutout_reordered.append(domain_reordered)
        field_descriptors_reordered.append(f_d_reordered)

    return coordinates_domain_nested_cutout_reordered, field_descriptors_reordered


def reorder_domain(domain, descriptors):
    """
    Sort domains by length. Rearrange descriptors accordingly.
    Solution from: https://stackoverflow.com/a/9764364

    # TODO: Investigate whether this is needed.
    # TODO: Domains may already be presorted by the nature of the preceding algorithms.
    """

    return zip(*sorted(zip(domain, descriptors), key=lambda x: len(x[0])))
