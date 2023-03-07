from math import atan, cos, pi, sin, sqrt

import numpy as np

from ghedesigner.constants import DEG_TO_RAD, RAD_TO_DEG, PI_OVER_2
from ghedesigner.shape import Shapes, sort_intersections


def gen_shape(prop_bound, ng_zones=None):
    """Returns an array of shapes objects representing the coordinates given"""
    r_a = [Shapes(prop_bound)]
    if ng_zones is not None:
        r_n = []
        for ngZone in ng_zones:
            r_n.append(Shapes(ngZone))
        r_a.append(r_n)
    else:
        r_a.append(None)
    return r_a


def field_optimization_wp_space_fr(
        p_space,
        space_start,
        rotate_step,
        prop_bound,
        ng_zones=None,
        rotate_start=None,
        rotate_stop=None,
):
    """Optimizes a Field by iterating over input values w/o perimeter spacing

    Parameters: p_space(float): Ratio of perimeter spacing to other target spacing space_start(float): the initial
    target spacing that the optimization program will start with rotate_step(float): the amount of rotation that will
    be changed per step (in degrees) prop_bound([[float,float]]): 2d array of floats that represent the property
    boundary (counter clockwise) ng_zones([[[float,float]]]): 3d array representing the different zones on the
    property where no boreholes can be placed rotate_start(float): the rotation that the field will start at (-pi/2 <
    rotateStart < pi/2) rotate_stop(float): the rotation that the field will stop at (exclusive) (-pi/2 < rotateStop
    < pi/2)

    Outputs: CSVs containing the coordinates for the max field for each target spacing, their respective graphs,
    and their respective data

    """
    if rotate_start is None:
        rotate_start = (-90.0 + rotate_step) * DEG_TO_RAD
    if rotate_stop is None:
        rotate_stop = PI_OVER_2
    if rotate_start > PI_OVER_2 or rotate_start < -PI_OVER_2 or rotate_stop > PI_OVER_2 or rotate_stop < -PI_OVER_2:
        raise ValueError("Invalid Rotation")

    space = space_start
    rt = rotate_start

    y_s = space
    x_s = y_s

    max_l = 0
    max_hole = None
    max_rt = None

    while rt < rotate_stop:
        hole = two_space_gen_bhc(
            prop_bound,
            y_s,
            x_s,
            rotate=rt,
            no_go=ng_zones,
            p_space=p_space * x_s,
            intersection_tolerance=1e-5,
        )

        # Assuming that the rotation with the maximum number of boreholes is most efficiently using space
        if len(hole) > max_l:
            max_l = len(hole)
            max_rt = rt * RAD_TO_DEG
            max_hole = hole

        rt += rotate_step * DEG_TO_RAD

    # Ensures that there are no repeated boreholes
    max_hole = np.array(remove_duplicates(max_hole, p_space * x_s))

    field = max_hole
    field_name = "P" + str(p_space) + "_S" + str(space) + "_rt" + str(max_rt)
    return [field, field_name]


def field_optimization_fr(
        space_start,
        rotate_step,
        prop_bound,
        ng_zones=None,
        rotate_start=None,
        rotate_stop=None,
        intersection_tolerance=1e-5,
):
    """Optimizes a Field by iterating over input values w/o perimeter spacing

    Parameters: space_start(float): the initial target spacing that the optimization program will start with
    rotate_step(float): the amount of rotation that will be changed per step (in degrees) prop_bound([[float,
    float]]): 2d array of floats that represent the property boundary (counter clockwise) ng_zones([[[float,
    float]]]): 3d array representing the different zones on the property where no boreholes can be placed
    rotate_start(float): the rotation that the field will start at (-pi/2 < rotateStart < pi/2) rotate_stop(float):
    the rotation that the field will stop at (exclusive) (-pi/2 < rotateStop < pi/2) intersection_tolerance:

    Outputs: CSVs containing the coordinates for the max field for each target spacing, their respective graphs,
    and their respective data

    """
    if rotate_start is None:
        rotate_start = -90.0 * DEG_TO_RAD
    if rotate_stop is None:
        rotate_stop = PI_OVER_2
    if (
            rotate_start > PI_OVER_2
            or rotate_start < -PI_OVER_2
            or rotate_stop > PI_OVER_2
            or rotate_stop < -PI_OVER_2
    ):
        raise ValueError("Invalid Rotation")

    # Target Spacing iterates

    space = space_start
    rt = rotate_start

    y_s = space
    x_s = y_s

    max_l = 0
    max_hole = None
    max_rt = None

    while rt < rotate_stop:
        hole = gen_borehole_config(
            prop_bound,
            y_s,
            x_s,
            rotate=rt,
            no_go=ng_zones,
            intersection_tolerance=intersection_tolerance,
        )

        # Assuming that the rotation with the maximum number of boreholes is most efficiently using space
        if len(hole) > max_l:
            max_l = len(hole)
            max_rt = rt * RAD_TO_DEG
            max_hole = hole

        rt += rotate_step * DEG_TO_RAD

    # Ensures that there are no repeated boreholes
    max_hole = np.array(remove_duplicates(max_hole, x_s * 1.2))

    field = max_hole
    field_name = "S" + str(space) + "_rt" + str(max_rt)
    return [field, field_name]


def find_duplicates(borefield, space, disp=False):
    """
    The distance method :func:`Borehole.distance` is utilized to find all
    duplicate boreholes in a boreField.
    This function considers a duplicate to be any pair of points that fall
    within each other's radius. The lower index (i) is always stored in the
    0 position of the tuple, while the higher index (j) is stored in the 1
    position.
    Parameters
    ----------
    borefield : list
        A list of :class:`Borehole` objects
    space:
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.
    Returns
    -------
    duplicate_pairs : list
        A list of tuples where the tuples are pairs of duplicates
    """

    duplicate_pairs = []  # define an empty list to be appended to
    for i, borehole_1 in enumerate(borefield):
        for j in range(i, len(borefield)):  # only loop unique interactions
            borehole_2 = borefield[j]
            if i == j:  # skip the borehole itself
                continue
            else:
                dist = sq_dist(borehole_1, borehole_2)
            if abs(dist) < (space * 10 ** -1):
                duplicate_pairs.append((i, j))
    if disp:
        # pad with '-' align in center
        output = f"{'*gt.boreholes.find_duplicates()*' :-^50}"
        # keep a space between the function name
        print(output.replace("*", " "))
        print(f"The duplicate pairs of boreholes found: {duplicate_pairs}")
    return duplicate_pairs


def sq_dist(p1, p2):
    """Returns the cartesian distance between two points"""
    return sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))


def remove_duplicates(borefield, space, disp=False):
    """
    Removes all the duplicates found from the duplicate pairs returned in
    :func:`check_duplicates`.
    For each pair of duplicates, the first borehole (with the lower index) is
    kept and the other (with the higher index) is removed.
    Parameters
    ----------
    borefield : list
        A list of :class:`Borehole` objects
    space:
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.
    Returns
    -------
    new_borefield : list
        A boreField without duplicates
    """
    # get a list of tuple
    duplicate_pairs = find_duplicates(borefield, space, disp=disp)

    new_borefield = []

    # values not to be included
    duplicate_bores = []
    for i in range(len(duplicate_pairs)):
        duplicate_bores.append(duplicate_pairs[i][1])

    for i in range(len(borefield)):
        if i in duplicate_bores:
            continue
        else:
            new_borefield.append(borefield[i])
    if disp:
        # pad with '-' align in center
        print(
            f"{'*gt.boreholes.remove_duplicates()*' :-^50}".replace("*", " ")
        )  # keep a space between the function name
        n_duplicates = len(borefield) - len(new_borefield)
        print(f"The number of duplicates removed: {n_duplicates}")

    return new_borefield


def two_space_gen_bhc(
        field,
        y_space,
        x_space,
        no_go=None,
        rotate=0,
        p_space=None,
        i_space=None,
        intersection_tolerance=1e-5,
):
    """Generates a borefield that has perimeter spacing

    Parameters:
        field: The outer boundary of the property represented as an array of points
        y_space: Target Spacing in y-dir
        x_space: Target Spacing in x-dir
        no_go: a 3d array representing all the areas where boreholes cannot be placed
        rotate: the amount of rotation (rad)
        p_space: Perimeter spacing
        i_space: Spacing required between perimeter and the rest
        i_space: Min spacing required from all edges
        intersection_tolerance:

    """
    if p_space is None:
        p_space = 0.9 * x_space
    if i_space is None:
        i_space = x_space

    # calls the standard row-wise coord generator w/ the adjusted vertices
    holes = gen_borehole_config(
        field,
        y_space,
        x_space,
        no_go=no_go,
        rotate=rotate,
        intersection_tolerance=intersection_tolerance,
    )

    holes = holes.tolist()
    remove_points_too_close(field, holes, i_space, no_go_zones=no_go)

    # places the boreholes along the perimeter of the property boundary and no_go zone(s)
    perimeter_distribute(field, p_space, holes)
    if no_go is not None:
        for ng in no_go:
            perimeter_distribute(ng, p_space, holes)

    # returns the Holes as a numpy array for easier manipulation
    return_array = np.array(holes)
    return return_array


def remove_points_too_close(field, holes, i_space, no_go_zones=None):
    """
    Will remove all points too close to the field and no-go zones

    Parameters:
        field: The outer boundary of the property represented as an array of points
        no_go_zones: a 3d array representing all the areas where boreholes cannot be placed
        holes: 2d array containing all the current boreholes
        i_space: Min spacing required from all edges
    """
    field = field.c
    len_field = len(field)
    for i in range(len_field):
        p1 = field[i]
        if i == len_field - 1:
            p2 = field[0]
        else:
            p2 = field[i + 1]
        remove_points_close_too_line(p1, p2, holes, i_space)
    if no_go_zones is not None:
        len_no_go_zones = len(no_go_zones)
        for i in range(len_no_go_zones):
            ng = no_go_zones[i].c
            len_ng = len(ng)
            for j in range(len_ng):
                p1 = ng[j]
                if j == len_ng - 1:
                    p2 = ng[0]
                else:
                    p2 = ng[j + 1]
                remove_points_close_too_line(p1, p2, holes, i_space)


def remove_points_close_too_line(p1, p2, holes, i_space):
    """Removes points that are close to the given line

    Parameters:
        p1([float,float]): first point in line
        p2([float,float]): second point in line
        holes: 2d array containing a bunch of points
        i_space(float): distance cutoff for how close points can be

    """
    len_holes = len(holes)
    i = 0
    while i < len_holes:
        hole = holes[i]
        dp = dist_from_line(p1, p2, hole)
        if dp < i_space:
            del holes[i]
            len_holes -= 1
        else:
            i += 1


def dist_from_line(p1, p2, other_point):
    """Calculates the distance from a point to a line (closest distance):
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    Parameter:
        p1: first point on the line
        p2: second point on the line
        other_point: point which is being measured to

    """
    dxl = p2[0] - p1[0]
    dyl = p2[1] - p1[1]
    dx = p1[0] - other_point[0]
    dy = p1[1] - other_point[1]
    num = abs(dxl * dy - dx * dyl)
    den = sqrt(dxl * dxl + dyl * dyl)
    dp = num / den
    dist_l = sq_dist(p1, p2)
    d01 = sq_dist(p1, other_point)
    d02 = sq_dist(p2, other_point)
    if d01 * d01 - dp * dp < 0:
        return d01
    if sqrt(d01 * d01 - dp * dp) / dist_l > 1:
        return min(d01, d02)
    if min(p1[0], p2[0]) < other_point[0] < max(p1[0], p2[0]) or min(p1[1], p2[1]) < other_point[1] < max(p1[1], p2[1]):
        return min(d01, d02, dp)
    else:
        return min(d01, d02)


def perimeter_distribute(field, space, r):
    """
    Distributes boreholes along the perimeter of a given shape

    Parameters:
        field: array of points representing closed polygon
        space (float): spacing that the boreholes should have from one another
        r (dict{}): existing dictionary of boreholes which will be appended to
    """
    # print(r)
    for i in range(len(field.c)):
        if i == len(field.c) - 1:
            vert1 = field.c[i]
            vert2 = field.c[0]
        else:
            vert1 = field.c[i]
            vert2 = field.c[i + 1]
        dx = vert2[0] - vert1[0]
        dy = vert2[1] - vert1[1]

        # Checking how many boreholes can be distributed along the line
        dist = sq_dist(vert1, vert2)
        num_holes = int(dist // space)

        # Distributing the spacing to the x and y directions
        x_space = None
        y_space = None
        if num_holes > 0:
            x_space = dx / num_holes
            y_space = dy / num_holes

        current_p = [vert1[0], vert1[1]]

        # for loop is tuned to leave one spot empty for the next line
        for _ in range(num_holes):
            r.append([current_p[0], current_p[1]])
            current_p[0] += x_space
            current_p[1] += y_space


def gen_borehole_config(
        field,
        y_space,
        x_space,
        no_go=None,
        rotate=0,
        intersection_tolerance=1e-6,
):
    """
    Function generates a series of x,y points representing a field of boreholes
    in a trapezoidal shape. Returns empty if boreHole field does not meet given requirements

    Parameters
    -------------
        :param field:
        :param y_space: float
            the minimum spacing between points in the y-dir
        :param x_space: float
            the minimum spacing between points in the x-dir
        :param no_go:
        :param rotate:
        :param intersection_tolerance:

        :return: [[float]] -> 2 col + n rows
    """

    if no_go is None:
        no_go = []
    # Decides which vertex to start generating boreholes at by finding the "lowest" vertex relative to a rotated x-axis
    lowest_vert_val = float("inf")
    highest_vert_val = float("-inf")
    lowest_vert = None
    highest_vert = None
    for vert in field.c:
        if vert[0] != 0:
            phi = atan(vert[1] / vert[0])
        else:
            phi = PI_OVER_2
        dist_vert = sqrt(vert[1] ** 2 + vert[0] ** 2)
        ref_angle = phi
        if phi > PI_OVER_2:
            if phi > pi:
                if phi > 3 * PI_OVER_2:
                    ref_angle = 2 * rotate + 3 * PI_OVER_2 - phi
                else:
                    ref_angle = 2 * rotate + pi - phi
            else:
                ref_angle = pi - phi + 2 * rotate
        yp = dist_vert * sin(ref_angle - rotate)
        if yp < lowest_vert_val:
            lowest_vert_val = yp
            lowest_vert = vert
        if yp > highest_vert_val:
            highest_vert_val = yp
            highest_vert = vert

    # Determines the number of rows as well as the distance between the rows
    num_rows = int((highest_vert_val - lowest_vert_val) // y_space)
    d = highest_vert_val - lowest_vert_val
    s = d / num_rows
    row_space = [-1 * s * cos(PI_OVER_2 - rotate), s * sin(PI_OVER_2 - rotate)]

    # Establishes the dictionary where the boreholes will be added two as well as establishing a point on the first row
    boreholes = {}
    row_point = [lowest_vert[0], lowest_vert[1]]

    # This is just a value that is combined with the slope of the row's to establish two points defining a row (could
    # be any value)
    point_shift = 1000.0

    for _ in range(num_rows + 1):

        # Row Defined by two points
        if row_space[1] == 0:
            row = [
                row_point[0],
                row_point[1],
                row_point[0],
                row_point[1] + point_shift,
            ]
        else:
            row = [
                row_point[0],
                row_point[1],
                row_point[0] + point_shift,
                row_point[1] + (-row_space[0] / row_space[1]) * point_shift,
            ]

        # Gets Intersection between current row and property boundary
        f_inters = field.line_intersect(row, rotate, intersection_tolerance)

        # Stores the number of intersections with the row
        len_f_inters = len(f_inters)

        # Checks for edge case where a single intersection is reported as two and treats it as one
        if (
                len_f_inters > 1
                and abs(f_inters[0][0] - f_inters[1][0]) <= intersection_tolerance
                and abs(f_inters[0][1] - f_inters[1][1]) <= intersection_tolerance
        ):
            fi = 0
            fij = 0
            while fi < len_f_inters:
                while fij < len_f_inters:
                    if fij == fi:
                        fij += 1
                        continue
                    if (
                            abs(f_inters[fi][0] - f_inters[fij][0]) <= intersection_tolerance
                            and abs(f_inters[fi][1] - f_inters[fij][1])
                            <= intersection_tolerance
                    ):
                        f_inters.pop(fij)
                        if fi >= fij:
                            fi -= 1
                        fij -= 1
                        len_f_inters -= 1
                    fij += 1
                fi += 1

        # Checks for edge case where there are no intersections detected due to a rounding error (can sometimes
        # happen with the last row)
        """

        if len_f_inters == 0 and ri == num_rows:
            ins = False

            # Checks if the predicted point (ghost point that was expected but not found) is inside one of the no_go
            zones for shape in no_go: if shape.point_intersect(highest_vert): ins = True if not ins: #Double checks
            that this borehole has not already been included if len(boreholes)==0 or not (boreholes[len(boreholes) -
            1][0] == highest_vert[0] and boreholes[len(boreholes) - 1][1] ==highest_vert[1]): boreholes[len(
            boreholes)] = highest_vert """
        # Handles cases with odd number of intersections
        if len_f_inters % 2 == 0:

            # Specific case with two intersections
            if len_f_inters == 2:

                # Checks for the edge case where two intersections are very close together and replaces them with one
                # point
                if (
                        sqrt(
                            (f_inters[0][0] - f_inters[1][0])
                            * (f_inters[0][0] - f_inters[1][0])
                            + (f_inters[0][1] - f_inters[1][1])
                            * (f_inters[0][1] - f_inters[1][1])
                        )
                        < x_space
                ):
                    ins = False
                    for ng_shape in no_go:
                        if ng_shape.point_intersect(highest_vert):
                            ins = True
                    if not ins:
                        boreholes[len(boreholes)] = f_inters[0]
                        len_f_inters = 0  # skips the while loop

            i = 0
            while i < len_f_inters - 1:

                left_offset = [0, 0]
                right_offset = [0, 0]

                # Checks if there is enough distance between this point and another and then will offset the point if
                # there is not enough room
                dls_check = sqrt(
                    (f_inters[i][0] - f_inters[i - 1][0])
                    * (f_inters[i][0] - f_inters[i - 1][0])
                    + (f_inters[i][1] - f_inters[i - 1][1])
                    * (f_inters[i][1] - f_inters[i - 1][1])
                )

                drs_check = sqrt(
                    (f_inters[i][0] - f_inters[i + 1][0])
                    * (f_inters[i][0] - f_inters[i + 1][0])
                    + (f_inters[i][1] - f_inters[i + 1][1])
                    * (f_inters[i][1] - f_inters[i + 1][1])
                )

                if i > 0 and (dls := dls_check) < x_space:
                    left_offset = [dls * cos(rotate), dls * sin(rotate)]
                elif i < len_f_inters - 1 and (drs := drs_check) < x_space:
                    right_offset = [-drs * cos(rotate), -drs * sin(rotate)]

                process_rows(
                    row,
                    [f_inters[i][0] + left_offset[0], f_inters[i][1] + left_offset[1]],
                    [
                        f_inters[i + 1][0] + right_offset[0],
                        f_inters[i + 1][1] + right_offset[1],
                    ],
                    no_go,
                    x_space,
                    boreholes,
                    rotate=rotate,
                )

                i += 2
        elif len_f_inters == 1:
            ins = False
            for ng_shape in no_go:
                if ng_shape.point_intersect(highest_vert):
                    ins = True
            if not ins:
                if len(boreholes) == 0:
                    boreholes[len(boreholes)] = f_inters[0]
                if not (
                        boreholes[len(boreholes) - 1][0] == f_inters[0][0]
                        and boreholes[len(boreholes) - 1][1] == f_inters[0][1]
                ):
                    boreholes[len(boreholes)] = f_inters[0]
        else:
            i = 0
            while i < len_f_inters - 1:
                if field.point_intersect(
                        [
                            (f_inters[i][0] + f_inters[i + 1][0]) / 2,
                            (f_inters[i][1] + f_inters[i + 1][1]) / 2,
                        ]
                ):
                    process_rows(
                        row,
                        f_inters[i],
                        f_inters[i + 1],
                        no_go,
                        x_space,
                        boreholes,
                        rotate=rotate,
                    )
                i += 1
        row_point[0] += row_space[0]
        row_point[1] += row_space[1]
    r_a = [boreholes[element] for element in boreholes]
    r_a = np.array(remove_duplicates(r_a, x_space))
    return r_a


def process_rows(row, row_sx, row_ex, no_go, row_space, r_a, rotate, intersection_tolerance=1e-5):
    """
    Function generates a row of the borefield
    *Note: the formatting from the rows can be a little unexpected. Some adjustment
    may be required to correct the formatting. The genBoreHoleConfig function already accounts for this.
    Parameters
    -------------
    :param row:
    :param row_sx:
    :param row_ex:
    :param no_go:
    :param row_space:
    :param r_a:
    :param rotate:
    :param intersection_tolerance:
    """

    if no_go is None:
        distribute(row_sx, row_ex, row_space, r_a, rotate)
        return r_a
    num_col = int(
        sqrt(
            (row_sx[0] - row_ex[0]) * (row_sx[0] - row_ex[0])
            + (row_sx[1] - row_ex[1]) * (row_sx[1] - row_ex[1])
        )
        // row_space
    )

    inters = [
        point
        for shape in no_go
        for point in shape.line_intersect(
            row, rotate=rotate, intersection_tolerance=intersection_tolerance
        )
    ]
    inters = sort_intersections(inters, rotate)
    num_inters = len(inters)

    if num_inters > 1:
        if less_than(
                inters[0],
                row_sx,
                rotate=rotate,
                intersection_tolerance=intersection_tolerance,
        ) and less_than(
            row_ex,
            inters[len(inters) - 1],
            rotate=rotate,
            intersection_tolerance=intersection_tolerance,
        ):
            inside = False
            for _ in inters:
                if less_than(
                        row_sx,
                        inters[0],
                        rotate=rotate,
                        intersection_tolerance=intersection_tolerance,
                ) and less_than(
                    inters[0],
                    row_ex,
                    rotate=rotate,
                    intersection_tolerance=intersection_tolerance,
                ):
                    inside = True
            if not inside:
                point_in = False
                for ngShape in no_go:
                    if ngShape.point_intersect(
                            [(row_ex[0] + row_sx[0]) / 2, (row_ex[1] + row_sx[1]) / 2]
                    ):
                        point_in = True
                if point_in:
                    return []
    inters = np.array(inters)
    indices = []
    for j in range(num_inters):
        less_than_1 = less_than(row_ex, inters[j], rotate=rotate, intersection_tolerance=intersection_tolerance)
        less_than_2 = less_than(inters[j], row_sx, rotate=rotate, intersection_tolerance=intersection_tolerance)
        if not (less_than_1 or less_than_2):
            indices.append(j)
    inters = inters[indices]
    num_inters = len(inters)
    for i in range(num_inters - 1):
        space = sqrt((inters[i + 1][0] - inters[i][0]) * (inters[i + 1][0] - inters[i][0])
                     + (inters[i + 1][1] - inters[i][1]) * (inters[i + 1][1] - inters[i][1]))
        if space < row_space:
            i_none = False
            for shape in no_go:
                if shape.point_intersect(
                        [
                            (inters[i + 1][0] + inters[i][0]) / 2,
                            (inters[i + 1][1] + inters[i][1]) / 2,
                        ]
                ):
                    i_none = True
            if i_none:
                d = (row_space - space) / 2
                inters[i + 1][0] += d * cos(rotate)
                inters[i + 1][1] += d * sin(rotate)
                inters[i][0] -= d * cos(rotate)
                inters[i][1] -= d * sin(rotate)
    if num_col < 1:
        ins = False
        for shape in no_go:
            if shape.point_intersect(
                    [(row_ex[0] + row_sx[0]) / 2, (row_ex[1] + row_sx[1]) / 2]
            ):
                ins = True
        if not ins:
            if len(r_a) == 0 or not (
                    r_a[len(r_a) - 1][0] == (row_ex[0] + row_sx[0]) / 2
                    and r_a[len(r_a) - 1][1] == (row_ex[1] + row_sx[1]) / 2
            ):
                r_a[len(r_a)] = [(row_ex[0] + row_sx[0]) / 2, (row_ex[1] + row_sx[1]) / 2]
            return r_a
    else:
        if num_inters == 0:
            if not_inside(row_sx, no_go) and not_inside(row_ex, no_go):
                distribute(row_sx, row_ex, row_space, r_a, rotate)
        elif num_inters == 2:
            distribute(row_sx, inters[0], row_space, r_a, rotate)
            distribute(inters[1], row_ex, row_space, r_a, rotate)
        elif num_inters == 1:
            ins = False
            for shape in no_go:
                if shape.point_intersect(
                        [(inters[0][0] + row_sx[0]) / 2, (inters[0][1] + row_sx[1]) / 2]
                ):
                    ins = True
            if not ins:
                distribute(row_sx, inters[0], row_space, r_a, rotate)
            else:
                distribute(inters[0], row_ex, row_space, r_a, rotate)
        elif num_inters % 2 == 0:
            i = 0
            while i < num_inters:
                if i == 0:
                    distribute(row_sx, inters[0], row_space, r_a, rotate)
                    i = 1
                    continue
                elif i == num_inters - 1:
                    distribute(inters[num_inters - 1], row_ex, row_space, r_a, rotate)
                else:
                    distribute(inters[i], inters[i + 1], row_space, r_a, rotate)
                i += 2
        else:
            ins = False
            for shape in no_go:
                if shape.point_intersect(
                        [(inters[0][0] + row_sx[0]) / 2, (inters[0][1] + row_sx[1]) / 2]
                ):
                    ins = True
            if not ins:
                i = 0
                while i < num_inters:
                    if i == 0:
                        distribute(row_sx, inters[0], row_space, r_a, rotate)
                        i = 1
                        continue
                    elif i == num_inters - 1:
                        i += 2
                        continue
                    else:
                        distribute(inters[i], inters[i + 1], row_space, r_a, rotate)
                    i += 2
            else:
                i = 0
                while i < num_inters:
                    if i == 0:
                        distribute(inters[0], inters[1], row_space, r_a, rotate)
                        i = 2
                        continue
                    elif i == num_inters - 1:
                        distribute(inters[i], row_ex, row_space, r_a, rotate)
                        i += 2
                        continue
                    else:
                        distribute(inters[i], inters[i + 1], row_space, r_a, rotate)
                    i += 2

    return r_a


def not_inside(p, ngs):
    inside = False
    for ng in ngs:
        if ng.point_intersect(p):
            inside = True
    return not inside


def less_than(p1, p2, rotate=0, intersection_tolerance=1e-5):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1

    if abs(dx) < intersection_tolerance:
        dx_sign = 0
    elif dx > 0:
        dx_sign = 1
    else:
        dx_sign = -1

    if abs(dy) < intersection_tolerance:
        dy_sign = 0
    elif dy > 0:
        dy_sign = 1
    else:
        dy_sign = -1

    if rotate >= 0:
        if dx_sign == 0:
            if dy_sign == 1:
                return True
            else:
                return False
        elif dy_sign == 0:
            if dx_sign == 1:
                return True
            else:
                return False
        elif dx_sign == dy_sign:
            if dx_sign == 1:
                return True
            else:
                return False
        else:
            raise ValueError("Slope between points does not match field orientation.")
    else:
        if dx_sign == 0:
            if dy_sign == 1:
                return False
            else:
                return False
        elif dy_sign == 0:
            if dx_sign == 1:
                return True
            else:
                return False
        elif dx_sign != dy_sign:
            if dx_sign == 1:
                return True
            else:
                return False
        else:
            raise ValueError("Slope between points does not match field orientation.")


def distribute(x1, x2, spacing, r, rotate):
    """
      Function generates a series of boreholes between x1 and x2
    Parameters
    -------------
    :param x1: float
        left x value
    :param x2: float
        right x value
    :param spacing: float
        spacing between columns
    :param r: [[float]]
        existing array of points
    :param rotate:
    :return:
    """
    dx = sqrt((x1[0] - x2[0]) * (x1[0] - x2[0]) + (x1[1] - x2[1]) * (x1[1] - x2[1]))
    if dx < spacing:
        if len(r) == 0 or not (
                r[len(r) - 1][0] == (x1[0] + x2[0]) / 2
                and r[len(r) - 1][1] == (x1[1] + x2[1]) / 2
        ):
            r[len(r)] = [(x1[0] + x2[0]) / 2, (x1[1] + x2[1]) / 2]
        return
    current_x = x1
    act_num_col = int(dx // spacing)
    act_space = dx / act_num_col
    while (
            sqrt(
                (current_x[0] - x2[0]) * (current_x[0] - x2[0])
                + (current_x[1] - x2[1]) * (current_x[1] - x2[1])
            )
    ) >= 1e-8:
        if len(r) == 0 or not (
                r[len(r) - 1][0] == current_x[0] and r[len(r) - 1][1] == current_x[1]
        ):
            r[len(r)] = [current_x[0], current_x[1]]
        current_x[0] += act_space * cos(rotate)
        current_x[1] += act_space * sin(rotate)
    if not (r[len(r) - 1][0] == x2[0] and r[len(r) - 1][1] == x2[1]):
        r[len(r)] = [x2[0], x2[1]]
    return
