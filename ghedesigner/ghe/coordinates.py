def transpose_coordinates(coordinates) -> list[tuple[float, float]]:
    coordinates_transposed = []
    for x, y in coordinates:
        coordinates_transposed.append((y, x))
    return coordinates_transposed


def rectangle(
    num_bh_x: int,
    num_bh_y: int,
    spacing_x: int | float,
    spacing_y: int | float,
    origin=(0, 0),
) -> list[tuple[float, float]]:
    """
    Creates a rectangular borehole field.

    X   X   X   X
    X   X   X   X
    X   X   X   X
    X   X   X   X

    Args:
        num_bh_x: number of borehole rows in x-direction
        num_bh_y: number of borehole rows in y-direction
        spacing_x: spacing between borehole rows in x-direction
        spacing_y: spacing between borehole rows in y-direction
        origin: coordinates for origin at lower-left corner

    Returns:
        list of tuples (x, y) containing borehole coordinates
    """

    r = []
    x_0 = origin[0]
    y_0 = origin[1]
    for i in range(num_bh_x):
        for j in range(num_bh_y):
            r.append((x_0 + i * spacing_x, y_0 + j * spacing_y))
    return r

def general_field_nbh_adjustment(coordinates, desired_nbh):
    sorted_coordinates = sorted(coordinates)
    return sorted_coordinates[0:desired_nbh]

def rectangle_adjusted_nbh(
    num_bh_x: int,
    num_bh_y: int,
    spacing_x: int | float,
    spacing_y: int | float,
    desired_nbh: int,
    origin=(0, 0),
) -> list[tuple[float, float]]:
    """
    Creates a rectangular borehole field. Boreholes are removed from the middle row(s) to achieve a given
    desired nbh.

    X   X   X   X
    X   X   X   X
    X   X   X   X
    X   X   X   X

    Args:
        num_bh_x: number of borehole rows in x-direction
        num_bh_y: number of borehole rows in y-direction
        spacing_x: spacing between borehole rows in x-direction
        spacing_y: spacing between borehole rows in y-direction
        desired_nbh: the number of boreholes desired for this field layout
        origin: coordinates for origin at lower-left corner

    Returns:
        list of tuples (x, y) containing borehole coordinates
    """
    nominal_nbh = num_bh_x * num_bh_y
    boreholes_to_remove = nominal_nbh - desired_nbh
    if  boreholes_to_remove  >= num_bh_x or boreholes_to_remove < 0:
        raise ValueError("The given desired nbh either exceeds the nominal nbh or requires"
                         " the removal of an entire row.")
    r = []
    x_0 = origin[0]
    y_0 = origin[1]
    if num_bh_y % 2 == 0:
        row_to_modify_1 = int(num_bh_y / 2) - 1
        row_to_modify_2 = int(num_bh_y / 2)
        modified_row_1_nbh = num_bh_x - int(boreholes_to_remove * 0.5)
        modified_row_2_nbh = num_bh_x - (int(boreholes_to_remove * 0.5) + 1)
        modified_spacing_1 = (num_bh_x - 1) * spacing_x / (modified_row_1_nbh - 1)
        if modified_row_2_nbh == 1:
            modified_spacing_2 = (num_bh_x - 1) * spacing_x * 0.5
        else:
            modified_spacing_2 = (num_bh_x - 1) * spacing_x / (modified_row_2_nbh - 1)
        for j in range(num_bh_y):
            if j == row_to_modify_1:
                row_nbh = modified_row_1_nbh
                row_spacing = modified_spacing_1
            elif j == row_to_modify_2:
                row_nbh = modified_row_2_nbh
                row_spacing = modified_spacing_2
            else:
                row_nbh = num_bh_x
                row_spacing = spacing_x
            for i in range(row_nbh):
                if j == row_to_modify_2 and modified_row_2_nbh == 1:
                    r.append((x_0 + (i + 1) * row_spacing, y_0 + j * spacing_y))
                else:
                    r.append((x_0 + i * row_spacing, y_0 + j * spacing_y))
        return r
    else:
        row_to_modify = int(num_bh_y / 2)
        modified_row_nbh = num_bh_x - boreholes_to_remove
        if modified_row_nbh == 1:
            modified_spacing = (num_bh_x - 1) * spacing_x * 0.5
        else:
            modified_spacing = (num_bh_x - 1) * spacing_x / (modified_row_nbh - 1)
        for j in range(num_bh_y):
            if j == row_to_modify:
                row_nbh = modified_row_nbh
                row_spacing = modified_spacing
            else:
                row_nbh = num_bh_x
                row_spacing = spacing_x
            for i in range(row_nbh):
                if j == row_to_modify and modified_row_nbh == 1:
                    r.append((x_0 + (i + 1) * row_spacing, y_0 + j * spacing_y))
                else:
                    r.append((x_0 + i * row_spacing, y_0 + j * spacing_y))
        return r

def open_rectangle(
    num_bh_x: int, num_bh_y: int, spacing_x: int | float, spacing_y: int | float
) -> list[tuple[float, float]]:
    """
    Creates a rectangular borehole field without center boreholes.

    X   X   X   X
    X           X
    X           X
    X   X   X   X

    Args:
        num_bh_x: number of borehole rows in x-direction
        num_bh_y: number of borehole rows in y-direction
        spacing_x: spacing between borehole rows in x-direction
        spacing_y: spacing between borehole rows in y-direction

    Returns:
        list of tuples (x, y) containing borehole coordinates
    """

    open_r = []
    if num_bh_x > 2 and num_bh_y > 2:  # noqa: PLR2004
        for i in range(num_bh_x):
            open_r.append((i * spacing_x, 0.0))
        for j in range(1, num_bh_y - 1):
            open_r.append((0, j * spacing_y))
            open_r.append(((num_bh_x - 1) * spacing_x, j * spacing_y))
        for i in range(num_bh_x):
            open_r.append((i * spacing_x, (num_bh_y - 1) * spacing_y))
        # nbh = num_bh_y * 2 + (num_bh_x - 2) * 2
    else:
        open_r = rectangle(num_bh_x, num_bh_y, spacing_x, spacing_y)
        # nbh = num_bh_x * num_bh_y
    return open_r


def c_shape(n_x_1: int, n_y: int, b_x: int | float, b_y: int | float, n_x_2: int) -> list[tuple[float, float]]:
    c = []
    for i in range(n_x_1):
        c.append((i * b_x, 0.0))
    x_loc = (n_x_1 - 1) * b_x
    for j in range(1, n_y):
        c.append((0.0, j * b_y))
    for j in range(1, n_y):
        c.append((x_loc, j * b_y))
    y_loc = (n_y - 1) * b_y
    for i in range(1, n_x_2 + 1):
        c.append((i * b_x, y_loc))
    return c


def lop_u(n_x: int, n_y_1: int, b_x: int | float, b_y: int | float, n_y_2: int) -> list[tuple[float, float]]:
    _lop_u = []
    for i in range(n_x):
        _lop_u.append((i * b_x, 0.0))
    for j in range(1, n_y_1):
        _lop_u.append((0.0, j * b_y))
    x_loc = (n_x - 1) * b_x
    for j in range(1, n_y_2):
        _lop_u.append((x_loc, j * b_y))
    return _lop_u


def l_shape(n_x: int, n_y: int, b_x: int | float, b_y: int | float) -> list[tuple[float, float]]:
    l_shape_object = []
    for i in range(n_x):
        l_shape_object.append((i * b_x, 0.0))
    for j in range(1, n_y):
        l_shape_object.append((0.0, j * b_y))
    return l_shape_object


def zoned_rectangle(
    n_x: int, n_y: int, b_x: int | float, b_y: int | float, n_ix: int, n_it: int
) -> list[tuple[float, float]]:
    """
    Create a zoned rectangle

    :param n_x:
    :param n_y:
    :param b_x:
    :param b_y:
    :param n_ix:
    :param n_it:
    :return:
    """

    if n_ix > (n_x - 2):
        raise ValueError("To many interior x boreholes.")
    if n_it > (n_y - 2):
        raise ValueError("Too many interior y boreholes.")

    # Create a list of (x, y) coordinates
    zoned = []

    # Boreholes on the perimeter
    zoned.extend(open_rectangle(n_x, n_y, b_x, b_y))

    # Create the interior coordinates
    bix = (n_x - 1) * b_x / (n_ix + 1)
    biy = (n_y - 1) * b_y / (n_it + 1)

    zoned.extend(rectangle(n_ix, n_it, bix, biy, origin=(bix, biy)))

    return zoned
