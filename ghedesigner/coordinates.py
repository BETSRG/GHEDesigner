from typing import List, Tuple, Union


def transpose_coordinates(coordinates) -> List[Tuple[float, float]]:
    coordinates_transposed = []
    for x, y in coordinates:
        coordinates_transposed.append((y, x))
    return coordinates_transposed


def rectangle(num_bh_x: int, num_bh_y: int, spacing_x: Union[int, float], spacing_y: Union[int, float],
              origin=(0, 0), ) -> List[Tuple[float, float]]:
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


def open_rectangle(num_bh_x: int, num_bh_y: int, spacing_x: Union[int, float],
                   spacing_y: Union[int, float]) -> List[Tuple[float, float]]:
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
    if num_bh_x > 2 and num_bh_y > 2:
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


def c_shape(n_x_1: int, n_y: int, b_x: Union[int, float],
            b_y: Union[int, float], n_x_2: int) -> List[Tuple[float, float]]:
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


def lop_u(n_x: int, n_y_1: int, b_x: Union[int, float],
          b_y: Union[int, float], n_y_2: int) -> List[Tuple[float, float]]:
    _lop_u = []
    for i in range(n_x):
        _lop_u.append((i * b_x, 0.0))
    for j in range(1, n_y_1):
        _lop_u.append((0.0, j * b_y))
    x_loc = (n_x - 1) * b_x
    for j in range(1, n_y_2):
        _lop_u.append((x_loc, j * b_y))

    return _lop_u


def l_shape(n_x: int, n_y: int, b_x: Union[int, float], b_y: Union[int, float]) -> List[Tuple[float, float]]:
    l_shape_object = []
    for i in range(n_x):
        l_shape_object.append((i * b_x, 0.0))
    for j in range(1, n_y):
        l_shape_object.append((0.0, j * b_y))

    return l_shape_object


def zoned_rectangle(n_x: int, n_y: int, b_x: Union[int, float], b_y: Union[int, float],
                    n_ix: int, n_iy: int) -> List[Tuple[float, float]]:
    """
    Create a zoned rectangle

    :param n_x:
    :param n_y:
    :param b_x:
    :param b_y:
    :param n_ix:
    :param n_iy:
    :return:
    """

    if n_ix > (n_x - 2):
        raise ValueError("To many interior x boreholes.")
    if n_iy > (n_y - 2):
        raise ValueError("Too many interior y boreholes.")

    # Create a list of (x, y) coordinates
    zoned = []

    # Boreholes on the perimeter
    zoned.extend(open_rectangle(n_x, n_y, b_x, b_y))

    # Create the interior coordinates
    bix = (n_x - 1) * b_x / (n_ix + 1)
    biy = (n_y - 1) * b_y / (n_iy + 1)

    zoned.extend(rectangle(n_ix, n_iy, bix, biy, origin=(bix, biy)))

    return zoned
