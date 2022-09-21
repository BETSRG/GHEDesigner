from typing import Union


def transpose_coordinates(coordinates):
    coordinates_transposed = []
    for i in range(len(coordinates)):
        x, y = coordinates[i]
        coordinates_transposed.append((y, x))
    return coordinates_transposed


def rectangle(num_bh_x: int, num_bh_y: int,
              spacing_x: Union[int, float], spacing_y: Union[int, float],
              origin=(0, 0)):
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
    assert len(r) == num_bh_x * num_bh_y
    return r


def open_rectangle(num_bh_x: int, num_bh_y: int,
                   spacing_x: Union[int, float], spacing_y: Union[int, float]):
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
            open_r.append((i * spacing_x, 0.))
        for j in range(1, num_bh_y - 1):
            open_r.append((0, j * spacing_y))
            open_r.append(((num_bh_x - 1) * spacing_x, j * spacing_y))
        for i in range(num_bh_x):
            open_r.append((i * spacing_x, (num_bh_y - 1) * spacing_y))
        nbh = num_bh_y * 2 + (num_bh_x - 2) * 2
    else:
        open_r = rectangle(num_bh_x, num_bh_y, spacing_x, spacing_y)
        nbh = num_bh_x * num_bh_y
    assert len(open_r) == nbh
    return open_r


def c_shape(Nx_1, Ny, Bx, By, Nx_2):
    nbh = Nx_1 + (Ny * 2) - 1 + Nx_2 - 1
    c = []
    for i in range(Nx_1):
        c.append((i * Bx, 0.))
    x_loc = (Nx_1 - 1) * Bx
    for j in range(1, Ny):
        c.append((0., j * By))
    for j in range(1, Ny):
        c.append((x_loc, j * By))
    y_loc = (Ny - 1) * By
    for i in range(1, Nx_2 + 1):
        c.append((i * Bx, y_loc))
    assert len(c) == nbh
    return c


def u_shape(Nx, Ny, Bx, By):
    # Create a list of (x, y) pairs for a U-shape
    U = []
    if Nx > 2 and Ny > 1:
        nbh = 2 * Ny + (Nx - 2)
        for i in range(Nx):
            U.append((i * Bx, 0.))
        for j in range(1, Ny):
            U.append((0., j * By))
    else:
        nbh = Nx * Ny
        U = rectangle(Nx, Ny, Bx, By)
    assert len(U) == nbh
    return U


def lop_u(Nx, Ny_1, Bx, By, Ny_2):
    nbh = Nx + Ny_1 - 1 + Ny_2 - 1
    lop_u = []
    for i in range(Nx):
        lop_u.append((i * Bx, 0.))
    for j in range(1, Ny_1):
        lop_u.append((0., j * By))
    x_loc = (Nx - 1) * Bx
    for j in range(1, Ny_2):
        lop_u.append((x_loc, j * By))
    assert len(lop_u) == nbh
    return lop_u


def l_shape(Nx, Ny, Bx, By):
    nbh = Nx + Ny - 1
    L = []
    for i in range(Nx):
        L.append((i * Bx, 0.))
    for j in range(1, Ny):
        L.append((0., j * By))
    assert len(L) == nbh
    return L


def zoned_rectangle(Nx, Ny, Bx, By, Nix, Niy):
    # Create a zoned rectangle
    # The creator of the idea behind the "zoned rectangle" is
    # Dr. Jeffrey D. Spitler

    if Nix > (Nx - 2):
        raise ValueError('To many interior x boreholes.')
    if Niy > (Ny - 2):
        raise ValueError('Too many interior y boreholes.')

    # Create a list of (x, y) coordinates
    zoned = []

    # Boreholes on the perimeter
    zoned.extend(open_rectangle(Nx, Ny, Bx, By))

    # Create the interior coordinates
    Bix = (Nx - 1) * Bx / (Nix + 1)
    Biy = (Ny - 1) * By / (Niy + 1)

    zoned.extend(rectangle(Nix, Niy, Bix, Biy, origin=(Bix, Biy)))

    return zoned
