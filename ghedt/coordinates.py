# Jack C. Cook
# Tuesday, October 26, 2021


def transpose_coordinates(coordinates):
    coordinates_transposed = []
    for i in range(len(coordinates)):
        x,y = coordinates[i]
        coordinates_transposed.append((y, x))
    return coordinates_transposed


def rectangle(Nx, Ny, Bx, By, origin=(0, 0)):
    # Create a list of (x, y) pairs for a rectangle
    r = []
    nbh = Nx * Ny
    _x = origin[0]
    _y = origin[1]
    for i in range(Nx):
        for j in range(Ny):
            r.append((_x + i * Bx, _y + j * By))
    assert len(r) == nbh
    return r


def open_rectangle(Nx, Ny, Bx, By):
    # Create a list of (x, y) pairs for an open rectangle
    open_r = []
    if Nx > 2 and Ny > 2:
        nbh = Ny * 2 + (Nx - 2) * 2
        for i in range(Nx):
            open_r.append((i * Bx, 0.))
        for j in range(1, Ny-1):
            open_r.append((0, j * By))
            open_r.append(((Nx-1) * Bx, j * By))
        for i in range(Nx):
            open_r.append((i * Bx, (Ny-1) * By))
    else:
        nbh = Nx * Ny
        open_r = rectangle(Nx, Ny, Bx, By)

    assert len(open_r) == nbh
    return open_r


def C_shape(Nx_1, Ny, Bx, By, Nx_2):
    nbh = Nx_1 + (Ny * 2) - 1 + Nx_2 - 1
    c = []
    for i in range(Nx_1):
        c.append((i * Bx, 0.))
    x_loc = (Nx_1 - 1) * Bx
    for j in range(1, Ny):
        c.append((0., j * By))
    for j in range(1, Ny):
        c.append((x_loc, j * By))
    y_loc = (Ny-1) * By
    for i in range(1, Nx_2+1):
        c.append((i * Bx, y_loc))
    assert len(c) == nbh
    return c


def U_shape(Nx, Ny, Bx, By):
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


def lop_U(Nx, Ny_1, Bx, By, Ny_2):
    nbh = Nx + Ny_1 - 1 + Ny_2 - 1
    lop_u = []
    for i in range(Nx):
        lop_u.append((i * Bx, 0.))
    for j in range(1, Ny_1):
        lop_u.append((0., j * By))
    x_loc = (Nx-1)*Bx
    for j in range(1, Ny_2):
        lop_u.append((x_loc, j * By))
    assert len(lop_u) == nbh
    return lop_u


def L_shape(Nx, Ny, Bx, By):
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


def visualize_coordinates(coordinates):
    """
    Visualize the (x,y) coordinates.
    Returns
    -------
    **fig, ax**
        Figure and axes information.
    """
    import matplotlib.pyplot as plt
    plt.rc('font', size=9)
    plt.rc('xtick', labelsize=9)
    plt.rc('ytick', labelsize=9)
    plt.rc('lines', lw=1.5, markersize=5.0)
    plt.rc('savefig', dpi=500)
    # fig, ax = plt.subplots(figsize=(3.5, 5))
    import pygfunction as gt
    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)

    x, y = list(zip(*coordinates))

    ax.scatter(x, y)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    ax.set_aspect('equal')

    return fig, ax
