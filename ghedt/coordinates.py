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
