# Jack C. Cook
# Friday, October 29, 2021

import matplotlib.pyplot as plt
import ghedt
import ghedt.PLAT.pygfunction as gt


def main():
    # Plot a zoned rectangle
    length_x = 80.
    length_y = 36.5

    n_x = 10
    n_y = 11

    b_x = length_x / (n_x - 1)
    b_y = length_y / (n_y - 1)

    n_ix = 4
    n_iy = 5

    zoned_rectangle = \
        ghedt.coordinates.zoned_rectangle(n_x, n_y, b_x, b_y, n_ix, n_iy)

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)

    x, y = list(zip(*zoned_rectangle))

    ax.scatter(x, y)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    fig.tight_layout()

    fig.savefig('zoned_01.png')

    n_ix = 2
    n_iy = 3

    zoned_rectangle = \
        ghedt.coordinates.zoned_rectangle(n_x, n_y, b_x, b_y, n_ix, n_iy)

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)

    x, y = list(zip(*zoned_rectangle))

    ax.scatter(x, y)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    fig.tight_layout()

    fig.savefig('zoned_02.png')


if __name__ == '__main__':
    main()
