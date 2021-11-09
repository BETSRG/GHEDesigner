# Jack C. Cook
# Saturday, November 6, 2021

import ghedt.PLAT.pygfunction as gt
import ghedt


def main():
    property_boundary = [[20, 0],
                         [90, 0],
                         [70, 100],
                         [20, 100],
                         [0, 40],
                         [20, 20],
                         [20, 0]]

    building_description = [[11, 50],
                            [53.4264068711929, 7.57359312880715],
                            [64.7401153701776, 18.8873016277919],
                            [33.6274169979695, 50],
                            [64.7401153701776, 81.1126983722081],
                            [53.4264068711929, 92.4264068711929],
                            [11, 50]]

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)

    x, y = list(zip(*property_boundary))
    ax.plot(x, y, 'g', label='Property Boundary')

    x, y = list(zip(*building_description))
    ax.plot(x, y, 'r', label='Building Description')

    outer_rectangle = \
        ghedt.feature_recognition.determine_largest_rectangle(property_boundary)

    x, y = list(zip(*outer_rectangle))
    ax.plot(x, y, 'k--', label='Largest Rectangle')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    fig.gca().set_aspect('equal')

    fig.legend(bbox_to_anchor=(.9, 1.05), ncol=3)

    fig.tight_layout()

    fig.savefig('alternative_03_constraint_plot.png',
                bbox_inches='tight', pad_inches=0.1)

    x, y = list(zip(*outer_rectangle))
    length = max(x)
    width = max(y)
    B_min = 4.45  # m
    B_max_x = 10.  # m
    B_max_y = 12.

    coordinates_domain = \
        ghedt.domains.bi_rectangular(length, width, B_min, B_max_x, B_max_y)

    # output_folder = 'Alternative_03_Domain'
    # ghedt.domains.visualize_domain(coordinates_domain, output_folder)

    coordinates = coordinates_domain[27]

    new_coordinates = \
        ghedt.feature_recognition.remove_cutout(
            coordinates, boundary=property_boundary, remove_inside=False)

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)

    x, y = list(zip(*new_coordinates))

    ax.scatter(x, y)

    ax.set_ylim([0, 100])

    fig.show()

if __name__ == '__main__':
    main()
