# Jack C. Cook
# Friday, October 29, 2021

import ghedt


def main():
    # Plot a zoned rectangle
    length_x = 80.
    length_y = 36.5

    n_x = 10
    n_y = 11

    zoned_rectangle_domain = \
        ghedt.domains.zoned_rectangle_domain(length_x, length_y, n_x, n_y)

    output_folder = 'zoned_rectangle_domain'

    ghedt.domains.visualize_domain(zoned_rectangle_domain, output_folder)


if __name__ == '__main__':
    main()
