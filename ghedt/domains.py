# Jack C. Cook
# Wednesdday, October 27, 2021

import ghedt


def square_and_near_square(lower: int,
                           upper: int,
                           B: float):
    coordinates_domain = []

    for i in range(lower, upper):
        for j in range(2):
            coordinates = \
                ghedt.coordinates.rectangle(i, i+j, B, B)

            coordinates_domain.append(coordinates)

    return coordinates_domain
