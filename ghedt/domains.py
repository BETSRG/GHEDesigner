# Jack C. Cook
# Wednesdday, October 27, 2021

import ghedt


def square_and_near_square(lower: int,
                           upper: int,
                           B: float):
    if lower or upper <= 0:
        raise ValueError('The lower and upper arguments must be positive'
                         'integer values.')
    if upper < lower:
        raise ValueError('The lower argument should be less than or equal to'
                         'the upper.')

    coordinates_domain = []

    for i in range(lower, upper+1):
        for j in range(2):
            coordinates = \
                ghedt.coordinates.rectangle(i, i+j, B, B)

            coordinates_domain.append(coordinates)

    return coordinates_domain
