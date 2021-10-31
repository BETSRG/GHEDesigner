# Jack C. Cook
# Saturday, October 30, 2021

import ghedt


def main():
    a = 1

    # Plot go and no-go zone with corrected borefield
    # -----------------------------------------------
    # coordinates = bisection_search.selected_coordinates
    coordinates = ghedt.coordinates.rectangle(10, 10, 5, 5)

    x,y = list(zip(*coordinates))

    # feature = ghedt.feature_recognition.FeatureRecognition(x, y)
    # fig, ax = feature.plot_field_convex()

    # fig.show()

    perimeter = [[0., 0.], [85., 0.], [85., 80.], [0., 80.]]
    l_x_building = 50
    l_y_building = 33.3
    origin_x, origin_y = (15, 36.5)
    no_go = [[origin_x, origin_y], [origin_x + l_x_building, origin_y],
             [origin_x + l_x_building, origin_y + l_y_building],
             [origin_x, origin_y + l_y_building]]

    fig, ax = ghedt.gfunction.GFunction.visualize_area_and_constraints(
        perimeter, coordinates, no_go=no_go)

    fig.savefig('bi-rectangle_cutout_placement.png')

    coordinates = ghedt.feature_recognition.remove_cutout(coordinates, no_go=no_go)

    fig, ax = ghedt.gfunction.GFunction.visualize_area_and_constraints(
        perimeter, coordinates, no_go=no_go)

    fig.show()


if __name__ == '__main__':
    main()
