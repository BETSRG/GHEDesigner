from ghedesigner.shape import point_polygon_check


def remove_cutout(coordinates, boundaries, remove_inside=True, keep_contour=True, on_edge_tolerance=0.01):

    if type(boundaries[0][0]) != list:
        boundaries = [boundaries]

    new_coordinates = []
    inside = 1
    outside = -1
    on_edge = 0
    for idx, coordinate in enumerate(coordinates):
        coordinate = coordinates[idx]
        boundary_results = []
        for boundary in boundaries:
            boundary_results.append(point_polygon_check(boundary, coordinate, on_edge_tolerance=on_edge_tolerance))
        if remove_inside:
            if (not inside in boundary_results) and not (on_edge in boundary_results and not keep_contour):
                new_coordinates.append(coordinate)
        else:
            if (inside in boundary_results) or (on_edge in boundary_results and keep_contour):
                new_coordinates.append(coordinate)

    return new_coordinates


def determine_largest_rectangle(property_boundary):
    x_max = float('-inf')
    y_max = float('-inf')
    x_min = float('inf')
    y_min = float('inf')
    for bf_outline in property_boundary:
        for x, y in bf_outline:
            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y

    rectangle = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]]

    return rectangle
