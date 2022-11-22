import cv2
import numpy as np


def scale_coordinates(coordinates, scale):
    new_coordinates = []
    for x, y in coordinates:
        x *= scale
        y *= scale
        new_coordinates.append((x, y))
    return new_coordinates


def remove_cutout(coordinates, boundary=None, remove_inside=True, keep_contour=True):
    if boundary is None:
        boundary = []

    # cv2.pointPolygonTest only takes integers, so we scale by 10000 and then
    # scale back to keep precision
    scale = 10000.0
    coordinates = scale_coordinates(coordinates, scale)
    boundary = scale_coordinates(boundary, scale)

    _boundary = np.array(boundary, dtype=np.uint64)

    # https://stackoverflow.com/a/50670359/11637415
    # Positive - point is inside the contour
    # Negative - point is outside the contour
    # Zero - point is on the contour

    inside_points_idx = []
    outside_points_idx = []
    boundary_points_idx = []
    for idx, coordinate in enumerate(coordinates):
        coordinate = coordinates[idx]
        dist = cv2.pointPolygonTest(_boundary, coordinate, False)
        if dist > 0.0:
            inside_points_idx.append(idx)
        elif dist < 0.0:
            outside_points_idx.append(idx)
        elif dist == 0.0:
            boundary_points_idx.append(idx)

    new_coordinates = []
    for idx, _ in enumerate(coordinates):
        # if we want to remove inside points and keep contour points
        if remove_inside and keep_contour:
            if idx in inside_points_idx:
                continue
            else:
                new_coordinates.append(coordinates[idx])
        # if we want to remove inside points and remove contour points
        elif remove_inside and not keep_contour:
            if idx in inside_points_idx or idx in boundary_points_idx:
                continue
            else:
                new_coordinates.append(coordinates[idx])
        # if we want to keep outside points and remove contour points
        elif not remove_inside and not keep_contour:
            if idx in outside_points_idx or idx in boundary_points_idx:
                continue
            else:
                new_coordinates.append(coordinates[idx])
        # if we want to keep outside points and keep contour points
        else:
            if idx in outside_points_idx:
                continue
            else:
                new_coordinates.append(coordinates[idx])

    new_coordinates = scale_coordinates(new_coordinates, 1 / scale)

    return new_coordinates


def determine_largest_rectangle(property_boundary):
    x_max = 0
    y_max = 0
    for x, y in property_boundary:
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y

    rectangle = [[0, 0], [x_max, 0], [x_max, y_max], [0, y_max], [0, 0]]

    return rectangle
