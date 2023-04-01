import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from ghedesigner.shape import point_polygon_check

def scale_coordinates(coordinates, scale):
    new_coordinates = []
    for x, y in coordinates:
        x *= scale
        y *= scale
        new_coordinates.append((x, y))
    return new_coordinates

def get_position(x):
    if x == 1:
        return "INSIDE"
    if x == 0:
        return "ON EDGE"
    if x == -1:
        return "OUTSIDE"

def plot_var(boundary, point, name, ocv, ghd):
    fig, ax = plt.subplots()

    boundary.append(boundary[0])

    x, y = zip(*boundary)
    ax.plot(x, y)
    ax.scatter(point[0], point[1])

    x_l, x_u = ax.get_xlim()
    y_l, y_u = ax.get_ylim()

    x_pos = (x_u - x_l) * 0.1 + x_l
    y_pos_1 = (y_u - y_l) * 0.15 + y_l
    y_pos_2 = (y_u - y_l) * 0.10 + y_l

    ax.text(x_pos, y_pos_1, f"OpenCV: {get_position(ocv)}")
    ax.text(x_pos, y_pos_2, f"GHEDesigner: {get_position(ghd)}")
    plt.grid()
    plt.savefig(f"{datetime.now().strftime('%Y-%m-%d_%H.%M.%S.%f')}-{name}.png")


def remove_cutout(coordinates, boundary=None, remove_inside=True, keep_contour=True):
    if boundary is None:
        boundary = []

    coords_orig = coordinates
    bound_orig = boundary

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
        ret = point_polygon_check(bound_orig, coords_orig[idx])

        if dist != ret:
            plot_var(bound_orig, coords_orig[idx], idx, dist, ret)

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
