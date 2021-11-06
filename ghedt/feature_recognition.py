# Jack C. Cook
# Wednesday, January 15, 2020

from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import math


def points_in_polygon(polygon, pts):
    # https://stackoverflow.com/a/67460792/11637415
    pts = np.asarray(pts,dtype='float32')
    polygon = np.asarray(polygon,dtype='float32')
    contour2 = np.vstack((polygon[1:], polygon[:1]))
    test_diff = contour2-polygon
    mask1 = (pts[:,None] == polygon).all(-1).any(-1)
    m1 = (polygon[:,1] > pts[:,None,1]) != (contour2[:,1] > pts[:,None,1])
    slope = ((pts[:,None,0]-polygon[:,0])*test_diff[:,1])-(test_diff[:,0]*(pts[:,None,1]-polygon[:,1]))
    m2 = slope == 0
    mask2 = (m1 & m2).any(-1)
    m3 = (slope < 0) != (contour2[:,1] < polygon[:,1])
    m4 = m1 & m3
    count = np.count_nonzero(m4,axis=-1)
    mask3 = ~(count%2==0)
    mask = mask1 | mask2 | mask3
    return mask


def remove_cutout(coordinates, no_go=None):
    if no_go is None:
        no_go = []

    m = points_in_polygon(no_go, coordinates)
    b = np.argwhere(m == True)

    b = b.tolist()
    indices = []
    for i in range(len(b)):
        idx = b[i][0]
        indices.append(idx)

    new_coordinates = []
    for i in range(len(coordinates)):
        if i in indices:
            pass
        else:
            new_coordinates.append(coordinates[i])

    return new_coordinates
