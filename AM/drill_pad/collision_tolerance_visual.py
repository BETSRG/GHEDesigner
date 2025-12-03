# This script generates a single drill pad layout, builds rectangular prism envelopes
# for each tilted borehole, and renders a 3D visualization.
#
# It is self-contained and mirrors the logic discussed:
# - tilted_drill_pad: makes n boreholes around a circle (coords, tilts, azimuths)
# - borehole_prism_polygon: returns 4 plan-view corners for the prism (p1..p4)
# - We lift those into 3D as a skewed box defined by vertices:
#     top edge:    p1(z=0)---p2(z=0)
#     bottom edge: p4(z=-H)---p3(z=-H)
#     long edges:  p1(z=0)---p4(z=-H), p2(z=0)---p3(z=-H)
#
# The result is a visual "rectangular prism" aligned with the borehole tilt.
#
# We'll also save this as a standalone .py users can download and run.

from math import cos, sin, pi
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------- Geometry helpers -----------------------

def tilted_drill_pad(n: int, radius: float, tilt: float, center_x: float = 0.0, center_y: float = 0.0):
    """Return (coords, tilts, azimuths) for n boreholes on a circle (pad)."""
    coords: List[Tuple[float, float]] = []
    tilts: List[float] = []
    orients: List[float] = []
    for i in range(n):
        angle = 2 * pi * i / n
        x_top = center_x + radius * cos(angle)
        y_top = center_y + radius * sin(angle)
        coords.append((x_top, y_top))
        tilts.append(tilt)
        orients.append(angle)  # lean outward
    return coords, tilts, orients


def borehole_prism_polygon(
    x0: float,
    y0: float,
    H: float,
    tilt: float,
    azimuth: float,
    clearance: float,
):
    """Plan-view rectangular footprint corners [p1,p2,p3,p4,p1]."""
    horiz = H * np.sin(tilt)  # horizontal reach at depth H
    dx = horiz * np.cos(azimuth)
    dy = horiz * np.sin(azimuth)
    x1 = x0 + dx
    y1 = y0 + dy

    # Perpendicular in plan to the azimuth
    nx = -np.sin(azimuth)
    ny =  np.cos(azimuth)

    p1 = (x0 + clearance * nx, y0 + clearance * ny)
    p2 = (x0 - clearance * nx, y0 - clearance * ny)
    p3 = (x1 - clearance * nx, y1 - clearance * ny)
    p4 = (x1 + clearance * nx, y1 + clearance * ny)
    return [p1, p2, p3, p4, p1]


def build_prism_vertices_3d(polygon_2d, H: float):
    """
    Lift plan-view polygon to 3D "skewed box" (parallelepiped) using
    z=0 for top and z=-H for bottom. The four unique 2D points define
    the top and bottom edges as described above.
    """
    p1, p2, p3, p4, _ = polygon_2d  # p1..p4 unique
    # Top (z=0): p1, p2
    p1_3d = (p1[0], p1[1], 0.0)
    p2_3d = (p2[0], p2[1], 0.0)
    # Bottom (z=-H): p4, p3  (note order to keep faces consistent)
    p3_3d = (p3[0], p3[1], -H)
    p4_3d = (p4[0], p4[1], -H)

    # Faces (each is a quad): we will draw 3 faces to give a good sense of volume
    # Face A: side connecting p1->p2 (top) to p3->p4 (bottom)
    face_a = [p1_3d, p2_3d, p3_3d, p4_3d]
    # Face B: long edge face p1(top) -> p4(bottom) with a thin "thickness" using p2/p3
    face_b = [p1_3d, p4_3d, p3_3d, p2_3d]
    # Face C: the opposite thin face (top edge and bottom edge)
    face_c = [p2_3d, p1_3d, p4_3d, p3_3d]

    # Edges (lines) for clarity
    edges = [
        (p1_3d, p2_3d),  # top width
        (p4_3d, p3_3d),  # bottom width
        (p1_3d, p4_3d),  # long edge 1
        (p2_3d, p3_3d),  # long edge 2
    ]
    return [face_a, face_b, face_c], edges, ((p1_3d, p2_3d, p3_3d, p4_3d))


# ----------------------- Demo parameters -----------------------

# Single pad
NBH = 12              # boreholes per pad
RADIUS = 4.0         # [m] radius of pad
TILT = 0.2617993877992  # [rad] tilt from vertical
CENTER = (0.0, 0.0)  # pad center
H = 125.0            # [m] borehole depth
CLEARANCE = 0.5      # [m] half-width of prism

coords, tilts, orients = tilted_drill_pad(NBH, RADIUS, TILT, *CENTER)

# ----------------------- Build 3D prisms -----------------------
all_faces = []
all_edges = []
centerlines = []

for (x0, y0), tilt, az in zip(coords, tilts, orients):
    poly2d = borehole_prism_polygon(x0, y0, H, tilt, az, CLEARANCE)
    faces, edges, verts = build_prism_vertices_3d(poly2d, H)
    all_faces.extend(faces)
    all_edges.extend(edges)

    # Centerline from surface to tip
    horiz = H * np.sin(tilt)
    dx = horiz * np.cos(az)
    dy = horiz * np.sin(az)
    x1 = x0 + dx
    y1 = y0 + dy
    centerlines.append(((x0, y0, 0.0), (x1, y1, -H)))

# ----------------------- Plotting -----------------------

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='3d')

# Add prism faces
poly_collection = Poly3DCollection(all_faces, alpha=0.25)
ax.add_collection3d(poly_collection)

# Add edges
for e in all_edges:
    xs = [e[0][0], e[1][0]]
    ys = [e[0][1], e[1][1]]
    zs = [e[0][2], e[1][2]]
    ax.plot(xs, ys, zs, linewidth=1.0)

# Add centerlines
for cl in centerlines:
    xs = [cl[0][0], cl[1][0]]
    ys = [cl[0][1], cl[1][1]]
    zs = [cl[0][2], cl[1][2]]
    ax.plot(xs, ys, zs, linewidth=1.0)

# Format axes
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m] (down)')
ax.set_title('Single Drill Pad: Tilted Borehole Rectangular Prisms')

# Set equal aspect ratio
xs = [c[0] for c in coords]
ys = [c[1] for c in coords]
x_min, x_max = min(xs) - 2, max(xs) + 2
y_min, y_max = min(ys) - 2, max(ys) + 2
z_min, z_max = -H - 5, 5
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

plt.tight_layout()

plt.show()

