import matplotlib.pyplot as plt
from math import sin, cos
from mpl_toolkits.mplot3d import Axes3D
from ghedesigner.ghe.coordinates import tilted_drill_pad# noqa: F401

def multi_pad_field(pad_centers: list[tuple[float, float]], nbh: int, tilt: float, radius: float, borehole_height: float):
    all_coordinates = []
    all_tilts = []
    all_orientations = []

    for pad_id, (cx, cy) in enumerate(pad_centers):
        coords, tilts, orients = tilted_drill_pad(
            n=nbh,
            tilt=tilt,
            radius=radius,
            center_x=cx,
            center_y=cy,
        )
        all_coordinates.extend(coords)
        all_tilts.extend(tilts)
        all_orientations.extend(orients)

    return all_coordinates, all_tilts, all_orientations

def plot_3d_borefield(collars, tilts, azimuths, H, D):
    """
    collars   : list of (x_top, y_top)
    tilts     : list of tilt angles [rad]
    azimuths  : list of azimuth angles [rad]
    H         : borehole length (active length), m
    D         : buried depth to top of active section, m
    """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    for (x_top, y_top), tilt_rad, azimuth_rad in zip(collars, tilts, azimuths):
        # Top of active borehole segment in 3D
        z_top = D  # depth positive downward

        # Direction cosines for the tilted borehole
        vx = sin(tilt_rad) * cos(azimuth_rad)
        vy = sin(tilt_rad) * sin(azimuth_rad)
        vz = cos(tilt_rad)

        # Bottom of active section
        x_bot = x_top + vx * H
        y_bot = y_top + vy * H
        z_bot = z_top + vz * H

        ax.plot(
            [x_top, x_bot],
            [y_top, y_bot],
            [z_top, z_bot],
            lw=1.5
        )
        ax.scatter([x_top], [y_top], [z_top], c="k", marker="o", s=10)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Depth [m]")
    ax.invert_zaxis()  # so that deeper = visually down
    ax.set_title("3D Borefield Geometry (Tilted Drill Pads)")
    plt.tight_layout()
    plt.show()
