import matplotlib.pyplot as plt
from math import sin, cos
from mpl_toolkits.mplot3d import Axes3D
from ghedesigner.ghe.coordinates import tilted_drill_pad  # noqa: F401

def single_drill_pad(nbh: int, tilt: float, radius: float, center=(0.0, 0.0)):
    """
    Create a single drill pad with NBH boreholes.
    center = (x, y) coordinates of pad center.
    """
    cx, cy = center

    coords, tilts, azimuths = tilted_drill_pad(
        n=nbh,
        tilt=tilt,
        radius=radius,
        center_x=cx,
        center_y=cy,
    )
    return coords, tilts, azimuths


def plot_3d_borefield(collars, tilts, azimuths, H, D):
    """ Plot tilted boreholes in 3D. """
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    for (x_top, y_top), tilt_rad, azimuth_rad in zip(collars, tilts, azimuths):
        z_top = D  # depth offset

        # Tilt direction
        vx = sin(tilt_rad) * cos(azimuth_rad)
        vy = sin(tilt_rad) * sin(azimuth_rad)
        vz = cos(tilt_rad)

        # Bottom of borehole
        x_bot = x_top + vx * H
        y_bot = y_top + vy * H
        z_bot = z_top + vz * H

        ax.plot([x_top, x_bot], [y_top, y_bot], [z_top, z_bot], lw=1.5)
        ax.scatter([x_top], [y_top], [z_top], c="k", marker="o", s=12)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Depth [m]")
    ax.invert_zaxis()
    ax.set_title("Single Drill Pad (Tilted Boreholes)")
    plt.tight_layout()
    plt.show()


# --------------------------------------
# Example usage: ONE pad, radius = 12 m
# --------------------------------------

nbh = 13            # number of boreholes
tilt_deg = 15       # tilt angle in degrees
tilt_rad = tilt_deg * 3.14159 / 180
radius = 12         # **YOUR REQUEST**
center = (0.0, 0.0) # pad center

coords, tilts, azimuths = single_drill_pad(
    nbh=nbh,
    tilt=tilt_rad,
    radius=radius,
    center=center
)

# Plot with your chosen borehole height H and top depth D
plot_3d_borefield(coords, tilts, azimuths, H=100, D=0)
