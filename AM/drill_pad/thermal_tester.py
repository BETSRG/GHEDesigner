import numpy as np
import matplotlib.pyplot as plt
from pygfunction.boreholes import Borehole
from pygfunction import gfunction as gt
from pygfunction.utilities import time_geometric


def build_time_vector_for_gfunc(years=25.0, Nt=30):
    """
    Build a time vector (in seconds) spanning ~1 hour to `years` years,
    with geometrically expanding step sizes.
    """
    dt = 3600.0  # 1 hour
    tmax = years * 365.0 * 24.0 * 3600.0
    return time_geometric(dt=dt, tmax=tmax, Nt=Nt)


def build_borefield_for_physics(collars, tilts, orientations, H=135.0, r_b=0.07, D=2.0):
    bh_objs = [
        Borehole(
            H=H,
            D=D,
            r_b=r_b,
            x=x,
            y=y,
            tilt=tilt,
            orientation=orientation,
        )
        for (x, y), tilt, orientation in zip(collars, tilts, orientations)
    ]
    return bh_objs


def compute_and_plot_gfunc(bfield, alpha=1.0e-6):
    """
    bfield : list of Borehole objects
    alpha  : ground thermal diffusivity [m^2/s]
    """
    # 1. Build times
    time_s = build_time_vector_for_gfunc()

    # 2. Check if any boreholes are tilted to select appropriate solver
    # The 'equivalent' solver only works with vertical boreholes
    has_tilted_boreholes = any(b.is_tilted() for b in bfield)
    
    # Use 'similarities' for tilted boreholes, 'equivalent' for vertical
    method = 'similarities' if has_tilted_boreholes else 'equivalent'
    
    # 3. Compute g-function with modern pygfunction API
    gfunc_obj = gt.gFunction(
        bfield,
        alpha,
        time_s,
        method=method
    )

    g_vals = gfunc_obj.gFunc

    # 4. Plot result
    time_yr = time_s / (3600.0 * 24.0 * 365.0)
    plt.semilogx(time_yr, g_vals)
    plt.xlabel("Time [years]")
    plt.ylabel("g-function [-]")
    plt.title(f"Field Thermal Response (method: {method})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return time_s, g_vals
