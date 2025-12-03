
from AM.drill_pad.drill_field_visual import plot_3d_borefield, multi_pad_field
from AM.drill_pad.thermal_tester import build_borefield_for_physics, compute_and_plot_gfunc

# 1. Geometry
collars, tilts, orients = multi_pad_field(
    pad_centers=[(0,0), (100,0), (0,100), (-100, 0), (0, -100)],
    nbh=12,
    tilt=0.2617993877992,
    radius=4.0,
    borehole_height=135.0
)

# 2. Visualize in 3D
plot_3d_borefield(
    collars,
    tilts,
    orients,
    H=135.0,  # borehole length
    D=2.0,    # burial depth to top
)

# 3. Build physics borefield
bfield = build_borefield_for_physics(
    collars=collars,
    tilts=tilts,
    orientations=orients,
    H=135.0,
    r_b=0.14/2.0,
    D=2.0,
)

# 4. Compute + plot g-function
time_s, g_vals = compute_and_plot_gfunc(
    bfield,
    alpha=1.0e-6,
)
