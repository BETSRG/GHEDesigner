import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- User parameters (customize) ---
H = 100                 # borehole length (m)
theta_deg = 15          # tilt angle (degrees)
theta = np.deg2rad(theta_deg)
L = 50                  # field length (m)
n = 10                  # number of boreholes
x_entries = np.linspace(0, L, n)
signs = [1 if i % 2 == 0 else -1 for i in range(n)]

# Projections
y_proj = H * np.sin(theta)
z_depth = -H * np.cos(theta)

# Initial beta
beta0 = 0.5

# --- Create figure & subplots ---
fig, (ax_side, ax_top) = plt.subplots(1, 2, figsize=(12, 6))
# Make room at bottom for legends & slider
plt.subplots_adjust(bottom=0.18, left=0.1, right=0.9, top=0.9)

# --- Draw static tilted boreholes ---
for sign in signs:
    ax_side.plot([0, sign*y_proj], [0, z_depth], color='lightgray', linewidth=1)
ax_side.scatter([0]*n, [0]*n, color='blue', marker='o', s=50, label='Entry')
ax_side.scatter([sign*y_proj for sign in signs], [z_depth]*n,
                color='gray', marker='x', s=50, label='Tilted Bottom')

ax_side.set_xlabel('Y (m)')
ax_side.set_ylabel('Depth (m)')
ax_side.set_ylim(z_depth*1.1, 0)
ax_side.set_title(f'Side View (θ={theta_deg}°)')
# Legend just above slider
ax_side.legend(loc='upper center',
               bbox_to_anchor=(0.5, -0.08),
               bbox_transform=ax_side.transAxes,
               ncol=2, frameon=False)

for x, sign in zip(x_entries, signs):
    ax_top.plot([x, x], [0, sign*y_proj], color='black', linewidth=1)
ax_top.scatter(x_entries, [0]*n, marker='s', facecolor='skyblue',
               edgecolor='black', s=50, label='Entry')
ax_top.scatter(x_entries, [sign*y_proj for sign in signs],
               marker='^', color='black', s=50, label='Tilted Bottom')

ax_top.set_xlabel('X (m)')
ax_top.set_ylabel('Y (m)')
ax_top.set_aspect('equal', 'box')
ax_top.set_title('Top View')
# Legend just above slider
ax_top.legend(loc='upper center',
              bbox_to_anchor=(0.5, -0.08),
              bbox_transform=ax_top.transAxes,
              ncol=2, frameon=False)

# --- Dynamic: full vertical boreholes & intercept markers ---
side_dyn = []
for sign in signs:
    line, = ax_side.plot([], [], '--', color='green', linewidth=1)
    marker = ax_side.scatter([], [], marker='^', color='green', s=50,
                             label='Staggered Borehole' if sign==signs[0] else "")
    side_dyn.append((line, marker, sign))

top_dyn = []
for x, sign in zip(x_entries, signs):
    line, = ax_top.plot([], [], '--', color='green', linewidth=1)
    marker = ax_top.scatter([], [], marker='^', color='green', s=50,
                             label='Staggered Borehole' if x==x_entries[0] else "")
    top_dyn.append((line, marker, x, sign))

# Slider below legends
ax_beta = plt.axes([0.25, 0.05, 0.5, 0.03])
beta_slider = Slider(ax_beta, 'β', 0.0, 1.0, valinit=beta0, valstep=.01)

def update(val):
    b = beta_slider.val
    for line, marker, sign in side_dyn:
        y_int = sign * b * y_proj
        line.set_data([y_int, y_int], [0, z_depth])
        marker.set_offsets([y_int, b*z_depth])
    for line, marker, x, sign in top_dyn:
        y_int = sign * b * y_proj
        line.set_data([x, x], [0, y_int])
        marker.set_offsets([x, y_int])
    fig.canvas.draw_idle()

beta_slider.on_changed(update)
update(beta0)

plt.show()
