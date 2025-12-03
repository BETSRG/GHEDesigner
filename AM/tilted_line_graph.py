import numpy as np
import plotly.graph_objects as go

def plot_tilted_only(
    n=13, b=5.0, tilt_deg=15.0, x0=0, y0=0, z0=0, depth=-100
):
    tilt = np.deg2rad(tilt_deg)

    # Entry coordinates (top of boreholes)
    x_entry = x0 + np.arange(n) * b
    y_entry = np.full(n, y0)
    z_entry = np.full(n, z0)

    # Tilted ends (bottom of boreholes)
    signs = (-1)**np.arange(n)         # alternating tilt directions
    y_end = y_entry + signs * np.tan(tilt) * depth
    z_end_arr = np.full(n, z0 - depth)

    # Build tilted borehole lines
    x_lines, y_lines, z_lines = [], [], []
    for xe, ye, ze, ye2 in zip(x_entry, y_entry, z_entry, y_end):
        x_lines += [xe, xe, None]
        y_lines += [ye, ye2, None]
        z_lines += [ze, z0 - depth, None]

    # Create the plot
    fig = go.Figure()

    # 1. Tilted borehole lines
    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color='gray', width=4),
        name='Tilted Boreholes'
    ))

    # 2. Entry markers (top)
    fig.add_trace(go.Scatter3d(
        x=x_entry, y=y_entry, z=z_entry,
        mode='markers',
        marker=dict(color='blue', symbol='circle', size=5),
        name='Tilted Entry'
    ))

    # 3. End markers (bottom)
    fig.add_trace(go.Scatter3d(
        x=x_entry, y=y_end, z=z_end_arr,
        mode='markers',
        marker=dict(color='red', symbol='x', size=6),
        name='Tilted End'
    ))

    # Layout
    fig.update_layout(
        title=f"{n} Tilted Boreholes ({tilt_deg}° Tilt)",
        scene=dict(
            xaxis_title="X Position (m)",
            yaxis_title="Y Position (m)",
            zaxis_title="Depth (m)",
            zaxis=dict(autorange="reversed")  # Depth goes downward
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    fig.show()

# Run it with 13 boreholes
plot_tilted_only(n=13, b=5.0, tilt_deg=15.0)
