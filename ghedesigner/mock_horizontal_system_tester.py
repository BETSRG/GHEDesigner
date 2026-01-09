import matplotlib.pyplot as plt
import numpy as np
import pygfunction as gt

from ghedesigner.horizontal_interpolator import ResponseInterpolator

# Import your classes
from ghedesigner.horizontal_system_solver import SystemSolver

# -------------------------------------------------------------------------
# 1. SIMULATION CONFIG
# -------------------------------------------------------------------------
HOURS = 72
DT = 3600.0
LOAD_WATTS = -4000.0  # Heating Load (Extraction)
FLOW_RATE = 0.0002  # ~3 GPM
FLUID = {'cp': 4182.0, 'rho': 998.0}

# Pulse Timing
START_HOUR = 24
STOP_HOUR = 48

# Physics
k_s = 2.0
Tg = 10.0
Rb = 0.1
alpha = 1.0e-6

# Horizontal Geometry (Fixed Cross-Section)
H_DEPTH = 1.5
H_SPACING = 0.5
H_BETA = 0.7

# SCENARIOS: (Vertical Length, Total Horizontal Length)
# We keep total length roughly 104m
scenarios = [
    (100.0, 4.0),  # Baseline: 96% Vertical
    (80.0, 24.0),  # Mixed
    (60.0, 44.0),  # Mixed
    (40.0, 64.0),  # Shallow: 60% Horizontal
]

# -------------------------------------------------------------------------
# 2. LOAD PHYSICS ENGINE
# -------------------------------------------------------------------------
time_values = np.arange(1, HOURS + 1) * DT

# Load Horizontal Factors ONCE (Geometry doesn't change, only Length scales)
print("Loading Horizontal Physics...")
try:
    interpolator = ResponseInterpolator("interpolator_table_huge.pkl")
    horiz_factors = interpolator.interpolate(H_DEPTH, H_SPACING, H_BETA, time_values)
except Exception as e:  # noqa: BLE001
    print(f"Warning: {e}. Using zeros.")
    horiz_factors = np.zeros(HOURS)

# -------------------------------------------------------------------------
# 3. RUN SCENARIOS
# -------------------------------------------------------------------------
results = {}

print(f"Running {len(scenarios)} Scenarios...")

for L_vert, L_horiz in scenarios:
    label = f"Vert: {int(L_vert)}m / Horiz: {int(L_horiz)}m"
    print(f"  > Simulating: {label}")

    # A. Recalculate Vertical g-functions for this specific Depth (H)
    # Note: Shorter boreholes have different g-functions!
    borefield = gt.boreholes.rectangle_field(N_1=1, N_2=1, B_1=5, B_2=5, H=L_vert, D=4, r_b=0.075)
    gfunc = gt.gfunction.gFunction(borefield, alpha=alpha, time=time_values)
    g_values = gfunc.gFunc

    # B. Calculate Cn for this specific borehole
    # Cn = (1/2pi*k) * g(dt) + Rb
    C_n = (1.0 / (2 * np.pi * k_s) * g_values[0]) + Rb

    # C. Setup Solver with new lengths
    geo_params = {'L_sup': L_horiz / 2.0, 'L_ret': L_horiz / 2.0, 'L_ghx': L_vert}

    # Mock pipe object for k_s access
    pipe_sys = type('obj', (object,), {'soil': type('obj', (object,), {'k': k_s})})
    solver = SystemSolver(FLUID, geo_params, pipe_sys, beta=H_BETA)

    # D. Run Loop
    temp_history = []
    history_q_ghx = []  # Local history for H_n calculation

    for i in range(HOURS):
        # 1. Define Load Profile
        if i < START_HOUR:
            current_load = 0.0  # Phase 1: Rest
        elif i < STOP_HOUR:
            current_load = LOAD_WATTS  # Phase 2: Pulse ON
        else:
            current_load = 0.0  # Phase 3: Recovery

        # Convert to r1/r2 (assuming constant COP 3.5)
        r2 = 0 if current_load == 0 else current_load * (1 - 1 / 3.5)
        r1 = 0.0

        # 2. Calculate H_n (History)
        H_n = Tg
        if i > 0:
            q_hist_per_m = np.array(history_q_ghx) / L_vert  # Normalize by CURRENT length
            q_deltas = np.diff(np.concatenate(([0], q_hist_per_m)))

            rise = 0.0
            for j, dq in enumerate(q_deltas):
                steps_ago = i - j
                if steps_ago > 0:
                    rise += dq * (g_values[steps_ago - 1] / (2 * np.pi * k_s))
            H_n = Tg + rise

        # 3. Solve
        x = solver.solve_timestep(FLOW_RATE, Tg, H_n, C_n, r1, r2, horiz_factors)

        history_q_ghx.append(x[0])
        temp_history.append(x[1])  # EFT

    results[label] = temp_history

# -------------------------------------------------------------------------
# 4. PLOT COMPARISON
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Distinct colors

for i, (label, data) in enumerate(results.items()):
    plt.plot(data, linewidth=2, label=label, color=colors[i])

plt.title(f"Geometry Impact (Fixed Total Length ~ {int(scenarios[0][0]+scenarios[0][1])}m)")
plt.xlabel("Hours")
plt.ylabel("Fluid Temperature (°C)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
