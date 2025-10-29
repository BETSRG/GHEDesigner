import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from ghedesigner.constants import TWO_PI
from ghedesigner.horizontal_pipe_heat_exchange import ParallelPipeSystem


# --- Mock Classes ---
class MockPipe:
    def __init__(self, r_out, k):
        self.r_out = r_out
        self.k = k


class MockSoil:
    def __init__(self, k, rhocp):
        self.k = k
        self.rhocp = rhocp


# --- Base Objects ---
pipe = MockPipe(r_out=0.1, k=0.4)
soil = MockSoil(k=1.5, rhocp=2.3e6)

# --- Base System (for the first 5 plots) ---
system = ParallelPipeSystem(x_coord=0.2, y_coord=0.8, pipe=pipe, soil=soil)

# --- Simulation Parameters ---
BETA = 0.7
YEARS = 30
HOURS_IN_YEAR = 365.25 * 24
KNOWN_HEAT_LOSS = 25.0


# --- Interpolator Function (FIXED) ---
# Now takes the system object as an argument
def build_interpolator(system_obj, epsilon_x_case):
    print(f"Building interpolator for epsilon_x = {epsilon_x_case} at depth = {system_obj.D}...")
    start_time = time.perf_counter()

    tau_years_precalc = np.logspace(-4, np.log10(YEARS), 100)
    tau_seconds_precalc = tau_years_precalc * HOURS_IN_YEAR * 3600

    q_prime_at_zero = 1.0 / BETA

    # Use the passed-in system_obj to do the calculation
    q_prime_values = [system_obj.heat_transfer(t, epsilon_x_case, BETA) for t in tau_seconds_precalc]

    final_tau_seconds = np.insert(tau_seconds_precalc, 0, 0.0)
    final_q_prime_values = np.insert(q_prime_values, 0, q_prime_at_zero)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"  Interpolator built in {elapsed_time:.2f} seconds.")

    return interpolate.interp1d(final_tau_seconds, final_q_prime_values, kind='cubic', fill_value="extrapolate")


# --- Build interpolators for the base system ---
fast_q_prime_even = build_interpolator(system, epsilon_x_case=1.0)
fast_q_prime_odd = build_interpolator(system, epsilon_x_case=-1.0)
print("Base interpolators built successfully!")

# --- Generate data for the base plots ---
time_array_hours = np.arange(1, (YEARS * HOURS_IN_YEAR) + 1)
time_array_seconds = time_array_hours * 3600
time_array_years = time_array_hours / HOURS_IN_YEAR

q_prime_plot_even = fast_q_prime_even(time_array_seconds)
q_prime_plot_odd = fast_q_prime_odd(time_array_seconds)

u_values = np.linspace(0.01, 25, 500)
n_func_real_even = []
n_func_real_odd = []

print("Calculating n_function values for base system...")
start_time_n_func = time.perf_counter()
for u in u_values:
    sigma = 1j * u
    n_even = system.n_function(sigma, epsilon_x=1.0, beta=BETA)
    n_func_real_even.append(np.real(n_even))
    n_odd = system.n_function(sigma, epsilon_x=-1.0, beta=BETA)
    n_func_real_odd.append(np.real(n_odd))
end_time_n_func = time.perf_counter()
print(f"n_function values calculated in {end_time_n_func - start_time_n_func:.2f} seconds.")

print("Calculating fluid temperatures for base system...")
lambda_soil = system.soil.k
T_f0_plot_even = KNOWN_HEAT_LOSS / (TWO_PI * lambda_soil * q_prime_plot_even)
T_f0_plot_odd = KNOWN_HEAT_LOSS / (TWO_PI * lambda_soil * q_prime_plot_odd)

print("Plotting base graphs...")

# --- Plot 1: q' vs. Time (log scale) ---
plt.figure(1)
plt.plot(time_array_years, q_prime_plot_even, label='Even Case ($\epsilon_x=1$)')
plt.plot(time_array_years, q_prime_plot_odd, label='Odd Case ($\epsilon_x=-1$)')
plt.title("Dimensionless Heat Transfer (q') Over Time (log scale)")
plt.xlabel("Time (Years)")
plt.ylabel("Dimensionless Heat Transfer (q')")
plt.legend()
plt.grid(True)
plt.xscale('log')

# --- Plot 2: n_function vs. u ---
plt.figure(2)
plt.plot(u_values, n_func_real_even, label='Even Case ($\epsilon_x=1$)')
plt.plot(u_values, n_func_real_odd, label='Odd Case ($\epsilon_x=-1$)')
plt.title("Real Part of N-Function vs. 'u'")
plt.xlabel("u (Integration Variable)")
plt.ylabel("Re(N(i*u, $\epsilon_x$))")
plt.legend()
plt.grid(True)

# --- Plot 3: q' vs. Time (linear scale) ---
plt.figure(3)
plt.plot(time_array_years, q_prime_plot_even, label='Even Case ($\epsilon_x=1$)')
plt.plot(time_array_years, q_prime_plot_odd, label='Odd Case ($\epsilon_x=-1$)')
plt.title("Dimensionless Heat Transfer (q') Over Time (linear scale)")
plt.xlabel("Time (Years)")
plt.ylabel("Dimensionless Heat Transfer (q')")
plt.legend()
plt.grid(True)

# --- Plot 4: Temperature vs. Time (linear scale) ---
plt.figure(4)
plt.plot(time_array_years, T_f0_plot_even, label='Even Case ($\epsilon_x=1$)')
plt.plot(time_array_years, T_f0_plot_odd, label='Odd Case ($\epsilon_x=-1$)')
plt.title(f"Fluid Temperature for Constant Heat Loss ({KNOWN_HEAT_LOSS} W/m)")
plt.xlabel("Time (Years)")
plt.ylabel("Fluid Temperature (T_f0) [°C]")
plt.legend()
plt.grid(True)

# --- Plot 5: Temperature vs. Time (log scale) ---
plt.figure(5)
plt.plot(time_array_years, T_f0_plot_even, label='Even Case ($\epsilon_x=1$)')
plt.plot(time_array_years, T_f0_plot_odd, label='Odd Case ($\epsilon_x=-1$)')
plt.title(f"Fluid Temperature for Constant Heat Loss ({KNOWN_HEAT_LOSS} W/m) (log scale)")
plt.xlabel("Time (Years)")
plt.ylabel("Fluid Temperature (T_f0) [°C]")
plt.legend()
plt.grid(True)
plt.xscale('log')

# --- Start of Multi-Depth Plot Logic ---
print("\n--- Starting Multi-Depth Plot Calculation ---")
depths_to_plot = [0.8, 5.0, 25.0, 50.0]

interpolators = {}
systems = {}
for depth in depths_to_plot:
    systems[depth] = ParallelPipeSystem(x_coord=0.2, y_coord=depth, pipe=pipe, soil=soil)
    # Correctly pass the new system object and epsilon_x
    interpolators[depth] = build_interpolator(systems[depth], 1.0)

print("All depth interpolators built successfully!")

# --- Calculate Temperature Curves for Each Depth ---
temperature_plots = {}
print("Calculating temperature curves for each depth...")
for depth in depths_to_plot:
    q_prime_plot = interpolators[depth](time_array_seconds)
    temperature_plots[depth] = KNOWN_HEAT_LOSS / (TWO_PI * lambda_soil * q_prime_plot)

# --- Plot 6: The Multi-Depth Plot ---
print("Plotting multi-depth graph...")
plt.figure(6, figsize=(10, 6))  # Give it a new figure number

for depth in depths_to_plot:
    plt.plot(time_array_years, temperature_plots[depth], label=f'Depth D = {depth} m')

plt.title(
    f"Fluid Temperature Over Time for Different Depths\n(Constant Heat Loss={KNOWN_HEAT_LOSS} W/m, Even Case $\epsilon_x=1$)"
)
plt.xlabel("Time (Years)")
plt.ylabel("Fluid Temperature (T_f0) [°C]")
plt.legend()
plt.grid(True)
plt.xscale('log')

# --- Plot 7: The Multi-Depth Plot linear ---
print("Plotting multi-depth graph linear...")
plt.figure(7, figsize=(10, 6))  # Give it a new figure number

for depth in depths_to_plot:
    plt.plot(time_array_years, temperature_plots[depth], label=f'Depth D = {depth} m')

plt.title(
    f"Fluid Temperature Over Time for Different Depths\n(Constant Heat Loss={KNOWN_HEAT_LOSS} W/m, Even Case $\epsilon_x=1$)"
)
plt.xlabel("Time (Years)")
plt.ylabel("Fluid Temperature (T_f0) [°C]")
plt.legend()
plt.grid(True)

# --- Show All Plots ---
plt.show()
