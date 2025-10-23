import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from ghedesigner.constants import TWO_PI
from ghedesigner.horizontal_pipe_heat_exchange import ParallelPipeSystem


class MockPipe:
    def __init__(self, r_out, k):
        self.r_out = r_out
        self.k = k


class MockSoil:
    def __init__(self, k, rhocp):
        self.k = k
        self.rhocp = rhocp


pipe = MockPipe(r_out=0.1, k=0.4)
soil = MockSoil(k=1.5, rhocp=2.3e6)

system = ParallelPipeSystem(x_coord=0.2, y_coord=0.8, pipe=pipe, soil=soil)

KNOWN_HEAT_LOSS = 250.0
EPSILON_X = 1.0
BETA = 0.7
YEARS = 1 / 12
HOURS_IN_YEAR = 365.25 * 24

print("Building fast interpolator... (this may take a minute)")

tau_years_precalc = np.logspace(-4, np.log10(YEARS), 100)
tau_seconds_precalc = tau_years_precalc * HOURS_IN_YEAR * 3600

q_prime_at_zero = 1.0 / BETA

q_prime_values = [system.heat_transfer(t, EPSILON_X, BETA) for t in tau_seconds_precalc]

final_tau_seconds = np.insert(tau_seconds_precalc, 0, 0.0)
final_q_prime_values = np.insert(q_prime_values, 0, q_prime_at_zero)

fast_q_prime_function = interpolate.interp1d(
    final_tau_seconds, final_q_prime_values, kind='cubic', fill_value="extrapolate"
)

print("Interpolator built successfully!")

time_array_hours = np.arange(1, (YEARS * HOURS_IN_YEAR) + 1)
time_array_seconds = time_array_hours * 3600
time_array_years = time_array_hours / HOURS_IN_YEAR

average_loss = 25.0
amplitude = 0

heat_loss_array = average_loss + amplitude * np.sin(2 * np.pi * time_array_hours / HOURS_IN_YEAR)

q_prime_plot = fast_q_prime_function(time_array_seconds)

lambda_soil = system.soil.k
T_f0_plot = heat_loss_array / (TWO_PI * lambda_soil * q_prime_plot)

print("Plotting...")
plt.figure()
plt.plot(time_array_years, T_f0_plot)
plt.title(f"Fluid Temperature for Constant Heat Loss ({KNOWN_HEAT_LOSS} W/m)")
plt.xlabel("Time (Years)")
plt.ylabel("Fluid Temperature (T_f0) [°C]")
plt.grid(True)
plt.show()
