import math

import numpy as np
from scipy import integrate, special

from ghedesigner.constants import TWO_PI
from ghedesigner.media import Pipe, Soil


class ParallelPipeSystem:
    def __init__(
        self,
        x_coord: float,
        y_coord: float,
        # fluid_temp: float,
        # epsilon_x: float,
        # beta: float,
        pipe: Pipe,
        soil: Soil,
    ):
        self.pipe = pipe
        self.soil = soil
        # self.T_f0 = fluid_temp
        # self.epsilon_x = epsilon_x
        # self.beta = beta
        self.B = x_coord
        self.D = y_coord
        self.r_p = pipe.r_out

        kappa = soil.k / soil.rhocp
        self.characteristic_time = (self.r_p) ** 2 / kappa

    def radial_distance(self, radius, psi, n):
        pipe_one = 1
        pipe_two = 2
        pipe_three = 3
        pipe_four = 4
        if n == pipe_one:
            return radius
        elif n == pipe_two:
            return math.sqrt(4 * self.B**2 + radius**2 + 4 * self.B * radius * math.cos(psi))
        elif n == pipe_three:
            return math.sqrt(
                4 * self.B**2
                + 4 * self.D**2
                + radius**2
                + 4 * self.B * radius * math.cos(psi)
                + 4 * self.D * radius * math.sin(psi)
            )
        elif n == pipe_four:
            return math.sqrt(4 * self.D**2 + radius**2 + 4 * self.D * radius * math.sin(psi))

    def radial_distance_derivative(self, radius, psi, n):  # rename variables to make logical sense
        pipe_one = 1
        pipe_two = 2
        pipe_three = 3
        if n == pipe_one:
            return 1
        elif n == pipe_two:
            return (radius + 2 * self.B * math.cos(psi)) / (self.radial_distance(radius, psi, n))
        elif n == pipe_three:
            return (radius + 2 * self.B * math.cos(psi) + 2 * self.D * math.sin(psi)) / (
                self.radial_distance(radius, psi, n)
            )
        else:  # add error handling for n greater than 4
            return (radius + 2 * self.D * math.sin(psi)) / (self.radial_distance(radius, psi, n))

    def laplace_integral_sum(self, sigma: complex, epsilon_x: float):
        epsilons = {2: epsilon_x, 3: -epsilon_x, 4: -1.0}

        def integrand_real(psi):
            sum_val = 0.0
            for n in range(2, 5):
                dist = self.radial_distance(self.r_p, psi, n)
                argument = (dist / self.r_p) * sigma
                sum_val += epsilons[n] * special.kv(0, argument)
            return np.real(sum_val)  # Return only the real part

        def integrand_imag(psi):
            sum_val = 0.0
            for n in range(2, 5):
                dist = self.radial_distance(self.r_p, psi, n)
                argument = (dist / self.r_p) * sigma
                sum_val += epsilons[n] * special.kv(0, argument)
            return np.imag(sum_val)  # Return only the imaginary part

        real_result, _ = integrate.quad(integrand_real, -math.pi, math.pi)
        imag_result, _ = integrate.quad(integrand_imag, -math.pi, math.pi)

        integral_result = complex(real_result, imag_result)
        return (1 / (TWO_PI)) * integral_result

    def laplace_integral_sum_derivative(self, sigma: complex, epsilon_x: float):
        epsilons = {2: epsilon_x, 3: -epsilon_x, 4: -1.0}

        def integrand_real(psi):
            sum_val = 0.0
            for n in range(2, 5):
                dist = self.radial_distance(self.r_p, psi, n)
                dist_deriv = self.radial_distance_derivative(self.r_p, psi, n)
                argument = (dist / self.r_p) * sigma
                sum_val += epsilons[n] * special.kv(1, argument) * dist_deriv
            return np.real(sum_val)

        def integrand_imag(psi):
            sum_val = 0.0
            for n in range(2, 5):
                dist = self.radial_distance(self.r_p, psi, n)
                dist_deriv = self.radial_distance_derivative(self.r_p, psi, n)
                argument = (dist / self.r_p) * sigma
                sum_val += epsilons[n] * special.kv(1, argument) * dist_deriv
            return np.imag(sum_val)

        real_result, _ = integrate.quad(integrand_real, -math.pi, math.pi)
        imag_result, _ = integrate.quad(integrand_imag, -math.pi, math.pi)

        integral_result = complex(real_result, imag_result)
        return (1 / (TWO_PI)) * integral_result

    def steady_state_heat_flow(self, epsilon_x, beta):
        return 1 / (
            math.log((2 * self.D) / self.r_p)
            + epsilon_x * math.log((math.sqrt(self.B**2 + self.D**2)) / (self.B))
            + beta
        )

    def n_function(self, sigma: complex, epsilon_x: float, beta: float):
        numerator = special.kv(1, sigma) + self.laplace_integral_sum_derivative(sigma, epsilon_x)
        denominator = (
            special.kv(0, sigma)
            + self.laplace_integral_sum(sigma, epsilon_x)
            + beta * sigma * (special.kv(1, sigma) + self.laplace_integral_sum_derivative(sigma, epsilon_x))
        )

        return numerator / denominator

    def heat_transfer(self, time, epsilon_x, beta):
        # Handle the t=0 edge case first
        if time <= 0:
            return 1.0 / beta

        tau = time / self.characteristic_time

        def integrand(u):
            sigma = 1j * u

            real_n = np.real(self.n_function(sigma, epsilon_x, beta))

            return math.exp(-tau * u**2) * real_n

        integral_result, _ = integrate.quad(integrand, 0, 5 / math.sqrt(tau))

        return (2 / math.pi) * integral_result + self.steady_state_heat_flow(epsilon_x, beta)

    def calculate_fluid_temp(
        self,
        known_heat_loss: float,
        time: float,
        epsilon_x: float,
        beta: float,
    ) -> float:
        heat_flux = self.heat_transfer(time, epsilon_x, beta)

        lambda_soil = self.soil.k

        # handle divide by 0 error
        if heat_flux == 0:
            return 0

        fluid_temp = known_heat_loss / (TWO_PI * lambda_soil * heat_flux)

        return fluid_temp

    def calculate_temps_from_asymmetric_heat_loss(
        self, q_p1: float, q_p2: float, time: float, beta: float
    ) -> tuple[float, float]:
        """
        Calculates the required fluid temperatures for two different,
        asymmetric heat loss inputs (q_p1 and q_p2) at a given time.

        :param q_p1: The heat loss for pipe 1 (W/m).
        :param q_p2: The heat loss for pipe 2 (W/m).
        :param time: The time for the calculation [s].
        :param beta: The dimensionless pipe thermal resistance parameter.
        :return: A tuple containing (T_f1, T_f2), the required
                 temperatures for pipe 1 and pipe 2.
        """

        # 1. Decompose the target heat losses into even and odd components
        q_pe = (q_p1 + q_p2) / 2.0  # Even heat loss
        q_po = (q_p1 - q_p2) / 2.0  # Odd heat loss

        # 2. Get the dimensionless heat flux (q') for both basic cases
        q_prime_even = self.heat_transfer(time, epsilon_x=1.0, beta=beta)
        q_prime_odd = self.heat_transfer(time, epsilon_x=-1.0, beta=beta)

        lambda_soil = self.soil.k

        # 3. Calculate the required even and odd TEMPERATURES

        # Handle potential divide-by-zero if q' is zero
        t_fe = 0.0 if q_prime_even == 0 else q_pe / (TWO_PI * lambda_soil * q_prime_even)

        t_f0 = 0.0 if q_prime_odd == 0 else q_po / (TWO_PI * lambda_soil * q_prime_odd)

        # 4. Superpose the temperatures to find the final answer
        t_f1 = t_fe + t_f0
        t_f2 = t_fe - t_f0

        return (t_f1, t_f2)

    def simulate_temperature_response(
        self, heat_load_series: np.ndarray, response_factors: np.ndarray, beta: float
    ) -> np.ndarray:
        """
        Calculates the fluid temperature evolution for a variable heat load
        using temporal superposition (thermal history).

        :param heat_load_series: Array of heat loads (W/m) at each time step.
        :param response_factors: Array of q' values corresponding to the time steps.
                                 Must be the same length or longer than heat_load_series.
        :param beta: The dimensionless pipe thermal resistance parameter.
        :return: Array of fluid temperatures (T_f0) in Celsius.
        """

        num_steps = len(heat_load_series)
        temperature_history = np.zeros(num_steps)
        delta_t_history = []  # Stores the step-changes in temperature

        # Calculate q'(0) - the immediate response
        # From PDF page 7, q'(0) = 1 / beta
        q_prime_0 = 1.0 / beta

        lambda_soil = self.soil.k
        current_t = 0.0

        # Run the superposition loop
        for n in range(num_steps):
            # 1. Current Load Term
            q_n = heat_load_series[n]
            load_term = q_n / (TWO_PI * lambda_soil)

            # 2. History Term (Convolution of past changes * response)
            history_term = 0.0
            if n > 0:
                # We need q' values for elapsed times [dt, 2dt, ... (n)*dt]
                # These correspond to indices [0, 1, ... n-1] in the response_factors array
                relevant_q = response_factors[:n]

                # Reverse deltas so the most recent change multiplies the earliest q'
                reversed_deltas = delta_t_history[::-1]

                # Use dot product for speed
                history_term = np.dot(reversed_deltas, relevant_q)

            # 3. Solve for NEW temperature step
            # Formula derived from superposition of step pulses
            delta_t = (load_term - history_term) / q_prime_0

            # 4. Update state
            current_t += delta_t

            delta_t_history.append(delta_t)
            temperature_history[n] = current_t

        return temperature_history


class SinglePipeSystem:
    def __init__(
        self,
        pipe: Pipe,
        soil: Soil,
    ):
        self.pipe = pipe
        self.soil = soil
        self.r_p = pipe.r_out

    # add functions from page 10 for single pipe system
