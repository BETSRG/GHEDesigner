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
