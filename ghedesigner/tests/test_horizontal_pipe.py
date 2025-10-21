import math

import numpy as np
import pytest

from ghedesigner.horizontal_pipe_heat_exchange import ParallelPipeSystem


# create mock classes that only contain needed variables
class MockPipe:
    def __init__(self, r_out, k):
        self.r_out = r_out
        self.k = k


class MockSoil:
    def __init__(self, k, rhocp):
        self.k = k
        self.rhocp = rhocp


# Create test objects using simple mock classes
pipe = MockPipe(r_out=0.1, k=0.4)
soil = MockSoil(k=1.5, rhocp=2.3e6)


@pytest.fixture
def pipe_system():
    system = ParallelPipeSystem(x_coord=0.2, y_coord=0.8, pipe=pipe, soil=soil)
    return system


def test_radial_distance(pipe_system):
    known_output_n3 = 1.746424919657298
    radius = 0.1
    psi = math.pi / 2

    calculated_output_n3 = pipe_system.radial_distance(radius, psi, 3)

    assert calculated_output_n3 == pytest.approx(known_output_n3)


def test_radial_distance_deriv(pipe_system):
    known_output_n4 = 0.8804612902116
    radius = 0.1
    psi = math.pi / 3

    calculated_output_n4 = pipe_system.radial_distance_derivative(radius, psi, 4)

    assert calculated_output_n4 == pytest.approx(known_output_n4)


def test_n_function_real_even_case(pipe_system):
    known_output_real = 0.248010569636388
    epsilon_x = 1
    beta = 0.7
    u = 2.0
    sigma = 1j * u

    result_complex = pipe_system.n_function(sigma, epsilon_x, beta)
    calculated_output_real = np.real(result_complex)

    assert calculated_output_real == pytest.approx(known_output_real, rel=1e-3)


def test_n_function_real_odd_case(pipe_system):
    known_output_real = 0.344060649883869
    epsilon_x = -1
    beta = 0.7
    u = 2.0
    sigma = 1j * u

    result_complex = pipe_system.n_function(sigma, epsilon_x, beta)
    calculated_output_real = np.real(result_complex)

    assert calculated_output_real == pytest.approx(known_output_real, rel=1e-3)


def test_heat_transfer_even_case(pipe_system):
    known_heat_transfer_output = 0.206198804352911
    epsilon_x = 1
    beta = 0.7
    time = 10

    calculated_heat_transfer_output = pipe_system.heat_transfer(time, epsilon_x, beta)

    assert calculated_heat_transfer_output == pytest.approx(known_heat_transfer_output)


def test_heat_transfer_odd_case(pipe_system):
    known_heat_transfer_output = 0.486392129985014
    epsilon_x = -1
    beta = 0.7
    time = 10

    calculated_heat_transfer_output = pipe_system.heat_transfer(time, epsilon_x, beta)

    assert calculated_heat_transfer_output == pytest.approx(known_heat_transfer_output)


def test_heat_transfer_initial_conditions(pipe_system):
    epsilon_x = 1
    beta = 0.7
    time = 0

    known_output = 1.0 / beta

    # Test for t=0
    calculated_output = pipe_system.heat_transfer(time, epsilon_x, beta)

    assert calculated_output == pytest.approx(known_output)


def test_heat_transfer_after_long_time(pipe_system):
    epsilon_x = 1
    beta = 0.7
    time = 1e12

    known_steady_state_value = pipe_system.steady_state_heat_flow(epsilon_x, beta)

    calculated_steady_state_value = pipe_system.heat_transfer(time, epsilon_x, beta)

    assert calculated_steady_state_value == pytest.approx(known_steady_state_value)
