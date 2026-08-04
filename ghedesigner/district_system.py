import copy
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from ghedesigner.constants import DAYS_IN_YEAR, HOURS_IN_YEAR, PI_OVER_2, SEC_IN_DAY, SEC_IN_HR, SEC_IN_YEAR, TWO_PI, HORZ_LIBRARY_FILENAME
from ghedesigner.enums import BHType, CentralLoopType, SimCompType, SourceSinkOpMode
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.boreholes.factory import get_bhe_object

from ghedesigner.ghe.gfunction import calc_g_func_for_multiple_lengths, GFunction
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.utilities import combine_sts_lts, get_loads, load_input_file, eskilson_log_times, float_tuple_to_string
from scipy import interpolate

# Horizontal Piping
import json
import math
import pickle
import time
from importlib import resources
from itertools import product
from math import cos, isclose, sin
from pathlib import Path
from typing import Any, cast


class DynamicAggregator:
    """
    Implements dynamic load aggregation to reduce temporal superposition
    complexity using expanding time bins.
    """

    def __init__(
        self,
        total_sim_time_sec: float,
        exp_rate: float = 1.62,
        bins_per_level: int = 9,
        base_dt_sec: float = 3600.0,
    ):
        self.exp_rate = exp_rate
        self.bins_per_level = bins_per_level

        dt = base_dt_sec
        t = 0.0
        dts_list = []

        # Initialize expanding bins covering the full simulation runtime
        while t < total_sim_time_sec + base_dt_sec:
            for _ in range(bins_per_level):
                t += dt
                dts_list.append(dt)
                if t >= total_sim_time_sec + base_dt_sec:
                    break
            dt *= exp_rate

        self.dts = np.array(dts_list, dtype=float)
        self.num_bins = len(self.dts)
        self.energy_bins = np.zeros(self.num_bins, dtype=float)

        # Pre-compute bin ages for evaluating response functions
        self.bin_ages = np.cumsum(self.dts)
        self.last_idx = 0

    def shift_and_add(self, new_value: float, current_dt_sec: float, idx_timestep: int):
        """
        Shifts historical values further into the load history and adds the new timestep value.
        Tracking last_idx ensures idempotent shifts if called multiple times by coupled components.
        """
        if idx_timestep > self.last_idx:
            frac_shift = current_dt_sec / self.dts
            frac_shift[-1] = 0.0
            delta = self.energy_bins * frac_shift
            self.energy_bins = self.energy_bins - delta + np.roll(delta, 1)
            self.energy_bins[0] += new_value * current_dt_sec
            self.last_idx = idx_timestep

    def get_step_changes(self) -> np.ndarray:
        """Returns the discrete step changes between consecutive averaged bins."""
        avg_vals = self.energy_bins / self.dts
        return -np.diff(avg_vals, append=0.0)


class BaseSimComp(ABC):
    def __init__(self) -> None:
        self.name: str | None = None
        self.comp_type: SimCompType | None = None
        self.matrix_size: int | None = None
        self.row_index: int | None = None
        self.downstream_index: int | None = None
        self.inlet_index: int | None = None
        self.downstream_device = None

    @abstractmethod
    def generate_matrix(self, mass_bldg, mass_loop, mass_loop_bldg, mass_flow_ghe, mass_loop_ghe, idx_timestep: int, configuration, method):
        pass

    def calc_energy(self) -> None:
        pass


class IsolatedHorizontalPipe(BaseSimComp):
    def __init__(
        self,
        name: str,
        length: float,
        num_segments: int,
        pipe: Pipe,
        soil: Soil,
        fluid: Fluid,
        num_timesteps: int,
        time_array: np.ndarray,
        q_prime_interp,
        beta: float,
        ugt_avg: float,
        ugt_amp1: float,
        ugt_phase1: float,
        ugt_amp2: float,
        ugt_phase2: float,
        depth: float,
        load_method: str = "hourly",
    ):
        super().__init__()
        self.name = name
        self.comp_type = None
        self.num_timesteps = num_timesteps
        self.time_array = time_array

        self.num_segments = num_segments
        self.matrix_rows = 3 * num_segments + 1

        self.q_prime_interp = q_prime_interp
        self.beta = beta
        self.soil = soil
        self.fluid = fluid
        self.cp = fluid.cp


        # UGT Model Parameters (Two-Harmonic)
        self.ugt_avg = ugt_avg
        self.ugt_amp1 = ugt_amp1
        self.ugt_phase1 = ugt_phase1
        self.ugt_amp2 = ugt_amp2
        self.ugt_phase2 = ugt_phase2
        self.depth = depth
        self.alpha_s = self.soil.k / self.soil.rho_cp
        if isinstance(pipe.r_out, list):
            raise TypeError("Expected pipe.r_out to be a float, but got a list.")

        self.t_p = (pipe.r_out**2) / self.alpha_s

        # Geometry & Discretization
        self.length = length
        self.L_seg = length / float(self.num_segments)
        self.V_seg = np.pi * cast(float, pipe.r_in) ** 2 * self.L_seg
        self.C_f_seg = self.V_seg * fluid.rho * self.cp  # * 2.2 #testing value
        self.two_pi_k = TWO_PI * self.soil.k

        initial_ugt = self.calculate_current_ugt(self.time_array[0] * SEC_IN_HR)

        # --- DYNAMIC STATE ARRAYS ---
        self.t_mean_seg = np.full((self.num_segments, num_timesteps), initial_ugt, dtype=float)
        self.q_seg = np.zeros((self.num_segments, num_timesteps), dtype=float)
        self.dtheta_seg = np.zeros((self.num_segments, num_timesteps), dtype=float)  # Tracking temp steps
        self.t_out_seg = np.full((self.num_segments, num_timesteps), initial_ugt, dtype=float)
        self.history_term_seg = np.zeros((self.num_segments, num_timesteps), dtype=float)

        self.t_in = np.full(num_timesteps, initial_ugt, dtype=float)
        self.t_out = np.full(num_timesteps, initial_ugt, dtype=float)
        self.y_n = np.zeros(num_timesteps, dtype=float)

        # for bidirectional flow
        self.temp_index_one = None
        self.temp_index_mean = None
        self.index_q = None
        self.temp_index_two = None
        self.mass_flow_rate = None
        self.q = np.zeros((self.num_segments, num_timesteps), dtype=float)
        self.t_mean = np.full((self.num_segments, num_timesteps), initial_ugt, dtype=float)

        # Corresponding hydraulic NetworkPipe
        self.network_pipe = None

        # Initialize load aggregation if specified
        self.load_method = load_method
        if self.load_method == "hourlyloadagg":
            total_sim_time_sec = (self.time_array[-1] - self.time_array[0]) * SEC_IN_HR
            self.aggregators = [
                DynamicAggregator(total_sim_time_sec, exp_rate=1.62, bins_per_level=9, base_dt_sec=SEC_IN_HR)
                for _ in range(self.num_segments)
            ]
            tau_agg = self.aggregators[0].bin_ages / self.t_p
            self.y_agg_evals = self.two_pi_k * self.q_prime_interp(tau_agg)

    def calculate_current_ugt(self, current_time_sec: float) -> float:
        t_days = current_time_sec / (24.0 * 3600.0)
        t_p = 365.0

        t_p_sec = 365.0 * 24.0 * 3600.0
        attenuation1 = self.depth * math.sqrt((1.0 * math.pi) / (self.alpha_s * t_p_sec))
        attenuation2 = self.depth * math.sqrt((2.0 * math.pi) / (self.alpha_s * t_p_sec))

        term1 = (
            math.exp(-attenuation1)
            * self.ugt_amp1
            * math.cos(((2.0 * math.pi * 1.0) / t_p) * (t_days - self.ugt_phase1) - attenuation1)
        )
        term2 = (
            math.exp(-attenuation2)
            * self.ugt_amp2
            * math.cos(((2.0 * math.pi * 2.0) / t_p) * (t_days - self.ugt_phase2) - attenuation2)
        )

        return self.ugt_avg - term1 - term2

    def compute_history_terms(self, idx_timestep: int):
        if getattr(self, "load_method", "hourly") == "hourlyloadagg":
            if idx_timestep > 1:
                dt_sec = (self.time_array[idx_timestep - 1] - self.time_array[idx_timestep - 2]) * SEC_IN_HR
                prev_time_sec = self.time_array[idx_timestep - 1] * SEC_IN_HR
                prev_ugt = self.calculate_current_ugt(prev_time_sec)

                for k in range(self.num_segments):
                    theta_prev = self.t_mean_seg[k, idx_timestep - 1] - prev_ugt
                    self.aggregators[k].shift_and_add(theta_prev, dt_sec, idx_timestep)
                    dtheta_b = self.aggregators[k].get_step_changes()
                    self.history_term_seg[k, idx_timestep] = np.dot(dtheta_b, self.y_agg_evals)

            current_dt_sec = (self.time_array[idx_timestep] - self.time_array[idx_timestep - 1]) * SEC_IN_HR
            current_tau = current_dt_sec / self.t_p
            self.y_n[idx_timestep] = self.two_pi_k * self.q_prime_interp(current_tau)
            return

        y_transient_array = np.zeros(idx_timestep, dtype=float)

        if idx_timestep > 0:
            dt_sec_array = (self.time_array[idx_timestep] - self.time_array[0:idx_timestep]) * SEC_IN_HR
            tau_array = dt_sec_array / self.t_p  # Convert to dimensionless time

            # Ask for q' using tau
            q_prime_array = self.q_prime_interp(tau_array)
            y_transient_array[0:idx_timestep] = self.two_pi_k * q_prime_array

        current_dt_sec = (self.time_array[idx_timestep] - self.time_array[idx_timestep - 1]) * SEC_IN_HR
        current_tau = current_dt_sec / self.t_p  # Convert to dimensionless time

        # Ask for q' using tau
        q_prime_current = self.q_prime_interp(current_tau)
        self.y_n[idx_timestep] = self.two_pi_k * q_prime_current

        for k in range(self.num_segments):
            sum_k = np.dot(self.dtheta_seg[k, 1:idx_timestep], y_transient_array[0 : idx_timestep - 1])

            self.history_term_seg[k, idx_timestep] = sum_k

    def generate_matrix(
        self,
        _mass_bldg,
        mass_loop,
        _mass_loop_bldg,
        mass_flow_pipe,
        _mass_loop_ghe,
        idx_timestep,
        configuration,
        _method,
    ):
        self.compute_history_terms(idx_timestep)

        rows = [np.zeros(self.matrix_size, dtype=np.float64) for _ in range(self.matrix_rows)]
        rhs = [0.0 for _ in range(self.matrix_rows)]

        dt_sec = (self.time_array[idx_timestep] - self.time_array[idx_timestep - 1]) * SEC_IN_HR
        cap_coeff = self.C_f_seg / dt_sec
        m_cp = mass_flow_pipe * self.cp
        yn = self.y_n[idx_timestep]

        idx_t_in = self.row_index
        idx_t_out_final = self.row_index + 3 * self.num_segments

        # for bidirectional flow
        if configuration == CentralLoopType.TWOPIPE_RING:
            if self.num_segments != 1:
                raise NotImplementedError(
                    "TWOPIPE_RING currently supports one "
                    "horizontal-pipe segment."
                )
            if mass_flow_pipe >= 0.0:
                idx_t_in = self.temp_index_one
                idx_t_out_final = self.temp_index_two
            else:
                idx_t_in = self.temp_index_two
                idx_t_out_final = self.temp_index_one

        elif configuration == CentralLoopType.ONEPIPE:
            rows[0][idx_t_in] = (mass_loop - mass_flow_pipe) * self.cp
            rows[0][idx_t_out_final] = m_cp
            rows[0][self.downstream_index] = -mass_loop * self.cp

        elif configuration == CentralLoopType.TWOPIPE:
            rows[0][idx_t_in] = 1.0
            rows[0][self.inlet_index] = -1

        current_time_sec = self.time_array[idx_timestep] * SEC_IN_HR
        prev_time_sec = self.time_array[idx_timestep - 1] * SEC_IN_HR
        current_ugt = self.calculate_current_ugt(current_time_sec)
        prev_ugt = self.calculate_current_ugt(prev_time_sec)

        for k in range(self.num_segments):
            if configuration == CentralLoopType.TWOPIPE_RING:
                idx_t_m = self.temp_index_mean
                idx_q = self.index_q
                idx_t_in_seg = idx_t_in
                idx_t_out = idx_t_out_final
            else:
                idx_t_m = self.row_index + 3 * k + 1
                idx_q = self.row_index + 3 * k + 2
                idx_t_out = self.row_index + 3 * k + 3
                idx_t_in_seg = self.row_index if k == 0 else self.row_index + 3 * (k - 1) + 3

            t_m_prev = self.t_mean_seg[k, idx_timestep - 1]

            # Eq 1: Ground Admittance formulation
            rows[3 * k + 1][idx_q] = 1.0
            rows[3 * k + 1][idx_t_m] = -yn
            rhs[3 * k + 1] = self.history_term_seg[k, idx_timestep] + yn * (-current_ugt - t_m_prev + prev_ugt)

            # Eq 2: Mean Temp
            rows[3 * k + 2][idx_t_in_seg] = -1.0
            rows[3 * k + 2][idx_t_m] = 2.0
            rows[3 * k + 2][idx_t_out] = -1.0

            # Eq 3: Energy Bal w/ Capacitance
            rows[3 * k + 3][idx_t_in_seg] = abs(m_cp)
            rows[3 * k + 3][idx_t_out] = -abs(m_cp)
            rows[3 * k + 3][idx_q] = -self.L_seg
            rows[3 * k + 3][idx_t_m] = -cap_coeff
            rhs[3 * k + 3] = -cap_coeff * t_m_prev

        return rows, rhs

    def update_post_solve(self, x_vector, idx_timestep, configuration):
        current_time_sec = self.time_array[idx_timestep] * SEC_IN_HR
        prev_time_sec = self.time_array[idx_timestep - 1] * SEC_IN_HR
        current_ugt = self.calculate_current_ugt(current_time_sec)
        prev_ugt = self.calculate_current_ugt(prev_time_sec)

        if configuration == CentralLoopType.TWOPIPE_RING:
            if self.network_pipe.mass_flow_rate >= 0.0:
                idx_t_in = self.temp_index_one
                idx_t_out = self.temp_index_two
            else:
                idx_t_in = self.temp_index_two
                idx_t_out = self.temp_index_one

            self.t_in[idx_timestep] = x_vector[idx_t_in]
            self.t_mean_seg[0, idx_timestep] = x_vector[self.temp_index_mean]
            self.q_seg[0, idx_timestep] = x_vector[self.index_q]
            self.t_out_seg[0, idx_timestep] = x_vector[idx_t_out]
        else:
            self.t_in[idx_timestep] = x_vector[self.row_index]

            for k in range(self.num_segments):
                self.t_mean_seg[k, idx_timestep] = x_vector[self.row_index + 3 * k + 1]
                self.q_seg[k, idx_timestep] = x_vector[self.row_index + 3 * k + 2]
                self.t_out_seg[k, idx_timestep] = x_vector[self.row_index + 3 * k + 3]

        for k in range(self.num_segments):
            # Calculate and store the discrete driving potential step (dtheta) that just occurred
            theta_n = self.t_mean_seg[k, idx_timestep] - current_ugt
            theta_n_minus_1 = self.t_mean_seg[k, idx_timestep - 1] - prev_ugt
            self.dtheta_seg[k, idx_timestep] = theta_n - theta_n_minus_1

        self.t_out[idx_timestep] = self.t_out_seg[-1, idx_timestep]


class CoupledHorizontalPipe(BaseSimComp):
    def __init__(
        self,
        name: str,
        length: float,
        num_segments: int,
        pipe: Pipe,
        soil: Soil,
        fluid: Fluid,
        num_timesteps: int,
        time_array: np.ndarray,
        q_prime_even_interp,
        q_prime_odd_interp,
        beta: float,
        ugt_avg: float,
        ugt_amp1: float,
        ugt_phase1: float,
        ugt_amp2: float,
        ugt_phase2: float,
        depth: float,
        counter_flow: bool = False,
        load_method: str = "hourly",
    ):
        super().__init__()
        self.name = name
        self.comp_type = SimCompType.COUPLED_HORIZONTAL_PIPE
        self.counter_flow = counter_flow
        self.num_timesteps = num_timesteps
        self.time_array = time_array
        self.num_segments = num_segments
        self.matrix_rows = 3 * num_segments + 1

        self.q_prime_even_interp = q_prime_even_interp
        self.q_prime_odd_interp = q_prime_odd_interp

        self.coupled_pipe: CoupledHorizontalPipe | None = None

        self.beta = beta
        self.soil = soil
        self.fluid = fluid
        self.cp = fluid.cp

        self.ugt_avg = ugt_avg
        self.ugt_amp1 = ugt_amp1
        self.ugt_phase1 = ugt_phase1
        self.ugt_amp2 = ugt_amp2
        self.ugt_phase2 = ugt_phase2
        self.depth = depth
        self.alpha_s = self.soil.k / self.soil.rho_cp
        if isinstance(pipe.r_out, list):
            raise TypeError("Expected pipe.r_out to be a float, but got a list.")

        self.t_p = (pipe.r_out**2) / self.alpha_s

        self.length = length
        self.L_seg = length / float(self.num_segments)
        self.V_seg = np.pi * cast(float, pipe.r_in) ** 2 * self.L_seg
        self.C_f_seg = self.V_seg * fluid.rho * self.cp  # * 2.2 #testing value
        self.two_pi_k = TWO_PI * self.soil.k

        initial_ugt = self.calculate_current_ugt(self.time_array[0] * SEC_IN_HR)

        self.t_mean_seg = np.full((self.num_segments, num_timesteps), initial_ugt, dtype=float)
        self.q_seg = np.zeros((self.num_segments, num_timesteps), dtype=float)
        self.dtheta_seg = np.zeros((self.num_segments, num_timesteps), dtype=float)
        self.t_out_seg = np.full((self.num_segments, num_timesteps), initial_ugt, dtype=float)
        self.history_term_seg = np.zeros((self.num_segments, num_timesteps), dtype=float)

        self.t_in = np.full(num_timesteps, initial_ugt, dtype=float)
        self.t_out = np.full(num_timesteps, initial_ugt, dtype=float)

        self.y_n = np.zeros(num_timesteps, dtype=float)
        self.y_cross = np.zeros(num_timesteps, dtype=float)

        # for bidirectional flow
        self.temp_index_one = None
        self.temp_index_two = None
        self.temp_index_mean = None
        self.pipe_heat_rejection_index = None

        self.load_method = load_method
        if self.load_method == "hourlyloadagg":
            total_sim_time_sec = (self.time_array[-1] - self.time_array[0]) * SEC_IN_HR
            self.aggregators = [
                DynamicAggregator(total_sim_time_sec, exp_rate=1.62, bins_per_level=9, base_dt_sec=SEC_IN_HR)
                for _ in range(self.num_segments)
            ]
            tau_agg = self.aggregators[0].bin_ages / self.t_p
            y_even_agg = self.two_pi_k * self.q_prime_even_interp(tau_agg)
            y_odd_agg = self.two_pi_k * self.q_prime_odd_interp(tau_agg)
            self.y_self_agg_evals = (y_even_agg + y_odd_agg) / 2.0
            self.y_cross_agg_evals = (y_even_agg - y_odd_agg) / 2.0

    def calculate_current_ugt(self, current_time_sec: float) -> float:
        t_days = current_time_sec / SEC_IN_DAY
        t_p = DAYS_IN_YEAR
        t_p_sec = SEC_IN_YEAR
        attenuation1 = self.depth * math.sqrt((1.0 * math.pi) / (self.alpha_s * t_p_sec))
        attenuation2 = self.depth * math.sqrt((2.0 * math.pi) / (self.alpha_s * t_p_sec))
        term1 = (
            math.exp(-attenuation1)
            * self.ugt_amp1
            * math.cos(((2.0 * math.pi * 1.0) / t_p) * (t_days - self.ugt_phase1) - attenuation1)
        )
        term2 = (
            math.exp(-attenuation2)
            * self.ugt_amp2
            * math.cos(((2.0 * math.pi * 2.0) / t_p) * (t_days - self.ugt_phase2) - attenuation2)
        )
        return self.ugt_avg - term1 - term2

    def compute_history_terms(self, idx_timestep: int):
        if self.coupled_pipe is None:
            raise ValueError("History terms cannot be computed without a defined coupled pipe.")

        if getattr(self, "load_method", "hourly") == "hourlyloadagg":
            if idx_timestep > 1:
                dt_sec = (self.time_array[idx_timestep - 1] - self.time_array[idx_timestep - 2]) * SEC_IN_HR
                prev_time_sec = self.time_array[idx_timestep - 1] * SEC_IN_HR
                prev_ugt = self.calculate_current_ugt(prev_time_sec)

                for k in range(self.num_segments):
                    # Safely shift self
                    theta_prev = self.t_mean_seg[k, idx_timestep - 1] - prev_ugt
                    self.aggregators[k].shift_and_add(theta_prev, dt_sec, idx_timestep)

                    # Safely shift neighbor ensuring asynchronous state consistency
                    neighbor_k = self.num_segments - 1 - k if self.counter_flow else k
                    theta_prev_neighbor = self.coupled_pipe.t_mean_seg[neighbor_k, idx_timestep - 1] - prev_ugt
                    self.coupled_pipe.aggregators[neighbor_k].shift_and_add(theta_prev_neighbor, dt_sec, idx_timestep)

                    dtheta_b_self = self.aggregators[k].get_step_changes()
                    sum_self = np.dot(dtheta_b_self, self.y_self_agg_evals)

                    dtheta_b_cross = self.coupled_pipe.aggregators[neighbor_k].get_step_changes()
                    sum_cross = np.dot(dtheta_b_cross, self.y_cross_agg_evals)

                    self.history_term_seg[k, idx_timestep] = sum_self + sum_cross

            current_dt_sec = (self.time_array[idx_timestep] - self.time_array[idx_timestep - 1]) * SEC_IN_HR
            current_tau = current_dt_sec / self.t_p
            y_even_cur = self.two_pi_k * self.q_prime_even_interp(current_tau)
            y_odd_cur = self.two_pi_k * self.q_prime_odd_interp(current_tau)
            self.y_n[idx_timestep] = (y_even_cur + y_odd_cur) / 2.0
            self.y_cross[idx_timestep] = (y_even_cur - y_odd_cur) / 2.0
            return

        y_self_array = np.zeros(idx_timestep, dtype=float)
        y_cross_array = np.zeros(idx_timestep, dtype=float)

        if idx_timestep > 0:
            dt_sec_array = (self.time_array[idx_timestep] - self.time_array[0:idx_timestep]) * SEC_IN_HR
            tau_array = dt_sec_array / self.t_p  # Convert to dimensionless time

            # Ask for q' using tau
            y_even_array = self.two_pi_k * self.q_prime_even_interp(tau_array)
            y_odd_array = self.two_pi_k * self.q_prime_odd_interp(tau_array)

            y_self_array[0:idx_timestep] = (y_even_array + y_odd_array) / 2.0
            y_cross_array[0:idx_timestep] = (y_even_array - y_odd_array) / 2.0

        current_dt_sec = (self.time_array[idx_timestep] - self.time_array[idx_timestep - 1]) * SEC_IN_HR
        current_tau = current_dt_sec / self.t_p  # Convert to dimensionless time

        # Ask for q' using tau
        y_even_cur = self.two_pi_k * self.q_prime_even_interp(current_tau)
        y_odd_cur = self.two_pi_k * self.q_prime_odd_interp(current_tau)

        self.y_n[idx_timestep] = (y_even_cur + y_odd_cur) / 2.0
        self.y_cross[idx_timestep] = (y_even_cur - y_odd_cur) / 2.0

        for k in range(self.num_segments):
            sum_self = np.dot(self.dtheta_seg[k, 1:idx_timestep], y_self_array[0 : idx_timestep - 1])

            neighbor_k = self.num_segments - 1 - k if self.counter_flow else k
            sum_cross = np.dot(
                self.coupled_pipe.dtheta_seg[neighbor_k, 1:idx_timestep], y_cross_array[0 : idx_timestep - 1]
            )

            self.history_term_seg[k, idx_timestep] = sum_self + sum_cross

    def generate_matrix(
        self,
        _mass_bldg,
        mass_loop,
        _mass_loop_bldg,
        mass_flow_pipe,
        _mass_loop_ghe,
        idx_timestep,
        configuration,
        _method,
    ):
        if self.coupled_pipe is None:
            raise ValueError("Simulation matrix cannot be computed without a defined coupled pipe.")
        self.compute_history_terms(idx_timestep)

        rows = [np.zeros(self.matrix_size, dtype=np.float64) for _ in range(self.matrix_rows)]
        rhs = [0.0 for _ in range(self.matrix_rows)]

        dt_sec = (self.time_array[idx_timestep] - self.time_array[idx_timestep - 1]) * SEC_IN_HR
        cap_coeff = self.C_f_seg / dt_sec
        m_cp = mass_flow_pipe * self.cp

        idx_t_in = self.row_index
        idx_t_out_final = self.row_index + 3 * self.num_segments

        if configuration == CentralLoopType.ONEPIPE:
            rows[0][idx_t_in] = (mass_loop - mass_flow_pipe) * self.cp
            rows[0][idx_t_out_final] = m_cp
            rows[0][self.downstream_index] = -mass_loop * self.cp
        elif configuration == CentralLoopType.TWOPIPE:
            rows[0][idx_t_in] = 1.0
            rows[0][self.inlet_index] = -1.0

        current_time_sec = self.time_array[idx_timestep] * SEC_IN_HR
        prev_time_sec = self.time_array[idx_timestep - 1] * SEC_IN_HR
        current_ugt = self.calculate_current_ugt(current_time_sec)
        prev_ugt = self.calculate_current_ugt(prev_time_sec)

        for k in range(self.num_segments):
            idx_t_m = self.row_index + 3 * k + 1
            idx_q_self = self.row_index + 3 * k + 2
            idx_t_out = self.row_index + 3 * k + 3

            idx_t_in_seg = self.row_index if k == 0 else self.row_index + 3 * (k - 1) + 3

            # Eq 1: Ground Admittance formulation
            rows[3 * k + 1][idx_q_self] = 1.0
            rows[3 * k + 1][idx_t_m] = -self.y_n[idx_timestep]

            # THE THERMAL BRIDGE: Linking to the neighbor's temperature state
            neighbor_k = self.num_segments - 1 - k if self.counter_flow else k
            idx_t_m_neighbor = self.coupled_pipe.row_index + 3 * neighbor_k + 1
            rows[3 * k + 1][idx_t_m_neighbor] = -self.y_cross[idx_timestep]

            t_m_prev = self.t_mean_seg[k, idx_timestep - 1]
            t_m_neighbor_prev = self.coupled_pipe.t_mean_seg[neighbor_k, idx_timestep - 1]

            rhs_self = self.y_n[idx_timestep] * (-current_ugt - t_m_prev + prev_ugt)
            rhs_cross = self.y_cross[idx_timestep] * (-current_ugt - t_m_neighbor_prev + prev_ugt)

            rhs[3 * k + 1] = self.history_term_seg[k, idx_timestep] + rhs_self + rhs_cross

            # Eq 2: Mean Temp
            rows[3 * k + 2][idx_t_in_seg] = -1.0
            rows[3 * k + 2][idx_t_m] = 2.0
            rows[3 * k + 2][idx_t_out] = -1.0

            # Eq 3: Energy Bal w/ Capacitance
            rows[3 * k + 3][idx_t_in_seg] = m_cp
            rows[3 * k + 3][idx_t_out] = -m_cp
            rows[3 * k + 3][idx_q_self] = -self.L_seg
            rows[3 * k + 3][idx_t_m] = -cap_coeff
            rhs[3 * k + 3] = -cap_coeff * t_m_prev

        return rows, rhs

    def update_post_solve(self, x_vector, idx_timestep):
        self.t_in[idx_timestep] = x_vector[self.row_index]

        current_time_sec = self.time_array[idx_timestep] * SEC_IN_HR
        prev_time_sec = self.time_array[idx_timestep - 1] * SEC_IN_HR
        current_ugt = self.calculate_current_ugt(current_time_sec)
        prev_ugt = self.calculate_current_ugt(prev_time_sec)

        for k in range(self.num_segments):
            self.t_mean_seg[k, idx_timestep] = x_vector[self.row_index + 3 * k + 1]
            self.q_seg[k, idx_timestep] = x_vector[self.row_index + 3 * k + 2]
            self.t_out_seg[k, idx_timestep] = x_vector[self.row_index + 3 * k + 3]

            # Calculate and store the discrete driving potential step (dtheta) that just occurred
            theta_n = self.t_mean_seg[k, idx_timestep] - current_ugt
            theta_n_minus_1 = self.t_mean_seg[k, idx_timestep - 1] - prev_ugt
            self.dtheta_seg[k, idx_timestep] = theta_n - theta_n_minus_1

        self.t_out[idx_timestep] = self.t_out_seg[-1, idx_timestep]


class SourceSinkHeatExchanger(BaseSimComp):
    MATRIX_ROWS = 1

    def __init__(self, hx_id: str, hx_data: dict, tg: float, num_timesteps: int) -> None:
        super().__init__()
        self.name = hx_id
        self.comp_type = SimCompType.SOURCE_SINK_HEAT_EXCHANGER
        self.cp: float | None = None
        self.num_timesteps = num_timesteps

        self.effectiveness = hx_data["effectiveness"]
        self.source_temp = hx_data["source_temperature"]
        self.source_flow_rate = hx_data["source_flow_rate"]
        self.cut_in_temp = hx_data["cut_in_temperature"]
        self.cut_out_temp = hx_data["cut_out_temperature"]
        self.t_in = np.full(self.num_timesteps, tg, dtype=float)
        self.t_out = np.full(self.num_timesteps, tg, dtype=float)
        self.op_mode = SourceSinkOpMode.SOURCE if self.cut_out_temp > self.cut_in_temp else SourceSinkOpMode.SINK
        self.was_running_last_time = False
        self.operating = np.full(self.num_timesteps, False, dtype=bool)

        # Validate hysteresis definition
        if self.op_mode == SourceSinkOpMode.SOURCE and self.cut_in_temp >= self.cut_out_temp:
            raise ValueError("SOURCE mode requires cut_in_temp < cut_out_temp")

        if self.op_mode == SourceSinkOpMode.SINK and self.cut_in_temp <= self.cut_out_temp:
            raise ValueError("SINK mode requires cut_in_temp > cut_out_temp")

        # for bidirectional flow
        self.node_network_inlet_ID = None
        self.node_network_outlet_ID = None
        self.node_HP_inlet_ID = None
        self.node_HP_outlet_ID = None
        self.bldgIDs = []
        self.bldgs = []
        self.type = "HX"

    def is_running(self, t_in: float) -> bool:
        # Hysteresis assumes a proper band:
        #  - SOURCE (heating): cut_in_temp < cut_out_temp
        #  - SINK (cooling): cut_out_temp < cut_in_temp (on at higher temp, off at lower)
        if self.op_mode == SourceSinkOpMode.SOURCE:
            # Turn ON when cold
            if t_in < self.cut_in_temp:
                self.was_running_last_time = True
                return True

            # Stay ON until it rises above cut_out
            if self.was_running_last_time and t_in <= self.cut_out_temp:
                return True

            self.was_running_last_time = False
            return False

        elif self.op_mode == SourceSinkOpMode.SINK:
            # Turn ON when hot
            if t_in > self.cut_in_temp:
                self.was_running_last_time = True
                return True

            # Stay ON until it drops below cut_out
            if self.was_running_last_time and t_in >= self.cut_out_temp:
                return True

            self.was_running_last_time = False
            return False

        self.was_running_last_time = False
        return False

    def generate_matrix(self, mass_bldg, mass_loop, mass_loop_bldg, mass_flow_ghe, mass_loop_ghe, idx_timestep, configuration, method):
        if self.cp is None:
            raise ValueError("cp is uninitialized")
        if self.matrix_size is None:
            raise ValueError("matrix_size is uninitialized")

        t_in = self.t_in[idx_timestep - 1]
        is_running = self.is_running(t_in)
        self.operating[idx_timestep] = is_running
        m_flow_source: float = self.source_flow_rate if is_running else 0.0
        c_source = m_flow_source * self.cp
        c_loop = mass_loop * self.cp
        c_min = min(c_source, c_loop)
        eff_c_min = self.effectiveness * c_min
        row = np.zeros(self.matrix_size, dtype=np.float64)

        # (C_loop - εCmin)*T_d,in - C_loop*T_d,out = -(εCmin)*T_s,in
        row[self.row_index] = c_loop - eff_c_min
        row[self.downstream_index] = -c_loop
        rhs = -eff_c_min * self.source_temp

        rows = [row]
        rhs = [rhs]

        return rows, rhs


class GHX(BaseSimComp):
    MATRIX_ROWS = 4

    def __init__(self, ghe_id: str, ghe_data: dict, fluid: Fluid, loop_config: CentralLoopType, num_timesteps: int, time_array):
        super().__init__()
        self.name = ghe_id
        self.comp_type = SimCompType.GROUND_HEAT_EXCHANGER
        self.height = None
        self.m_dot_total = None
        self.loop_config = loop_config

        self.pipe = Pipe.init_single_u_tube(
            inner_diameter=ghe_data["pipe"]["inner_diameter"],
            outer_diameter=ghe_data["pipe"]["outer_diameter"],
            shank_spacing=ghe_data["pipe"]["shank_spacing"],
            roughness=ghe_data["pipe"]["roughness"],
            conductivity=ghe_data["pipe"]["conductivity"],
            rho_cp=ghe_data["pipe"]["rho_cp"],
        )

        self.soil = Soil(
            k=ghe_data["soil"]["conductivity"],
            rho_cp=ghe_data["soil"]["rho_cp"],
            ugt=ghe_data["soil"]["undisturbed_temp"],
        )

        self.grout = Grout(k=ghe_data["grout"]["conductivity"], rho_cp=ghe_data["grout"]["rho_cp"])

        self.borehole = Borehole(
            burial_depth=ghe_data["borehole"]["buried_depth"],
            borehole_radius=ghe_data["borehole"]["diameter"] / 2.0,
            borehole_height=ghe_data["pre_designed"]["H"],
        )

        self.fluid = fluid
        self.bh_type = BHType.SINGLEUTUBE
        self.split_ratio = None
        self.g_function = GFunction

        self.two_pi_k = TWO_PI * self.soil.k

        # Computed properties
        self.bhe = None
        self.bh_effective_resist = None
        self.gFunction = GFunction(b=0.0, d=0.0, r_b_values={},g_lts={},log_time=[], bore_locations=[])
        self.depth = None

        self.mass_flow_borehole_design = None
        self.history_terms = None
        self.total_values_ghe = None

        # for output
        self.t_in = None
        self.t_mean = None
        self.t_out = None
        self.q_ghe = None
        self.dq_ghe = None
        self.log_lag = None

        self.n_rows = ghe_data["pre_designed"]["boreholes_in_x_dimension"]
        self.n_cols = ghe_data["pre_designed"]["boreholes_in_y_dimension"]
        self.row_spacing = ghe_data["pre_designed"]["spacing_in_x_dimension"]
        self.col_spacing = ghe_data["pre_designed"]["spacing_in_y_dimension"]
        self.max_height = ghe_data["pre_designed"]["max_height"]
        self.min_height = ghe_data["pre_designed"]["min_height"]
        self.nbh = self.n_rows * self.n_cols
        self.mass_flow_ghe_design = ghe_data["flow_rate"] * self.nbh
        self.matrix_size = None
        self.num_timesteps = num_timesteps
        self.history_terms, self.total_values_ghe, self.q_ghe, self.dq_ghe = (
            np.full(self.num_timesteps, self.soil.ugt, dtype=float),
            np.zeros(self.num_timesteps, dtype=float),
            np.zeros(self.num_timesteps, dtype=float),
            np.zeros(self.num_timesteps, dtype=float),
        )

        self.t_in = np.full(self.num_timesteps, self.soil.ugt, dtype=float)
        self.t_mean = np.full(self.num_timesteps, self.soil.ugt, dtype=float)
        self.t_mix_out = np.full(self.num_timesteps, self.soil.ugt, dtype=float)
        self.t_out = np.full(self.num_timesteps, self.soil.ugt, dtype=float)
        self.time_array = np.array(time_array, dtype=float)
        self.g = None
        self.c_n = None

        self.height = self.borehole.H
        self.nbh = self.n_rows * self.n_cols
        self.total_length = self.height * self.nbh
        self.mass_flow_borehole_design = self.mass_flow_ghe_design / self.nbh
        self.bhe = get_bhe_object(
            self.bh_type,
            self.mass_flow_borehole_design,
            self.fluid,
            self.borehole,
            self.pipe,
            self.grout,
            self.soil,
        )
        self.bhe_eq = self.bhe.to_single()
        self.bhe_eq.calc_sts_g_functions()

        self.ts = self.bhe_eq.t_s
        self.cp = self.bhe.fluid.cp
        self.tg = self.bhe.soil.ugt

        self.log_time = eskilson_log_times()

        self.initialize_gFunction_object()
        self.gFunction = self.compute_g_functions()
        self.g, _ = self.grab_g_function()
        self.bh_effective_resist = self.bhe.calc_effective_borehole_resistance()
        self.c_n = self.calc_cn_constant()

        # for bidirectional flow
        self.inlet = None
        self.outlet = None
        self.temp_index_one = None
        self.temp_index_two = None
        self.temp_index_mean = None
        self.heat_rejection_index = None
        self.mass_flow_ghe = None
        self.input = None
        self.output = None
        self.inlet_nodeID = ghe_data["inlet_nodeID"]
        self.outlet_nodeID = ghe_data["outlet_nodeID"]
        self.m_ghe_array = None
        self.type = "GHX"
        self.ID = ghe_data["id"]
        self.m_ghe_array = np.zeros(num_timesteps, dtype=float)

    def initialize_gFunction_object(self):
        self.gFunction.bore_locations = [(i * self.row_spacing, j * self.row_spacing) for i in range(int(self.n_rows)) for j in range(int(self.n_cols))]
        self.gFunction.log_time = eskilson_log_times()

    def compute_g_functions(self):
        # Compute g-functions for a bracketed solution, based on min and max
        # height
        min_height = self.min_height
        max_height = self.max_height
        avg_height = (min_height + max_height) / 2.0
        h_values = [min_height, avg_height, max_height]

        coordinates = self.gFunction.bore_locations
        log_time = self.gFunction.log_time

        self.gFunction = calc_g_func_for_multiple_lengths(
            self.row_spacing,
            h_values,
            self.bhe.borehole.r_b,
            self.bhe.borehole.D,
            self.mass_flow_borehole_design,
            self.bh_type,
            log_time,
            coordinates,
            self.bhe.fluid,
            self.bhe.pipe,
            self.bhe.grout,
            self.bhe.soil,
        )

        return self.gFunction

    def grab_g_function(self):
        """
        Interpolates g-function values using self.gFunction and self.bhe,
        and returns g and g_bhw arrays.
        """

        # Interpolate LTS g-function
        g_function, rb_value, _, _ = self.gFunction.g_function_interpolation(self.row_spacing / self.height)

        # Correct the g-function for borehole radius
        g_function_corrected = self.gFunction.borehole_radius_correction(g_function, rb_value, self.bhe.borehole.r_b)

        # Combine STS and LTS g-functions
        g = combine_sts_lts(
            self.log_time,
            g_function_corrected,
            self.bhe.lntts.tolist(),
            self.bhe.g.tolist(),
        )

        g_bhw = combine_sts_lts(
            self.log_time,
            g_function_corrected,
            self.bhe.lntts.tolist(),
            self.bhe.g_bhw.tolist(),
        )

        return g, g_bhw

    def calc_cn_constant(self):
        """
        Calculate C_n values for three GHEs based on their g-functions.

        Cn = 1 / (2 * pi * K_s) * g((tn - tn-1) / t_s) + R_b
        """

        c_n = np.zeros(self.num_timesteps, dtype=float)

        for i in range(1, self.num_timesteps):
            delta_log_time = np.log((self.time_array[i] - self.time_array[i - 1]) / (self.ts / SEC_IN_HR))
            g_val = self.g(delta_log_time)

            c_n[i] = (1 / self.two_pi_k * g_val) + self.bh_effective_resist

        return c_n

    def compute_history_term(self, i, method):
        # if i == 0:
        #     raise IndexError("Timestep index error")

        ts_hr = self.ts / SEC_IN_HR

        if method == "hourly":
            dim_less_time = self.log_lag[i:1:-1]
            dim1_less_time = self.log_lag[1]
        else:
            past_times = self.time_array[:i - 1]
            dim_less_time = np.log((self.time_array[i] - past_times) / ts_hr)
            dim1_less_time = np.log((self.time_array[i] - self.time_array[i - 1]) / ts_hr)

        delta_q_ghe = self.dq_ghe[:i - 1]
        g_vals = self.g(dim_less_time)
        values = np.sum(delta_q_ghe * g_vals)

        self.total_values_ghe[i] = values

        self.history_terms[i] = (
                self.soil.ugt
                + self.total_values_ghe[i]
                - (self.q_ghe[i - 1] / self.two_pi_k * self.g(dim1_less_time))
        )

        return self.history_terms[i]

    def generate_matrix(self, mass_bldg, mass_loop, mass_loop_bldg, mass_flow_ghe, mass_loop_ghe, idx_timestep, configuration, method):
        # self.history_terms, self.total_values_ghe = self.calc_history_term(
        #     idx_timestep, self.history_terms, self.total_values_ghe,
        # )
        self.compute_history_term(idx_timestep, method)

        row1 = np.zeros(self.matrix_size, dtype=np.float64)
        row2 = np.zeros(self.matrix_size, dtype=np.float64)
        row3 = np.zeros(self.matrix_size, dtype=np.float64)
        row4 = np.zeros(self.matrix_size, dtype=np.float64)

        if configuration == CentralLoopType.ONEPIPE:
            row1[self.row_index] = (mass_loop - mass_flow_ghe) * self.cp
            row1[self.row_index + 3] = mass_flow_ghe * self.cp
            row1[self.downstream_index] = -mass_loop * self.cp

            row2[self.row_index + 1] = 1
            row2[self.row_index + 2] = -self.c_n[idx_timestep]

            row3[self.row_index] = -1
            row3[self.row_index + 1] = 2
            row3[self.row_index + 3] = -1

            row4[self.row_index] = mass_flow_ghe * self.cp
            row4[self.row_index + 2] = -self.height * self.nbh
            row4[self.row_index + 3] = -mass_flow_ghe * self.cp

            rhs1, rhs2, rhs3, rhs4 = 0, self.history_terms[idx_timestep], 0, 0

            rows = [row1, row2, row3, row4]
            rhs = [rhs1, rhs2, rhs3, rhs4]

        elif configuration == CentralLoopType.TWOPIPE:
            row1[self.row_index + 1] = 1
            row1[self.row_index + 2] = -self.c_n[idx_timestep]

            row2[self.row_index + 1] = 2
            row2[self.inlet_index] = -1
            row2[self.row_index + 3] = -1

            row3[self.inlet_index] = mass_flow_ghe * self.cp
            row3[self.row_index + 3] = -mass_flow_ghe * self.cp
            row3[self.row_index + 2] = -self.height * (self.n_rows * self.n_cols)

            row4[self.row_index] = (mass_loop_ghe - mass_flow_ghe) * self.cp
            row4[self.row_index + 3] = mass_flow_ghe * self.cp

            if self.downstream_device.comp_type == SimCompType.GROUND_HEAT_EXCHANGER:
                row4[self.downstream_index] = -mass_loop_ghe * self.cp
            else:
                row4[self.downstream_device.inlet_index] = -mass_loop_ghe * self.cp

            rhs1, rhs2, rhs3, rhs4 = self.history_terms[idx_timestep], 0, 0, 0

            rows = [row1, row2, row3, row4]
            rhs = [rhs1, rhs2, rhs3, rhs4]

        elif configuration == CentralLoopType.TWOPIPE_RING:
            row1[self.temp_index_one] = -1.0
            row1[self.temp_index_two] = -1.0
            row1[self.temp_index_mean] = 2.0

            row2[self.temp_index_mean] = 1
            row2[self.heat_rejection_index] = -self.c_n[idx_timestep]

            if mass_flow_ghe < 0:
                row3[self.temp_index_two] = abs(mass_flow_ghe) * self.cp
                row3[self.temp_index_one] = - abs(mass_flow_ghe) * self.cp
                row3[self.heat_rejection_index] = - self.height * (self.n_rows * self.n_cols)
            else:
                row3[self.temp_index_one] = abs(mass_flow_ghe) * self.cp
                row3[self.temp_index_two] = - abs(mass_flow_ghe) * self.cp
                row3[self.heat_rejection_index] = - self.height * (self.n_rows * self.n_cols)

            rhs1, rhs2, rhs3 = 0.0, self.history_terms[idx_timestep], 0.0

            rows = [row1, row2, row3]
            rhs = [rhs1, rhs2, rhs3]

        else:
            raise ValueError(f"Unknown configuration: {configuration}")

        return rows, rhs


class Building(BaseSimComp):
    MATRIX_ROWS = 1

    def __init__(
        self,
        bldg_id: str,
        bldg_data: dict,
        hp_data: dict,
        tg,
        fluid: Fluid,
        loop_config: CentralLoopType,
        num_timesteps: int,
        load_method: str = "hourly",
        external_loads: dict | None = None,
    ):
        super().__init__()
        self.name = bldg_id
        self.comp_type = SimCompType.BUILDING
        self.matrix_size: int | None = None
        self.cp: float | None = None

        self.fluid = fluid
        self.loop_config = loop_config
        self.heating_exists = bool("heating_load" in bldg_data)
        self.cooling_exists = bool("cooling_load" in bldg_data)
        self.num_timesteps = num_timesteps
        self.sim_years = num_timesteps // HOURS_IN_YEAR

        self.htg_vals: np.ndarray = np.zeros(self.num_timesteps, dtype=float)
        self.clg_vals: np.ndarray = np.zeros(self.num_timesteps, dtype=float)

        if self.heating_exists:
            hp_htg_name = bldg_data["heating_load"]["heat_pump_name"]
            hp_htg_data = hp_data[hp_htg_name]
            self.hp_htg = HPmodel(hp_htg_name, hp_htg_data)

        if self.cooling_exists:
            hp_clg_name = bldg_data["cooling_load"]["heat_pump_name"]
            hp_clg_data = hp_data[hp_clg_name]
            self.hp_clg = HPmodel(hp_clg_name, hp_clg_data)

        if load_method == "hourly":
            if self.heating_exists:
                one_yr_htg_vals = np.array(
                    get_loads(hp_htg_name, SimCompType.HEAT_PUMP.name, bldg_data["heating_load"]),
                    dtype=float,
                )
                hourly_htg = np.tile(one_yr_htg_vals, self.sim_years)
                self.htg_vals = np.insert(hourly_htg, 0, 0.0)

            if self.cooling_exists:
                one_yr_clg_vals = np.array(
                    get_loads(hp_clg_name, SimCompType.HEAT_PUMP.name, bldg_data["cooling_load"]),
                    dtype=float,
                )
                hourly_clg = np.tile(one_yr_clg_vals, self.sim_years)
                self.clg_vals = np.insert(hourly_clg, 0, 0.0)

        elif load_method == "hybrid":
            if external_loads is None:
                raise ValueError(f"Hybrid loads missing for building '{bldg_id}'")

            self.htg_vals = np.array(
                external_loads.get("q_htg", np.zeros(self.num_timesteps)),
                dtype=float,
            )
            self.clg_vals = np.array(
                external_loads.get("q_clg", np.zeros(self.num_timesteps)),
                dtype=float,
            )

            if len(self.htg_vals) != self.num_timesteps or len(self.clg_vals) != self.num_timesteps:
                raise ValueError(f"Hybrid load length mismatch for building '{bldg_id}'")

        else:
            raise ValueError(f"Unknown load_method: {load_method}")

        self.q_net = self.htg_vals - self.clg_vals
        self.t_in = np.full(self.num_timesteps, tg, dtype=float)
        self.t_out = np.full(self.num_timesteps, tg, dtype=float)
        self.m_flow = np.zeros(self.num_timesteps, dtype=float)
        self.power_hp_htg = np.zeros(self.num_timesteps, dtype=float)
        self.power_hp_clg = np.zeros(self.num_timesteps, dtype=float)
        self.power_hp_tot = np.zeros(self.num_timesteps, dtype=float)
        self.power_circ_pump = np.zeros(self.num_timesteps, dtype=float)

        # for bidirectional flow
        self.inlet = None
        self.outlet = None
        self.loop = None
        self.temp_index_one = None
        self.temp_index_two = None
        self.q_ext = None
        self.q_rej = None
        self.mass_bldg = None
        self.q_net_c = self.clg_vals - self.htg_vals
        self.mass_bldg_array = np.zeros(num_timesteps,dtype=float)
        self.input = None
        self.output = None
        self.inlet_nodeID = bldg_data["inlet_nodeID"]
        self.outlet_nodeID = bldg_data["outlet_nodeID"]
        self.type = "bldg"
        self.ID = bldg_data["id"]

    def calc_mass_flow_rate(self, t_in, idx_timestep):
        if self.heating_exists:
            cap_htg = self.hp_htg.c1_htg * t_in**2 + self.hp_htg.c2_htg * t_in + self.hp_htg.c3_htg
            m_single_hp_htg = self.hp_htg.m_flow_single_hp
        else:
            cap_htg = 0.0
            m_single_hp_htg = 0.0

        if self.cooling_exists:
            cap_clg = self.hp_clg.c1_clg * t_in**2 + self.hp_clg.c2_clg * t_in + self.hp_clg.c3_clg
            m_single_hp_clg = self.hp_clg.m_flow_single_hp
        else:
            cap_clg = 0.0
            m_single_hp_clg = 0.0

        m_single_hp = max(m_single_hp_htg, m_single_hp_clg)

        if cap_clg == 0:
            rtf = abs(self.htg_vals[idx_timestep] / cap_htg)
        else:
            rtf = abs(self.htg_vals[idx_timestep]/cap_htg + abs(self.clg_vals[idx_timestep]/cap_clg))

        mass_flow_bldg = rtf * m_single_hp

        self.m_flow[idx_timestep] = mass_flow_bldg

        return mass_flow_bldg

    def calc_r1_r2(self, t_in, idx_timestep):
        """
        Calculate r1 and r2 for this building based on entering fluid temperature and HP coefficients.
        """

        a = 0
        b = 0
        u = 0
        v = 0

        # Extract loads
        h = self.htg_vals[idx_timestep]   # NB changed it from h = self.htg_vals[idx_timestep-1]??
        c = self.clg_vals[idx_timestep]    # NB changed it from c = self.clg_vals[idx_timestep - 1]

        # Heating calculations
        if self.heating_exists:
            slope_htg = 2 * self.hp_htg.a_htg * t_in + self.hp_htg.b_htg
            ratio_htg = self.hp_htg.a_htg * t_in**2 + self.hp_htg.b_htg * t_in + self.hp_htg.c_htg
            u = ratio_htg - slope_htg * t_in
            v = slope_htg

        # Cooling calculations
        if self.cooling_exists:
            slope_clg = 2 * self.hp_clg.a_clg * t_in + self.hp_clg.b_clg
            ratio_clg = self.hp_clg.a_clg * t_in**2 + self.hp_clg.b_clg * t_in + self.hp_clg.c_clg
            a = ratio_clg - slope_clg * t_in
            b = slope_clg

        # Final arrays
        r1 = b * c - v * h
        r2 = a * c - u * h

        return r1, r2

    def generate_matrix(self, mass_bldg, mass_loop, mass_loop_bldg, mass_flow_ghe, mass_loop_ghe, idx_timestep, configuration, method):
        t_in = self.t_in[idx_timestep - 1]
        r1, r2 = self.calc_r1_r2(t_in, idx_timestep)
        if configuration == CentralLoopType.ONEPIPE:
            row = np.zeros(self.matrix_size, dtype=float)
            row[self.row_index] = 1 + r1 / (mass_loop * self.cp)
            row[self.downstream_index] = -1
            rhs = -r2 / (mass_loop * self.cp)
            rows = [row]
            rhs_list = [rhs]

        elif configuration == CentralLoopType.TWOPIPE:
            row1 = np.zeros(self.matrix_size)
            row2 = np.zeros(self.matrix_size)

            if mass_bldg == 0:
                row1[self.inlet_index] = 1
                row1[self.row_index + 1] = -1
            else:
                row1[self.inlet_index] = r1 + mass_bldg * self.cp
                row1[self.row_index + 1] = -mass_bldg * self.cp

            row2[self.row_index] = (mass_loop_bldg - mass_bldg) * self.cp
            row2[self.row_index + 1] = mass_bldg * self.cp
            row2[self.downstream_index] = -mass_loop_bldg * self.cp

            rhs1, rhs2 = -r2, 0

            rows = [row1, row2]
            rhs_list = [rhs1, rhs2]

        elif configuration == CentralLoopType.TWOPIPE_RING:
            row = np.zeros(self.matrix_size)

            if abs(mass_bldg) == 0:
                row[self.temp_index_one] = 1
                row[self.temp_index_two] = -1
                rhs = 0

            elif mass_bldg < 0:
                row[self.temp_index_two] = r1 + abs(self.mass_bldg) * self.cp
                row[self.temp_index_one] = - abs(self.mass_bldg) * self.cp
                rhs = -r2
            else:
                row[self.temp_index_one] = r1 + abs(self.mass_bldg) * self.cp
                row[self.temp_index_two] = - abs(self.mass_bldg) * self.cp
                rhs = -r2

            rows = [row]
            rhs_list = [rhs]

        else:
            raise ValueError(f"Unknown configuration: {configuration}")
        return rows, rhs_list

    def calc_energy(self):
        """Calculate energy consumption of the heat pump system."""

        if self.cooling_exists:
            ratio_clg = self.hp_clg.a_clg * self.t_in**2 + self.hp_clg.b_clg * self.t_in + self.hp_clg.c_clg
            self.power_hp_clg = np.abs(self.clg_vals * (ratio_clg - 1))

        if self.heating_exists:
            ratio_htg = self.hp_htg.a_htg * self.t_in**2 + self.hp_htg.b_htg * self.t_in + self.hp_htg.c_htg
            self.power_hp_htg = self.htg_vals * (1 - ratio_htg)

        self.power_hp_tot = self.power_hp_clg + self.power_hp_htg

        # power consumed by circulating pump
        if self.heating_exists:
            self.power_circ_pump = (
                self.m_flow / (self.fluid.rho * self.hp_htg.pump_efficiency) * self.hp_htg.design_pressure_loss
            )
        if self.cooling_exists:
            self.power_circ_pump = (
                self.m_flow / (self.fluid.rho * self.hp_clg.pump_efficiency) * self.hp_clg.design_pressure_loss
            )


class HPmodel:
    def __init__(self, hp_id: str, hp_data: dict):
        self.name = hp_id

        self.a_htg = hp_data["heating_performance"]["a"]
        self.b_htg = hp_data["heating_performance"]["b"]
        self.c_htg = hp_data["heating_performance"]["c"]

        self.a_clg = hp_data["cooling_performance"]["a"]
        self.b_clg = hp_data["cooling_performance"]["b"]
        self.c_clg = hp_data["cooling_performance"]["c"]

        self.c1_htg = hp_data["heating_performance"]["c1"]
        self.c2_htg = hp_data["heating_performance"]["c2"]
        self.c_ = hp_data["heating_performance"]["c3"]
        self.c3_htg = self.c_

        self.c1_clg = hp_data["cooling_performance"]["c1"]
        self.c2_clg = hp_data["cooling_performance"]["c2"]
        self.c3_clg = hp_data["cooling_performance"]["c3"]

        self.m_flow_single_hp = hp_data["design_flow_rate"]
        self.design_pressure_loss = hp_data["design_pressure_loss"]
        self.pump_efficiency = hp_data["pump_efficiency"]
        self.design_htg_cap_single_hp = hp_data["heating_performance"]["design_cap"]
        self.design_clg_cap_single_hp = hp_data["cooling_performance"]["design_cap"]


class Node:
    def __init__(self, node_id: str, node_data: dict):
        self.ID = node_id
        self.type = node_data["type"]

        self.x = float(node_data["x"])
        self.y = float(node_data["y"])
        self.z = float(node_data["z"])

        self.connection = node_data["connection"]
        self.connection_id = node_data["connection_id"]

        # Assigned later
        self.input = None
        self.output = None
        self.device_pipe = None
        self.diversion = None
        self.merger = None

        # for bidirectional flow
        self.inlet = None
        self.outlet = None
        self.row_index = None
        self.temp_type = None

    def resolve_thermal_output_pipe(self, pipe):
        """
        Skip graphical dummy pipes and return the next physical ring pipe.

        NetworkPipe.output -> Node
        Node.output        -> next NetworkPipe
        """
        current = pipe

        while current is not None and current.type in ("main_dir", "main_dor"):
            next_node = current.output

            if next_node is None or next_node.output is None:
                raise ValueError(
                    f"Dummy pipe chain starting from '{current.ID}' "
                    "does not reach a physical ring pipe."
                )

            current = next_node.output

        return current

    def generate_node_matrix(self, matrix_size, cp, bldg_lookup, GHX_lookup):
        if self.type not in ("branching", "merging"):
            return [], []

        row1 = np.zeros(matrix_size)
        row2 = np.zeros(matrix_size)

        input_pipe = self.input
        output_pipe = self.resolve_thermal_output_pipe(self.output)

        if self.type in ("branching", "merging"):
            if self.connection == "building":
                bldg = bldg_lookup[self.connection_id]
            if self.connection == "ground_heat_exchanger":
                ghx = GHX_lookup[self.connection_id]

            row1[input_pipe.temp_index_two] = input_pipe.mass_flow_rate * cp
            row1[output_pipe.temp_index_one] = - output_pipe.mass_flow_rate * cp
            if input_pipe.type == "main_or" and self.connection == "building":
                row1[bldg.temp_index_one] = -bldg.mass_bldg * cp
            elif input_pipe.type == "main_ir" and self.connection == "building":
                row1[bldg.temp_index_two] = bldg.mass_bldg * cp
            elif input_pipe.type == "main_or" and self.connection == "ground_heat_exchanger":
                row1[ghx.temp_index_two] = ghx.mass_flow_ghe * cp
            elif input_pipe.type == "main_ir" and self.connection == "ground_heat_exchanger":
                row1[ghx.temp_index_one] = - ghx.mass_flow_ghe * cp

        rows = [row1]
        rhs = [0.0]

        if self.temp_type == "branching":
            row2[input_pipe.temp_index_two] = 1.0
            if input_pipe.type == "main_or" and self.connection == "building":
                row2[bldg.temp_index_one] = -1.0
            if input_pipe.type == "main_ir" and self.connection == "building":
                row2[bldg.temp_index_two] = -1.0
            if input_pipe.type == "main_or" and self.connection == "ground_heat_exchanger":
                row2[ghx.temp_index_two] = -1.0
            if input_pipe.type == "main_ir" and self.connection == "ground_heat_exchanger":
                row2[ghx.temp_index_one] = -1.0

            rows.append(row2)
            rhs.append(0.0)

        return rows, rhs


class NetworkPipe:

    def __init__(self, pipe_id: str, pipe_data: dict, num_timesteps: int):
        # Pipe identification
        self.ID = pipe_id
        self.type = pipe_data["type"]

        # Node IDs read from JSON
        self.node_in_name = pipe_data["node_in_name"]
        self.node_out_name = pipe_data["node_out_name"]

        # Physical properties
        self.length = float(pipe_data["length"])
        self.nominal_mass_flow = float(pipe_data["nominal_mass_flow"])
        self.diameter = float(pipe_data["diameter"])

        # Horizontal thermal model
        self.horizontal_pipe_name = pipe_data.get(
            "horizontal_pipe"
        )
        self.horizontal_pipe = None

        # Actual Node objects will be assigned later
        self.input = None
        self.output = None
        self.roughness = 0.000001
        self.matrix_index = None

        # for bidirectional flow
        self.inlet = None
        self.outlet = None
        self.temp_index = None
        self.temp_index_one = None
        self.temp_index_two = None
        self.temp_index_mean = None
        self.heat_rejection_index = None
        self.mass_flow_rate = None
        self.mass_flow_rate_array = np.zeros(num_timesteps,dtype=float)
        self.num_timesteps = num_timesteps

    def calc_pipe_resistance(self, density, kinematic_viscosity):
        vol_flow_rate = self.nominal_mass_flow / density
        velocity = vol_flow_rate / (np.pi/4*self.diameter**2)
        Reynolds_num = velocity * self.diameter / kinematic_viscosity
        friction_factor = 1/(-1.8*np.log10(((self.roughness/self.diameter)/3.7)**1.11 + 6.9/Reynolds_num))**2
        pressure_drop = (friction_factor*self.length*density*velocity**2)/(2*self.diameter)
        pipe_resistance = pressure_drop/self.nominal_mass_flow

        return pipe_resistance


class GHEHPSystem:
    def __init__(self, f_path_json: Path):
        self.components: list[Building | GHX | SourceSinkHeatExchanger] = []
        self.nbh_total = None
        self.matrix_size = 0

        json_data = load_input_file(f_path_json)

        self.loop_config = CentralLoopType[json_data["central_loop"]["pipe_configuration"].upper()]
        self.loop_flow_factor = json_data["central_loop"]["flow_factor"]
        self.loop_pump_efficiency = json_data["central_loop"]["pump_efficiency"]
        self.loop_length = json_data["central_loop"]["loop_length"]
        self.loop_design_pressure_loss_per_meter = json_data["central_loop"]["design_pressure_loss"]

        fluid_data = json_data["fluid"]
        topology_data = json_data["topology"]
        heat_pump_data = json_data["heat_pump"]
        building_data = json_data.get("building", {})
        ghe_data = json_data.get("ground_heat_exchanger", {})
        hx_data = json_data.get("source_sink_heat_exchanger", {})

        # addition for bidirectional flow
        node_data = json_data.get("node", {})
        pipe_data = json_data.get("pipe", {})

        # addition for horizontal piping
        horiz_data = json_data.get("horizontal_piping", {})
        ugt_data = json_data.get("ground_temperature_model", {})

        self.use_horizontal = json_data.get("simulation_control", {}).get("horizontal_simulation_considered", True)

        if horiz_data and not ugt_data:
            raise ValueError("A 'ground_temperature_model' block is required when simulating horizontal piping.")

        horiz_axes = {}
        if self.use_horizontal and horiz_data:
            try:
                with resources.files("ghedesigner.ghe").joinpath(HORZ_LIBRARY_FILENAME).open("rb") as f:
                    lib_data = json.load(f)
                table_single = lib_data["table_single"]
                table_parallel = lib_data["table_parallel"]
                horiz_axes = {
                    k: np.asarray(v, dtype=float)
                    for k, v in lib_data["axes"].items()
                }
            except FileNotFoundError:
                raise FileNotFoundError(
                    "The interpolation library 'unified_horizontal_library.pkl' is required for horizontal"
                    " simulation but was not found in the installed package."
                )

        self.fluid = Fluid(
            fluid_name=fluid_data["fluid_name"],
            percent=fluid_data["concentration_percent"],
            temperature=fluid_data["temperature"],
        )

        self.cp = self.fluid.cp
        tg = json_data["ground_heat_exchanger"]["ghe1"]["soil"]["undisturbed_temp"]  # TODO: fix this

        self.sim_years = json_data["simulation_control"]["simulation_years"]
        self.load_method = json_data["simulation_control"].get("load_method", "hourly").lower()
        self.horiz_segments = json_data["simulation_control"].get("horizontal_segments", 3)

        self.hybrid_load_data = None

        if self.load_method == "hourly":
            self.time_array = np.arange(self.sim_years * HOURS_IN_YEAR + 1, dtype=float)
            self.num_timesteps = len(self.time_array)

        elif self.load_method == "hybrid":
            from ghedesigner.ghe.HP_hybrid_loads_processor import ProcessLoads
            processor = ProcessLoads()
            processor.read_data_from_json_file(json_data)
            processor.read_HP_load_from_json(json_data)
            self.hybrid_load_data = processor.run_hybrid_pipeline()

            first_bldg = next(iter(self.hybrid_load_data))
            self.time_array = np.array(self.hybrid_load_data[first_bldg]["time"], dtype=float)
            self.num_timesteps = len(self.time_array)

        else:
            raise ValueError(f"Unknown load_method: {self.load_method}")

        # get component names we need to build, validate they exist and are referenced correctly
        def get_comp_names(topology: dict, comp_list: dict, comp_type_to_check: SimCompType) -> list[str]:
            comp_names = [c["name"].upper() for c in topology if SimCompType[c["type"].upper()] == comp_type_to_check]

            avail_comps = {k.upper() for k in comp_list}
            for name in comp_names:
                if name not in avail_comps:
                    c_type_name = comp_type_to_check.name
                    msg = f"{c_type_name} name '{name}' in 'topology' not found in key '{c_type_name}'"
                    raise ValueError(msg)

            return comp_names

        building_names = get_comp_names(topology_data, building_data, SimCompType.BUILDING)
        ghx_names = get_comp_names(topology_data, ghe_data, SimCompType.GROUND_HEAT_EXCHANGER)
        hx_names = get_comp_names(topology_data, hx_data, SimCompType.SOURCE_SINK_HEAT_EXCHANGER)

        # # for horizontal piping
        # isolated_names = get_comp_names(topology_data, horiz_data, SimCompType.ISOLATED_HORIZONTAL_PIPE)
        # coupled_names = get_comp_names(topology_data, horiz_data, SimCompType.COUPLED_HORIZONTAL_PIPE)

        # get needed buildings
        buildings: list[Building] = []
        for this_building_id, this_bldg_data in building_data.items():
            if this_building_id.upper() in building_names:
                external_loads = None
                if self.load_method == "hybrid":
                    external_loads = self.hybrid_load_data[this_building_id]

                this_bldg = Building(
                    this_building_id,
                    this_bldg_data,
                    heat_pump_data,
                    tg,
                    self.fluid,
                    self.loop_config,
                    self.num_timesteps,
                    load_method=self.load_method,
                    external_loads=external_loads,
                )
                buildings.append(this_bldg)

        self.buildings = buildings
        self.num_buildings = len(buildings)

        heat_exchangers: list[SourceSinkHeatExchanger] = []
        for this_hx_id, this_hx_data in hx_data.items():
            if this_hx_id.upper() in hx_names:
                this_hx = SourceSinkHeatExchanger(this_hx_id, this_hx_data, tg, self.num_timesteps)
                heat_exchangers.append(this_hx)

        self.heat_exchangers = heat_exchangers
        self.num_heat_exchangers = len(heat_exchangers)

        cp = self.fluid.cp

        ground_heat_exchangers: list[GHX] = []
        for ghx_id, ghe_data in ghe_data.items():
            if ghx_id.upper() in ghx_names:
                this_ghx = GHX(ghx_id, ghe_data, self.fluid, self.loop_config, self.num_timesteps, self.time_array)
                cp = this_ghx.cp
                ground_heat_exchangers.append(this_ghx)

        self.ground_heat_exchangers = ground_heat_exchangers
        self.num_ground_heat_exchangers = len(ground_heat_exchangers)   # why this, I think self.num_ghx = len(ground_heat_exchangers) takes care of this

        nodes: list[Node] = []
        for node_id, node_data in node_data.items():
            this_node = Node(node_id, node_data)
            nodes.append(this_node)

        self.nodes = nodes
        self.num_nodes = len(nodes)

        pipes: list[NetworkPipe] = []
        for pipe_id, pipe_data in pipe_data.items():
            this_pipe = NetworkPipe(pipe_id, pipe_data, self.num_timesteps)
            pipes.append(this_pipe)

        self.pipes = pipes
        self.num_pipes = len(pipes)

        if self.load_method == "hourly":
            for ghx in ground_heat_exchangers:
                ts_hr = ghx.ts / SEC_IN_HR
                lags = np.arange(1, self.num_timesteps + 1, dtype=float)
                ghx.log_lag = np.zeros(self.num_timesteps + 1, dtype=float)
                ghx.log_lag[1:] = np.log(lags / ts_hr)

        self.nbh_total = sum(x.nbh for x in ground_heat_exchangers)
        self.num_ghx = len(ground_heat_exchangers)

        isolated_names = get_comp_names(topology_data, horiz_data, SimCompType.ISOLATED_HORIZONTAL_PIPE)
        coupled_names = get_comp_names(topology_data, horiz_data, SimCompType.COUPLED_HORIZONTAL_PIPE)

        # Helper function to snap to nearest table grid value
        def get_nearest(value, array):
            idx = (np.abs(array - value)).argmin()
            # Cast the NumPy float back to a native Python float
            return float(array[idx])

        isolated_pipes = []
        coupled_pipes_dict = {}

        if self.use_horizontal:
            # PASS 1: Build the components
            for h_id, h_data in horiz_data.items():
                is_isolated = h_id.upper() in isolated_names
                is_coupled = h_id.upper() in coupled_names

                if not is_isolated and not is_coupled:
                    continue

                h_soil = Soil(k=h_data["soil"]["conductivity"], rho_cp=h_data["soil"]["rho_cp"], ugt=0)
                h_pipe = Pipe.init_single_u_tube(
                    inner_diameter=h_data["pipe"]["inner_diameter"],
                    outer_diameter=h_data["pipe"]["outer_diameter"],
                    shank_spacing=0.0,
                    roughness=h_data["pipe"]["roughness"],
                    conductivity=h_data["pipe"]["conductivity"],
                    rho_cp=h_data["pipe"]["rho_cp"],
                )

                r_pipe = 0.1  # TODO: Placeholder: update to actual resistance later
                beta = r_pipe * (TWO_PI * h_soil.k)

                # target_d = get_nearest(h_data["trench_depth"], horiz_axes["depths"])
                # target_beta = get_nearest(beta, horiz_axes["betas"])
                # target_r = get_nearest(h_pipe.r_out, horiz_axes["radii"])
                # target_k = get_nearest(h_soil.k, horiz_axes["soil_ks"])

                target_d = get_nearest(h_data["trench_depth"], np.array(horiz_axes["depths"], dtype=float))
                target_beta = get_nearest(beta, np.array(horiz_axes["betas"], dtype=float))
                target_r = get_nearest(h_pipe.r_out, np.array(horiz_axes["radii"], dtype=float))
                target_k = get_nearest(h_soil.k, np.array(horiz_axes["soil_ks"], dtype=float))

                # this_horiz: IsolatedHorizontalPipe | CoupledHorizontalPipe
                # if is_isolated:
                #     q_prime_interp = table_single[(target_d, target_beta, target_r, target_k)]

                if is_isolated:
                    q_prime_data = table_single[float_tuple_to_string((target_d, target_beta, target_r, target_k))]
                    q_prime_interp = interpolate.interp1d(
                        q_prime_data["x"], q_prime_data["y"], kind="cubic", fill_value="extrapolate"
                    )

                    this_horiz = IsolatedHorizontalPipe(
                        name=h_id,
                        length=h_data["length"],
                        num_segments=self.horiz_segments,
                        pipe=h_pipe,
                        soil=h_soil,
                        fluid=self.fluid,
                        num_timesteps=len(self.time_array),
                        time_array=self.time_array,
                        q_prime_interp=q_prime_interp,
                        beta=beta,
                        ugt_avg=ugt_data["annual_average"],
                        ugt_amp1=ugt_data["amplitude_1"],
                        ugt_phase1=ugt_data["phase_lag_1"],
                        ugt_amp2=ugt_data["amplitude_2"],
                        ugt_phase2=ugt_data["phase_lag_2"],
                        depth=h_data["trench_depth"],
                        load_method=self.load_method,
                    )
                    this_horiz.comp_type = SimCompType.ISOLATED_HORIZONTAL_PIPE
                    isolated_pipes.append(this_horiz)

                elif is_coupled:
                    target_b = get_nearest(h_data["spacing"], horiz_axes["spacings"])
                    q_prime_even, q_prime_odd = table_parallel[(target_d, target_b, target_beta, target_r, target_k)]

                    this_horiz = CoupledHorizontalPipe(
                        name=h_id,
                        length=h_data["length"],
                        num_segments=self.horiz_segments,
                        pipe=h_pipe,
                        soil=h_soil,
                        fluid=self.fluid,
                        num_timesteps=len(self.time_array),
                        time_array=self.time_array,
                        q_prime_even_interp=q_prime_even,
                        q_prime_odd_interp=q_prime_odd,
                        beta=beta,
                        ugt_avg=ugt_data["annual_average"],
                        ugt_amp1=ugt_data["amplitude_1"],
                        ugt_phase1=ugt_data["phase_lag_1"],
                        ugt_amp2=ugt_data["amplitude_2"],
                        ugt_phase2=ugt_data["phase_lag_2"],
                        depth=h_data["trench_depth"],
                        counter_flow=h_data.get("counter_flow", False),
                        load_method=self.load_method,
                    )
                    coupled_pipes_dict[h_id] = this_horiz

            # PASS 2: Link the Coupled Pipes
            for h_id, pipe in coupled_pipes_dict.items():
                partner_id = horiz_data[h_id].get("coupled_to")

                if not partner_id or partner_id not in coupled_pipes_dict:
                    raise ValueError(
                        f"Coupled pipe '{h_id}' is missing a valid 'coupled_to' partner in the horizontal_piping block."
                    )

                partner_pipe = coupled_pipes_dict[partner_id]

                if pipe.length != partner_pipe.length:
                    raise ValueError(
                        f"Coupled pipes '{pipe.name}' and '{partner_pipe.name}' must have identical lengths."
                    )

                pipe.coupled_pipe = partner_pipe

        # Flatten into the master horizontal list
        horizontal_pipes = isolated_pipes + list(coupled_pipes_dict.values())
        self.horizontal_pipes = horizontal_pipes

        # Update MATRIX_ROWS handling
        if self.loop_config == CentralLoopType.ONEPIPE:
            Building.MATRIX_ROWS = 1
            GHX.MATRIX_ROWS = 4
            node_matrix_rows = 0

        elif self.loop_config == CentralLoopType.TWOPIPE:
            Building.MATRIX_ROWS = 2
            GHX.MATRIX_ROWS = 1
            node_matrix_rows = 0

        elif self.loop_config == CentralLoopType.TWOPIPE_RING:
            Building.MATRIX_ROWS = 1
            GHX.MATRIX_ROWS = 3
            node_matrix_rows = sum(
                2 if node.type == "branching"
                else 1 if node.type == "merging"
                else 0
                for node in self.nodes
    )
        else:
            raise ValueError("Invalid CentralLoopType")

        # Horizontal pipes contribute one fewer unknown in TWO_PIPE_RING
        if self.loop_config == CentralLoopType.TWOPIPE_RING:
            horizontal_matrix_size = sum(
                pipe.matrix_rows - 1
                for pipe in horizontal_pipes
            )
        else:
            horizontal_matrix_size = sum(
                pipe.matrix_rows
                for pipe in horizontal_pipes
            )

        self.matrix_size = np.dot(
            [GHX.MATRIX_ROWS, Building.MATRIX_ROWS, SourceSinkHeatExchanger.MATRIX_ROWS],
            [self.num_ghx, self.num_buildings, self.num_heat_exchangers],
        ) + node_matrix_rows + horizontal_matrix_size

        self.m_flow_loop = np.zeros(self.num_timesteps)
        self.pump_power_loop = np.zeros(self.num_timesteps)

        def get_bldg(name: str) -> Building | None:
            return copy.deepcopy(
                next((obj for obj in buildings if obj.name and obj.name.upper() == name.upper()), None)
            )

        def get_ghx(name: str) -> GHX | None:
            return copy.deepcopy(
                next((obj for obj in ground_heat_exchangers if obj.name and obj.name.upper() == name.upper()), None)
            )

        def get_hx(name: str) -> SourceSinkHeatExchanger | None:
            return copy.deepcopy(
                next((obj for obj in heat_exchangers if obj.name and obj.name.upper() == name.upper()), None)
            )

        def get_horiz(name: str) -> IsolatedHorizontalPipe | CoupledHorizontalPipe | None:
            return (
                next((obj for obj in horizontal_pipes if obj.name and obj.name.upper() == name.upper()), None)
            )

        # Topology Assembly
        comp: GHX | Building | SourceSinkHeatExchanger | IsolatedHorizontalPipe | CoupledHorizontalPipe | None
        for v in topology_data:
            comp_type = v["type"]
            #comp: Building | GHX | SourceSinkHeatExchanger | None
            if SimCompType[comp_type.upper()] == SimCompType.BUILDING:
                comp = get_bldg(v["name"])
                if comp is not None:
                    self.components.append(comp)
            elif SimCompType[comp_type.upper()] == SimCompType.GROUND_HEAT_EXCHANGER:
                comp = get_ghx(v["name"])
                if comp is not None:
                    self.components.append(comp)
            elif SimCompType[comp_type.upper()] == SimCompType.SOURCE_SINK_HEAT_EXCHANGER:
                comp = get_hx(v["name"])
                if comp is not None:
                    self.components.append(comp)
            elif SimCompType[comp_type.upper()] in (
                    SimCompType.ISOLATED_HORIZONTAL_PIPE,
                    SimCompType.COUPLED_HORIZONTAL_PIPE,
            ):
                if self.use_horizontal:
                    comp = get_horiz(v["name"])
                    if comp is not None:
                        self.components.append(comp)

        # Link each physical NetworkPipe to its horizontal thermal model
        for network_pipe in self.pipes:
            horizontal_pipe_name = network_pipe.horizontal_pipe_name

            # Dummy or non-horizontal pipes have no thermal model
            if horizontal_pipe_name is None:
                continue

            horizontal_pipe = next(
                (
                    comp
                    for comp in self.components
                    if isinstance(
                    comp,
                    (
                        IsolatedHorizontalPipe,
                        CoupledHorizontalPipe,
                    ),
                )
                       and comp.name
                       and comp.name.upper()
                       == horizontal_pipe_name.upper()
                ),
                None,
            )

            if horizontal_pipe is None:
                raise ValueError(
                    f"NetworkPipe '{network_pipe.ID}' references "
                    f"horizontal pipe '{horizontal_pipe_name}', "
                    "but that horizontal pipe was not found in "
                    "self.components."
                )

            # Establish the two-way connection
            network_pipe.horizontal_pipe = horizontal_pipe
            horizontal_pipe.network_pipe = network_pipe

        for this_comp in self.components:
            this_comp.matrix_size = self.matrix_size
            if isinstance(this_comp, GHX):
                this_comp.split_ratio = this_comp.nbh / self.nbh_total
            elif isinstance(this_comp, (Building, SourceSinkHeatExchanger, IsolatedHorizontalPipe, CoupledHorizontalPipe)):
                this_comp.cp = cp

        # Assigning downstream device to each component
        for i in range(len(self.components)):
            self.components[i].downstream_device = self.components[(i+1) % len(self.components)]

        # Updating connections
        self.UpdateConnections()

        # Assigning row_indices
        idx_comp = 0
        for this_comp in self.components:
            this_comp.row_index = idx_comp
            # Use matrix_rows for horizontal pipes; MATRIX_ROWS is not defined for them.
            if isinstance(
                    this_comp,
                    (IsolatedHorizontalPipe, CoupledHorizontalPipe),
            ):
                idx_comp += this_comp.matrix_rows
            else:
                idx_comp += this_comp.MATRIX_ROWS

            this_comp.downstream_index = idx_comp

        # set the last component to loops back to the start
        self.components[-1].downstream_index = 0

        # Assigning inlet_index
        common_inlet_index_bldg = None
        common_inlet_index_ghx = None

        for comp in self.components:
            if comp.comp_type == SimCompType.BUILDING:
                common_inlet_index_bldg = comp.row_index
                break

        for comp in self.components:
            if comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER:
                common_inlet_index_ghx = comp.row_index
                break

        for comp in self.components:
            if comp.comp_type == SimCompType.BUILDING:
                comp.inlet_index = common_inlet_index_bldg

            elif comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER:
                comp.inlet_index = common_inlet_index_ghx

            else:
                pass

        # Assign temperature/heat-transfer indices for bidirectional flow
        index = 0

        bldg_lookup = {
            bldg.ID: bldg
            for bldg in self.buildings
        }

        ghx_lookup = {
            ghx.ID: ghx
            for ghx in self.ground_heat_exchangers
        }

        for this_comp in self.components:

            if isinstance(this_comp, Building):
                idx_one = index
                index += 1

                idx_two = index
                index += 1

                # Component in topology
                this_comp.temp_index_one = idx_one
                this_comp.temp_index_two = idx_two

                # Original building object
                original_bldg = bldg_lookup[this_comp.ID]
                original_bldg.temp_index_one = idx_one
                original_bldg.temp_index_two = idx_two

            elif isinstance(this_comp, GHX):
                idx_one = index
                index += 1

                idx_two = index
                index += 1

                idx_mean = index
                index += 1

                idx_q = index
                index += 1

                # Component in topology
                this_comp.temp_index_one = idx_one
                this_comp.temp_index_two = idx_two
                this_comp.temp_index_mean = idx_mean
                this_comp.heat_rejection_index = idx_q

                # Original GHX object
                original_ghx = ghx_lookup[this_comp.ID]
                original_ghx.temp_index_one = idx_one
                original_ghx.temp_index_two = idx_two
                original_ghx.temp_index_mean = idx_mean
                original_ghx.heat_rejection_index = idx_q

            elif isinstance(this_comp, IsolatedHorizontalPipe):
                if this_comp.network_pipe is None:
                    raise ValueError(
                        f"Horizontal pipe '{this_comp.name}' "
                        "is not linked to a NetworkPipe."
                    )

                if this_comp.num_segments != 1:
                    raise NotImplementedError(
                        f"Horizontal pipe '{this_comp.name}' currently supports "
                        "only num_segments = 1 in the integrated ring solver."
                    )

                # Four unknowns for the physical horizontal pipe
                idx_one = index
                index += 1

                idx_mean = index
                index += 1

                idx_q = index
                index += 1

                idx_two = index
                index += 1

                # Thermal horizontal-pipe model
                this_comp.temp_index_one = idx_one
                this_comp.temp_index_mean = idx_mean
                this_comp.index_q = idx_q
                this_comp.temp_index_two = idx_two

                # Hydraulic NetworkPipe representing the same physical pipe
                network_pipe = this_comp.network_pipe

                network_pipe.temp_index_one = idx_one
                network_pipe.temp_index_mean = idx_mean
                network_pipe.heat_rejection_index = idx_q
                network_pipe.temp_index_two = idx_two

    def solve_system(self):
        # for bidirectional flow - calculating fluid resistances
        density = self.fluid.rho
        kinematic_viscosity = self.fluid.mu / self.fluid.rho

        for pipe in self.pipes:
            if pipe.type in ("main_ir", "main_or"):
                pipe.resistance = abs(pipe.calc_pipe_resistance(density, kinematic_viscosity))

        for idx_timestep in range(1, self.num_timesteps):  # loop over all timestep
            matrix_rows = []
            matrix_rhs = []
            total_hp_flow = 0
            m_bldg_cum = 0
            m_ghe_cum = 0

            # ---- RESET flows for all components ----
            for comp in self.components:
                comp.mass_bldg = 0.0
                comp.mass_flow_ghe = 0.0
                comp.mass_loop_bldg = 0.0
                comp.mass_loop_ghe = 0.0

            for this_comp in self.components:
                if isinstance(this_comp, Building):
                    t_in = this_comp.t_in[idx_timestep - 1]
                    this_comp.mass_bldg = this_comp.calc_mass_flow_rate(t_in, idx_timestep)
                    total_hp_flow += this_comp.mass_bldg
                    m_bldg_cum += this_comp.mass_bldg

                this_comp.mass_loop_bldg = m_bldg_cum

            # for bidirectional flow

            # Calculating zone mass flow rates
            if self.loop_config == CentralLoopType.TWOPIPE_RING:
                total_bldg_flow = 0.0
                for bldg in self.buildings:
                    t_in = bldg.t_in[idx_timestep - 1]
                    mass_bldg = bldg.calc_mass_flow_rate(t_in, idx_timestep)

                    if bldg.q_net_c[idx_timestep] > 0:
                        bldg.mass_bldg = mass_bldg
                    else:
                        bldg.mass_bldg = -mass_bldg

                    # Copy the same value to the component copy
                    for comp in self.components:
                        if isinstance(comp, Building) and comp.ID == bldg.ID:
                            comp.mass_bldg = bldg.mass_bldg
                            comp.mass_bldg_array[idx_timestep] = bldg.mass_bldg
                            break

                    bldg.mass_bldg_array[idx_timestep] = bldg.mass_bldg

                    # Add to total flow
                    total_bldg_flow += bldg.mass_bldg

                    # my convention is cooling-reference (clockwise) flow heating-reverse flow (counter-clockwise)

                    bldg.input.input.mass_flow_rate = bldg.mass_bldg
                    bldg.output.output.mass_flow_rate = bldg.mass_bldg

                # Calculating mass flow rate of GHE
                nbh_total = sum(GHE.nbh for GHE in self.ground_heat_exchangers)
                for GHE in self.ground_heat_exchangers:
                    GHE.nbh = len(GHE.gFunction.bore_locations)
                    split_ratio = GHE.nbh / nbh_total
                    GHE.mass_flow_ghe = total_bldg_flow * split_ratio

                # Put check for mass flow rate of GHE going very low

                low_ghe_flow = any(
                    abs(ghx.mass_flow_ghe) < ghx.mass_flow_ghe_design * 0.05
                    for ghx in self.ground_heat_exchangers
                )

                if low_ghe_flow:
                    max_zone = max(
                        self.buildings,
                        key=lambda bldg: abs(bldg.mass_bldg),
                    )

                    max_zone.mass_bldg *= 1.5

                    # Update stored zone flow for this timestep
                    max_zone.mass_bldg_array[idx_timestep] = max_zone.mass_bldg

                    # Update all zone connecting-pipe flows
                    for bldg in self.buildings:
                        bldg.input.input.mass_flow_rate = bldg.mass_bldg
                        bldg.output.output.mass_flow_rate = bldg.mass_bldg

                    # Recalculate total signed zone flow
                    total_zone_flow = sum(
                        bldg.mass_bldg for bldg in self.buildings
                    )
                    # Update every GHE
                    for ghe in self.ground_heat_exchangers:
                        split_ratio = ghe.nbh / nbh_total
                        ghe.mass_flow_ghe = total_zone_flow * split_ratio

                # building the mass_matrix

                index = 0
                for pipe in self.pipes:
                    if pipe.type == "main_ir":
                        pipe.matrix_index = index
                        index += 1

                for pipe in self.pipes:
                    if pipe.type == "main_or":
                        pipe.matrix_index = index
                        index += 1

                # Assigning appropriate matrix_index and temp_index to dummy pipes
                for pipe in self.pipes:
                    if pipe.type == "main_dir":
                        current = pipe
                        while current.type != "main_ir":
                            current = current.output
                        pipe.matrix_index = current.matrix_index
                        pipe.temp_index = current.temp_index

                for pipe in self.pipes:
                    if pipe.type == "main_dor":
                        current = pipe
                        while current.type != "main_or":
                            current = current.output
                        pipe.matrix_index = current.matrix_index
                        pipe.temp_index = current.temp_index

                mass_matrix_size = 2 * len(self.buildings) + 2 * (len(self.ground_heat_exchangers) - 1) + 2

                mass_matrix = []
                mass_matrix_rhs = []

                # generating matrix for pipes
                row1 = np.zeros(mass_matrix_size)
                row2 = np.zeros(mass_matrix_size)

                main_ir_index = 0
                main_or_index = len(self.buildings) + (len(self.ground_heat_exchangers) - 1) + 1

                for pipe in self.pipes:
                    if pipe.type == "main_ir":
                        row1[main_ir_index] = pipe.resistance
                        main_ir_index += 1

                    elif pipe.type == "main_or":
                        row2[main_or_index] = pipe.resistance
                        main_or_index += 1

                mass_matrix.append(row1)
                mass_matrix_rhs.append(0.0)

                mass_matrix.append(row2)
                mass_matrix_rhs.append(0.0)

                # generating matrix for nodes
                bldg_lookup = {bldg.ID: bldg for bldg in self.buildings}
                GHX_lookup = {ghx.ID: ghx for ghx in self.ground_heat_exchangers}
                last_ghe_id = self.ground_heat_exchangers[-1].ID

                for node in self.nodes:
                    if node.connection == "ground_heat_exchanger" and node.connection_id == last_ghe_id:
                        continue

                    row = np.zeros(mass_matrix_size)

                    if node.type == "merging" and node.connection == "building":
                        row[node.output.matrix_index] = 1
                        row[node.input.matrix_index] = -1
                    elif node.type == "branching" and node.connection == "building":
                        row[node.input.matrix_index] = 1
                        row[node.output.matrix_index] = -1
                    elif node.type == "merging" and node.connection == "ground_heat_exchanger":
                        row[node.output.matrix_index] = 1
                        row[node.input.matrix_index] = -1
                    elif node.type == "branching" and node.connection == "ground_heat_exchanger":
                        row[node.input.matrix_index] = 1
                        row[node.output.matrix_index] = -1
                    else:
                        continue

                    if node.connection == "building":
                        rhs = bldg_lookup[node.connection_id].mass_bldg
                    elif node.connection == "ground_heat_exchanger":
                        rhs = GHX_lookup[node.connection_id].mass_flow_ghe

                    mass_matrix.append(row)
                    mass_matrix_rhs.append(rhs)

                mass_matrix = np.array(mass_matrix)
                mass_matrix_rhs = np.array(mass_matrix_rhs)

                solution = np.linalg.solve(mass_matrix, mass_matrix_rhs)

                # Assigning mass flow rates to loop segments
                inner_index = 0
                outer_index = len(self.buildings) + len(self.ground_heat_exchangers)

                for pipe in self.pipes:
                    if pipe.type == "main_ir":
                        pipe.mass_flow_rate = solution[inner_index]
                        inner_index += 1

                    elif pipe.type == "main_or":
                        pipe.mass_flow_rate = solution[outer_index]
                        outer_index += 1

                for pipe in self.pipes:
                    if pipe.type == "main_dir":
                        current = pipe
                        while current.output.type != "main_ir":
                            current = current.output
                        pipe.mass_flow_rate = current.output.mass_flow_rate

                for pipe in self.pipes:
                    if pipe.type == "main_dor":
                        current = pipe
                        while current.output.type != "main_or":
                            current = current.output
                        pipe.mass_flow_rate = current.output.mass_flow_rate

                for pipe in self.pipes:
                    pipe.mass_flow_rate_array[idx_timestep] = pipe.mass_flow_rate

                # solving node mass balance to find mass flow rate of GHE and assigning flows to GHE connecting pipes
                for ghx in self.ground_heat_exchangers:
                    node_upstream = ghx.input.input.input
                    ghx.mass_flow_ghe = node_upstream.input.mass_flow_rate - node_upstream.output.mass_flow_rate  # comment by NB: I may not need to do this as I already have GHE flow before building matrix, check and remove!!
                    ghx.input.input.mass_flow_rate = ghx.mass_flow_ghe
                    ghx.output.output.mass_flow_rate = ghx.mass_flow_ghe
                    ghx.m_ghe_array[idx_timestep] = ghx.mass_flow_ghe

                    # Copy the same value to the component copy
                    for comp in self.components:
                        if isinstance(comp, GHX) and comp.ID == ghx.ID:
                            comp.mass_flow_ghe = ghx.mass_flow_ghe
                            break

                self.UpdateThermalConnections()

            for this_comp in self.components:
                if isinstance(this_comp, Building):
                    t_in = this_comp.t_in[idx_timestep - 1]
                    if self.loop_config == CentralLoopType.TWOPIPE_RING:
                        this_comp.mass_bldg = this_comp.mass_bldg
                    else:
                        this_comp.mass_bldg = this_comp.calc_mass_flow_rate(t_in, idx_timestep)
                    total_hp_flow += this_comp.mass_bldg
                    m_bldg_cum += this_comp.mass_bldg

                this_comp.mass_loop_bldg = m_bldg_cum
            mass_loop = max(total_hp_flow * self.loop_flow_factor, 0.1)

            for this_comp in self.components:
                if isinstance(this_comp, GHX):
                    if self.loop_config != CentralLoopType.TWOPIPE_RING:
                        this_comp.mass_flow_ghe = (
                                mass_loop * this_comp.split_ratio
                        )

                    m_ghe_cum += this_comp.mass_flow_ghe

                this_comp.mass_loop_ghe = m_ghe_cum

                # The 4th generate_matrix() argument is the component-specific mass flow.
                if isinstance(this_comp, IsolatedHorizontalPipe):
                    component_mass_flow = (
                        this_comp.network_pipe.mass_flow_rate
                    )
                else:
                    component_mass_flow = this_comp.mass_flow_ghe
                rows, rhs = this_comp.generate_matrix(this_comp.mass_bldg, mass_loop, this_comp.mass_loop_bldg, component_mass_flow, this_comp.mass_loop_ghe, idx_timestep, self.loop_config, self.load_method)
                #rows, rhs = this_comp.generate_matrix(this_comp.mass_bldg, mass_loop, this_comp.mass_loop_bldg, this_comp.mass_flow_ghe, this_comp.mass_loop_ghe, idx_timestep, self.loop_config, self.load_method)

                if (
                        self.loop_config == CentralLoopType.TWOPIPE_RING
                        and isinstance(this_comp, IsolatedHorizontalPipe)
                ):
                    # Skip Drew's unused connection row; node equations provide the connection.
                    matrix_rows.extend(rows[1:])
                    matrix_rhs.extend(rhs[1:])
                else:
                    matrix_rows.extend(rows)
                    matrix_rhs.extend(rhs)

            # Generating matrix for nodes

            if self.loop_config == CentralLoopType.TWOPIPE_RING:
                bldg_lookup = {bldg.ID: bldg for bldg in self.buildings}
                GHX_lookup = {ghx.ID: ghx for ghx in self.ground_heat_exchangers}
                for node in self.nodes:
                    rows, rhs_values = node.generate_node_matrix(self.matrix_size, self.fluid.cp, bldg_lookup, GHX_lookup)

                    matrix_rows.extend(rows)
                    matrix_rhs.extend(rhs_values)

            # Solve the system = A * X = B
            a_matrix = np.array(matrix_rows, dtype=float)
            b_vector = np.array(matrix_rhs, dtype=float)
            x_vector = np.linalg.solve(a_matrix, b_vector)

            # save output data
            self.m_flow_loop[idx_timestep] = mass_loop

            for this_comp in self.components:
                row_index = this_comp.row_index

                if this_comp.comp_type == SimCompType.BUILDING:

                    if self.loop_config == CentralLoopType.TWOPIPE:
                        this_comp.t_in[idx_timestep] = x_vector[this_comp.inlet_index]
                        this_comp.t_out[idx_timestep] = x_vector[row_index + 1]

                    elif self.loop_config == CentralLoopType.TWOPIPE_RING:

                        if this_comp.mass_bldg >= 0.0:
                            this_comp.t_in[idx_timestep] = x_vector[this_comp.temp_index_one]
                            this_comp.t_out[idx_timestep] = x_vector[this_comp.temp_index_two]
                        else:
                            this_comp.t_in[idx_timestep] = x_vector[this_comp.temp_index_two]
                            this_comp.t_out[idx_timestep] = x_vector[this_comp.temp_index_one]
                    else:
                        this_comp.t_in[idx_timestep] = x_vector[row_index]
                        this_comp.t_out[idx_timestep] = x_vector[this_comp.downstream_index]

                elif this_comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER:

                    if self.loop_config == CentralLoopType.TWOPIPE:
                        this_comp.t_in[idx_timestep] = x_vector[this_comp.inlet_index]
                        this_comp.t_mix_out[idx_timestep] = x_vector[row_index]
                        this_comp.t_mean[idx_timestep] = x_vector[row_index + 1]
                        this_comp.q_ghe[idx_timestep] = x_vector[row_index + 2]
                        this_comp.t_out[idx_timestep] = x_vector[row_index + 3]

                    elif self.loop_config == CentralLoopType.TWOPIPE_RING:

                        if this_comp.mass_flow_ghe >= 0.0:
                            this_comp.t_in[idx_timestep] = x_vector[this_comp.temp_index_one]
                            this_comp.t_out[idx_timestep] = x_vector[this_comp.temp_index_two]
                        else:
                            this_comp.t_in[idx_timestep] = x_vector[this_comp.temp_index_two]
                            this_comp.t_out[idx_timestep] = x_vector[this_comp.temp_index_one]

                        this_comp.t_mean[idx_timestep] = x_vector[this_comp.temp_index_mean]
                        this_comp.q_ghe[idx_timestep] = x_vector[this_comp.heat_rejection_index]

                    else:
                        this_comp.t_in[idx_timestep] = x_vector[row_index]
                        this_comp.t_mix_out[idx_timestep] = x_vector[this_comp.downstream_index]
                        this_comp.t_mean[idx_timestep] = x_vector[row_index + 1]
                        this_comp.q_ghe[idx_timestep] = x_vector[row_index + 2]
                        this_comp.t_out[idx_timestep] = x_vector[row_index + 3]

                    this_comp.dq_ghe[idx_timestep - 1] = (this_comp.q_ghe[idx_timestep] - this_comp.q_ghe[idx_timestep - 1]) / this_comp.two_pi_k

                elif this_comp.comp_type == SimCompType.ISOLATED_HORIZONTAL_PIPE:
                    if self.loop_config == CentralLoopType.TWOPIPE_RING:
                        if this_comp.network_pipe.mass_flow_rate >= 0.0:
                            this_comp.t_in[idx_timestep] = x_vector[this_comp.temp_index_one]
                            this_comp.t_out[idx_timestep] = x_vector[this_comp.temp_index_two]
                        else:
                            this_comp.t_in[idx_timestep] = x_vector[this_comp.temp_index_two]
                            this_comp.t_out[idx_timestep] = x_vector[this_comp.temp_index_one]

                        this_comp.t_mean_seg[0, idx_timestep] = x_vector[this_comp.temp_index_mean]
                        this_comp.q_seg[0, idx_timestep] = x_vector[this_comp.index_q]

                elif this_comp.comp_type == SimCompType.SOURCE_SINK_HEAT_EXCHANGER:

                    this_comp.t_in[idx_timestep] = x_vector[row_index]
                    this_comp.t_out[idx_timestep] = x_vector[this_comp.downstream_index]

                    if self.loop_config == CentralLoopType.TWOPIPE:

                        this_comp.t_in[idx_timestep] = x_vector[
                            this_comp.inlet_index
                        ]
                        this_comp.t_mix_out[idx_timestep] = x_vector[row_index]

                        this_comp.t_mean[idx_timestep] = x_vector[row_index + 1]
                        this_comp.q_ghe[idx_timestep] = x_vector[row_index + 2]
                        this_comp.t_out[idx_timestep] = x_vector[row_index + 3]

    def calc_energy(self):
        self.pump_power_loop = (
            self.m_flow_loop
            / (self.fluid.rho * self.loop_pump_efficiency)
            * self.loop_design_pressure_loss_per_meter
            * self.loop_length
        )

    def create_output(self, output_path: Path):
        output_data = pd.DataFrame(index=self.time_array[1:])
        output_data.index.name = "Time"

        network_q_net_bldg_tot = np.zeros(self.num_timesteps, dtype=float)
        network_q_net_ghe_tot = np.zeros(self.num_timesteps, dtype=float)

        # compute energy use for central loop
        self.calc_energy()

        # compute energy use for all components
        for this_comp in self.components:
            this_comp.calc_energy()

        for this_comp in self.components:
            if isinstance(this_comp, Building):
                output_data[f"{this_comp.name}:EFT [C]"] = this_comp.t_in[1:]
                output_data[f"{this_comp.name}:ExFT [C]"] = this_comp.t_out[1:]
                output_data[f"{this_comp.name}:Q_htg [W]"] = this_comp.htg_vals[1:]
                output_data[f"{this_comp.name}:Q_clg [W]"] = this_comp.clg_vals[1:]
                output_data[f"{this_comp.name}:Q_net [W]"] = this_comp.q_net[1:]
                output_data[f"{this_comp.name}:M_flow [kg/s]"] = this_comp.mass_bldg_array[1:]
                network_q_net_bldg_tot += this_comp.q_net
                output_data[f"{this_comp.name}:P_hp_htg [W]"] = this_comp.power_hp_htg[1:]
                output_data[f"{this_comp.name}:P_hp_clg [W]"] = this_comp.power_hp_clg[1:]
                output_data[f"{this_comp.name}:P_hp_tot [W]"] = this_comp.power_hp_tot[1:]
                output_data[f"{this_comp.name}:P_pump [W]"] = this_comp.power_circ_pump[1:]

                q_src_clg = this_comp.clg_vals + this_comp.power_hp_clg
                q_src_htg = this_comp.htg_vals - this_comp.power_hp_htg

                output_data[f"{this_comp.name}:Q_src_clg [W]"] = q_src_clg[1:]
                output_data[f"{this_comp.name}:Q_src_htg [W]"] = q_src_htg[1:]
                output_data[f"{this_comp.name}:Q_src_het [W]"] = (q_src_htg - q_src_clg)[1:]

        for this_comp in self.components:
            if isinstance(this_comp, GHX):
                output_data[f"{this_comp.name}:EFT [C]"] = this_comp.t_in[1:]
                output_data[f"{this_comp.name}:ExFT [C]"] = this_comp.t_out[1:]
                if self.loop_config != CentralLoopType.TWOPIPE_RING:
                    output_data[f"{this_comp.name}:ExFT Mixed Loop [C]"] = this_comp.t_mix_out[1:]
                output_data[f"{this_comp.name}:MFT [C]"] = this_comp.t_mean[1:]
                output_data[f"{this_comp.name}:Q [W/m]"] = this_comp.q_ghe[1:]
                output_data[f"{this_comp.name}:Q_tot [W]"] = (this_comp.q_ghe * this_comp.nbh * this_comp.height)[1:]
                network_q_net_ghe_tot += (this_comp.q_ghe * this_comp.nbh * this_comp.height)

        if self.loop_config == CentralLoopType.TWOPIPE_RING:
            for this_comp in self.components:
                if isinstance(this_comp, IsolatedHorizontalPipe):
                    output_data[f"{this_comp.name}:EFT [C]"] = this_comp.t_in[1:]
                    output_data[f"{this_comp.name}:ExFT [C]"] = this_comp.t_out[1:]
                    output_data[f"{this_comp.name}:MFT [C]"] = this_comp.t_mean_seg[0, 1:]
                    output_data[f"{this_comp.name}:Q [W/m]"] = this_comp.q_seg[0, 1:]
                    output_data[f"{this_comp.name}:M_flow [kg/s]"] = this_comp.network_pipe.mass_flow_rate_array[1:]

        for this_comp in self.components:
            if isinstance(this_comp, SourceSinkHeatExchanger):
                output_data[f"{this_comp.name}:EFT [C]"] = this_comp.t_in[1:]
                output_data[f"{this_comp.name}:ExFT [C]"] = this_comp.t_out[1:]
                output_data[f"{this_comp.name}:Operating [T/F]"] = this_comp.operating[1:]
                output_data[f"{this_comp.name}:Q [W]"] = (
                    this_comp.operating * self.m_flow_loop * self.fluid.cp * (this_comp.t_out - this_comp.t_in)
                )[1:]

        output_data["Network:M_flow [kg/s]"] = self.m_flow_loop[1:]
        output_data["Network:P_pump [W]"] = self.pump_power_loop[1:]
        output_data["Network:Q_net_bldg [W]"] = network_q_net_bldg_tot[1:]
        output_data["Network:Q_net_ghe [W]"] = network_q_net_ghe_tot[1:]

        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        output_data.to_csv(output_path, float_format="%0.4f")


    def UpdateConnections(self):

        for pipe in self.pipes:
            pipe.input = FindItemByID(pipe.node_in_name, self.nodes)
            pipe.output = FindItemByID(pipe.node_out_name, self.nodes)
            if pipe.type in ("main_ir", "main_or", "main_dir", "main_dor", "main"):
                pipe.input.output = pipe
                pipe.output.input = pipe
            elif pipe.type == "branch":
                pipe.input.diversion = pipe
                pipe.input.device_pipe = pipe
                pipe.output.input = pipe
            else:
                pipe.output.merger = pipe
                pipe.output.device_pipe = pipe
                pipe.input.output = pipe

        for bldg in self.buildings:
            bldg.input = FindItemByID(bldg.inlet_nodeID, self.nodes)
            if bldg.input is None:
                raise ValueError(
                    f"Inlet node '{bldg.inlet_nodeID}' for building "
                    f"'{bldg.name}' was not found."
                )

            bldg.input.output = bldg
            if self.loop_config in (CentralLoopType.TWOPIPE, CentralLoopType.TWOPIPE_RING):
                bldg.output = FindItemByID(bldg.outlet_nodeID, self.nodes)
                bldg.output.input = bldg

        for ghx in self.ground_heat_exchangers:
            ghx.input = FindItemByID(ghx.inlet_nodeID, self.nodes)
            ghx.input.output = ghx
            if self.loop_config in (CentralLoopType.TWOPIPE, CentralLoopType.TWOPIPE_RING):
                ghx.output = FindItemByID(ghx.outlet_nodeID, self.nodes)
                ghx.output.input = ghx

        for HX in self.heat_exchangers:
            HX.input = FindItemByID(HX.node_network_inlet_ID, self.nodes)
            HX.output = FindItemByID(HX.node_network_outlet_ID, self.nodes)
            HX.HP_input = FindItemByID(HX.node_HP_inlet_ID, self.nodes)
            HX.HP_output = FindItemByID(HX.node_HP_outlet_ID, self.nodes)

            HX.input.output = HX
            HX.HP_output.input = HX
            HX.HP_input.output = HX
            if self.loop_config in (CentralLoopType.TWOPIPE, CentralLoopType.TWOPIPE_RING):
                HX.output.input = HX

        for HX in self.heat_exchangers:
            for bldgID in HX.bldgIDs:
                bldg = FindItemByID(bldgID, self.buildings)
                HX.bldgs.append(bldg)

        # finding upstream and downstream device for GHE

        for ghx in self.ground_heat_exchangers:

            # find the first upstream branching node
            device = ghx.input
            while device.type != "branching":
                device = device.input

            # find the second upstream branching node
            device = device.input
            while device.type != "branching" and device.type != "merging":
                device = device.input

            # find the upstream device
            if device.type == "branching":
                device = device.diversion
                while device.type != "GHX" and device.type != "bldg" and device.type != "HX":
                    device = device.output
            else:
                device = device.merger
                while device.type != "GHX" and device.type != "bldg":
                    device = device.input

            ghx.upstream_device = device
            device.downstream_device = ghx

        # finding upstream and downstream device for zones

        for bldg in self.buildings:
            # find the first upstream branching node
            device = bldg.input
            while device.type != "branching":
                device = device.input

            # find the second upstream branching node or upstream device, if it is connected to ISHX
            device = device.input
            while device.type != "branching" and device.type != "device" and device.type != "merging":
                device = device.input

            # find the upstream device
            if device.type == "branching":
                device = device.diversion
                while device.type != "GHX" and device.type != "bldg" and device.type != "HX":
                    device = device.output
            elif device.type == "merging":
                device = device.merger
                while device.type != "GHX":
                    device = device.input

            else:
                device = device.input

            bldg.upstream_device = device
            if device.type != "HX":
                device.downstream_device = bldg

        # finding upstream and downstream device for ISHX
        for HX in self.heat_exchangers:
            # find first upstream node
            device = HX.input
            while device.type != "branching":
                device = device.input

            # find the second upstream branching node
            device = device.input
            while device.type != "branching" and device.type != "merging":
                device = device.input

            # finding upstream device
            if device.type == "branching":
                device = device.diversion
                while device.type != "GHX" and device.type != "bldg":
                    device = device.output
            else:
                device = device.merger
            while device.type != "GHX" and device.type != "bldg":
                device = device.input

            HX.upstream_device = device
            device.downstream_device = HX

        # finding ISHX upstream device in HP side

        for HX in self.heat_exchangers:
            device = HX.HP_input
            # finding first upstream branching node
            while device.type != "branching" and device.type != "merging":
                device = device.input

            # finding upstream device
            if device.type == "branching":
                device = device.diversion
                while device.type != "bldg":
                    device = device.output
            else:
                device = device.merger
                while device.type != "bldg":
                    device = device.input

            HX.upstream_device_HP = device
            device.downstream_device = HX

        # finding ISHX downstream device in HP side

        for HX in self.heat_exchangers:
            device = HX.HP_output

            # finding first branching node
            while device.type != "branching":
                device = device.output

            # finding downstream device
            device = device.diversion
            while device.type != "bldg":
                device = device.output

            HX.downstream_device_HP = device
            device.upstream_device = HX

    def UpdateThermalConnections(self):

        # Updating pipe directions

        for pipe in self.pipes:
            if pipe.mass_flow_rate < 0:
                pipe.inlet = pipe.output
                pipe.outlet = pipe.input
                # assigning inlets and outlets to connecting nodes
                pipe.inlet.outlet = pipe
                pipe.outlet.inlet = pipe
            else:
                pipe.inlet = pipe.input
                pipe.outlet = pipe.output
                # assigning inlets and outlets to connecting nodes
                pipe.inlet.outlet = pipe
                pipe.outlet.inlet = pipe

        # Updating zone/ heat pump directions

        for bldg in self.buildings:
            if bldg.mass_bldg < 0:
                # reversing input and output nodes
                bldg.inlet = bldg.output
                bldg.outlet = bldg.input
                # reversing pipe flow directions for pipes connecting zones to inner and outer rings
                bldg.input.input.inlet = bldg.input.input.output
                bldg.input.input.outlet = bldg.input.input.input
                bldg.output.output.inlet = bldg.output.output.output
                bldg.output.output.outlet = bldg.output.output.input
                # assigning inlets and outlets to connecting nodes
                bldg.inlet.outlet = bldg
                bldg.outlet.inlet = bldg

            else:
                bldg.inlet = bldg.input
                bldg.outlet = bldg.output
                # the flow remains same if mass flow rate is not negative
                bldg.input.input.inlet = bldg.input.input.input
                bldg.input.input.outlet = bldg.input.input.output
                bldg.output.output.inlet = bldg.output.output.input
                bldg.output.output.outlet = bldg.output.output.output
                # assigning inlets and outlets to connecting nodes
                bldg.inlet.outlet = bldg
                bldg.outlet.inlet = bldg

        # Updating GHE directions

        for ghx in self.ground_heat_exchangers:
            if ghx.mass_flow_ghe < 0:
                # reversing input and output nodes
                ghx.inlet = ghx.output
                ghx.outlet = ghx.input
                # reversing pipe flow directions for pipes connecting zones to inner and outer rings
                ghx.output.output.inlet = ghx.output.output.output
                ghx.output.output.outlet = ghx.output.output.input
                ghx.input.input.inlet = ghx.input.input.output
                ghx.input.input.outlet = ghx.input.input.input
                # assigning inlets and outlets to connecting nodes
                ghx.inlet.outlet = ghx
                ghx.outlet.inlet = ghx

            else:
                ghx.inlet = ghx.input
                ghx.outlet = ghx.output
                ghx.output.output.inlet = ghx.output.output.input
                ghx.output.output.outlet = ghx.output.output.output
                ghx.input.input.inlet = ghx.input.input.input
                ghx.input.input.outlet = ghx.input.input.output
                # assigning inlets and outlets to connecting nodes
                ghx.inlet.outlet = ghx
                ghx.outlet.inlet = ghx

        self.UpdateNodeTypes()

    def UpdateNodeTypes(self):

        for node in self.nodes:

            if node.type not in ("branching", "merging"):
                continue

            components = [
                node.input,
                node.output,
                node.device_pipe
            ]

            n_in = sum(component.outlet is node for component in components)
            n_out = sum(component.inlet is node for component in components)

            if n_in == 2 and n_out == 1:
                node.temp_type = "merging"

            elif n_in == 1 and n_out == 2:
                node.temp_type = "branching"

            else:
                raise ValueError(
                    f"Invalid node flow pattern: "
                    f"{n_in} incoming, {n_out} outgoing."
                )

    def is_zero_flow_timestep(self):

        for pipe in self.pipes:
            if abs(pipe.mass_flow_rate) > 0:
                return False

        for bldg in self.buildings:
            if abs(bldg.mass_bldg) > 0:
                return False

        for ghx in self.ground_heat_exchangers:
            if abs(ghx.mass_flow_ghe) > 0:
                return False

        return True


def FindItemByID(ID, objectlist):
    # search a list of objects to find one with a particular name
    # of course, the objects must have a "name" member
    for item in objectlist:  # all objects in the list
        if item.ID == ID:  # does it have the ID I am seeking?
            return item  # then return this one
    # next item
    return None  # couldn't find it