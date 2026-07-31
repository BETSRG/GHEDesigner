import json
import math
import time
from abc import ABC, abstractmethod
from importlib import resources
from itertools import product
from math import cos, isclose, sin
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize

from ghedesigner.constants import (
    BOREHOLES_PER_SQUARE_METER,
    DAYS_IN_YEAR,
    DLA_BINS_PER_LEVEL,
    DLA_EXPANSION_RATE,
    HORZ_LIBRARY_FILENAME,
    HOURS_IN_YEAR,
    IDX_COMPARISON_OFFSET_1,
    IDX_COMPARISON_OFFSET_2,
    PI_OVER_2,
    SEC_IN_DAY,
    SEC_IN_HR,
    SEC_IN_YEAR,
    SIMULATION_CONSTANT_COP_OFFSET,
    TWO_PI,
)
from ghedesigner.enums import CentralLoopType, DesignGeomType, SimCompType, SourceSinkOpMode
from ghedesigner.ghe.domains import polygonal_land_constraint_multi_field
from ghedesigner.ghe.hp_hybrid_loads_processor import ProcessLoads
from ghedesigner.ghe.manager import GroundHeatExchanger
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Soil
from ghedesigner.utilities import HPmodel, float_tuple_to_string, get_loads, load_input_file


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
        constant_time_step=True,
    ):
        self.exp_rate = exp_rate
        self.bins_per_level = bins_per_level
        self.constant_time_step = constant_time_step
        self.base_dt_sec = base_dt_sec

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
        self.dts_reciprocal = 1.0 / self.dts
        if constant_time_step:
            self.constant_frac_shift = np.zeros(self.num_bins, dtype=float)
            self.constant_frac_shift[:-1] = base_dt_sec * self.dts_reciprocal[:-1]
        self.deltas = np.zeros(self.num_bins, dtype=float)
        self.average_vals = np.zeros(self.num_bins, dtype=float)

    def shift_and_add(self, new_value: float, current_dt_sec: float, idx_timestep: int):
        """
        Shifts historical values further into the load history and adds the new timestep value.
        Tracking last_idx ensures idempotent shifts if called multiple times by coupled components.
        """
        if idx_timestep > self.last_idx:
            if self.constant_time_step:
                frac_shift = self.constant_frac_shift
            else:
                frac_shift = current_dt_sec * self.dts_reciprocal
                frac_shift[-1] = 0.0
            np.multiply(self.energy_bins, frac_shift, out=self.deltas)
            self.energy_bins -= self.deltas
            self.energy_bins[1:] += self.deltas[:-1]
            if self.constant_time_step:
                self.energy_bins[0] += new_value * self.base_dt_sec
            else:
                self.energy_bins[0] += new_value * current_dt_sec
            self.last_idx = idx_timestep

    def get_step_changes(self) -> np.ndarray:
        """Returns the discrete step changes between consecutive averaged bins."""
        np.multiply(self.energy_bins, self.dts_reciprocal, out=self.average_vals)
        self.average_vals[:-1] = self.average_vals[:-1] - self.average_vals[1:]
        return self.average_vals

    def clear_history(self):
        self.energy_bins = np.zeros(self.num_bins, dtype=float)
        self.last_idx = 0


def timestep_params_generator(time_array: np.ndarray) -> tuple[float, bool]:
    time_differences_sec = np.diff(time_array) * SEC_IN_HR
    if time_differences_sec.size == 0:
        return SEC_IN_HR, True

    base_dt_sec = float(time_differences_sec[0])
    constant_time_step = bool(np.allclose(time_differences_sec, base_dt_sec))
    return base_dt_sec, constant_time_step


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
    def generate_matrix(
        self,
        mass_bldg,
        mass_loop,
        mass_loop_bldg,
        mass_flow_ghe,
        mass_loop_ghe,
        idx_timestep: int,
        configuration,
        method,
    ):
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
        time_step_params: tuple[float, bool],
        load_method: str = "hourly",
    ):
        super().__init__()
        self.name = name
        self.comp_type = None
        self.num_timesteps = num_timesteps
        self.time_array = time_array
        self.base_dt_sec, self.constant_time_step = time_step_params

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

        # Initialize load aggregation if specified
        self.load_method = load_method
        if self.load_method == "hourlyloadagg":
            total_sim_time_sec = (self.time_array[-1] - self.time_array[0]) * SEC_IN_HR
            self.aggregators = [
                DynamicAggregator(
                    total_sim_time_sec,
                    exp_rate=DLA_EXPANSION_RATE,
                    bins_per_level=DLA_BINS_PER_LEVEL,
                    base_dt_sec=self.base_dt_sec,
                    constant_time_step=self.constant_time_step,
                )
                for _ in range(self.num_segments)
            ]
            tau_agg = self.aggregators[0].bin_ages / self.t_p
            self.y_agg_evals = self.two_pi_k * self.q_prime_interp(tau_agg)
        elif self.constant_time_step:
            tau_vals = (self.time_array[-1] - self.time_array[0:-1]) * SEC_IN_HR / self.t_p
            self.y_evals = self.two_pi_k * self.q_prime_interp(tau_vals)

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
            if idx_timestep > IDX_COMPARISON_OFFSET_1:
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

        if idx_timestep > IDX_COMPARISON_OFFSET_1:
            if self.constant_time_step:
                y_transient_array[0 : idx_timestep - 1] = self.y_evals[-idx_timestep:-1]
            else:
                dt_sec_array = (self.time_array[idx_timestep] - self.time_array[0:idx_timestep]) * SEC_IN_HR
                tau_array = dt_sec_array / self.t_p  # Convert to dimensionless time

                # Ask for q' using tau
                q_prime_array = self.q_prime_interp(tau_array)
                y_transient_array[0:idx_timestep] = self.two_pi_k * q_prime_array

        if self.constant_time_step:
            self.y_n[idx_timestep] = self.y_evals[-1]
        else:
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
            idx_q = self.row_index + 3 * k + 2
            idx_t_out = self.row_index + 3 * k + 3

            idx_t_in_seg = self.row_index if k == 0 else self.row_index + 3 * (k - 1) + 3

            # Eq 1: Ground Admittance formulation
            rows[3 * k + 1][idx_q] = 1.0
            rows[3 * k + 1][idx_t_m] = -yn

            t_m_prev = self.t_mean_seg[k, idx_timestep - 1]
            rhs[3 * k + 1] = self.history_term_seg[k, idx_timestep] + yn * (-current_ugt - t_m_prev + prev_ugt)

            # Eq 2: Mean Temp
            rows[3 * k + 2][idx_t_in_seg] = -1.0
            rows[3 * k + 2][idx_t_m] = 2.0
            rows[3 * k + 2][idx_t_out] = -1.0

            # Eq 3: Energy Bal w/ Capacitance
            rows[3 * k + 3][idx_t_in_seg] = m_cp
            rows[3 * k + 3][idx_t_out] = -m_cp
            rows[3 * k + 3][idx_q] = -self.L_seg
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
        time_step_params: tuple[float, bool],
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
        self.base_dt_sec, self.constant_time_step = time_step_params
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

        self.load_method = load_method
        if self.load_method == "hourlyloadagg":
            total_sim_time_sec = (self.time_array[-1] - self.time_array[0]) * SEC_IN_HR
            self.aggregators = [
                DynamicAggregator(
                    total_sim_time_sec,
                    exp_rate=DLA_EXPANSION_RATE,
                    bins_per_level=DLA_BINS_PER_LEVEL,
                    base_dt_sec=self.base_dt_sec,
                    constant_time_step=self.constant_time_step,
                )
                for _ in range(self.num_segments)
            ]
            tau_agg = self.aggregators[0].bin_ages / self.t_p
            y_even_agg = self.two_pi_k * self.q_prime_even_interp(tau_agg)
            y_odd_agg = self.two_pi_k * self.q_prime_odd_interp(tau_agg)
            self.y_self_agg_evals = (y_even_agg + y_odd_agg) / 2.0
            self.y_cross_agg_evals = (y_even_agg - y_odd_agg) / 2.0
        elif self.constant_time_step:
            tau_vals = (self.time_array[-1] - self.time_array[:-1]) * SEC_IN_HR / self.t_p
            y_even_evals = self.two_pi_k * self.q_prime_even_interp(tau_vals)
            y_odd_evals = self.two_pi_k * self.q_prime_odd_interp(tau_vals)
            self.y_self_evals = (y_even_evals + y_odd_evals) * 0.5
            self.y_cross_evals = (y_even_evals - y_odd_evals) * 0.5

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
            if idx_timestep > IDX_COMPARISON_OFFSET_1:
                dt_sec = (self.time_array[idx_timestep - 1] - self.time_array[idx_timestep - 2]) * SEC_IN_HR
                prev_time_sec = self.time_array[idx_timestep - 1] * SEC_IN_HR
                prev_ugt = self.calculate_current_ugt(prev_time_sec)
                prev_ugt_neighbor = self.coupled_pipe.calculate_current_ugt(prev_time_sec)

                for k in range(self.num_segments):
                    # Safely shift self
                    theta_prev = self.t_mean_seg[k, idx_timestep - 1] - prev_ugt
                    self.aggregators[k].shift_and_add(theta_prev, dt_sec, idx_timestep)

                    # Safely shift neighbor ensuring asynchronous state consistency
                    neighbor_k = self.num_segments - 1 - k if self.counter_flow else k
                    theta_prev_neighbor = self.coupled_pipe.t_mean_seg[neighbor_k, idx_timestep - 1] - prev_ugt_neighbor
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

        if idx_timestep > IDX_COMPARISON_OFFSET_1:
            if self.constant_time_step:
                y_self_array[0 : idx_timestep - 1] = self.y_self_evals[-idx_timestep:-1]
                y_cross_array[0 : idx_timestep - 1] = self.y_cross_evals[-idx_timestep:-1]
            else:
                dt_sec_array = (self.time_array[idx_timestep] - self.time_array[0:idx_timestep]) * SEC_IN_HR
                tau_array = dt_sec_array / self.t_p  # Convert to dimensionless time

                # Ask for q' using tau
                y_even_array = self.two_pi_k * self.q_prime_even_interp(tau_array)
                y_odd_array = self.two_pi_k * self.q_prime_odd_interp(tau_array)

                y_self_array[0:idx_timestep] = (y_even_array + y_odd_array) / 2.0
                y_cross_array[0:idx_timestep] = (y_even_array - y_odd_array) / 2.0

        if self.constant_time_step:
            self.y_n[idx_timestep] = self.y_self_evals[-1]
            self.y_cross[idx_timestep] = self.y_cross_evals[-1]
        else:
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

    def generate_matrix(
        self,
        _mass_bldg: float,
        mass_loop: float,
        _mass_loop_bldg: float,
        _mass_flow_ghe: float,
        _mass_loop_ghe: float,
        idx_timestep: int,
        _configuration: CentralLoopType,
        _method: str,
    ) -> tuple[list[np.ndarray], list[float]]:
        if self.cp is None:
            raise ValueError("cp is uninitialized")
        if self.matrix_size is None:
            raise ValueError("matrix_size is uninitialized")
        t_in = self.t_in[0] if idx_timestep == 1 else self.t_in[idx_timestep - 1]
        is_running = self.is_running(t_in)
        self.operating[idx_timestep - 1] = is_running
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

    def __init__(
        self,
        ghe_id: str,
        ghe_data: dict,
        fluid: Fluid,
        loop_config: CentralLoopType,
        num_timesteps: int,
        time_array: np.ndarray[tuple[int], np.dtype[np.float64]],
        time_step_params: tuple[float, bool],
        sizing_end_month=240,
        load_method: str = "hourly",
    ):
        super().__init__()
        self.name = ghe_id
        self.comp_type = SimCompType.GROUND_HEAT_EXCHANGER
        self.height = None
        self.m_dot_total = None
        self.loop_config = loop_config
        self.sizing_end_month = sizing_end_month
        self.search, self.search_time = None, None
        self.num_timesteps = num_timesteps
        self.is_bypassed = False
        self.time_differences = np.diff(time_array)
        self.base_dt_sec, self.constant_time_step = time_step_params

        self.ghe_manager = GroundHeatExchanger.init_from_dictionary(
            ghe_data,
            {
                "fluid_name": fluid.name,
                "concentration_percent": fluid.concentration_percent,
                "temperature": fluid.temperature,
            },
        )
        self.ghe_manager.ghe_setup(ghe_data)
        self.ghe_manager.continue_if_design_unmet = True

        self.fluid = self.ghe_manager.fluid
        self.bh_type = self.ghe_manager.pipe.type
        self.split_ratio: float = 0.1

        self.two_pi_k_recip = 1.0 / (TWO_PI * self.ghe_manager.soil.k)
        self.time_array = time_array

        self.num_timesteps = num_timesteps
        self.history_terms, self.total_values_ghe, self.q_ghe = (
            np.full(self.num_timesteps + 1, self.ghe_manager.soil.ugt, dtype=float),
            np.zeros(self.num_timesteps, dtype=float),
            np.zeros(self.num_timesteps, dtype=float),
        )

        self.t_in = np.full(self.num_timesteps, self.ghe_manager.soil.ugt, dtype=float)
        self.t_mean = np.full(self.num_timesteps, self.ghe_manager.soil.ugt, dtype=float)
        self.t_mix_out = np.full(self.num_timesteps, self.ghe_manager.soil.ugt, dtype=float)
        self.t_out = np.full(self.num_timesteps, self.ghe_manager.soil.ugt, dtype=float)

        self.cp = self.ghe_manager.fluid.cp
        self.tg = self.ghe_manager.soil.ugt

        self.b = None
        self.g = None
        self.ts = None
        self.nbh: int = 0
        self.bh_effective_resist = None
        self.c_n = None
        self.borefield_coordinates = None
        self.dq = None
        self.dim_less_time = None
        self.gfunction_evals = None

        self.load_method = load_method

        if load_method == "hourlyloadagg":
            total_sim_time_sec = (self.time_array[-1] - self.time_array[0]) * SEC_IN_HR
            self.aggregator = DynamicAggregator(
                total_sim_time_sec,
                exp_rate=DLA_EXPANSION_RATE,
                bins_per_level=DLA_BINS_PER_LEVEL,
                base_dt_sec=self.base_dt_sec,
                constant_time_step=self.constant_time_step,
            )
            self.g_agg = None

        if self.ghe_manager.is_sizable:
            self.ghe_designed = False
            self.base_max_eft = self.ghe_manager.max_eft
            self.base_min_eft = self.ghe_manager.min_eft
        else:
            self.ghe_designed = True
            self.ghe_manager.initialize_pre_designed_ghe()
            self.update_ghe_parameters()

    def design_new_ghe(self, load_profile=None, max_eft=None, min_eft=None):
        if not self.ghe_manager.is_sizable:
            self.ghe_designed = True
            self.ghe_manager.initialize_pre_designed_ghe()
            self.borefield_coordinates = self.ghe_manager.pre_designed_locations
        else:
            self.ghe_designed = True
            self.ghe_manager.max_eft = max_eft
            self.ghe_manager.min_eft = min_eft
            self.search, self.search_time, _ = self.ghe_manager.design_and_size_ghe(
                self.sizing_end_month, loads_override=load_profile
            )
            self.borefield_coordinates = self.ghe_manager.current_ghe.gFunction.bore_locations

    def update_ghe_design_desired_nbh(self, desired_nbh):
        if desired_nbh == 0:
            self.is_bypassed = True
            self.ghe_manager.current_ghe.nbh = 0
            self.nbh = 0
            self.borefield_coordinates = []
        else:
            self.ghe_designed = True
            self.is_bypassed = False
            self.ghe_manager.new_nbh_design(desired_nbh)
            self.borefield_coordinates = self.ghe_manager.current_ghe.gFunction.bore_locations

    def update_ghe_design_coordinate(self, new_coordinates):
        if len(new_coordinates) == 0:
            self.is_bypassed = True
            self.ghe_manager.current_ghe.nbh = 0
            self.nbh = 0
            self.borefield_coordinates = []
        else:
            self.ghe_designed = True
            self.ghe_manager.pre_designed_height = self.ghe_manager.max_height
            self.is_bypassed = False
            self.borefield_coordinates = new_coordinates
            self.ghe_manager.pre_designed_locations = self.borefield_coordinates
            self.ghe_manager.initialize_pre_designed_ghe()

    def update_ghe_design_target_spacing(self, target_spacing):
        if self.ghe_manager.geom_type != DesignGeomType.ROWWISE:
            raise ValueError(
                '"update_ghe_design_target_spacing" can only be used on GHEs which have RowWisegeometric constraints.'
            )
        self.ghe_designed = True
        self.is_bypassed = False
        self.ghe_manager.new_ts_design(target_spacing)
        self.borefield_coordinates = self.ghe_manager.current_ghe.gFunction.bore_locations.tolist()

    def update_ghe_design_height(self, new_height):
        if self.is_bypassed:
            pass
        else:
            self.ghe_manager.pre_designed_height = new_height
            self.ghe_manager.initialize_pre_designed_ghe()

    def update_ghe_parameters(self):
        if not self.ghe_designed and not self.is_bypassed:
            raise ValueError("A GHE must be either pre-provided or designed before the parameters can be updated.")
        self.history_terms, self.total_values_ghe, self.q_ghe = (
            np.full(self.num_timesteps + 1, self.ghe_manager.soil.ugt, dtype=float),
            np.zeros(self.num_timesteps, dtype=float),
            np.zeros(self.num_timesteps, dtype=float),
        )
        if self.is_bypassed:
            return
        ghe_object = self.ghe_manager.current_ghe
        self.b = ghe_object.b_spacing
        self.height = ghe_object.bhe.borehole.H
        self.nbh = ghe_object.nbh

        b_over_h = self.b / self.height
        ghe_object.bhe_eq.calc_sts_g_functions()
        self.g, _ = ghe_object.grab_g_function(b_over_h)
        self.ts = ghe_object.bhe_eq.t_s
        self.bh_effective_resist = ghe_object.bhe.calc_effective_borehole_resistance()
        self.dq = np.zeros(self.num_timesteps, dtype=float)
        self.dim_less_time = np.log((self.time_array[-1] - self.time_array[0:-1]) / (self.ts / SEC_IN_HR))
        self.gfunction_evals = self.g(self.dim_less_time)
        self.c_n = self.calc_cn_constant()

        if self.load_method == "hourlyloadagg":
            self.aggregator.clear_history()
            lntts_agg = np.log(self.aggregator.bin_ages / self.ts)
            self.g_agg = self.g(lntts_agg)

        self.t_in = np.full(self.num_timesteps, self.ghe_manager.soil.ugt, dtype=float)
        self.t_mean = np.full(self.num_timesteps, self.ghe_manager.soil.ugt, dtype=float)
        self.t_mix_out = np.full(self.num_timesteps, self.ghe_manager.soil.ugt, dtype=float)
        self.t_out = np.full(self.num_timesteps, self.ghe_manager.soil.ugt, dtype=float)

    def calc_cn_constant(self):
        """
        Calculate C_n values for three GHEs based on their g-functions.

        Cn = 1 / (2 * pi * K_s) * g((tn - tn-1) / t_s) + R_b
        """
        g_vals = np.ones(self.num_timesteps, dtype=float)
        g_vals *= self.gfunction_evals[-1]
        c_n = g_vals * self.two_pi_k_recip + self.bh_effective_resist

        return c_n

    def calc_history_term(self, idx_timestep):
        """
        Computes the history term H_n for this GHX at time index `i`.
        Updates self.total_values_ghe and self.H_n_ghe in place.
        """
        if idx_timestep == 0:
            raise IndexError("Timestep index error")
        # Compute contributions from all previous steps

        if getattr(self, "load_method", "hourly") == "hourlyloadagg":
            if idx_timestep > IDX_COMPARISON_OFFSET_1:
                dt_sec = (self.time_array[idx_timestep - 1] - self.time_array[idx_timestep - 2]) * SEC_IN_HR
                self.aggregator.shift_and_add(self.q_ghe[idx_timestep - 2], dt_sec, idx_timestep)
                dq_b = self.aggregator.get_step_changes()
                values = np.dot(dq_b * self.two_pi_k_recip, self.g_agg)
            else:
                values = 0.0

        else:
            if idx_timestep > IDX_COMPARISON_OFFSET_2:
                self.dq[idx_timestep - 2] -= self.q_ghe[idx_timestep - 3] * self.two_pi_k_recip
            if idx_timestep > IDX_COMPARISON_OFFSET_1:
                self.dq[idx_timestep - 2] += self.q_ghe[idx_timestep - 2] * self.two_pi_k_recip
                if self.constant_time_step:  # Handles hourly (or other constant timesteps)
                    values = np.dot(self.dq[0 : idx_timestep - 1], self.gfunction_evals[-idx_timestep + 1 :])
                else:  # Handles hybrid (or other uneven timesteps)
                    gfunction_evals = self.g(
                        np.log(
                            (self.time_array[idx_timestep - 1] - self.time_array[0 : idx_timestep - 1])
                            / (self.ts / SEC_IN_HR)
                        )
                    )
                    values = np.dot(self.dq[0 : idx_timestep - 1], gfunction_evals)
            else:
                values = 0

        self.total_values_ghe[idx_timestep - 1] = values

        # Contribution from the last time step only
        self.history_terms[idx_timestep] = (
            self.ghe_manager.soil.ugt
            - self.total_values_ghe[idx_timestep - 1]
            + (self.q_ghe[idx_timestep - 2] * self.two_pi_k_recip * self.gfunction_evals[-1])
        )

        return self.history_terms[idx_timestep]

    def generate_matrix(
        self, _mass_bldg, mass_loop, _mass_loop_bldg, mass_flow_ghe, mass_loop_ghe, idx_timestep, configuration, _method
    ):
        row_1 = np.zeros(self.matrix_size, dtype=np.float64)
        row_2 = np.zeros(self.matrix_size, dtype=np.float64)
        row_3 = np.zeros(self.matrix_size, dtype=np.float64)
        row_4 = np.zeros(self.matrix_size, dtype=np.float64)
        if self.is_bypassed:
            # If this ghe is bypassed q_ghe is 0 and the outgoing temperature is the same as the entering one.
            if configuration == CentralLoopType.ONEPIPE:
                row_1[self.row_index] = -1.0
                row_1[self.downstream_index] = 1.0

                row_2[self.row_index + 2] = 1.0

                row_3[self.row_index + 1] = 1.0
                row_3[self.row_index] = -1.0

                row_4[self.row_index + 3] = 1.0
                row_4[self.row_index] = -1.0

                rhs_1, rhs_2, rhs_3, rhs_4 = 0, 0, 0, 0
            elif configuration == CentralLoopType.TWOPIPE:
                row_4[self.row_index] = -1.0
                if self.downstream_device.comp_type == SimCompType.GROUND_HEAT_EXCHANGER:
                    row_4[self.downstream_index] = 1.0
                else:
                    row_4[self.downstream_device.inlet_index] = 1.0

                row_1[self.row_index + 2] = 1.0

                row_2[self.row_index + 1] = 1.0
                row_2[self.row_index] = -1.0

                row_3[self.row_index + 3] = 1.0
                row_3[self.row_index] = -1.0

                rhs_1, rhs_2, rhs_3, rhs_4 = 0, 0, 0, 0
            else:
                raise ValueError(f"Unknown configuration: {configuration}")
        else:
            _ = self.calc_history_term(idx_timestep)
            if configuration == CentralLoopType.ONEPIPE:
                row_1[self.row_index] = (mass_loop - mass_flow_ghe) * self.cp
                row_1[self.row_index + 3] = mass_flow_ghe * self.cp
                row_1[self.downstream_index] = -mass_loop * self.cp

                row_2[self.row_index + 1] = 1
                row_2[self.row_index + 2] = self.c_n[idx_timestep - 1]

                row_3[self.row_index] = -1
                row_3[self.row_index + 1] = 2
                row_3[self.row_index + 3] = -1

                row_4[self.row_index] = mass_flow_ghe * self.cp
                row_4[self.row_index + 2] = self.height * self.nbh
                row_4[self.row_index + 3] = -mass_flow_ghe * self.cp

                rhs_1, rhs_2, rhs_3, rhs_4 = 0, self.history_terms[idx_timestep], 0, 0
            elif configuration == CentralLoopType.TWOPIPE:
                row_1[self.row_index + 1] = 1
                row_1[self.row_index + 2] = self.c_n[idx_timestep - 1]

                row_2[self.row_index + 1] = 2
                row_2[self.inlet_index] = -1
                row_2[self.row_index + 3] = -1

                row_3[self.inlet_index] = mass_flow_ghe * self.cp
                row_3[self.row_index + 3] = -mass_flow_ghe * self.cp
                row_3[self.row_index + 2] = -self.nbh * self.height

                row_4[self.inlet_index] = (mass_loop_ghe - mass_flow_ghe) * self.cp
                row_4[self.row_index + 3] = mass_flow_ghe * self.cp

                if self.downstream_device.comp_type == SimCompType.GROUND_HEAT_EXCHANGER:
                    row_4[self.downstream_index] = -mass_loop_ghe * self.cp
                else:
                    row_4[self.downstream_device.inlet_index] = -mass_loop_ghe * self.cp
                rhs_1, rhs_2, rhs_3, rhs_4 = self.history_terms[idx_timestep], 0, 0, 0
            else:
                raise ValueError(f"Unknown configuration: {configuration}")
        rows = [row_1, row_2, row_3, row_4]
        rhs = [rhs_1, rhs_2, rhs_3, rhs_4]
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
        constant_cop=False,
        load_method: str = "hourly",
        external_loads: dict | None = None,
    ):
        super().__init__()
        self.name = bldg_id
        self.comp_type = SimCompType.BUILDING
        self.matrix_size: int | None = None
        self.cp: float | None = None
        self.constant_cop = constant_cop
        if constant_cop:
            self.loads = None

        self.fluid = fluid
        self.loop_config = loop_config
        self.heating_exists = bool("heating_load" in bldg_data)
        self.cooling_exists = bool("cooling_load" in bldg_data)
        self.num_timesteps = num_timesteps
        self.sim_years = num_timesteps // HOURS_IN_YEAR

        self.htg_vals: np.ndarray = np.zeros(self.num_timesteps, dtype=float)
        self.clg_vals: np.ndarray = np.zeros(self.num_timesteps, dtype=float)
        if "max_eft" in bldg_data and "min_eft" in bldg_data:
            self.max_eft = bldg_data["max_eft"]
            self.min_eft = bldg_data["min_eft"]
        else:
            self.max_eft = 0.0
            self.min_eft = 0.0

        self.heating_fixed_cop: float | None = None
        self.hp_htg: HPmodel
        if self.heating_exists:
            if self.constant_cop and "heat_pump_cop" in bldg_data["heating_load"]:
                self.heating_fixed_cop = bldg_data["heating_load"]["heat_pump_cop"]
            else:
                hp_htg_name = bldg_data["heating_load"]["heat_pump_name"]
                hp_htg_data = hp_data[hp_htg_name]
                self.hp_htg = HPmodel(hp_htg_name, hp_htg_data)

        self.cooling_fixed_cop: float | None = None
        self.hp_clg: HPmodel
        if self.cooling_exists:
            if self.constant_cop and "heat_pump_cop" in bldg_data["cooling_load"]:
                self.cooling_fixed_cop = bldg_data["cooling_load"]["heat_pump_cop"]
            else:
                hp_clg_name = bldg_data["cooling_load"]["heat_pump_name"]
                hp_clg_data = hp_data[hp_clg_name]
                self.hp_clg = HPmodel(hp_clg_name, hp_clg_data)

        if load_method in ("hourly", "hourlyloadagg"):
            if self.heating_exists:
                one_yr_htg_vals = np.array(
                    get_loads(self.name + "_htg", SimCompType.HEAT_PUMP.name, bldg_data["heating_load"]),
                    dtype=float,
                )
                hourly_htg = np.tile(one_yr_htg_vals, self.sim_years)
                self.htg_vals = hourly_htg
                # self.htg_vals = np.insert(hourly_htg, 0, 0.0)

            if self.cooling_exists:
                one_yr_clg_vals = np.array(
                    get_loads(self.name + "_clg", SimCompType.HEAT_PUMP.name, bldg_data["cooling_load"]),
                    dtype=float,
                )
                hourly_clg = np.tile(one_yr_clg_vals, self.sim_years)
                self.clg_vals = hourly_clg
                # self.clg_vals = np.insert(hourly_clg, 0, 0.0)

        elif load_method == "hybrid":
            if external_loads is None:
                raise ValueError(f"Hybrid loads missing for building '{bldg_id}'")

            self.htg_vals = np.array(
                external_loads.get("q_htg", np.zeros(self.num_timesteps))[1:],
                dtype=float,
            )
            self.clg_vals = np.array(
                external_loads.get("q_clg", np.zeros(self.num_timesteps))[1:],
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

    def generate_ghe_load_estimate(self, ugt, beta=0.1):
        self.generate_constant_cop_loads(ugt, beta=beta)
        return self.loads

    def get_excess_temperature(self):
        max_temp = np.max(self.t_in)
        min_temp = np.min(self.t_in)
        if self.constant_cop:
            self.heat_transfers_calced = False
        return max(max_temp - self.max_eft, self.min_eft - min_temp)

    def calc_mass_flow_rate(self, t_in, idx_timestep):
        if self.heating_exists:
            if self.heating_fixed_cop is not None:
                cap_htg = abs(self.htg_vals[idx_timestep])
                m_single_hp_htg = cap_htg / (self.cp * SIMULATION_CONSTANT_COP_OFFSET)
            else:
                cap_htg = self.hp_htg.c1_htg * t_in**2 + self.hp_htg.c2_htg * t_in + self.hp_htg.c3_htg
                m_single_hp_htg = self.hp_htg.m_flow_single_hp
        else:
            cap_htg = 0.0
            m_single_hp_htg = 0.0

        if self.cooling_exists:
            if self.cooling_fixed_cop is not None:
                cap_clg = abs(self.clg_vals[idx_timestep])
                m_single_hp_clg = cap_clg / (self.cp * SIMULATION_CONSTANT_COP_OFFSET)
            else:
                cap_clg = self.hp_clg.c1_clg * t_in**2 + self.hp_clg.c2_clg * t_in + self.hp_clg.c3_clg
                m_single_hp_clg = self.hp_clg.m_flow_single_hp
        else:
            cap_clg = 0.0
            m_single_hp_clg = 0.0

        m_single_hp = max(m_single_hp_htg, m_single_hp_clg)
        rtf_htg = abs(self.htg_vals[idx_timestep] / cap_htg) if cap_htg != 0 else 0.0
        rtf_clg = abs(self.clg_vals[idx_timestep] / cap_clg) if cap_clg != 0 else 0.0

        rtf = rtf_htg + rtf_clg

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
        h = self.htg_vals[idx_timestep]  # NB changed it from h = self.htg_vals[idx_timestep-1]??
        c = self.clg_vals[idx_timestep]  # NB changed it from c = self.clg_vals[idx_timestep - 1]

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

    def generate_constant_cop_loads(self, ugt, beta=0.1):
        if self.min_eft == 0.0 and self.max_eft == 0.0:
            min_eft = ugt - SIMULATION_CONSTANT_COP_OFFSET
            max_eft = ugt + SIMULATION_CONSTANT_COP_OFFSET
        else:
            min_eft = self.min_eft
            max_eft = self.max_eft
        self.loads = np.zeros(self.num_timesteps, dtype=float)
        if self.cooling_exists:
            if self.cooling_fixed_cop is not None:
                q_rej_ratio = 1 + 1.0 / self.cooling_fixed_cop
            else:
                cooling_temp = (1 - beta) * max_eft + beta * ugt
                q_rej_ratio = (
                    self.hp_clg.a_clg * cooling_temp * cooling_temp
                    + self.hp_clg.b_clg * cooling_temp
                    + self.hp_clg.c_clg
                )
            self.loads -= q_rej_ratio * self.clg_vals
        if self.heating_exists:
            if self.heating_fixed_cop is not None:
                q_extr_ratio = 1 - 1.0 / self.heating_fixed_cop
            else:
                heating_temp = (1 - beta) * min_eft + beta * ugt
                q_extr_ratio = (
                    self.hp_htg.a_htg * heating_temp * heating_temp
                    + self.hp_htg.b_htg * heating_temp
                    + self.hp_htg.c_htg
                )
            self.loads += q_extr_ratio * self.htg_vals

    def generate_matrix_constant_cop(self, mass_bldg, mass_loop, mass_loop_bldg, idx_timestep, configuration):
        if configuration == CentralLoopType.ONEPIPE:
            row = np.zeros(self.matrix_size, dtype=float)
            row[self.row_index] = mass_loop * self.cp
            row[self.downstream_index] = -mass_loop * self.cp
            rhs = self.loads[idx_timestep]
            return [row], [rhs]
        elif configuration == CentralLoopType.TWOPIPE:
            row1 = np.zeros(self.matrix_size)
            row2 = np.zeros(self.matrix_size)

            if mass_bldg == 0:
                row1[self.inlet_index] = 1
                row1[self.row_index + 1] = -1
            else:
                row1[self.inlet_index] = mass_bldg * self.cp
                row1[self.row_index + 1] = -mass_bldg * self.cp

            row2[self.row_index] = (mass_loop_bldg - mass_bldg) * self.cp
            row2[self.row_index + 1] = mass_bldg * self.cp
            row2[self.downstream_index] = -mass_loop_bldg * self.cp

            rhs1, rhs2 = self.loads[idx_timestep], 0

            rows = [row1, row2]
            rhs = [rhs1, rhs2]

            return rows, rhs
        else:
            raise ValueError(f"Unknown configuration: {configuration}")

    def generate_matrix(
        self, mass_bldg, mass_loop, mass_loop_bldg, _mass_flow_ghe, _mass_loop_ghe, idx_timestep, configuration, _method
    ):
        if self.constant_cop:
            return self.generate_matrix_constant_cop(
                mass_bldg, mass_loop, mass_loop_bldg, idx_timestep - 1, configuration
            )
        else:
            t_in_idx = 0 if idx_timestep == 1 else idx_timestep - 2
            t_in = self.t_in[t_in_idx]
            r1, r2 = self.calc_r1_r2(t_in, idx_timestep - 1)
            if configuration == CentralLoopType.ONEPIPE:
                row = np.zeros(self.matrix_size, dtype=float)
                row[self.row_index] = 1 + r1 / (mass_loop * self.cp)
                row[self.downstream_index] = -1
                rhs = -r2 / (mass_loop * self.cp)
                return [row], [rhs]
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
                rhs = [rhs1, rhs2]

                return rows, rhs
            else:
                raise ValueError(f"Unknown configuration: {configuration}")

    def calc_energy(self):
        """Calculate energy consumption of the heat pump system."""

        if self.cooling_exists:
            if self.cooling_fixed_cop is not None:
                self.power_hp_clg = np.abs(self.clg_vals / self.cooling_fixed_cop)
            else:
                ratio_clg = self.hp_clg.a_clg * self.t_in**2 + self.hp_clg.b_clg * self.t_in + self.hp_clg.c_clg
                self.power_hp_clg = np.abs(self.clg_vals * (ratio_clg - 1))

        if self.heating_exists:
            if self.heating_fixed_cop is not None:
                self.power_hp_htg = self.htg_vals / self.heating_fixed_cop
            else:
                ratio_htg = self.hp_htg.a_htg * self.t_in**2 + self.hp_htg.b_htg * self.t_in + self.hp_htg.c_htg
                self.power_hp_htg = self.htg_vals * (1 - ratio_htg)

        self.power_hp_tot = self.power_hp_clg + self.power_hp_htg

        # power consumed by circulating pump
        if self.heating_exists:
            if self.heating_fixed_cop is not None:
                self.power_circ_pump = 0.0
            else:
                self.power_circ_pump = (
                    self.m_flow / (self.fluid.rho * self.hp_htg.pump_efficiency) * self.hp_htg.design_pressure_loss
                )
        if self.cooling_exists:
            if self.cooling_fixed_cop:
                self.power_circ_pump = 0.0
            else:
                self.power_circ_pump = (
                    self.m_flow / (self.fluid.rho * self.hp_clg.pump_efficiency) * self.hp_clg.design_pressure_loss
                )


class GHEHPSystem:
    total_loads: np.ndarray[tuple[int], np.dtype[np.float64]]
    nbh_selections: list[str]
    excess_temperatures: list[float]
    coordinate_locations: dict[str, list[tuple[float, float]]]
    nbh_values: list[int]
    total_drilling_values: list[float]
    objective_function_values: list[float]
    borehole_heights: list[float]
    previous_objective_function_evaluations: dict[str, dict[str, int | float]]
    guess_idx: int
    sample_rate: int

    def __init__(self, f_path_json: Path):
        self.components: list = []  # Will hold Building, GHX, SourceSinkHeatExchanger, Isolated/CoupledHorizontalPipe
        self.matrix_size = 0
        self.number_of_simulations = 0

        json_data = load_input_file(f_path_json)

        self.loop_config = CentralLoopType[json_data["central_loop"]["pipe_configuration"].upper()]
        self.loop_flow_factor = json_data["central_loop"]["flow_factor"]
        self.loop_pump_efficiency = json_data["central_loop"]["pump_efficiency"]
        self.loop_length = json_data["central_loop"]["loop_length"]
        self.loop_design_pressure_loss_per_meter = json_data["central_loop"]["design_pressure_loss_per_meter"]
        sim_controls = json_data["simulation_control"]
        self.sim_years = sim_controls["simulation_years"]
        if "search_method" in sim_controls:
            self.search_method = sim_controls["search_method"]
        else:
            self.search_method = "GLOBAL_BUPCRS"
        if "constant_cop" in sim_controls:
            self.constant_cop = sim_controls["constant_cop"]
        else:
            self.constant_cop = True

        if "exhaustive_search" in sim_controls:
            self.exhaustive_search = sim_controls["exhaustive_search"]
        else:
            self.exhaustive_search = False
        self.num_timesteps = self.sim_years * HOURS_IN_YEAR
        self.total_loads = np.zeros(self.num_timesteps, dtype=float)
        self.nbh_selections = []
        self.excess_temperatures = []
        self.coordinate_locations = {}
        self.nbh_values = []
        self.total_drilling_values = []
        self.objective_function_values = []
        self.borehole_heights = []
        self.previous_objective_function_evaluations = {}
        self.guess_idx = -1

        if self.search_method in ("GLOBAL_BUPCRS", "GLOBAL_BUPCRS_BR"):
            self.domain: list[list[list[tuple[float, float]]]] = [[[]]]
            self.field_descriptors: list[str] = []
            if self.exhaustive_search:
                self.sample_rate = 100
            if self.search_method == "GLOBAL_BUPCRS":
                self.remove_boreholes = False
            else:
                self.remove_boreholes = True
        elif self.search_method == "GLOBAL_ROWWISE":
            if self.exhaustive_search:
                self.sample_rate = 100
        elif self.search_method == "NELDER-MEAD":
            self.penalty_baseline: float = 0.0
            self.number_of_restarts: int = 1
            self.excess_temperature_tolerance: float = 1e-1
            self.nbh_bounds: list[list[float]] = []
            self.angles: list[float] = []
            self.nbh_vectors: list[str] = []
            if self.exhaustive_search:
                self.sample_rate = 5
        elif self.search_method == "SIMULATION_ONLY":
            pass
        else:
            raise ValueError("Given search method not recognized.")

        if self.search_method in ("GLOBAL_ROWWISE", "NELDER-MEAD"):
            self.max_iter: int = 50

        fluid_data = json_data["fluid"]
        topology_data = json_data["topology"]
        heat_pump_data = json_data.get("heat_pump", {})
        building_data = json_data.get("building", {})
        ghe_data = json_data.get("ground_heat_exchanger", {})
        hx_data = json_data.get("source_sink_heat_exchanger", {})

        horiz_data = json_data.get("horizontal_piping", {})
        ugt_data = json_data.get("ground_temperature_model", {})

        self.use_horizontal = json_data.get("simulation_control", {}).get("horizontal_simulation_considered", False)

        if horiz_data and not ugt_data:
            raise ValueError("A 'ground_temperature_model' block is required when simulating horizontal piping.")

        horiz_axes = {}
        if self.use_horizontal and horiz_data:
            try:
                with resources.files("ghedesigner.ghe").joinpath(HORZ_LIBRARY_FILENAME).open("rb") as f:
                    lib_data = json.load(f)
                table_single = lib_data["table_single"]
                table_parallel = lib_data["table_parallel"]
                horiz_axes = lib_data["axes"]
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
        first_ghe_key = next(iter(json_data["ground_heat_exchanger"]))
        tg = json_data["ground_heat_exchanger"][first_ghe_key]["soil"]["undisturbed_temp"]  # TODO: fix this

        self.sim_years = json_data["simulation_control"]["simulation_years"]
        self.load_method = json_data["simulation_control"].get("load_method", "hourly").lower()

        self.horiz_segments = json_data["simulation_control"].get("horizontal_segments", 3)

        self.hybrid_load_data: dict[str, dict[str, list[float]]] = {}

        if self.load_method in ("hourly", "hourlyloadagg"):
            self.time_array = np.arange(self.sim_years * HOURS_IN_YEAR + 1, dtype=float)
            self.num_timesteps = len(self.time_array) - 1
        elif self.load_method == "hybrid":
            processor = ProcessLoads()
            processor.read_data_from_json_file(json_data)
            processor.read_hp_load_from_json(json_data)
            self.hybrid_load_data = processor.run_hybrid_pipeline()

            first_bldg = next(iter(self.hybrid_load_data))
            self.time_array = (np.array(self.hybrid_load_data[first_bldg]["time"], dtype=float)).flatten()
            self.num_timesteps = len(self.time_array) - 1
            # self.time_array = (np.insert(self.time_array, 0, 0)).flatten()
        else:
            raise ValueError(f"Unknown load_method: {self.load_method}")

        self.time_step_params = timestep_params_generator(self.time_array)

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
        isolated_names = get_comp_names(topology_data, horiz_data, SimCompType.ISOLATED_HORIZONTAL_PIPE)
        coupled_names = get_comp_names(topology_data, horiz_data, SimCompType.COUPLED_HORIZONTAL_PIPE)

        # get needed buildings
        buildings = []
        for this_building_id, this_bldg_data in building_data.items():
            if this_building_id.upper() in building_names:
                external_loads: dict[str, list[float]] = {}
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
                    constant_cop=self.constant_cop,
                    load_method=self.load_method,
                    external_loads=external_loads,
                )
                buildings.append(this_bldg)

        self.num_buildings = len(buildings)
        self.buildings = buildings

        heat_exchangers = []
        for this_hx_id, this_hx_data in hx_data.items():
            if this_hx_id.upper() in hx_names:
                this_hx = SourceSinkHeatExchanger(this_hx_id, this_hx_data, tg, self.num_timesteps)
                heat_exchangers.append(this_hx)

        self.num_heat_exchangers = len(heat_exchangers)
        self.heat_exchangers = heat_exchangers

        # cp = self.fluid.cp

        ground_heat_exchangers = []
        for ghx_id, ghx_item_data in ghe_data.items():
            if ghx_id.upper() in ghx_names:
                this_ghx = GHX(
                    ghx_id,
                    ghx_item_data,
                    self.fluid,
                    self.loop_config,
                    self.num_timesteps,
                    self.time_array,
                    self.time_step_params,
                    load_method=self.load_method,
                )
                self.cp = this_ghx.cp
                ground_heat_exchangers.append(this_ghx)

        self.num_ghx = len(ground_heat_exchangers)
        self.ground_heat_exchangers = ground_heat_exchangers
        self.sizable_ground_heat_exchangers = []
        for ghe in self.ground_heat_exchangers:
            if ghe.ghe_manager.is_sizable:
                self.sizable_ground_heat_exchangers.append(ghe)
        ghx_matrix_rows = GHX.MATRIX_ROWS
        self.matrix_size = (
            ghx_matrix_rows * self.num_ghx
            + Building.MATRIX_ROWS * self.num_buildings
            + SourceSinkHeatExchanger.MATRIX_ROWS * self.num_heat_exchangers
        )

        self.nbh_total: int = sum([x.nbh for x in ground_heat_exchangers]) if ground_heat_exchangers is not None else 0
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

                target_d = get_nearest(h_data["trench_depth"], np.array(horiz_axes["depths"], dtype=float))
                target_beta = get_nearest(beta, np.array(horiz_axes["betas"], dtype=float))
                target_r = get_nearest(h_pipe.r_out, np.array(horiz_axes["radii"], dtype=float))
                target_k = get_nearest(h_soil.k, np.array(horiz_axes["soil_ks"], dtype=float))

                this_horiz: IsolatedHorizontalPipe | CoupledHorizontalPipe
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
                        time_step_params=self.time_step_params,
                        load_method=self.load_method,
                    )
                    this_horiz.comp_type = SimCompType.ISOLATED_HORIZONTAL_PIPE
                    isolated_pipes.append(this_horiz)

                elif is_coupled:
                    target_b = get_nearest(h_data["spacing"], np.array(horiz_axes["spacings"], dtype=float))
                    q_prime_data = table_parallel[
                        float_tuple_to_string((target_d, target_b, target_beta, target_r, target_k))
                    ]
                    q_prime_even = interpolate.interp1d(
                        q_prime_data["x1"], q_prime_data["y1"], kind="cubic", fill_value="extrapolate"
                    )
                    q_prime_odd = interpolate.interp1d(
                        q_prime_data["x2"], q_prime_data["y2"], kind="cubic", fill_value="extrapolate"
                    )

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
                        time_step_params=self.time_step_params,
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

        # Update MATRIX_ROWS handling
        if self.loop_config == CentralLoopType.ONEPIPE:
            Building.MATRIX_ROWS = 1
        else:
            Building.MATRIX_ROWS = 2

        ghx_matrix_rows = GHX.MATRIX_ROWS
        self.matrix_size = (
            ghx_matrix_rows * self.num_ghx
            + Building.MATRIX_ROWS * self.num_buildings
            + SourceSinkHeatExchanger.MATRIX_ROWS * self.num_heat_exchangers
            + sum(pipe.matrix_rows for pipe in horizontal_pipes)
        )

        self.m_flow_loop = np.zeros(self.num_timesteps)
        self.pump_power_loop = np.zeros(self.num_timesteps)

        def get_bldg(name: str) -> Building | None:
            return next((obj for obj in buildings if obj.name and obj.name.upper() == name.upper()), None)

        def get_ghx(name: str) -> GHX | None:
            return next((obj for obj in ground_heat_exchangers if obj.name and obj.name.upper() == name.upper()), None)

        def get_hx(name: str) -> SourceSinkHeatExchanger | None:
            return next((obj for obj in heat_exchangers if obj.name and obj.name.upper() == name.upper()), None)

        def get_horiz(name: str):
            return next((obj for obj in horizontal_pipes if obj.name and obj.name.upper() == name.upper()), None)

        # Topology assembly
        comp: GHX | Building | SourceSinkHeatExchanger | IsolatedHorizontalPipe | CoupledHorizontalPipe | None
        for v in topology_data:
            comp_type = v["type"]
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

        # for this_comp in self.components:
        #     this_comp.matrix_size = self.matrix_size
        #     if isinstance(this_comp, GHX):
        #         this_comp.split_ratio = this_comp.nbh / self.nbh_total
        #     elif isinstance(
        #         this_comp, (Building, SourceSinkHeatExchanger, IsolatedHorizontalPipe, CoupledHorizontalPipe)
        #     ):
        #         this_comp.cp = cp

        # Assigning downstream device to each component
        for i in range(len(self.components)):
            self.components[i].downstream_device = self.components[(i + 1) % len(self.components)]

        # Assigning row_indices
        idx_comp = 0
        for this_comp in self.components:
            this_comp.row_index = idx_comp
            # Use dynamic instance attribute if present, otherwise default to class attribute
            rows_required = getattr(this_comp, "matrix_rows", getattr(this_comp.__class__, "MATRIX_ROWS", 1))
            idx_comp += rows_required
            this_comp.downstream_index = idx_comp

        # set the last component to loop back to the start
        self.components[-1].downstream_index = 0

        # Assigning inlet_index
        common_inlet_index_bldg = None
        common_inlet_index_ghx = None
        if self.loop_config == CentralLoopType.TWOPIPE:
            common_inlet_index_bldg = next(
                (comp.row_index for comp in self.components if comp.comp_type == SimCompType.BUILDING),
                None,
            )
            common_inlet_index_ghx = next(
                (comp.row_index for comp in self.components if comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER),
                None,
            )

        for i, comp in enumerate(self.components):
            if comp.comp_type == SimCompType.BUILDING:
                comp.inlet_index = common_inlet_index_bldg
            elif comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER:
                comp.inlet_index = common_inlet_index_ghx
            elif comp.comp_type in (SimCompType.ISOLATED_HORIZONTAL_PIPE, SimCompType.COUPLED_HORIZONTAL_PIPE):
                # Transit lines simply take the outlet of the component right before them in the topology
                comp.inlet_index = self.components[i - 1].downstream_index
            else:
                pass

    def size_and_simulate(self):
        if np.any([ghe.ghe_manager.is_sizable for ghe in self.ground_heat_exchangers]):
            if self.search_method in ("GLOBAL_BUPCRS", "GLOBAL_BUPCRS_BR"):
                self.design_system_single_bupcrs()
            elif self.search_method == "GLOBAL_ROWWISE":
                self.design_system_single_rowwise()
            elif self.search_method == "NELDER-MEAD":
                self.design_system_optimizer()
            else:
                raise ValueError("search_method does not match any implemented.")
            self.size_ground_heat_exchangers()
        else:
            for ghe in self.ground_heat_exchangers:
                ghe.design_new_ghe()
                ghe.update_ghe_parameters()
            self.solve_system()
        print("Number of simulations completed: ", self.number_of_simulations)

    def calculate_building_excess(self):
        max_excess = None
        for building in self.buildings:
            current_excess = building.get_excess_temperature()
            if max_excess is None or current_excess > max_excess:
                max_excess = current_excess
        return max_excess

    def append_new_coordinates(self, iteration_name):
        new_index = len(self.coordinate_locations)
        self.coordinate_locations[new_index] = {}
        self.coordinate_locations[new_index]["name"] = iteration_name
        for i, ghe in enumerate(self.ground_heat_exchangers):
            self.coordinate_locations[new_index][ghe.name] = ghe.borefield_coordinates

    def set_ground_heat_exchanger_size(self, ratio):
        for ghe in self.sizable_ground_heat_exchangers:
            if not ghe.is_bypassed:
                ghe.update_ghe_design_height(ratio * ghe.ghe_manager.max_height)
                ghe.update_ghe_parameters()

    def size_ground_heat_exchangers(self, size_tolerance=1e-3):
        if len(self.sizable_ground_heat_exchangers) < 1:
            raise ValueError("In order to size ground heatexhangers, at least one must be sizable.")
        r_min = 0.0
        r_max = 1.0

        max_excess_temperature = self.calculate_building_excess()
        if max_excess_temperature > 0:
            raise Warning("Maximum Height GHE cannot meet temperature constraints. Continuing despite this.")
            return
        while True:
            r_mid = 0.5 * (r_min + r_max)
            self.set_ground_heat_exchanger_size(r_mid)
            self.solve_system()
            mid_et = self.calculate_building_excess()
            self.nbh_selections.append(self.nbh_selections[-1])
            self.excess_temperatures.append(mid_et)
            nbh_val, td_val, height_val = self.get_nbh_and_td()
            self.nbh_values.append(nbh_val)
            self.total_drilling_values.append(td_val)
            self.borehole_heights.append(height_val)
            self.objective_function_values.append(0.0)
            if self.search_method == "NELDER-MEAD":
                self.angles.append(self.angles[-1])
                self.nbh_vectors.append(self.nbh_vectors[-1])
            if mid_et > 0:
                r_min = r_mid
            else:
                r_max = r_mid
            if abs(r_max - r_min) < size_tolerance:
                break
        self.set_ground_heat_exchanger_size(r_max)
        self.solve_system()
        _ = self.calculate_building_excess()
        return

    def initialize_system_ghes(self, need_penalty=False):
        # Get penalty multiplier based on the estimated number of maximum boreholes and maximum borehole height.
        if need_penalty:
            self.penalty_baseline = 0
            for i, ghe in enumerate(self.ground_heat_exchangers):
                self.penalty_baseline += (
                    ghe.ghe_manager.get_design_area() * BOREHOLES_PER_SQUARE_METER * ghe.ghe_manager.max_height
                )
        for ghe in self.ground_heat_exchangers:
            if not ghe.ghe_manager.is_sizable:
                ghe.design_new_ghe()
            else:
                ghe.update_ghe_design_coordinate([[0.0, 0.0]])
            ghe.update_ghe_parameters()
        self.guess_idx = -1

    def design_system_single_bupcrs(self):
        # Perform initial sizing of GHEs
        self.initialize_system_ghes(need_penalty=False)

        property_boundaries = []
        nogo_zones = []
        b_min = None
        for ghe in self.sizable_ground_heat_exchangers:
            if ghe.ghe_manager.geom_type == DesignGeomType.BIRECTANGLECONSTRAINED:
                property_boundaries.append(ghe.ghe_manager.geometric_constraint.property_boundary)
                ng = ghe.ghe_manager.geometric_constraint.no_go_boundaries
                if ng is not None:
                    nogo_zones.extend(ng)
                if b_min is None:
                    b_min = ghe.ghe_manager.geometric_constraint.b_min
                else:
                    b_min = min(b_min, ghe.ghe_manager.geometric_constraint.b_min)
            else:
                raise ValueError(
                    'All GHEs must be of type "BIRECTANGLECONSTRAINED" for use in the global BUPCRS'
                    "system design algorithm."
                )

        self.domain, self.field_descriptors = polygonal_land_constraint_multi_field(
            b_min, property_boundaries, no_go_boundaries=nogo_zones
        )

        def objective(field_index, ignore_previous=False):
            self.guess_idx += 1
            eval_key = self.field_descriptors[field_index]
            self.nbh_selections.append(eval_key)
            if eval_key in self.previous_objective_function_evaluations and not ignore_previous:
                self.nbh_values.append(self.previous_objective_function_evaluations[eval_key]["NBH"])
                self.total_drilling_values.append(self.previous_objective_function_evaluations[eval_key]["TD"])
                self.excess_temperatures.append(self.previous_objective_function_evaluations[eval_key]["EXC"])
                self.objective_function_values.append(self.previous_objective_function_evaluations[eval_key]["EXC"])
                self.borehole_heights.append(
                    self.previous_objective_function_evaluations[eval_key]["TD"]
                    / self.previous_objective_function_evaluations[eval_key]["NBH"]
                )
                return self.previous_objective_function_evaluations[eval_key]["EXC"]
            coords = self.domain[field_index]
            for i, coord in enumerate(coords):
                ghe = self.sizable_ground_heat_exchangers[i]
                ghe.update_ghe_design_coordinate(coord)
            nbh_val = 0
            total_drilling_val = 0.0
            for ghe in self.ground_heat_exchangers:
                ghe.update_ghe_parameters()
                nbh_val += ghe.nbh
                total_drilling_val += ghe.nbh * ghe.ghe_manager.current_ghe.bhe.borehole.H
            self.solve_system()
            self.nbh_values.append(nbh_val)
            self.total_drilling_values.append(total_drilling_val)
            self.borehole_heights.append(total_drilling_val / nbh_val)
            self.append_new_coordinates(f"{self.guess_idx}")
            excess_temp = self.calculate_building_excess()
            self.excess_temperatures.append(excess_temp)
            self.objective_function_values.append(excess_temp)
            self.previous_objective_function_evaluations[eval_key] = {
                "NBH": nbh_val,
                "TD": total_drilling_val,
                "EXC": excess_temp,
                "OFE": excess_temp,
            }
            return excess_temp

        def final_bupcrs_adjustment(nbhs):
            self.guess_idx += 1
            self.nbh_selections.append("Final Placement Adjustment")
            for i, ghe in enumerate(self.sizable_ground_heat_exchangers):
                ghe.update_ghe_design_desired_nbh(len(nbhs[i]))
            nbh_val = 0
            total_drilling_val = 0.0
            for ghe in self.ground_heat_exchangers:
                ghe.update_ghe_parameters()
                nbh_val += ghe.nbh
                total_drilling_val += ghe.nbh * ghe.ghe_manager.current_ghe.bhe.borehole.H
            self.solve_system()
            self.nbh_values.append(nbh_val)
            self.total_drilling_values.append(total_drilling_val)
            self.borehole_heights.append(total_drilling_val / nbh_val)
            self.append_new_coordinates(f"{self.guess_idx}")
            excess_temp = self.calculate_building_excess()
            self.excess_temperatures.append(excess_temp)
            self.objective_function_values.append(excess_temp)

        if self.exhaustive_search:
            x_range = np.arange(0, len(self.domain))
            total = x_range.shape[0]
            sample_rate = self.sample_rate
            for i, index in enumerate(x_range):
                if index % sample_rate == 0:
                    print(f"Percent Completed: {100 * index / total}")
                _ = objective(i)
        else:
            min_idx = 0
            max_idx = len(self.domain) - 1
            min_result = objective(min_idx)
            max_result = objective(max_idx)
            if min_result <= 0 and max_result <= 0:
                _ = objective(min_idx, ignore_previous=True)
                if self.remove_boreholes:
                    final_bupcrs_adjustment(self.domain[min_idx])
            elif min_result > 0 and max_result > 0:
                # raise ValueError("Largest borefield cannot meet system temperature requirements. It is suggested"
                #                  "that the minimum spacing or available property area is adjusted to allow for "
                #                  "additional boreholes.")
                _ = objective(max_idx, ignore_previous=True)
                if self.remove_boreholes:
                    final_bupcrs_adjustment(self.domain[max_idx])
            elif min_result >= 0 >= max_result:
                while True:
                    m_idx = int(0.5 * (min_idx + max_idx))
                    if m_idx in (min_idx, max_idx):
                        break
                    m_result = objective(m_idx)
                    if m_result > 0:
                        min_idx = m_idx
                    elif m_result <= 0:
                        max_idx = m_idx
                    else:
                        raise ValueError(
                            "The has been an error in the bisection search logic of the "
                            '"design_system_single_bupcrs" algorithm. Please report.'
                        )
                _ = objective(max_idx, ignore_previous=True)
                if self.remove_boreholes:
                    final_bupcrs_adjustment(self.domain[max_idx])
            else:
                raise ValueError(
                    "There has been an error in the bracketing logic of the"
                    ' "design_system_single_bupcrs" algorithm. Please report.'
                )

    @staticmethod
    def _get_rowwise_spacing_bounds(ground_heat_exchangers):
        if not ground_heat_exchangers:
            raise ValueError("At least one sizable GHE is required for the global ROWWISE system design algorithm.")

        constraints = []
        for ghe in ground_heat_exchangers:
            if ghe.ghe_manager.geom_type == DesignGeomType.ROWWISE:
                constraints.append(ghe.ghe_manager.geometric_constraint)
            else:
                raise ValueError(
                    'All GHEs must be of type "ROWWISE" for use in the global ROWWISEsystem design algorithm.'
                )

        min_target_spacing = max(constraint.min_spacing for constraint in constraints)
        max_target_spacing = min(constraint.max_spacing for constraint in constraints)
        if min_target_spacing > max_target_spacing:
            raise ValueError("Row-wise GHE target-spacing constraints do not have a common range.")

        return min_target_spacing, max_target_spacing

    def design_system_single_rowwise(self):
        min_target_spacing, max_target_spacing = self._get_rowwise_spacing_bounds(self.sizable_ground_heat_exchangers)

        # Perform initial sizing of GHEs
        self.initialize_system_ghes(need_penalty=False)

        def objective(target_spacing):
            self.guess_idx += 1
            eval_key = f"{target_spacing:.3f}m"
            print(eval_key)
            self.nbh_selections.append(eval_key)
            for ghe in self.sizable_ground_heat_exchangers:
                ghe.update_ghe_design_target_spacing(target_spacing)
            nbh_val = 0
            total_drilling_val = 0.0
            for ghe in self.ground_heat_exchangers:
                ghe.update_ghe_parameters()
                nbh_val += ghe.nbh
                total_drilling_val += ghe.nbh * ghe.ghe_manager.current_ghe.bhe.borehole.H
            self.solve_system()
            self.nbh_values.append(nbh_val)
            self.total_drilling_values.append(total_drilling_val)
            self.borehole_heights.append(total_drilling_val / nbh_val)
            self.append_new_coordinates(f"{self.guess_idx}")
            excess_temp = self.calculate_building_excess()
            self.excess_temperatures.append(excess_temp)
            self.objective_function_values.append(excess_temp)
            return excess_temp

        if self.exhaustive_search:
            x_range = np.linspace(min_target_spacing, max_target_spacing, num=19)
            total = x_range.shape[0]
            sample_rate = self.sample_rate
            for i, spacing in enumerate(x_range):
                if i % sample_rate == 0:
                    print(f"Percent Completed: {100 * i / total}")
                _ = objective(spacing)
        else:
            min_result = objective(max_target_spacing)
            max_result = objective(min_target_spacing)
            if min_result <= 0 and max_result <= 0:
                _ = objective(max_target_spacing)
            elif min_result > 0 and max_result > 0:
                # raise ValueError("Largest borefield cannot meet system temperature requirements. It is suggested"
                #                  "that the minimum spacing or available property area is adjusted to allow for "
                #                  "additional boreholes.")
                _ = objective(min_target_spacing)
            elif min_result >= 0 >= max_result:
                n_iter = 0
                while n_iter < self.max_iter:
                    n_iter += 1
                    if isclose(max_target_spacing, min_target_spacing, abs_tol=0.001):
                        break
                    mid_spacing = 0.5 * (min_target_spacing + max_target_spacing)
                    mid_result = objective(mid_spacing)
                    if mid_result > 0:
                        max_target_spacing = mid_spacing
                    elif mid_result <= 0:
                        min_target_spacing = mid_spacing
                    else:
                        raise ValueError(
                            "The has been an error in the bisection search logic of the "
                            '"design_system_single_rowwise" algorithm. Please report.'
                        )
                _ = objective(min_target_spacing)
            else:
                raise ValueError(
                    "There has been an error in the bracketing logic of the"
                    ' "design_system_single_bupcrs" algorithm. Please report.'
                )

    def design_system_optimizer(
        self,
    ):
        # max_iter = self.max_iter
        number_of_restarts = self.number_of_restarts
        excess_temperature_tolerance = self.excess_temperature_tolerance

        self.initialize_system_ghes(need_penalty=True)

        self.nbh_bounds = []
        for ghe in self.sizable_ground_heat_exchangers:
            self.nbh_bounds.append(ghe.ghe_manager.design.get_bounds())
        number_of_sizable_ghes = len(self.sizable_ground_heat_exchangers)

        def nbh_sim(nbhs):
            if np.all(nbhs == 0):
                raise ValueError('Only empty borefields given to "nbh_sim".')
            nbhs = [round(nbh) for nbh in nbhs]
            eval_key = "_".join([str(nbh) for nbh in nbhs])
            if eval_key in self.previous_objective_function_evaluations:
                for i, ghe in enumerate(self.ground_heat_exchangers):
                    ghe.ghe_manager.current_ghe.nbh = self.previous_objective_function_evaluations[eval_key]["NBHS"][i]
                    ghe.borefield_coordinates = self.previous_objective_function_evaluations[eval_key]["coords"][i]
                return self.previous_objective_function_evaluations[eval_key]["EXC"], eval_key
            for i, nbh in enumerate(nbhs):
                ghe = self.sizable_ground_heat_exchangers[i]
                ghe.update_ghe_design_desired_nbh(nbh)
                ghe.update_ghe_parameters()
            self.solve_system()
            excess_temp = self.calculate_building_excess()
            self.previous_objective_function_evaluations[eval_key] = {
                "EXC": excess_temp,
                "NBHS": [ghe.ghe_manager.current_ghe.nbh for ghe in self.ground_heat_exchangers],
                "coords": [ghe.borefield_coordinates for ghe in self.ground_heat_exchangers],
            }
            return excess_temp, eval_key

        self.inner_obj_func = nbh_sim

        def objective(spherical_angles):
            self.guess_idx += 1
            # Find the nbh ratio unit vector from the spherical angles
            number_of_angles = len(spherical_angles)
            self.angles.append("_".join([str(angle) for angle in spherical_angles]))
            ratios = np.zeros(number_of_angles + 1, dtype=float)
            sin_multiplier = 1.0
            for sdx in range(number_of_angles + 1):
                if sdx == number_of_angles:
                    ratios[sdx] = sin_multiplier
                else:
                    current_angle = spherical_angles[sdx]
                    ratios[sdx] = sin_multiplier * cos(current_angle)
                    sin_multiplier *= sin(current_angle)
            # To make the search progression easier to understand, we will be keeping track of the nbh ratios
            # of the borefields rather than the angles. We would also like the ratios to sum up to 1.
            r_one = 1.0 / np.sum(ratios)
            self.nbh_vectors.append("_".join([str(rat * r_one) for rat in ratios]))
            # Next the minimum and maximum length of the nbh-vectors must be determined before rootfinding.
            # The minimum length is when any borefield has a non-zero number of boreholes.
            # The maximum length is when the vector intersects the Maximum NBH bounds.
            r_min = np.inf
            r_max = np.inf
            for r_idx, ratio in enumerate(ratios):
                if ratio == 0:
                    r_r_min = np.inf
                    r_r_max = np.inf
                else:
                    # We will be rounding the nbh vector to the nearest integer, so a value of 0.5 would round up to
                    # an NBH of 1.
                    r_r_min = (0.5 + 1e-8) / ratio
                    r_r_max = self.nbh_bounds[r_idx][1] / ratio
                r_min = min(r_min, r_r_min)
                r_max = min(r_max, r_r_max)

            # Next, we need to check if this angle actually brackets a root.
            min_eft, min_eval_key = self.inner_obj_func(ratios * r_min)
            _, min_td, _ = self.get_nbh_and_td()
            max_eft, max_eval_key = self.inner_obj_func(ratios * r_max)
            max_nbh, max_td, max_height = self.get_nbh_and_td()

            if min_eft > 0 and max_eft > 0:
                self.excess_temperatures.append(max_eft)
                nbh_val, total_drilling_val = max_nbh, max_td
                self.nbh_values.append(nbh_val)
                self.total_drilling_values.append(total_drilling_val)
                self.append_new_coordinates(f"{self.guess_idx}")
                total_drilling_val += self.penalty_baseline * (max_eft + 1.0) ** 2
                self.objective_function_values.append(total_drilling_val)
                self.nbh_selections.append(max_eval_key)
                self.borehole_heights.append(max_height)
                return total_drilling_val
            elif min_eft < 0 and max_eft < 0:
                _, _ = self.inner_obj_func(ratios * r_min)
                self.excess_temperatures.append(min_eft)
                nbh_val, total_drilling_val, height_val = self.get_nbh_and_td()
                self.nbh_values.append(nbh_val)
                self.total_drilling_values.append(total_drilling_val)
                self.append_new_coordinates(f"{self.guess_idx}")
                self.objective_function_values.append(total_drilling_val)
                self.nbh_selections.append(min_eval_key)
                self.borehole_heights.append(height_val)
                return total_drilling_val
            elif min_eft >= 0 >= max_eft:
                while True:
                    r_mid = 0.5 * (r_min + r_max)
                    mid_eft, mid_eval_key = self.inner_obj_func(ratios * r_mid)
                    _, mid_td, _ = self.get_nbh_and_td()
                    if mid_eval_key in (min_eval_key, max_eval_key):
                        break
                    if mid_eft > 0:
                        min_eft = mid_eft
                        min_td = mid_td
                        r_min = r_mid
                        min_eval_key = mid_eval_key
                    else:
                        max_eft = mid_eft
                        max_td = mid_td
                        r_max = r_mid
                        max_eval_key = mid_eval_key
                if min_eft <= -max_eft:
                    min_nbhs = np.array([round(nbh) for nbh in (ratios * r_min)], dtype=float)
                    r_max = min((min_nbhs + 1) / (ratios + 1e-16))
                    max_eft, max_eval_key = self.inner_obj_func(ratios * r_max)
                    max_nbh, max_td, max_height = self.get_nbh_and_td()
                else:
                    max_nbhs = np.array([round(nbh) for nbh in (ratios * r_max)], dtype=float)
                    r_min = max((max_nbhs - 1) / (ratios + 1e-16))
                    min_eft, min_eval_key = self.inner_obj_func(ratios * r_min)
                    _, min_td, _ = self.get_nbh_and_td()
                excess_temp, final_eval_key = self.inner_obj_func(ratios * r_max)
                nbh_val, total_drilling_value, height_val = self.get_nbh_and_td()
                self.nbh_values.append(nbh_val)
                self.total_drilling_values.append(total_drilling_value)
                self.append_new_coordinates(f"{self.guess_idx}")
                self.excess_temperatures.append(excess_temp)
                self.nbh_selections.append(final_eval_key)
                self.borehole_heights.append(height_val)
                if max_td != min_td:
                    interp_r = -min_eft * (r_max - r_min) / (max_eft - min_eft) + r_min
                    obf_val = (max_td - min_td) / (r_max - r_min) * (interp_r - r_min) + min_td
                else:
                    obf_val = total_drilling_value
                self.objective_function_values.append(obf_val)
                return obf_val
            else:
                raise ValueError("Bisection search in NBH ray-cast is producing unexpected results.")

        number_of_search_dimensions = number_of_sizable_ghes - 1
        if self.exhaustive_search:
            x_range = np.linspace(0, PI_OVER_2, num=19)
            total = x_range.shape[0] ** number_of_search_dimensions
            sample_rate = self.sample_rate
            for index, angles in enumerate(product(x_range, repeat=number_of_search_dimensions)):
                if index % sample_rate == 0:
                    print(f"Percent Completed: {100 * index / total}")
                objective(angles)
        else:
            initial_guess = [0.5 * PI_OVER_2 for _ in range(number_of_search_dimensions)]
            initial_simplex = [initial_guess]
            bounds = []
            for i in range(number_of_search_dimensions):
                local_guess = [PI_OVER_2 for _ in range(number_of_search_dimensions)]
                local_guess[i] = 0.0
                initial_simplex.append(local_guess)
                bounds.append((0.0, PI_OVER_2))
            for i in range(number_of_restarts + 1):
                if i == 0:
                    result = minimize(
                        objective,
                        initial_guess,
                        method="Nelder-Mead",
                        bounds=bounds,
                        options={"initial_simplex": initial_simplex, "fatol": excess_temperature_tolerance},
                    )
                else:
                    result = minimize(
                        objective,
                        initial_guess,
                        method="Nelder-Mead",
                        bounds=bounds,
                        options={"fatol": excess_temperature_tolerance},
                    )
                initial_guess = result.x
                print(f"Finished Nelder-Mead Round: {i}; Solver Successful?: {result.success}")

            _ = objective(initial_guess)

    def get_nbh_and_td(self):
        nbh_val = 0
        total_drilling_val = 0.0
        for ghe in self.ground_heat_exchangers:
            nbh_val += ghe.ghe_manager.current_ghe.nbh
            total_drilling_val += ghe.ghe_manager.current_ghe.nbh * ghe.ghe_manager.current_ghe.bhe.borehole.H
        return nbh_val, total_drilling_val, total_drilling_val / nbh_val

    def solve_system(self):
        self.number_of_simulations += 1
        self.solve_system_standard()

    def solve_system_standard(self):
        t_start = time.perf_counter()
        self.nbh_total = sum([x.nbh for x in self.ground_heat_exchangers])
        average_ugt = 0.0
        for this_comp in self.components:
            this_comp.matrix_size = self.matrix_size
            if isinstance(this_comp, GHX):
                this_comp.split_ratio = this_comp.nbh / self.nbh_total
                average_ugt += this_comp.ghe_manager.soil.ugt * this_comp.nbh / self.nbh_total
            elif isinstance(this_comp, (Building, SourceSinkHeatExchanger)):
                this_comp.cp = self.cp

        if self.constant_cop:
            for building in self.buildings:
                building.generate_constant_cop_loads(average_ugt)
                building.t_in = np.full(self.num_timesteps, average_ugt)

        for idx_timestep in range(1, self.num_timesteps + 1):  # loop over all timestep
            matrix_rows = []
            matrix_rhs = []
            total_hp_flow = 0
            m_bldg_cum = 0
            m_ghe_cum = 0

            # ---- RESET flows for all components ----
            for comp in self.components:
                comp.mass_bldg = 0.0
                comp.mass_flow_ghe = 0.0
                comp.mass_flow_pipe = 0.0
                comp.mass_loop_bldg = 0.0
                comp.mass_loop_ghe = 0.0

            for this_comp in self.components:
                if isinstance(this_comp, Building):
                    t_in = this_comp.t_in[idx_timestep - 2]
                    this_comp.mass_bldg = this_comp.calc_mass_flow_rate(t_in, idx_timestep - 1)
                    total_hp_flow += this_comp.mass_bldg
                    m_bldg_cum += this_comp.mass_bldg
                this_comp.mass_loop_bldg = m_bldg_cum

            mass_loop = max(total_hp_flow * self.loop_flow_factor, 0.1)

            for this_comp in self.components:
                if isinstance(this_comp, GHX):
                    this_comp.mass_flow_ghe = mass_loop * this_comp.split_ratio
                    m_ghe_cum += this_comp.mass_flow_ghe
                elif isinstance(this_comp, (IsolatedHorizontalPipe, CoupledHorizontalPipe)):
                    # For a series pipe, the mass flow is the total loop mass flow
                    this_comp.mass_flow_pipe = mass_loop

                this_comp.mass_loop_ghe = m_ghe_cum

                # Note: We pass this_comp.mass_flow_pipe in the mass_flow_ghe slot for Horizontal pipes
                flow_to_pass = getattr(this_comp, "mass_flow_ghe", 0.0)
                if isinstance(this_comp, (IsolatedHorizontalPipe, CoupledHorizontalPipe)):
                    flow_to_pass = this_comp.mass_flow_pipe

                rows, rhs = this_comp.generate_matrix(
                    this_comp.mass_bldg,
                    mass_loop,
                    this_comp.mass_loop_bldg,
                    flow_to_pass,
                    this_comp.mass_loop_ghe,
                    idx_timestep,
                    self.loop_config,
                    self.load_method,
                )
                matrix_rows.extend(rows)
                matrix_rhs.extend(rhs)

            # Solve the system = A * X = B
            a_matrix = np.array(matrix_rows, dtype=float)
            b_vector = np.array(matrix_rhs, dtype=float)
            x_vector = np.linalg.solve(a_matrix, b_vector)

            # save output data
            self.m_flow_loop[idx_timestep - 1] = mass_loop

            for this_comp in self.components:
                row_index = this_comp.row_index
                if this_comp.comp_type == SimCompType.BUILDING:
                    if self.loop_config == CentralLoopType.TWOPIPE:
                        this_comp.t_in[idx_timestep - 1] = x_vector[this_comp.inlet_index]
                        this_comp.t_out[idx_timestep - 1] = x_vector[row_index + 1]
                    else:
                        this_comp.t_in[idx_timestep - 1] = x_vector[row_index]
                        this_comp.t_out[idx_timestep - 1] = x_vector[this_comp.downstream_index]

                elif this_comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER:
                    if self.loop_config == CentralLoopType.TWOPIPE:
                        this_comp.t_in[idx_timestep - 1] = x_vector[this_comp.inlet_index]
                        if this_comp.downstream_device.comp_type == SimCompType.GROUND_HEAT_EXCHANGER:
                            this_comp.t_mix_out[idx_timestep - 1] = x_vector[this_comp.downstream_index]
                        else:
                            this_comp.t_mix_out[idx_timestep - 1] = x_vector[this_comp.downstream_device.inlet_index]
                    else:
                        this_comp.t_in[idx_timestep - 1] = x_vector[row_index]
                        this_comp.t_mix_out[idx_timestep - 1] = x_vector[this_comp.downstream_index]
                    this_comp.t_mean[idx_timestep - 1] = x_vector[row_index + 1]
                    this_comp.q_ghe[idx_timestep - 1] = x_vector[row_index + 2]
                    this_comp.t_out[idx_timestep - 1] = x_vector[row_index + 3]
                elif this_comp.comp_type == SimCompType.SOURCE_SINK_HEAT_EXCHANGER:
                    this_comp.t_in[idx_timestep - 1] = x_vector[row_index]
                    this_comp.t_out[idx_timestep - 1] = x_vector[this_comp.downstream_index]
                elif isinstance(this_comp, (IsolatedHorizontalPipe, CoupledHorizontalPipe)):
                    this_comp.update_post_solve(x_vector, idx_timestep)

            # Update the console every 1 timesteps or on the very last step
            if (idx_timestep - 1) % 1 == 0 or idx_timestep == self.num_timesteps - 1:
                elapsed = time.perf_counter() - t_start
                percent = ((idx_timestep - 1) / (self.num_timesteps - 1)) * 100
                print(
                    f"  Progress: {(idx_timestep - 1)}/{self.num_timesteps - 1} ({percent:.1f}%) |"
                    f" Elapsed time: {elapsed:.2f}s",
                    end="\r",
                    flush=True,
                )
        print(f"\n--- Solver finished in {time.perf_counter() - t_start:.2f} seconds! ---")

    def calc_energy(self):
        self.pump_power_loop = (
            self.m_flow_loop
            / (self.fluid.rho * self.loop_pump_efficiency)
            * self.loop_design_pressure_loss_per_meter
            * self.loop_length
        )

    def create_output(
        self,
        output_path: Path,
        output_path_2: Path | None = None,
        output_path_coordinates: Path | None = None,
    ):
        output_columns: dict[str, Any] = {}

        network_q_net_bldg_tot = np.zeros(self.num_timesteps, dtype=float)
        network_q_net_ghe_tot = np.zeros(self.num_timesteps, dtype=float)

        self.calc_energy()

        for this_comp in self.components:
            this_comp.calc_energy()

        for this_comp in self.components:
            if isinstance(this_comp, Building):
                output_columns[f"{this_comp.name}:EFT [C]"] = this_comp.t_in
                output_columns[f"{this_comp.name}:ExFT [C]"] = this_comp.t_out
                output_columns[f"{this_comp.name}:Q_htg [W]"] = this_comp.htg_vals
                output_columns[f"{this_comp.name}:Q_clg [W]"] = this_comp.clg_vals
                output_columns[f"{this_comp.name}:Q_net [W]"] = this_comp.q_net
                output_columns[f"{this_comp.name}:M_flow [kg/s]"] = this_comp.m_flow
                network_q_net_bldg_tot += this_comp.q_net
                output_columns[f"{this_comp.name}:P_hp_htg [W]"] = this_comp.power_hp_htg
                output_columns[f"{this_comp.name}:P_hp_clg [W]"] = this_comp.power_hp_clg
                output_columns[f"{this_comp.name}:P_hp_tot [W]"] = this_comp.power_hp_tot
                output_columns[f"{this_comp.name}:P_pump [W]"] = this_comp.power_circ_pump

                q_src_clg = this_comp.clg_vals + this_comp.power_hp_clg
                q_src_htg = this_comp.htg_vals - this_comp.power_hp_htg

                output_columns[f"{this_comp.name}:Q_src_clg [W]"] = q_src_clg
                output_columns[f"{this_comp.name}:Q_src_htg [W]"] = q_src_htg
                output_columns[f"{this_comp.name}:Q_src_het [W]"] = q_src_htg - q_src_clg

        for this_comp in self.components:
            if isinstance(this_comp, GHX):
                output_columns[f"{this_comp.name}:EFT [C]"] = this_comp.t_in
                output_columns[f"{this_comp.name}:ExFT [C]"] = this_comp.t_out
                output_columns[f"{this_comp.name}:ExFT Mixed Loop [C]"] = this_comp.t_mix_out
                output_columns[f"{this_comp.name}:MFT [C]"] = this_comp.t_mean
                output_columns[f"{this_comp.name}:Q [W/m]"] = this_comp.q_ghe
                output_columns[f"{this_comp.name}:Q_tot [W]"] = this_comp.q_ghe * this_comp.nbh * this_comp.height
                network_q_net_ghe_tot += this_comp.q_ghe * this_comp.nbh * this_comp.height

        for this_comp in self.components:
            if isinstance(this_comp, SourceSinkHeatExchanger):
                output_columns[f"{this_comp.name}:EFT [C]"] = this_comp.t_in
                output_columns[f"{this_comp.name}:ExFT [C]"] = this_comp.t_out
                output_columns[f"{this_comp.name}:Operating [T/F]"] = this_comp.operating
                output_columns[f"{this_comp.name}:Q [W]"] = (
                    this_comp.operating * self.m_flow_loop * self.fluid.cp * (this_comp.t_out - this_comp.t_in)
                )

        for this_comp in self.components:
            if isinstance(this_comp, (IsolatedHorizontalPipe, CoupledHorizontalPipe)):
                # Add [1:] to slice off the 0th hour and match the DataFrame length
                output_columns[f"{this_comp.name}:EFT [C]"] = this_comp.t_in[1:]

                # Loop through the dynamic array to print each segment's details
                for k in range(this_comp.num_segments):
                    output_columns[f"{this_comp.name}:Node{k + 1}_Out [C]"] = this_comp.t_out_seg[k, 1:]
                    output_columns[f"{this_comp.name}:Q{k + 1} [W/m]"] = this_comp.q_seg[k, 1:]

                output_columns[f"{this_comp.name}:ExFT [C]"] = this_comp.t_out[1:]

        output_columns["Network:M_flow [kg/s]"] = self.m_flow_loop
        output_columns["Network:P_pump [W]"] = self.pump_power_loop
        output_columns["Network:Q_net_bldg [W]"] = network_q_net_bldg_tot
        output_columns["Network:Q_net_ghe [W]"] = network_q_net_ghe_tot

        output_data = pd.DataFrame(output_columns, index=self.time_array[1:])
        output_data.index.name = "Time [hr]"

        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        output_data.to_csv(output_path, float_format="%0.4f")
        if output_path_2 is not None:
            output_data = pd.DataFrame()
            output_data.index.name = "Iteration"
            output_data["NBH Selections (-)"] = self.nbh_selections
            output_data["Excess Temperature (°C)"] = self.excess_temperatures
            output_data["NBH Total (-)"] = self.nbh_values
            output_data["Total Drilling (m)"] = self.total_drilling_values
            output_data["Objective Function Value (m)"] = self.objective_function_values
            output_data["Borehole Height (m)"] = self.borehole_heights
            if self.search_method == "NELDER-MEAD":
                output_data["Angles (RAD)"] = self.angles
                output_data["NBH Vectors"] = self.nbh_vectors
            if not output_path_2.parent.exists():
                output_path_2.parent.mkdir(parents=True)
            output_data.to_csv(output_path_2, float_format="%0.4f")
        if output_path_coordinates is not None:
            with open(output_path_coordinates, "w") as output_file:
                json.dump(self.coordinate_locations, output_file)
