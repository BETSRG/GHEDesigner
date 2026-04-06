import copy
from abc import ABC, abstractmethod
from math import copysign
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ghedesigner.constants import HOURS_IN_YEAR, SEC_IN_HR, TWO_PI
from ghedesigner.enums import CentralLoopType, SimCompType, SourceSinkOpMode
from ghedesigner.ghe.manager import GroundHeatExchanger
from ghedesigner.media import Fluid
from ghedesigner.utilities import get_loads, load_input_file, solve_root


class BaseSimComp(ABC):
    def __init__(self) -> None:
        self.name: str | None = None
        self.comp_type: SimCompType | None = None
        self.matrix_size: int | None = None
        self.row_index: int | None = None
        self.downstream_index: int | None = None

    @abstractmethod
    def generate_matrix(self, m_loop: float, idx_timestep: int) -> None:
        pass

    def calc_energy(self) -> None:
        pass


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

    def generate_matrix(self, m_loop: float, idx_timestep: int):
        if self.cp is None:
            raise ValueError("cp is uninitialized")
        if self.matrix_size is None:
            raise ValueError("matrix_size is uninitialized")

        t_in = self.t_in[idx_timestep - 1]
        is_running = self.is_running(t_in)
        self.operating[idx_timestep] = is_running
        m_flow_source: float = self.source_flow_rate if is_running else 0.0
        c_source = m_flow_source * self.cp
        c_loop = m_loop * self.cp
        c_min = min(c_source, c_loop)
        eff_c_min = self.effectiveness * c_min
        row = np.zeros(self.matrix_size, dtype=np.float64)

        # (C_loop - εCmin)*T_d,in - C_loop*T_d,out = -(εCmin)*T_s,in
        row[self.row_index] = c_loop - eff_c_min
        row[self.downstream_index] = -c_loop
        rhs = -eff_c_min * self.source_temp
        return [row], [rhs]


class GHX(BaseSimComp):
    MATRIX_ROWS = 4
    MATRIX_ROWS_FIXED_LOADS = 2

    def __init__(self, ghe_id: str, ghe_data: dict, fluid: Fluid, loop_config: CentralLoopType, num_timesteps: int,
                 sizing_end_month=240, fixed_loads=False):
        super().__init__()
        self.name = ghe_id
        self.comp_type = SimCompType.GROUND_HEAT_EXCHANGER
        self.height = None
        self.m_dot_total = None
        self.loop_config = loop_config
        self.sizing_end_month = sizing_end_month
        self.search, self.search_time = None, None
        self.num_timesteps = num_timesteps
        self.fixed_loads = fixed_loads
        if fixed_loads:
            self.convolution_completed = False

        self.ghe_manager = GroundHeatExchanger.init_from_dictionary(ghe_data,
                                                   {"fluid_name": fluid.name,
                                                                "concentration_percent": fluid.concentration_percent,
                                                                "temperature": fluid.temperature})
        self.ghe_manager.ghe_setup(ghe_data)
        self.ghe_manager.continue_if_design_unmet = True

        self.fluid = self.ghe_manager.fluid
        self.bh_type = self.ghe_manager.pipe.type
        self.split_ratio = None

        self.two_pi_k_recip = 1.0 / (TWO_PI * self.ghe_manager.soil.k)
        self.time_array = np.arange(1, self.num_timesteps + 1)

        self.num_timesteps = num_timesteps
        self.history_terms, self.total_values_ghe, self.q_ghe = (
            np.full(self.num_timesteps, self.ghe_manager.soil.ugt, dtype=float),
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
        self.nbh = None
        self.bh_effective_resist = None
        self.c_n = None
        self.borefield_coordinates = None

        if self.ghe_manager.is_sizable:
            self.ghe_designed = False
            self.base_max_eft = self.ghe_manager.max_eft
            self.base_min_eft = self.ghe_manager.min_eft
        else:
            self.ghe_designed = False

    def can_be_resized(self, upsize=True):
        if (upsize and not self.ghe_manager.at_maximum_size) or  (not upsize and not self.ghe_manager.at_minimum_size):
            return True
        else:
            return False

    def design_new_ghe(self, load_profile, max_eft, min_eft):
        if not self.ghe_manager.is_sizable:
            self.ghe_designed = True
            self.ghe_manager.initialize_pre_designed_ghe(1, self.sizing_end_month, load_profile)
            self.borefield_coordinates = self.ghe_manager.pre_designed_locations
        else:
            self.ghe_designed = True
            self.ghe_manager.max_eft = max_eft
            self.ghe_manager.min_eft = min_eft
            self.search, self.search_time, _ = \
            self.ghe_manager.design_and_size_ghe(self.sizing_end_month, loads_override=load_profile)
            self.borefield_coordinates = self.ghe_manager.current_ghe.gFunction.bore_locations

    def update_ghe_parameters(self):
        if not self.ghe_designed:
            raise ValueError("A GHE must be either pre-provided or designed before the parameters can be updated.")
        ghe_object = self.ghe_manager.current_ghe
        self.b = ghe_object.b_spacing
        self.height = ghe_object.bhe.borehole.H
        self.nbh = ghe_object.nbh

        b_over_h = self.b / self.height
        ghe_object.bhe_eq.calc_sts_g_functions()
        self.g, _ = ghe_object.grab_g_function(b_over_h)
        self.ts = ghe_object.bhe_eq.t_s
        self.bh_effective_resist = ghe_object.bhe.calc_effective_borehole_resistance()

        if self.fixed_loads:
            self.convolution_completed = False
        else:
            self.c_n = self.calc_cn_constant()
        self.history_terms, self.total_values_ghe, self.q_ghe = (
            np.full(self.num_timesteps, self.ghe_manager.soil.ugt, dtype=float),
            np.zeros(self.num_timesteps, dtype=float),
            np.zeros(self.num_timesteps, dtype=float),
        )

    def calc_cn_constant(self):
        """
        Calculate C_n values for three GHEs based on their g-functions.

        Cn = 1 / (2 * pi * K_s) * g((tn - tn-1) / t_s) + R_b
        """
        delta_log_times = np.log((self.time_array[1:] - self.time_array[0:-1]) / (self.ts / SEC_IN_HR))
        g_vals = self.g(delta_log_times)
        c_n = g_vals * self.two_pi_k_recip + self.bh_effective_resist

        return c_n

    def calc_history_term(self, idx_timestep, history_terms, total_values_ghe):
        """
        Computes the history term H_n for this GHX at time index `i`.
        Updates self.total_values_ghe and self.H_n_ghe in place.
        """
        if idx_timestep == 0:
            raise IndexError("Timestep index error")

        time_n = self.time_array[idx_timestep]

        q_ghe = self.q_ghe[:idx_timestep]

        # Compute dimensionless time for all indices from 1 to i-1
        indices = np.arange(1, idx_timestep)
        dim_less_time = np.log((time_n - self.time_array[indices - 1]) / (self.ts / SEC_IN_HR))

        # Compute contributions from all previous steps
        delta_q_ghe = (q_ghe[indices] - q_ghe[indices - 1]) * self.two_pi_k_recip
        values = np.sum(delta_q_ghe * self.g(dim_less_time))

        total_values_ghe[idx_timestep] = values

        # Contribution from the last time step only
        dim1_less_time = np.log((time_n - self.time_array[idx_timestep - 1]) / (self.ts / SEC_IN_HR))
        history_terms[idx_timestep] = (
            self.ghe_manager.soil.ugt
            - total_values_ghe[idx_timestep]
            + (q_ghe[idx_timestep - 1] * self.two_pi_k_recip * self.g(dim1_less_time))
        )

        return history_terms, total_values_ghe

    def generate_matrix_fixed_loads(self, m_loop, idx_timestep, load_profile=None):

        if not self.convolution_completed:
            assert(load_profile is not None)
            n = load_profile.size
            convolution_length = 2 * n - 1
            time_values = np.log((self.time_array[1:] * SEC_IN_HR) / self.ts)
            g_values = self.g(time_values)
            q_dot_b_dt = np.zeros(n, dtype=float)
            q_dot_b_dt[0] = load_profile[0]
            q_dot_b_dt[1:] = load_profile[1:] - load_profile[:-1]
            delta_tb = np.fft.irfft(
                np.fft.rfft(q_dot_b_dt * self.two_pi_k_recip, n=convolution_length) * np.fft.rfft(g_values, n=convolution_length),
                n=convolution_length,
            )[:n]
            self.t_mean = self.ghe_manager.soil.ugt + delta_tb + load_profile * self.bh_effective_resist
            self.q_ghe = load_profile
            self.convolution_completed = True
            return [0.0, 0.0]
        else:
            row_1 = np.zeros(self.matrix_size, dtype=np.float64)
            row_2 = np.zeros(self.matrix_size, dtype=np.float64)

            mass_flow_ghe = m_loop * self.split_ratio

            # m_ghe * (T_out - T_in) = m_loop * (T_mix_out - T_in) assuming constant c_p
            row_1[self.row_index] = (m_loop - mass_flow_ghe)
            row_1[self.row_index + 1] = mass_flow_ghe
            row_1[self.downstream_index] = -m_loop
            rhs_1 = 0.0

            # 2 * T_mean = T_in + T_out
            row_2[self.row_index] = 1
            row_2[self.row_index + 1] = 1
            rhs_2 = 2 * self.t_mean[idx_timestep]

        return [row_1, row_2], [rhs_1, rhs_2]

    def generate_matrix(self, m_loop, idx_timestep):
        self.history_terms, self.total_values_ghe = self.calc_history_term(
            idx_timestep, self.history_terms, self.total_values_ghe
        )

        row_1 = np.zeros(self.matrix_size, dtype=np.float64)
        row_2 = np.zeros(self.matrix_size, dtype=np.float64)
        row_3 = np.zeros(self.matrix_size, dtype=np.float64)
        row_4 = np.zeros(self.matrix_size, dtype=np.float64)

        mass_flow_ghe = m_loop * self.split_ratio

        row_1[self.row_index] = (m_loop - mass_flow_ghe) * self.cp
        row_1[self.row_index + 3] = mass_flow_ghe * self.cp
        row_1[self.downstream_index] = -m_loop * self.cp

        row_2[self.row_index + 1] = 1
        row_2[self.row_index + 2] = self.c_n[idx_timestep]

        row_3[self.row_index] = -1
        row_3[self.row_index + 1] = 2
        row_3[self.row_index + 3] = -1

        row_4[self.row_index] = mass_flow_ghe * self.cp
        row_4[self.row_index + 2] = self.height * self.nbh
        row_4[self.row_index + 3] = -mass_flow_ghe * self.cp

        rhs_1, rhs_2, rhs_3, rhs_4 = 0, self.history_terms[idx_timestep], 0, 0

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
        constant_cop=False
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

        if self.heating_exists:
            hp_htg_name = bldg_data["heating_load"]["heat_pump_name"]
            hp_htg_data = hp_data[hp_htg_name]
            one_yr_htg_vals = np.array(get_loads(hp_htg_name, SimCompType.HEAT_PUMP.name, bldg_data["heating_load"]))
            self.htg_vals = np.tile(one_yr_htg_vals, self.sim_years)
            self.hp_htg = HPmodel(hp_htg_name, hp_htg_data)

        if self.cooling_exists:
            hp_clg_name = bldg_data["cooling_load"]["heat_pump_name"]
            hp_clg_data = hp_data[hp_clg_name]
            one_yr_clg_vals = np.array(get_loads(hp_clg_name, SimCompType.HEAT_PUMP.name, bldg_data["cooling_load"]))
            self.clg_vals = np.tile(one_yr_clg_vals, self.sim_years)
            self.hp_clg = HPmodel(hp_clg_name, hp_clg_data)

        self.q_net = self.htg_vals - self.clg_vals
        self.t_in = np.full(self.num_timesteps, tg, dtype=float)
        self.t_out = np.full(self.num_timesteps, tg, dtype=float)
        self.m_flow = np.zeros(self.num_timesteps, dtype=float)
        self.power_hp_htg = np.zeros(self.num_timesteps, dtype=float)
        self.power_hp_clg = np.zeros(self.num_timesteps, dtype=float)
        self.power_hp_tot = np.zeros(self.num_timesteps, dtype=float)
        self.power_circ_pump = np.zeros(self.num_timesteps, dtype=float)

    def generate_ghe_load_estimate(self, ugt, beta=0.1):
        min_eft = self.min_eft
        max_eft = self.max_eft
        loads = np.zeros(self.num_timesteps, dtype=float)
        if self.cooling_exists:
            cooling_temp = ((1 - beta) * max_eft + beta * ugt) / 2.0
            q_rej_ratio = self.hp_clg.a_clg * cooling_temp * cooling_temp + self.hp_clg.b_clg * cooling_temp + \
                          self.hp_clg.c_clg
            loads -= q_rej_ratio * self.clg_vals
        if self.heating_exists:
            heating_temp = ((1 - beta) * min_eft + beta * ugt) / 2.0
            q_extr_ratio = self.hp_htg.a_htg * heating_temp * heating_temp + self.hp_htg.b_htg * heating_temp + \
                          self.hp_htg.c_htg
            loads += q_extr_ratio * self.htg_vals
        return loads

    def get_excess_temperature(self):
        max_temp = np.max(self.t_in)
        min_temp = np.min(self.t_in)
        if self.constant_cop:
            self.heat_transfers_calced = False
        return max(max_temp - self.max_eft, self.min_eft - min_temp)

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

        q_net_i = self.q_net[idx_timestep - 1]
        hp_capacity = cap_htg if q_net_i > 0 else cap_clg

        # compute mass flow rates
        mass_flow_bldg = max(np.abs(q_net_i) / hp_capacity * m_single_hp, m_single_hp)

        # save for later
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
        h = self.htg_vals[idx_timestep - 1]
        c = self.clg_vals[idx_timestep - 1]

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
        min_eft = self.min_eft
        max_eft = self.max_eft
        self.loads = np.zeros(self.num_timesteps, dtype=float)
        if self.cooling_exists:
            cooling_temp = ((1 - beta) * max_eft + beta * ugt) / 2.0
            q_rej_ratio = self.hp_clg.a_clg * cooling_temp * cooling_temp + self.hp_clg.b_clg * cooling_temp + \
                          self.hp_clg.c_clg
            self.loads -= q_rej_ratio * self.clg_vals
        if self.heating_exists:
            heating_temp = ((1 - beta) * min_eft + beta * ugt) / 2.0
            q_extr_ratio = self.hp_htg.a_htg * heating_temp * heating_temp + self.hp_htg.b_htg * heating_temp + \
                           self.hp_htg.c_htg
            self.loads += q_extr_ratio * self.htg_vals

    def generate_matrix_constant_cop(self, m_loop, idx_timestep):
        row = np.zeros(self.matrix_size, dtype=float)
        row[self.row_index] = -m_loop * self.cp
        row[self.downstream_index] = m_loop * self.cp
        rhs = self.loads[idx_timestep]
        return [row], [rhs]

    def generate_matrix(self, m_loop, idx_timestep):
        if self.constant_cop:
            return self.generate_matrix_constant_cop(m_loop, idx_timestep)
        else:
            t_in = self.t_in[idx_timestep - 1]
            r1, r2 = self.calc_r1_r2(t_in, idx_timestep)
            row = np.zeros(self.matrix_size, dtype=float)
            row[self.row_index] = 1 + r1 / (m_loop * self.cp)
            row[self.downstream_index] = -1
            rhs = -r2 / (m_loop * self.cp)
            return [row], [rhs]

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
        self.c3_htg = hp_data["heating_performance"]["c3"]

        self.c1_clg = hp_data["cooling_performance"]["c1"]
        self.c2_clg = hp_data["cooling_performance"]["c2"]
        self.c3_clg = hp_data["cooling_performance"]["c3"]

        self.m_flow_single_hp = hp_data["design_flow_rate"]
        self.design_pressure_loss = hp_data["design_pressure_loss"]
        self.pump_efficiency = hp_data["pump_efficiency"]
        self.design_htg_cap_single_hp = hp_data["heating_performance"]["design_cap"]
        self.design_clg_cap_single_hp = hp_data["cooling_performance"]["design_cap"]


class GHEHPSystem:
    def __init__(self, f_path_json: Path, constant_cop=True, fixed_loads=True):
        self.components: list[Building | GHX | SourceSinkHeatExchanger] = []
        self.nbh_total = None
        self.matrix_size = 0

        json_data = load_input_file(f_path_json)

        self.loop_config = CentralLoopType[json_data["central_loop"]["pipe_configuration"].upper()]
        self.loop_flow_factor = json_data["central_loop"]["flow_factor"]
        self.loop_pump_efficiency = json_data["central_loop"]["pump_efficiency"]
        self.loop_length = json_data["central_loop"]["loop_length"]
        self.loop_design_pressure_loss_per_meter = json_data["central_loop"]["design_pressure_loss"]
        self.constant_cop = constant_cop
        self.fixed_loads = fixed_loads

        fluid_data = json_data["fluid"]
        topology_data = json_data["topology"]
        heat_pump_data = json_data["heat_pump"]
        building_data = json_data.get("building", {})
        ghe_data = json_data.get("ground_heat_exchanger", {})
        hx_data = json_data.get("source_sink_heat_exchanger", {})

        self.fluid = Fluid(
            fluid_name=fluid_data["fluid_name"],
            percent=fluid_data["concentration_percent"],
            temperature=fluid_data["temperature"],
        )

        tg = json_data["ground_heat_exchanger"]["ghe1"]["soil"]["undisturbed_temp"]  # TODO: fix this

        self.sim_years = json_data["simulation_control"]["simulation_years"]
        self.num_timesteps = self.sim_years * HOURS_IN_YEAR
        self.total_loads = np.zeros(self.num_timesteps, dtype=float)
        self.previous_temp_adjustment = None
        self.final_excess_temperature_adjustment = None
        self.final_excess_temperature = None
        self.excess_temperature_adjustments = []
        self.excess_temperatures = []
        self.load_profiles = []
        self.ghe_loads = None
        self.previous_ghe_loads = None
        self.iteration_indices = []
        self.coordinate_locations = {}
        self.nbh_values = []
        self.total_drilling_values = []

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

        # get needed buildings
        buildings: list[Building] = []
        for this_building_id, this_bldg_data in building_data.items():
            if this_building_id.upper() in building_names:
                this_bldg = Building(
                    this_building_id,
                    this_bldg_data,
                    heat_pump_data,
                    tg,
                    self.fluid,
                    self.loop_config,
                    self.num_timesteps,
                    constant_cop=constant_cop
                )
                buildings.append(this_bldg)

        self.num_buildings = len(buildings)
        self.buildings = buildings

        heat_exchangers: list[SourceSinkHeatExchanger] = []
        for this_hx_id, this_hx_data in hx_data.items():
            if this_hx_id.upper() in hx_names:
                this_hx = SourceSinkHeatExchanger(this_hx_id, this_hx_data, tg, self.num_timesteps)
                heat_exchangers.append(this_hx)

        self.num_heat_exchangers = len(heat_exchangers)
        self.heat_exchangers = heat_exchangers

        self.cp = 0.0

        ground_heat_exchangers: list[GHX] = []
        for ghx_id, ghe_data in ghe_data.items():
            if ghx_id.upper() in ghx_names:
                this_ghx = GHX(ghx_id, ghe_data, self.fluid, self.loop_config, self.num_timesteps,
                               fixed_loads=fixed_loads)
                self.cp = this_ghx.cp
                ground_heat_exchangers.append(this_ghx)


        self.num_ghx = len(ground_heat_exchangers)
        self.ghe_load_differences = []
        self.ghe_load_magnitudes = []
        self.ground_heat_exchangers = ground_heat_exchangers
        self.sizable_ground_heat_exchangers = []
        for ghe in self.ground_heat_exchangers:
            if ghe.ghe_manager.is_sizable:
                self.sizable_ground_heat_exchangers.append(ghe)
        if fixed_loads:
            self.matrix_size = np.dot(
                [GHX.MATRIX_ROWS_FIXED_LOADS, Building.MATRIX_ROWS, SourceSinkHeatExchanger.MATRIX_ROWS],
                [self.num_ghx, self.num_buildings, self.num_heat_exchangers],
            )
        else:
            self.matrix_size = np.dot(
                [GHX.MATRIX_ROWS, Building.MATRIX_ROWS, SourceSinkHeatExchanger.MATRIX_ROWS],
                [self.num_ghx, self.num_buildings, self.num_heat_exchangers],
            )

        self.m_flow_loop = np.zeros(self.num_timesteps)
        self.pump_power_loop = np.zeros(self.num_timesteps)

        def get_bldg(name: str) -> Building | None:
            return \
                next((obj for obj in buildings if obj.name and obj.name.upper() == name.upper()), None)


        def get_ghx(name: str) -> GHX | None:
            return \
                next((obj for obj in ground_heat_exchangers if obj.name and obj.name.upper() == name.upper()), None)


        def get_hx(name: str) -> SourceSinkHeatExchanger | None:
            return \
                next((obj for obj in heat_exchangers if obj.name and obj.name.upper() == name.upper()), None)


        for v in topology_data:
            comp_type = v["type"]
            comp: Building | GHX | SourceSinkHeatExchanger | None
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

        # Assigning row_indices
        idx_comp = 0
        for this_comp in self.components:
            this_comp.row_index = idx_comp
            if self.fixed_loads and isinstance(this_comp, GHX):
                idx_comp += this_comp.MATRIX_ROWS_FIXED_LOADS
            else:
                idx_comp += this_comp.MATRIX_ROWS
            this_comp.downstream_index = idx_comp

        # set the last component to loops back to the start
        self.components[-1].downstream_index = 0

    def size_and_simulate(self):
        if np.any([not ghe.ghe_designed for ghe in self.ground_heat_exchangers]):
            self.size_system()
        else:
            self.solve_system()

    def calculate_building_excess(self):
        max_excess = None
        for building in self.buildings:
            current_excess = building.get_excess_temperature()
            if max_excess is None or current_excess > max_excess:
                max_excess = current_excess
        return max_excess

    def get_updated_load_profile(self, based_on_nbh=False):
        if self.ghe_loads is not None:
            self.previous_ghe_loads = self.ghe_loads
        self.total_loads = np.zeros(self.num_timesteps, dtype=float)
        self.ghe_loads = [np.zeros(self.num_timesteps, dtype=float) for ghe in self.ground_heat_exchangers]
        for i, ghe in enumerate(self.ground_heat_exchangers):
            current_loads = ghe.q_ghe * ghe.nbh * ghe.height
            self.total_loads += current_loads
            if not based_on_nbh:
                self.ghe_loads[i][:] = current_loads[:]
        if based_on_nbh:
            for i, ghe in enumerate(self.ground_heat_exchangers):
                self.ghe_loads[i][:] = self.total_loads * ghe.split_ratio

    def append_load(self, new_load, do_ghe_loads=True):
        if not do_ghe_loads:
            pass
            # self.ghe_load_residuals.append([0.0 for i in range(self.num_ghx)])
        else:
            new_differences = []
            new_magnitudes = []
            for i, ghe in enumerate(self.ground_heat_exchangers):
                current_ghe_loads = self.ghe_loads[i]
                average = np.average(current_ghe_loads)
                pos_vals = np.where(current_ghe_loads > 0, current_ghe_loads, 0)
                neg_vals = np.where(current_ghe_loads < 0, current_ghe_loads, 0)
                new_differences.append(np.sum((pos_vals - neg_vals)) / average)
                new_magnitudes.append(average)
            self.ghe_load_differences.append(new_differences)
            self.ghe_load_magnitudes.append(new_magnitudes)
        self.load_profiles.append(new_load)

    def append_new_coordinates(self, iteration_name):
        new_index = len(self.coordinate_locations)
        self.coordinate_locations[new_index] = {}
        self.coordinate_locations[new_index]["name"] = iteration_name
        for i, ghe in enumerate(self.ground_heat_exchangers):
            self.coordinate_locations[new_index][ghe.name] = ghe.borefield_coordinates

    def size_system(self, max_iter=50, lower_bound=0.05, upper_bound=1.0, iteration_per_value=1,
                    restart=False, max_ub_temperature_adjustment=30.0, min_ub_temperature_adjustment=30.0):

        # Perform initial sizing of GHEs
        average_ugt = 0
        total_design_volume = 0
        load_split = np.zeros(self.num_ghx, dtype=float)
        for i, ghe in enumerate(self.ground_heat_exchangers):
            design_volume = ghe.ghe_manager.get_design_volume()
            average_ugt += ghe.ghe_manager.soil.ugt * design_volume
            total_design_volume += design_volume
            load_split[i] = design_volume
        average_ugt /= total_design_volume
        load_split /= total_design_volume

        initial_load = None
        for building in self.buildings:
            if initial_load is None:
                initial_load = building.generate_ghe_load_estimate(average_ugt)
            else:
                initial_load += building.generate_ghe_load_estimate(average_ugt)
        initial_ghe_load = []
        for i, ghe in enumerate(self.ground_heat_exchangers):
            ghe.split_ratio = load_split[i]
            current_load = initial_load * ghe.split_ratio
            ugt = ghe.ghe_manager.soil.ugt
            ghe.design_new_ghe(current_load, ugt + 0.5 * max_ub_temperature_adjustment,
                               ugt - 0.5 * min_ub_temperature_adjustment)
            initial_ghe_load.append(current_load)
            ghe.update_ghe_parameters()
        self.append_load(initial_load, do_ghe_loads=False)
        self.append_new_coordinates("0_0")
        self.previous_ghe_loads = initial_ghe_load
        self.solve_system()
        self.previous_temp_adjustment = 0.0
        self.iteration_indices = []
        self.guess_idx = -1
        def objective(excess_temp_adjustment):
            # self.excess_temperature_adjustments.append(excess_temp_adjustment)
            direction = copysign(1, excess_temp_adjustment - self.previous_temp_adjustment)
            self.guess_idx += 1
            for itx in range(iteration_per_value):
                self.excess_temperature_adjustments.append(excess_temp_adjustment)
                self.iteration_indices.append(itx)
                if itx == 0 and restart:
                    self.total_loads = initial_load
                    self.ghe_loads = initial_ghe_load
                    self.previous_ghe_loads = initial_ghe_load
                else:
                    self.get_updated_load_profile(based_on_nbh=False)
                self.append_load(self.total_loads, do_ghe_loads=True)
                for ghx, ghe in enumerate(self.sizable_ground_heat_exchangers):
                    if ghe.can_be_resized(upsize=(direction == 1)):
                        ugt = ghe.ghe_manager.soil.ugt
                        ghe.design_new_ghe(self.ghe_loads[ghx], ugt + excess_temp_adjustment * max_ub_temperature_adjustment,
                                           ugt - excess_temp_adjustment * min_ub_temperature_adjustment)
                nbh_val = 0
                total_drilling_val = 0.0
                for ghe in self.ground_heat_exchangers:
                    nbh_val += ghe.ghe_manager.current_ghe.nbh
                    total_drilling_val += ghe.ghe_manager.current_ghe.nbh * ghe.ghe_manager.current_ghe.bhe.borehole.H
                self.nbh_values.append(nbh_val)
                self.total_drilling_values.append(total_drilling_val)
                self.append_new_coordinates(f"{self.guess_idx}_{itx}")
                for ghe in self.ground_heat_exchangers:
                    ghe.update_ghe_parameters()
                self.solve_system()
                excess_temp = self.calculate_building_excess()
                self.excess_temperatures.append(excess_temp)
            self.previous_temp_adjustment = excess_temp_adjustment
            return excess_temp

        self.final_excess_temperature_adjustment = solve_root(0.0, objective, lower_bound, upper_bound,
                                                              max_iter=max_iter, abs_tol=1.0e-3)
        self.final_excess_temperature = objective(self.final_excess_temperature_adjustment)

    def solve_system(self):
        if self.fixed_loads:
            assert self.constant_cop
            self.solve_system_fixed_loads()
        else:
            self.solve_system_standard()

    def solve_system_fixed_loads(self):
        self.nbh_total = sum([x.nbh for x in self.ground_heat_exchangers])
        total_loads = np.zeros(self.num_timesteps, dtype=float)
        average_ugt = 0.0
        for this_comp in self.components:
            this_comp.matrix_size = self.matrix_size
            if isinstance(this_comp, GHX):
                this_comp.split_ratio = this_comp.nbh / self.nbh_total
                average_ugt += this_comp.ghe_manager.soil.ugt * this_comp.nbh / self.nbh_total
            elif isinstance(this_comp, Building):
                this_comp.cp = self.cp
            elif isinstance(this_comp, SourceSinkHeatExchanger):
                this_comp.cp = self.cp
        for building in self.buildings:
            building.generate_constant_cop_loads(average_ugt)
            total_loads += building.loads
        for ghe in self.ground_heat_exchangers:
            ghe.generate_matrix_fixed_loads(0.0, 0, load_profile=total_loads * ghe.split_ratio / (ghe.nbh * ghe.height))
        for idx_timestep in range(1, self.num_timesteps):  # loop over all timestep
            matrix_rows = []
            matrix_rhs = []
            total_hp_flow = 0

            for this_comp in self.components:
                if isinstance(this_comp, Building):
                    t_in = this_comp.t_in[idx_timestep - 1]
                    m_bldg = this_comp.calc_mass_flow_rate(t_in, idx_timestep)
                    total_hp_flow += m_bldg

            m_loop = max(total_hp_flow * self.loop_flow_factor, 0.1)

            for this_comp in self.components:
                if isinstance(this_comp, GHX):
                    rows, rhs = this_comp.generate_matrix_fixed_loads(m_loop, idx_timestep)
                else:
                    rows, rhs = this_comp.generate_matrix(m_loop, idx_timestep)
                matrix_rows.extend(rows)
                matrix_rhs.extend(rhs)

            # Solve the system = A * X = B
            a_matrix = np.array(matrix_rows, dtype=float)
            b_vector = np.array(matrix_rhs, dtype=float)
            x_vector = np.linalg.solve(a_matrix, b_vector)

            # save output data
            self.m_flow_loop[idx_timestep] = m_loop

            for this_comp in self.components:
                row_index = this_comp.row_index
                if this_comp.comp_type == SimCompType.BUILDING:
                    this_comp.t_in[idx_timestep] = x_vector[row_index]
                    this_comp.t_out[idx_timestep] = x_vector[this_comp.downstream_index]
                elif this_comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER:
                    this_comp.t_in[idx_timestep] = x_vector[row_index]
                    this_comp.t_out[idx_timestep] = x_vector[row_index + 1]
                    this_comp.t_mix_out[idx_timestep] = x_vector[this_comp.downstream_index]
                elif this_comp.comp_type == SimCompType.SOURCE_SINK_HEAT_EXCHANGER:
                    this_comp.t_in[idx_timestep] = x_vector[row_index]
                    this_comp.t_out[idx_timestep] = x_vector[this_comp.downstream_index]

    def solve_system_standard(self):
        self.nbh_total = sum([x.nbh for x in self.ground_heat_exchangers])
        average_ugt = 0.0
        for this_comp in self.components:
            this_comp.matrix_size = self.matrix_size
            if isinstance(this_comp, GHX):
                this_comp.split_ratio = this_comp.nbh / self.nbh_total
                average_ugt += this_comp.ghe_manager.soil.ugt * this_comp.nbh / self.nbh_total
            elif isinstance(this_comp, Building):
                this_comp.cp = self.cp
            elif isinstance(this_comp, SourceSinkHeatExchanger):
                this_comp.cp = self.cp

        if self.constant_cop:
            for building in self.buildings:
                building.generate_constant_cop_loads(average_ugt)

        for idx_timestep in range(1, self.num_timesteps):  # loop over all timestep
            matrix_rows = []
            matrix_rhs = []
            total_hp_flow = 0

            for this_comp in self.components:
                if isinstance(this_comp, Building):
                    t_in = this_comp.t_in[idx_timestep - 1]
                    m_bldg = this_comp.calc_mass_flow_rate(t_in, idx_timestep)
                    total_hp_flow += m_bldg

            m_loop = max(total_hp_flow * self.loop_flow_factor, 0.1)

            for this_comp in self.components:
                rows, rhs = this_comp.generate_matrix(m_loop, idx_timestep)
                matrix_rows.extend(rows)
                matrix_rhs.extend(rhs)

            # Solve the system = A * X = B
            a_matrix = np.array(matrix_rows, dtype=float)
            b_vector = np.array(matrix_rhs, dtype=float)
            x_vector = np.linalg.solve(a_matrix, b_vector)

            # save output data
            self.m_flow_loop[idx_timestep] = m_loop

            for this_comp in self.components:
                row_index = this_comp.row_index
                if this_comp.comp_type == SimCompType.BUILDING:
                    this_comp.t_in[idx_timestep] = x_vector[row_index]
                    this_comp.t_out[idx_timestep] = x_vector[this_comp.downstream_index]
                elif this_comp.comp_type == SimCompType.GROUND_HEAT_EXCHANGER:
                    this_comp.t_in[idx_timestep] = x_vector[row_index]
                    this_comp.t_mean[idx_timestep] = x_vector[row_index + 1]
                    this_comp.q_ghe[idx_timestep] = x_vector[row_index + 2]
                    this_comp.t_out[idx_timestep] = x_vector[row_index + 3]
                    this_comp.t_mix_out[idx_timestep] = x_vector[this_comp.downstream_index]

                elif this_comp.comp_type == SimCompType.SOURCE_SINK_HEAT_EXCHANGER:
                    this_comp.t_in[idx_timestep] = x_vector[row_index]
                    this_comp.t_out[idx_timestep] = x_vector[this_comp.downstream_index]

    def calc_energy(self):
        self.pump_power_loop = (
            self.m_flow_loop
            / (self.fluid.rho * self.loop_pump_efficiency)
            * self.loop_design_pressure_loss_per_meter
            * self.loop_length
        )

    def create_output(self, output_path: Path, output_path_2: Path=None, output_path_load: Path=None,
                      output_path_coordinates: Path=None):
        output_data = pd.DataFrame()
        output_data.index.name = "Hour"

        network_q_net_bldg_tot = np.zeros(self.num_timesteps, dtype=float)
        network_q_net_ghe_tot = np.zeros(self.num_timesteps, dtype=float)

        # compute energy use for central loop
        self.calc_energy()

        # compute energy use for all components
        for this_comp in self.components:
            this_comp.calc_energy()

        for this_comp in self.components:
            if isinstance(this_comp, Building):
                output_data[f"{this_comp.name}:EFT [C]"] = this_comp.t_in
                output_data[f"{this_comp.name}:ExFT [C]"] = this_comp.t_out
                output_data[f"{this_comp.name}:Q_htg [W]"] = this_comp.htg_vals
                output_data[f"{this_comp.name}:Q_clg [W]"] = this_comp.clg_vals
                output_data[f"{this_comp.name}:Q_net [W]"] = this_comp.q_net
                output_data[f"{this_comp.name}:M_flow [kg/s]"] = this_comp.m_flow
                network_q_net_bldg_tot += this_comp.q_net
                output_data[f"{this_comp.name}:P_hp_htg [W]"] = this_comp.power_hp_htg
                output_data[f"{this_comp.name}:P_hp_clg [W]"] = this_comp.power_hp_clg
                output_data[f"{this_comp.name}:P_hp_tot [W]"] = this_comp.power_hp_tot
                output_data[f"{this_comp.name}:P_pump [W]"] = this_comp.power_circ_pump
                q_src_clg = this_comp.clg_vals + this_comp.power_hp_clg
                q_src_htg = this_comp.htg_vals - this_comp.power_hp_htg
                output_data[f"{this_comp.name}:Q_src_clg [W]"] = q_src_clg
                output_data[f"{this_comp.name}:Q_src_htg [W]"] = q_src_htg
                output_data[f"{this_comp.name}:Q_src_het [W]"] = q_src_htg - q_src_clg

        for this_comp in self.components:
            if isinstance(this_comp, GHX):
                output_data[f"{this_comp.name}:EFT [C]"] = this_comp.t_in
                output_data[f"{this_comp.name}:ExFT [C]"] = this_comp.t_out
                output_data[f"{this_comp.name}:ExFT Mixed Loop [C]"] = this_comp.t_mix_out
                output_data[f"{this_comp.name}:MFT [C]"] = this_comp.t_mean
                output_data[f"{this_comp.name}:Q [W/m]"] = this_comp.q_ghe
                output_data[f"{this_comp.name}:Q_tot [W]"] = this_comp.q_ghe * this_comp.nbh * this_comp.height
                network_q_net_ghe_tot += this_comp.q_ghe * this_comp.nbh * this_comp.height

        for this_comp in self.components:
            if isinstance(this_comp, SourceSinkHeatExchanger):
                output_data[f"{this_comp.name}:EFT [C]"] = this_comp.t_in
                output_data[f"{this_comp.name}:ExFT [C]"] = this_comp.t_out
                output_data[f"{this_comp.name}:Operating [T/F]"] = this_comp.operating
                output_data[f"{this_comp.name}:Q [W]"] = (
                    this_comp.operating * self.m_flow_loop * self.fluid.cp * (this_comp.t_out - this_comp.t_in)
                )

        output_data["Network:M_flow [kg/s]"] = self.m_flow_loop
        output_data["Network:P_pump [W]"] = self.pump_power_loop
        output_data["Network:Q_net_bldg [W]"] = network_q_net_bldg_tot
        output_data["Network:Q_net_ghe [W]"] = network_q_net_ghe_tot

        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        output_data.to_csv(output_path, float_format="%0.4f")
        if output_path_2 is not None:
            output_data = pd.DataFrame()
            output_data.index.name = "Iteration"
            output_data["Load Estimation Iterations"] = self.iteration_indices
            output_data["Excess Temperature Adjustments (°C)"] = self.excess_temperature_adjustments
            output_data["Excess Temperature (°C)"] = self.excess_temperatures
            for i, ghe in enumerate(self.ground_heat_exchangers):
                output_data["".join([ghe.name, "_Load_Differences"])] = [row[i] for row in self.ghe_load_differences]
                output_data["".join([ghe.name, "_Load_Magnitudes"])] = [row[i] for row in self.ghe_load_magnitudes]
            output_data["NBH Total (-)"] = self.nbh_values
            output_data["Total Drilling (m)"] = self.total_drilling_values
            if not output_path_2.parent.exists():
                output_path_2.parent.mkdir(parents=True)
            output_data.to_csv(output_path_2, float_format="%0.4f")
        if output_path_load is not None:
            output_data = pd.DataFrame()
            output_data.index.name = "Time (hr)"
            for i in range(len(self.load_profiles)):
                output_data[f"Iteration {i}"] = self.load_profiles[i]
            if not output_path_load.parent.exists():
                output_path_load.parent.mkdir(parents=True)
            output_data.to_csv(output_path_load, float_format="%0.4f")

        if output_path_2 is not None:
            with open(output_path_coordinates, "w") as output_file:
                json.dump(self.coordinate_locations, output_file)
