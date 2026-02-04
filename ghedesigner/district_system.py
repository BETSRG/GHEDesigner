import copy
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from ghedesigner.constants import HOURS_IN_YEAR, SEC_IN_HR, TWO_PI
from ghedesigner.enums import BHType, CentralLoopType, SimCompType, SourceSinkOpMode
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.boreholes.factory import get_bhe_object
from ghedesigner.ghe.gfunction import calc_g_func_for_multiple_lengths
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.utilities import combine_sts_lts, get_loads, load_input_file


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

    def __init__(self, ghe_id: str, ghe_data: dict, fluid: Fluid, loop_config: CentralLoopType, num_timesteps: int):
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

        self.two_pi_k = TWO_PI * self.soil.k

        # Computed properties
        self.bhe = None
        self.bh_effective_resist = None
        self.gFunction = None
        self.depth = None

        self.mass_flow_borehole_design = None
        self.history_terms = None
        self.total_values_ghe = None

        # for output
        self.t_in = None
        self.t_mean = None
        self.t_out = None
        self.q_ghe = None

        self.n_rows = ghe_data["pre_designed"]["boreholes_in_x_dimension"]
        self.n_cols = ghe_data["pre_designed"]["boreholes_in_y_dimension"]
        self.row_spacing = ghe_data["pre_designed"]["spacing_in_x_dimension"]
        self.col_spacing = ghe_data["pre_designed"]["spacing_in_y_dimension"]
        self.nbh = self.n_rows * self.n_cols
        self.mass_flow_ghe_design = ghe_data["flow_rate"] * self.nbh
        self.matrix_size = None
        self.num_timesteps = num_timesteps
        self.history_terms, self.total_values_ghe, self.q_ghe = (
            np.full(self.num_timesteps, self.soil.ugt, dtype=float),
            np.zeros(self.num_timesteps, dtype=float),
            np.zeros(self.num_timesteps, dtype=float),
        )

        self.t_in = np.full(self.num_timesteps, self.soil.ugt, dtype=float)
        self.t_mean = np.full(self.num_timesteps, self.soil.ugt, dtype=float)
        self.t_mix_out = np.full(self.num_timesteps, self.soil.ugt, dtype=float)
        self.t_out = np.full(self.num_timesteps, self.soil.ugt, dtype=float)
        self.q_ghe = np.zeros(self.num_timesteps)
        self.time_array = np.arange(1, self.num_timesteps + 1)
        self.gFunction = None
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

        self.log_time = np.linspace(-10, 4, 25).tolist()

        self.gFunction = self.generate_g_function_object()
        self.g, _ = self.grab_g_function()
        self.c_n = self.calc_cn_constant()

    def generate_g_function_object(self):
        self.bh_effective_resist = self.bhe.calc_effective_borehole_resistance()
        self.depth = self.bhe.borehole.D
        self.mass_flow_borehole_design = self.mass_flow_ghe_design / self.nbh
        h_values = [self.height]
        coordinates_ghe = [
            (i * self.row_spacing, j * self.row_spacing) for i in range(self.n_rows) for j in range(self.n_cols)
        ]
        self.gFunction = calc_g_func_for_multiple_lengths(
            self.row_spacing,
            h_values,
            self.bhe.borehole.r_b,
            self.depth,
            self.mass_flow_borehole_design,
            self.bh_type,
            self.log_time,
            coordinates_ghe,
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
        delta_q_ghe = (q_ghe[indices] - q_ghe[indices - 1]) / self.two_pi_k
        values = np.sum(delta_q_ghe * self.g(dim_less_time))

        total_values_ghe[idx_timestep] = values

        # Contribution from the last time step only
        dim1_less_time = np.log((time_n - self.time_array[idx_timestep - 1]) / (self.ts / SEC_IN_HR))
        history_terms[idx_timestep] = (
            self.soil.ugt
            - total_values_ghe[idx_timestep]
            + (q_ghe[idx_timestep - 1] / self.two_pi_k * self.g(dim1_less_time))
        )

        return history_terms, total_values_ghe

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

    def generate_matrix(self, m_loop, idx_timestep):
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

        self.fluid = Fluid(
            fluid_name=fluid_data["fluid_name"],
            percent=fluid_data["concentration_percent"],
            temperature=fluid_data["temperature"],
        )

        tg = json_data["ground_heat_exchanger"]["ghe1"]["soil"]["undisturbed_temp"]  # TODO: fix this

        self.sim_years = json_data["simulation_control"]["simulation_years"]
        self.num_timesteps = self.sim_years * HOURS_IN_YEAR

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
                )
                buildings.append(this_bldg)

        self.num_buildings = len(buildings)

        heat_exchangers: list[SourceSinkHeatExchanger] = []
        for this_hx_id, this_hx_data in hx_data.items():
            if this_hx_id.upper() in hx_names:
                this_hx = SourceSinkHeatExchanger(this_hx_id, this_hx_data, tg, self.num_timesteps)
                heat_exchangers.append(this_hx)

        self.num_heat_exchangers = len(heat_exchangers)

        cp = 0.0

        ground_heat_exchangers: list[GHX] = []
        for ghx_id, ghe_data in ghe_data.items():
            if ghx_id.upper() in ghx_names:
                this_ghx = GHX(ghx_id, ghe_data, self.fluid, self.loop_config, self.num_timesteps)
                cp = this_ghx.cp
                ground_heat_exchangers.append(this_ghx)

        self.nbh_total = sum(x.nbh for x in ground_heat_exchangers)
        self.num_ghx = len(ground_heat_exchangers)
        self.matrix_size = np.dot(
            [GHX.MATRIX_ROWS, Building.MATRIX_ROWS, SourceSinkHeatExchanger.MATRIX_ROWS],
            [self.num_ghx, self.num_buildings, self.num_heat_exchangers],
        )

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

        for this_comp in self.components:
            this_comp.matrix_size = self.matrix_size
            if isinstance(this_comp, GHX):
                this_comp.split_ratio = this_comp.nbh / self.nbh_total
            elif isinstance(this_comp, (Building, SourceSinkHeatExchanger)):
                this_comp.cp = cp

        # Assigning row_indices
        idx_comp = 0
        for this_comp in self.components:
            this_comp.row_index = idx_comp
            idx_comp += this_comp.MATRIX_ROWS
            this_comp.downstream_index = idx_comp

        # set the last component to loops back to the start
        self.components[-1].downstream_index = 0

    def solve_system(self):
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

    def create_output(self, output_path: Path):
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
        output_data.to_csv(output_path, float_format="%0.8f")
