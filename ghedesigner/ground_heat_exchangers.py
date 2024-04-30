from math import ceil, floor

import numpy as np
from scipy.interpolate import interp1d

from ghedesigner import VERSION
from ghedesigner.borehole import GHEBorehole
from ghedesigner.borehole_heat_exchangers import get_bhe_object
from ghedesigner.constants import SEC_IN_HR, TWO_PI
from ghedesigner.enums import BHPipeType, TimestepType
from ghedesigner.gfunction import GFunction, calc_g_func_for_multiple_lengths
from ghedesigner.ground_loads import HybridLoad
from ghedesigner.media import Grout, Pipe, Soil
from ghedesigner.radial_numerical_borehole import RadialNumericalBH
from ghedesigner.simulation import SimulationParameters
from ghedesigner.utilities import solve_root


class BaseGHE:
    def __init__(
            self,
            v_flow_system: float,
            b_spacing: float,
            bhe_type: BHPipeType,
            fluid,
            borehole: GHEBorehole,
            pipe: Pipe,
            grout: Grout,
            soil: Soil,
            g_function: GFunction,
            sim_params: SimulationParameters,
            hourly_extraction_ground_loads: list,
            field_type="N/A",
            field_specifier="N/A",
    ):

        self.fieldType = field_type
        self.fieldSpecifier = field_specifier
        self.V_flow_system = v_flow_system
        self.B_spacing = b_spacing
        self.nbh = len(g_function.bore_locations)
        self.V_flow_borehole = self.V_flow_system / self.nbh
        m_flow_borehole = self.V_flow_borehole / 1000.0 * fluid.rho
        self.m_flow_borehole = m_flow_borehole

        # Borehole Heat Exchanger
        self.bhe_type = bhe_type
        self.bhe = get_bhe_object(bhe_type, m_flow_borehole, fluid, borehole, pipe, grout, soil)

        # Equivalent borehole Heat Exchanger
        self.bhe_eq = self.bhe.to_single()

        # Radial numerical short time step
        self.radial_numerical = RadialNumericalBH(self.bhe_eq)
        self.radial_numerical.calc_sts_g_functions(self.bhe_eq)

        # gFunction object
        self.gFunction = g_function
        # Additional simulation parameters
        self.sim_params = sim_params
        # Hourly ground extraction loads
        # Building cooling is negative, building heating is positive
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.times = []
        self.loading = None

    def as_dict(self) -> dict:
        output = dict()
        output['title'] = f"GHEDesigner GHE Output - Version {VERSION}"
        output['number_of_boreholes'] = len(self.gFunction.bore_locations)
        output['borehole_depth'] = {'value': self.bhe.b.H, 'units': 'm'}
        output['borehole_spacing'] = {'value': self.B_spacing, 'units': 'm'}
        output['borehole_heat_exchanger'] = self.bhe.as_dict()
        output['equivalent_borehole_heat_exchanger'] = self.bhe_eq.as_dict()
        output['simulation_parameters'] = self.sim_params.as_dict()
        return output

    @staticmethod
    def combine_sts_lts(log_time_lts: list, g_lts: list, log_time_sts: list, g_sts: list) -> interp1d:
        # make sure the short time step doesn't overlap with the long time step
        max_log_time_sts = max(log_time_sts)
        min_log_time_lts = min(log_time_lts)

        if max_log_time_sts < min_log_time_lts:
            log_time = log_time_sts + log_time_lts
            g = g_sts + g_lts
        else:
            # find where to stop in sts
            i = 0
            value = log_time_sts[i]
            while value <= min_log_time_lts:
                i += 1
                value = log_time_sts[i]
            log_time = log_time_sts[0:i] + log_time_lts
            g = g_sts[0:i] + g_lts
        g = interp1d(log_time, g)

        return g

    def grab_g_function(self, b_over_h):
        # interpolate for the Long time step g-function
        g_function, rb_value, _, _ = self.gFunction.g_function_interpolation(b_over_h)
        # correct the long time step for borehole radius
        g_function_corrected = self.gFunction.borehole_radius_correction(
            g_function, rb_value, self.bhe.b.r_b
        )
        # Don't Update the HybridLoad (its dependent on the STS) because
        # it doesn't change the results much, and it slows things down a lot
        # combine the short and long time step g-function
        g = self.combine_sts_lts(
            self.gFunction.log_time,
            g_function_corrected,
            self.radial_numerical.lntts.tolist(),
            self.radial_numerical.g.tolist(),
        )

        g_bhw = self.combine_sts_lts(
            self.gFunction.log_time,
            g_function_corrected,
            self.radial_numerical.lntts.tolist(),
            self.radial_numerical.g_bhw.tolist(),
        )

        return g, g_bhw

    def cost(self, max_eft, min_eft):
        delta_t_max = max_eft - self.sim_params.max_EFT_allowable
        delta_t_min = self.sim_params.min_EFT_allowable - min_eft
        t_excess = max(delta_t_max, delta_t_min)
        return t_excess

    def _simulate_detailed(self, q_dot: np.ndarray, time_values: np.ndarray, g: interp1d):
        # Perform a detailed simulation based on a numpy array of heat rejection
        # rates, Q_dot (Watts) where each load is applied at the time_value
        # (seconds). The g-function can interpolate.
        # Source: Chapter 2 of Advances in Ground Source Heat Pumps

        n = q_dot.size

        # Convert the total load applied to the field to the average over
        # borehole wall rejection rate
        # At time t=0, make the heat rejection rate 0.
        q_dot_b = np.hstack((0.0, q_dot / float(self.nbh)))
        time_values = np.hstack((0.0, time_values))

        q_dot_b_dt = np.hstack((q_dot_b[1:] - q_dot_b[:-1]))

        ts = self.radial_numerical.t_s  # (-)
        two_pi_k = TWO_PI * self.bhe.soil.k  # (W/m.K)
        h = self.bhe.b.H  # (meters)
        tg = self.bhe.soil.ugt  # (Celsius)
        rb = self.bhe.calc_effective_borehole_resistance()  # (m.K/W)
        m_dot = self.bhe.m_flow_borehole  # (kg/s)
        cp = self.bhe.fluid.cp  # (J/kg.s)

        hp_eft = []
        delta_tb = []
        for i in range(1, n + 1):
            # Take the last i elements of the reversed time array
            _time = time_values[i] - time_values[0:i]
            # _time = time_values_reversed[n - i:n]
            g_values = g(np.log((_time * SEC_IN_HR) / ts))
            # Tb = Tg + (q_dt * g)  (Equation 2.12)
            delta_tb_i = (q_dot_b_dt[0:i] / h / two_pi_k).dot(g_values)
            # Tf = Tb + q_i * R_b^* (Equation 2.13)
            tb = tg + delta_tb_i
            # Bulk fluid temperature
            tf_bulk = tb + q_dot_b[i] / h * rb
            # T_out = T_f - Q / (2 * m_dot cp)  (Equation 2.14)
            tf_out = tf_bulk - q_dot_b[i] / (2 * m_dot * cp)
            hp_eft.append(tf_out)
            delta_tb.append(delta_tb_i)

        return hp_eft, delta_tb

    # Source:
    # 1. Claesson, J., & Javed, S. (2012). A load-aggregation method to calculate extraction temperatures of
    # borehole heat exchangers. ASHRAE Transactions, 118(1), 530-540.
    # 2. Mitchell, Matt S., and Jeffrey D. Spitler. "Characterization, testing, and optimization of load aggregation"
    # "methods for ground heat exchanger response-factor models." Science and Technology for the Built Environment 25,
    #  no. 8 (2019): 1036-1051.

    def simulate_dynamic_load_agg(self, q_dot_hourly, g):
        rb = self.bhe.calc_effective_borehole_resistance()  # (m.k/w)
        m_dot = self.bhe.m_flow_borehole  # (kg/s)
        cp = self.bhe.fluid.cp  # (j/kg.s)

        tg = self.bhe.soil.ugt  # (celsius)
        q_dot_b_watt = np.hstack((q_dot_hourly / float(self.nbh)))  # w/m
        height = self.bhe.b.H  # (meters)
        q_dot = q_dot_b_watt / height

        two_pi_k = TWO_PI * self.bhe.soil.k  # (w/m.k)

        n_max = len(q_dot)  # load_array_length
        rate_expansion = 1.62
        p_q = 9  # 5 # number of bins in each level

        q_max = int(np.log(n_max / p_q) / np.log(rate_expansion)) + 1  # eq. 9
        # v_max = p_q * (rate_expansion ** q - 1) / (rate_expansion - 1)

        v_matrix = np.zeros((q_max, p_q + 1))  # create v matrix

        q_array = np.arange(1, q_max + 1)  # creating levels
        r_q_array = rate_expansion ** (q_array - 1)  # rate of expansion

        v_matrix[0, 0] = 0  # eq. 10 top right

        for i in range(1, q_max):  # eq. 10 top left
            v_matrix[i, 0] = v_matrix[i - 1, 0] + r_q_array[i - 1] * p_q

        for q in range(0, q_max):  # eq. 10 bottom
            for p in range(1, p_q + 1):
                v_matrix[q, p] = v_matrix[q, 0] + r_q_array[q] * p

        ts = self.radial_numerical.t_s  # timescale

        v_values_matrix = (v_matrix[:, 1:])  # duration matrix
        v_array = v_values_matrix.flatten()  # duration array in 1d
        v_array = np.hstack((0.0, v_array))  # adding beginning time
        r_q_array_1d = np.zeros(q_max * p_q)  # duration in each level in 1d

        for i in range(0, len(r_q_array)):  # making 1d array of rq
            r_q_array_1d[i * p_q:(i + 1) * p_q] = r_q_array[i]

        hp_eft = np.zeros(len(q_dot))
        delta_tb = np.zeros(len(q_dot))
        g_s = g(np.log((v_array[1:] * 3600.0) / ts))  # g-values

        q_n = np.zeros(q_max * p_q)  # array holding current load
        q_n_old = np.zeros(q_max * p_q)  # array holding previous load
        q_dot = np.insert(q_dot, 0, 0)  # zero load in beginning

        # simulation
        q_n_old[0] = q_dot[0]
        for i in range(1, len(q_dot)):
            q_n[0] = q_dot[i]
            q_n[1:] = q_n_old[1:] + (1 / r_q_array_1d[1:]) * (q_n_old[:-1] - q_n_old[1:])
            q_p = np.insert(q_n, -1, 0)
            x = (q_p[:-1] - q_p[1:])
            delta_tb_i = np.dot(x / two_pi_k, g_s)
            tf_bulk = tg + delta_tb_i + q_dot[i] * rb  # (equation 3.32- ms thesis jack cook)
            tf_out = tf_bulk - q_dot_b_watt[i - 1] / (2 * m_dot * cp)  # (equation 3.33- ms thesis jack cook)
            hp_eft[i - 1] = tf_out
            delta_tb[i - 1] = delta_tb_i

            q_n_old = q_n.copy()

        hp_eft, delta_tb = list(hp_eft), list(delta_tb)

        return hp_eft, delta_tb

    def compute_g_functions(self):
        # Compute g-functions for a bracketed solution, based on min and max
        # height
        min_height = self.sim_params.min_height
        max_height = self.sim_params.max_height
        avg_height = (min_height + max_height) / 2.0
        h_values = [min_height, avg_height, max_height]

        coordinates = self.gFunction.bore_locations
        log_time = self.gFunction.log_time

        g_function = calc_g_func_for_multiple_lengths(
            self.B_spacing,
            h_values,
            self.bhe.b.r_b,
            self.bhe.b.D,
            self.bhe.m_flow_borehole,
            self.bhe_type,
            log_time,
            coordinates,
            self.bhe.fluid,
            self.bhe.pipe,
            self.bhe.grout,
            self.bhe.soil,
        )

        self.gFunction = g_function


class GHE(BaseGHE):
    def __init__(
            self,
            v_flow_system: float,
            b_spacing: float,
            bhe_type: BHPipeType,
            fluid,
            borehole: GHEBorehole,
            pipe: Pipe,
            grout: Grout,
            soil: Soil,
            g_function: GFunction,
            sim_params: SimulationParameters,
            hourly_extraction_ground_loads: list,
            field_type="N/A",
            field_specifier="N/A",
            load_years=None,
    ):
        BaseGHE.__init__(
            self,
            v_flow_system,
            b_spacing,
            bhe_type,
            fluid,
            borehole,
            pipe,
            grout,
            soil,
            g_function,
            sim_params,
            hourly_extraction_ground_loads,
            field_type=field_type,
            field_specifier=field_specifier,
        )

        # Split the extraction loads into heating and cooling for input to
        # the HybridLoad object
        if load_years is None:
            load_years = [2019]

        hybrid_load = HybridLoad(
            self.hourly_extraction_ground_loads,
            self.bhe_eq,
            self.radial_numerical,
            sim_params,
            years=load_years)

        # hybrid load object
        self.hybrid_load = hybrid_load

        # List of heat pump exiting fluid temperatures
        self.hp_eft = []
        # list of change in borehole wall temperatures
        self.dTb = []

    def as_dict(self) -> dict:
        output = dict()
        output['base'] = super().as_dict()

        results = dict()
        if len(self.hp_eft) > 0:
            max_hp_eft = max(self.hp_eft)
            min_hp_eft = min(self.hp_eft)
            results['max_hp_entering_temp'] = {'value': max_hp_eft, 'units': 'C'}
            results['min_hp_entering_temp'] = {'value': min_hp_eft, 'units': 'C'}
            t_excess = self.cost(max_hp_eft, min_hp_eft)
            results['excess_fluid_temperature'] = {'value': t_excess, 'units': 'C'}
        results['peak_load_analysis'] = self.hybrid_load.as_dict()

        g_function = dict()
        g_function['coordinates (x[m], y[m])'] = [(x, y) for x, y in self.gFunction.bore_locations]  # TODO: Verify form
        b_over_h = self.B_spacing / self.bhe.b.H
        g, _ = self.grab_g_function(b_over_h)
        total_g_values = g.x.size
        number_lts_g_values = 27
        number_sts_g_values = 50
        sts_step_size = floor((total_g_values - number_lts_g_values) / number_sts_g_values)
        lntts = []
        g_values = []
        for idx in range(0, (total_g_values - number_lts_g_values), sts_step_size):
            lntts.append(g.x[idx].tolist())
            g_values.append(g.y[idx].tolist())
        lntts += g.x[total_g_values - number_lts_g_values: total_g_values].tolist()
        g_values += g.y[total_g_values - number_lts_g_values: total_g_values].tolist()
        pairs = zip(lntts, g_values)
        for lntts_val, g_val in pairs:
            output += f"{lntts_val:0.4f}\t{g_val:0.4f}"
        g_function['lntts, g'] = [*pairs]

        results['g_function_information'] = g_function
        output['simulation_results'] = results

        return output

    def simulate(self, method: TimestepType):
        b = self.B_spacing
        b_over_h = b / self.bhe.b.H

        # Solve for equivalent single U-tube
        self.bhe_eq = self.bhe.to_single()
        # Update short time step object with equivalent single u-tube
        self.radial_numerical.calc_sts_g_functions(self.bhe_eq)
        # Combine the short and long-term g-functions. The long term g-function
        # is interpolated for specific B/H and rb/H values.
        g, _ = self.grab_g_function(b_over_h)

        if method == TimestepType.HYBRID:
            q_dot = self.hybrid_load.load[2:] * 1000.0  # convert to Watts
            time_values = self.hybrid_load.hour[2:]  # convert to seconds
            self.times = time_values
            self.loading = q_dot
            self.hp_eft, self.d_tb = self._simulate_detailed(q_dot, time_values, g)
        elif method == TimestepType.HOURLY or method == TimestepType.HOURLYNOLOADAGG:

            n_months = self.sim_params.end_month - self.sim_params.start_month + 1
            n_hours = int(n_months / 12.0 * 8760.0)
            q_dot = self.hourly_extraction_ground_loads

            n_years = ceil(n_hours / 8760)
            if len(q_dot) // 8760 < n_years:
                q_dot = q_dot * n_years
            else:
                n_hours = len(q_dot)

            q_dot = -1.0 * np.array(q_dot)  # Convert loads to rejection
            if len(self.times) == 0:
                self.times = np.arange(1, n_hours + 1, 1)

            self.loading = q_dot

            if TimestepType.HOURLY:
                self.hp_eft, self.d_tb = self.simulate_dynamic_load_agg(q_dot, g)
            else:
                self.hp_eft, self.d_tb = self._simulate_detailed(q_dot, self.times, g)

        else:
            raise ValueError("Only hybrid or hourly methods available.")

        return max(self.hp_eft), min(self.hp_eft)

    def size(self, method: TimestepType) -> None:
        # Size the ground heat exchanger
        def local_objective(h):
            self.bhe.b.H = h
            max_hp_eft, min_hp_eft = self.simulate(method)
            t_excess = self.cost(max_hp_eft, min_hp_eft)
            return t_excess

        # Make the initial guess variable the average of the heights given
        self.bhe.b.H = (self.sim_params.max_height + self.sim_params.min_height) / 2.0
        # bhe.b.H is updated during sizing
        returned_height = solve_root(
            self.bhe.b.H,
            local_objective,
            lower=self.sim_params.min_height,
            upper=self.sim_params.max_height,
            abs_tol=1.0e-6,
            rel_tol=1.0e-6,
            max_iter=50,
        )

        self.bhe.b.H = returned_height
