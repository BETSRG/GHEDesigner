from math import ceil

import numpy as np
from pygfunction.boreholes import Borehole
from scipy.interpolate import interp1d

from ghedesigner.constants import SEC_IN_HR, TWO_PI, VERSION
from ghedesigner.enums import PipeType, TimestepType
from ghedesigner.ghe.boreholes.factory import get_bhe_object
from ghedesigner.ghe.gfunction import GFunction, calc_g_func_for_multiple_lengths
from ghedesigner.ghe.ground_loads import HybridLoad
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Grout, Soil
from ghedesigner.utilities import combine_sts_lts, solve_root


class GHE:
    def __init__(
        self,
        v_flow_system: float,
        b_spacing: float,
        bhe_type: PipeType,
        fluid,
        borehole: Borehole,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        g_function: GFunction,
        start_month: int,
        end_month: int,
        hourly_extraction_ground_loads: list,
        field_type="N/A",
        field_specifier="N/A",
    ) -> None:
        self.field_type = field_type
        self.fieldSpecifier = field_specifier
        self.v_flow_system = v_flow_system
        self.b_spacing = b_spacing
        self.nbh = len(g_function.bore_locations)
        self.v_flow_borehole = self.v_flow_system / self.nbh
        m_flow_borehole = self.v_flow_borehole / 1000.0 * fluid.rho
        self.m_flow_borehole = m_flow_borehole

        # Borehole Heat Exchanger
        self.bhe_type = bhe_type
        self.bhe = get_bhe_object(bhe_type, m_flow_borehole, fluid, borehole, pipe, grout, soil)

        # Equivalent borehole Heat Exchanger
        self.bhe_eq = self.bhe.to_single()

        # Radial numerical short time step
        self.bhe_eq.calc_sts_g_functions()

        # gFunction object
        self.gFunction = g_function
        # Additional simulation parameters
        self.start_month = start_month
        self.end_month = end_month

        # Hourly ground extraction loads
        # Building cooling is negative, building heating is positive
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.times = np.empty((0,), dtype=np.float64)
        self.loading: list | None = None

        self.hybrid_load = HybridLoad(
            self.hourly_extraction_ground_loads, self.bhe_eq, self.bhe_eq, start_month, end_month
        )

        # List of heat pump exiting fluid temperatures
        self.hp_eft: list[float] = []
        # list of change in borehole wall temperatures
        self.dTb: list[float] = []

    def as_dict(self) -> dict:
        output = {
            "title": f"GHEDesigner GHE Output - Version {VERSION}",
            "number_of_boreholes": len(self.gFunction.bore_locations),
            "borehole_depth": {"value": self.bhe.borehole.H, "units": "m"},
            "borehole_spacing": {"value": self.b_spacing, "units": "m"},
            "borehole_heat_exchanger": self.bhe.as_dict(),
            "equivalent_borehole_heat_exchanger": self.bhe_eq.as_dict(),
            # "simulation_parameters": self.sim_params.as_dict(),
        }
        return output

    def grab_g_function(self, b_over_h):
        # interpolate for the Long time step g-function
        g_function, rb_value, _, _ = self.gFunction.g_function_interpolation(b_over_h)
        # correct the long time step for borehole radius
        g_function_corrected = self.gFunction.borehole_radius_correction(g_function, rb_value, self.bhe.borehole.r_b)
        # Don't Update the HybridLoad (its dependent on the STS) because
        # it doesn't change the results much, and it slows things down a lot
        # combine the short and long time step g-function
        g = combine_sts_lts(
            self.gFunction.log_time,
            g_function_corrected,
            self.bhe_eq.lntts.tolist(),
            self.bhe_eq.g.tolist(),
        )

        g_bhw = combine_sts_lts(
            self.gFunction.log_time,
            g_function_corrected,
            self.bhe_eq.lntts.tolist(),
            self.bhe_eq.g_bhw.tolist(),
        )

        return g, g_bhw

    def cost(self, max_eft: float, min_eft: float, design_max_eft: float, design_min_eft: float):
        delta_t_max = max_eft - design_max_eft
        delta_t_min = design_min_eft - min_eft
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

        q_dot_b_dt = np.hstack(q_dot_b[1:] - q_dot_b[:-1])

        ts = self.bhe_eq.t_s  # (-)
        two_pi_k = TWO_PI * self.bhe.soil.k  # (W/m.K)
        h = self.bhe.borehole.H  # (meters)
        tg = self.bhe.soil.ugt  # (Celsius)
        rb = self.bhe.calc_effective_borehole_resistance()  # (m.K/W)
        m_dot = self.bhe.m_flow_borehole  # (kg/s)
        cp = self.bhe.fluid.cp  # (J/kg.s)

        hp_eft: list[float] = []
        delta_tb: list[float] = []
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

    def compute_g_functions(self, h_min: float, h_max: float):
        # Compute g-functions for a bracketed solution, based on min and max height
        self.gFunction = calc_g_func_for_multiple_lengths(
            self.b_spacing,
            [h_min] if h_min == h_max else [h_min, (h_min + h_max) / 2.0, h_max],
            self.bhe.borehole.r_b,
            self.bhe.borehole.D,
            self.bhe.m_flow_borehole,
            self.bhe_type,
            self.gFunction.log_time,
            self.gFunction.bore_locations,
            self.bhe.fluid,
            self.bhe.pipe,
            self.bhe.grout,
            self.bhe.soil,
        )

    def simulate(self, method: TimestepType):
        b = self.b_spacing
        b_over_h = b / self.bhe.borehole.H

        # Solve for equivalent single U-tube
        self.bhe_eq = self.bhe.to_single()
        # Update short time step object with equivalent single u-tube
        self.bhe_eq.calc_sts_g_functions()
        # Combine the short and long-term g-functions. The long term g-function
        # is interpolated for specific B/H and rb/H values.
        g, _ = self.grab_g_function(b_over_h)

        if method == TimestepType.HYBRID:
            q_dot = self.hybrid_load.load[2:] * 1000.0  # convert to Watts
            time_values = self.hybrid_load.hour[2:]  # convert to seconds
            self.times = time_values
            self.loading = q_dot

            hp_eft, d_tb = self._simulate_detailed(q_dot, time_values, g)
        elif method == TimestepType.HOURLY:
            n_months = self.end_month - self.start_month + 1
            n_hours = int(n_months / 12.0 * 8760.0)
            q_dot = self.hourly_extraction_ground_loads
            # How many times does q need to be repeated?
            n_years = ceil(n_hours / 8760)
            if len(q_dot) // 8760 < n_years:
                q_dot = q_dot * n_years
            else:
                n_hours = len(q_dot)
            q_dot = -1.0 * np.array(q_dot)  # Convert loads to rejection
            # print("Times:",self.times)
            if len(self.times) == 0:
                self.times = np.arange(1, n_hours + 1, 1)
            t = self.times
            self.loading = q_dot

            hp_eft, d_tb = self._simulate_detailed(q_dot, t, g)
        else:
            raise ValueError("Only hybrid or hourly methods available.")

        self.hp_eft = hp_eft
        self.dTb = d_tb

        return max(hp_eft), min(hp_eft)

    def size(
        self, method: TimestepType, max_height: float, min_height: float, design_max_eft: float, design_min_eft: float
    ) -> None:
        # Size the ground heat exchanger
        def local_objective(h: float):
            self.bhe.borehole.H = h
            this_max_hp_eft, this_min_hp_eft = self.simulate(method=method)
            t_excess = self.cost(this_max_hp_eft, this_min_hp_eft, design_max_eft, design_min_eft)
            return t_excess

        # Make the initial guess variable the average of the heights given
        self.bhe.borehole.H = (max_height + min_height) / 2.0
        # bhe.b.H is updated during sizing
        returned_height = solve_root(
            self.bhe.borehole.H,
            local_objective,
            lower=min_height,
            upper=max_height,
            abs_tol=1.0e-6,
            rel_tol=1.0e-6,
            max_iter=50,
        )

        self.bhe.borehole.H = returned_height

    @staticmethod
    def calculate(_hour_index: int, inlet_temp: float, _flow_rate: float) -> float:
        effectiveness = 0.5
        soil_temp = 20

        return effectiveness * (soil_temp - inlet_temp) + inlet_temp
