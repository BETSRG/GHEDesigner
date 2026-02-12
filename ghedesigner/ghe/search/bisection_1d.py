from math import ceil

import numpy as np
from pygfunction.boreholes import Borehole

from ghedesigner.enums import FlowConfigType, TimestepType
from ghedesigner.ghe.gfunction import calc_g_func_for_multiple_lengths
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.utilities import borehole_spacing, check_bracket, eskilson_log_times, sign


class Bisection1D:
    def __init__(
        self,
        coordinates_domain: list,
        field_descriptors: list,
        v_flow: float,
        borehole: Borehole,
        fluid: Fluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        max_boreholes: int | None,
        min_height: float,
        max_height: float,
        continue_if_design_unmet: bool,
        start_month: int,
        end_month: int,
        min_eft: float,
        max_eft: float,
        hourly_extraction_ground_loads: list,
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        max_iter=15,
        disp=False,
        search=True,
        field_type="N/A",
        load_years=None,
    ) -> None:
        # Take the lowest part of the coordinates domain to be used for the
        # initial setup
        if load_years is None:
            load_years = [2019]
        self.load_years = load_years
        self.searchTracker: list[list] = []
        coordinates = coordinates_domain[0]
        current_field = field_descriptors[0]
        self.field_type = field_type
        # Flow rate tracking
        self.v_flow = v_flow
        self.flow_type = flow_type
        v_flow_system, m_flow_borehole = self.retrieve_flow(coordinates, fluid.rho)
        self.method = method
        self.min_height = min_height
        self.max_height = max_height
        self.start_month = start_month
        self.end_month = end_month
        self.max_boreholes = max_boreholes
        self.max_eft = max_eft
        self.min_eft = min_eft
        self.continue_if_design_unmet = continue_if_design_unmet
        self.log_time = eskilson_log_times()
        self.bhe_type = pipe.type
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.coordinates_domain = coordinates_domain
        self.fieldDescriptors = field_descriptors
        self.max_iter = max_iter
        self.disp = disp

        b = borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = calc_g_func_for_multiple_lengths(
            b,
            [borehole.H],
            borehole.r_b,
            borehole.D,
            m_flow_borehole,
            self.bhe_type,
            self.log_time,
            coordinates,
            fluid,
            pipe,
            grout,
            soil,
        )

        # Initialize the GHE object
        self.ghe = GHE(
            v_flow_system,
            b,
            pipe.type,
            fluid,
            borehole,
            pipe,
            grout,
            soil,
            g_function,
            self.start_month,
            self.end_month,
            hourly_extraction_ground_loads,
            field_specifier=current_field,
            field_type=field_type,
        )

        self.calculated_temperatures: dict[int, np.float64] = {}

        if search:
            self.selection_key, self.selected_coordinates = self.search()

    def retrieve_flow(self, coordinates, rho):
        if self.flow_type == FlowConfigType.BOREHOLE:
            v_flow_system = self.v_flow * len(coordinates)
            # Total fluid mass flow rate per borehole (kg/s)
            m_flow_borehole = self.v_flow / 1000.0 * rho
        elif self.flow_type == FlowConfigType.SYSTEM:
            v_flow_system = self.v_flow
            v_flow_borehole = self.v_flow / len(coordinates)
            m_flow_borehole = v_flow_borehole / 1000.0 * rho
        else:
            raise ValueError("The flow argument should be either `borehole` or `system`.")
        return v_flow_system, m_flow_borehole

    def initialize_ghe(self, coordinates, h, field_specifier="N/A"):
        v_flow_system, m_flow_borehole = self.retrieve_flow(coordinates, self.ghe.bhe.fluid.rho)

        self.ghe.bhe.borehole.H = h
        borehole = self.ghe.bhe.borehole
        fluid = self.ghe.bhe.fluid
        pipe = self.ghe.bhe.pipe
        grout = self.ghe.bhe.grout
        soil = self.ghe.bhe.soil

        b = borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = calc_g_func_for_multiple_lengths(
            b,
            [borehole.H],
            borehole.r_b,
            borehole.D,
            m_flow_borehole,
            self.bhe_type,
            self.log_time,
            coordinates,
            fluid,
            pipe,
            grout,
            soil,
        )

        # Initialize the GHE object
        self.ghe = GHE(
            v_flow_system,
            b,
            self.bhe_type,
            fluid,
            borehole,
            pipe,
            grout,
            soil,
            g_function,
            self.start_month,
            self.end_month,
            self.hourly_extraction_ground_loads,
            field_type=self.field_type,
            field_specifier=field_specifier,
        )

    def calculate_excess(self, coordinates, h, field_specifier="N/A"):
        self.initialize_ghe(coordinates, h, field_specifier=field_specifier)
        # Simulate after computing just one g-function
        max_hp_eft, min_hp_eft = self.ghe.simulate(method=self.method)
        t_excess = self.ghe.cost(max_hp_eft, min_hp_eft, self.max_eft, self.min_eft)
        self.searchTracker.append([field_specifier, t_excess, max_hp_eft, min_hp_eft])

        return t_excess

    def search(self):
        x_l_idx = 0

        # find upper bound that respects max_boreholes
        if self.max_boreholes is not None:
            num_coordinates_in_each = [len(x) for x in self.coordinates_domain]
            x_r_idx = [idx for idx, x in enumerate(num_coordinates_in_each) if x < self.max_boreholes][-1]
        else:
            x_r_idx = len(self.coordinates_domain) - 1

        if self.disp:
            print("Do some initial checks before searching.")
        # Get the lowest possible excess temperature from minimum height at the
        # smallest location in the domain
        t_0_lower = self.calculate_excess(
            self.coordinates_domain[x_l_idx],
            self.min_height,
            field_specifier=self.fieldDescriptors[x_l_idx],
        )
        t_0_upper = self.calculate_excess(
            self.coordinates_domain[x_l_idx],
            self.max_height,
            field_specifier=self.fieldDescriptors[x_l_idx],
        )
        t_m1 = self.calculate_excess(
            self.coordinates_domain[x_r_idx],
            self.max_height,
            field_specifier=self.fieldDescriptors[x_r_idx],
        )

        self.calculated_temperatures[x_l_idx] = t_0_upper
        self.calculated_temperatures[x_r_idx] = t_m1

        if check_bracket(sign(t_0_lower), sign(t_0_upper)):
            if self.disp:
                print("Size between min and max of lower bound in domain.")
            self.initialize_ghe(self.coordinates_domain[x_l_idx], self.max_height)
            return x_l_idx, self.coordinates_domain[x_l_idx]
        elif check_bracket(sign(t_0_upper), sign(t_m1)):
            if self.disp:
                print("Perform the integer bisection search routine.")
        elif t_0_lower < 0.0:
            condition_msg = (
                "The optimal design requires fewer or shorter boreholes \n"
                "than what is possible based on the current design parameters."
            )
            print(condition_msg)
            if self.continue_if_design_unmet:
                print("Smallest available configuration selected.")
                selection_key = x_l_idx
                self.initialize_ghe(
                    self.coordinates_domain[selection_key],
                    self.min_height,
                    self.fieldDescriptors[selection_key],
                )
                return selection_key, self.coordinates_domain[selection_key]
            else:
                raise ValueError("Search failed.")
        elif t_m1 > 0.0:
            condition_msg = (
                "The optimal design requires more or deeper boreholes \n"
                "than what is possible based on the current design parameters. \n"
                "Consider increasing the available land area, \n"
                "increasing the maximum borehole depth, \n"
                "or decreasing the maximum borehole spacing."
            )
            print(condition_msg)
            if self.continue_if_design_unmet:
                print("Largest available configuration selected.")
                selection_key = x_r_idx
                self.initialize_ghe(
                    self.coordinates_domain[selection_key],
                    self.max_height,
                    self.fieldDescriptors[selection_key],
                )
                return selection_key, self.coordinates_domain[selection_key]
            else:
                raise ValueError("Search failed.")
        else:
            # if we've gotten here, everything should be good for the bisection search.
            # can add catches for other cases here if they pop up.
            pass
        if self.disp:
            print("Beginning bisection search...")

        x_l_sign = sign(t_0_upper)

        i = 0

        while i < self.max_iter:
            c_idx = ceil((x_l_idx + x_r_idx) / 2)
            # if the solution is no longer making progress break the while
            # if c_idx == x_l_idx or c_idx == x_r_idx:
            if c_idx in (x_l_idx, x_r_idx):
                break

            c_t_excess = self.calculate_excess(
                self.coordinates_domain[c_idx],
                self.max_height,
                field_specifier=self.fieldDescriptors[c_idx],
            )

            self.calculated_temperatures[c_idx] = c_t_excess
            c_sign = sign(c_t_excess)

            if c_sign == x_l_sign:
                x_l_idx = c_idx
            else:
                x_r_idx = c_idx

            i += 1

        coordinates = self.coordinates_domain[i]

        self.calculate_excess(coordinates, self.max_height, self.fieldDescriptors[i])
        # Make sure the field being returned pertains to the index which is the
        # closest to 0 but also negative (the maximum of all 0 or negative
        # excess temperatures)
        keys = list(self.calculated_temperatures.keys())
        values = list(self.calculated_temperatures.values())

        # theoretically, the biggest negative value should be the field that is just undersized
        negative_excess_values = [v for v in values if v <= 0.0]
        excess_of_interest = max(negative_excess_values)

        # but some conditions don't yield this result
        # adding a check here to ensure we pick the smallest field with
        # negative excess temperature
        num_bh = [len(self.coordinates_domain[x]) for x in keys]
        sorted_num_bh, sorted_values = (list(t) for t in zip(*sorted(zip(num_bh, values))))
        for _, val in zip(sorted_num_bh, sorted_values):
            if val < 0:
                if excess_of_interest != val:
                    print(
                        "Loads resulted in odd behavior requiring the selected field configuration \n"
                        "to be reset to the smallest field with negative excess temperature. \n"
                        "Please forward the inputs to the developers for investigation."
                    )
                excess_of_interest = val
                break

        idx = values.index(excess_of_interest)
        selection_key = keys[idx]
        self.initialize_ghe(
            self.coordinates_domain[selection_key], self.max_height, self.fieldDescriptors[selection_key]
        )
        return selection_key, self.coordinates_domain[selection_key]
