# Jack C. Cook
# Wednesday, October 27, 2021

import ghedt
import ghedt.PLAT.pygfunction as gt
import ghedt.PLAT as PLAT
from ghedt.utilities import sign, check_bracket
import numpy as np
import copy


class Bisection1D:
    def __init__(self, coordinates_domain: list, V_flow_borehole: float,
                 borehole: gt.boreholes.Borehole,
                 bhe_object: PLAT.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: PLAT.media.Pipe,
                 grout: PLAT.media.ThermalProperty, soil: PLAT.media.Soil,
                 sim_params: PLAT.media.SimulationParameters,
                 hourly_extraction_ground_loads: list,
                 max_iter=15, disp=False, search=True):

        # Take the lowest part of the coordinates domain to be used for the
        # initial setup
        coordinates = coordinates_domain[0]

        V_flow_system = V_flow_borehole * float(len(coordinates))
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = V_flow_borehole / 1000. * fluid.rho

        self.log_time = ghedt.utilities.Eskilson_log_times()
        self.bhe_object = bhe_object
        self.sim_params = sim_params
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.coordinates_domain = coordinates_domain
        self.max_iter = max_iter
        self.disp = disp

        B = ghedt.utilities.borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = ghedt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            self.bhe_object, self.log_time, coordinates, fluid, pipe, grout,
            soil)

        # Initialize the GHE object
        self.ghe = ghedt.ground_heat_exchangers.GHE(
            V_flow_system, B, bhe_object, fluid, borehole, pipe, grout, soil,
            g_function, sim_params, hourly_extraction_ground_loads)

        self.calculated_temperatures = {}

        if search:
            self.selection_key, self.selected_coordinates = self.search()

    def initialize_ghe(self, coordinates, H):

        self.ghe.bhe.b.H = H
        borehole = self.ghe.bhe.b
        m_flow_borehole = self.ghe.bhe.m_flow_borehole
        fluid = self.ghe.bhe.fluid
        pipe = self.ghe.bhe.pipe
        grout = self.ghe.bhe.grout
        soil = self.ghe.bhe.soil
        V_flow_borehole = self.ghe.V_flow_borehole
        V_flow_system = V_flow_borehole * float(len(coordinates))

        B = ghedt.utilities.borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = ghedt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            self.bhe_object, self.log_time, coordinates, fluid, pipe, grout,
            soil)

        # Initialize the GHE object
        self.ghe = ghedt.ground_heat_exchangers.GHE(
            V_flow_system, B, self.bhe_object, fluid, borehole, pipe, grout,
            soil, g_function, self.sim_params,
            self.hourly_extraction_ground_loads)

    def calculate_excess(self, coordinates, H):
        self.initialize_ghe(coordinates, H)
        # Simulate after computing just one g-function
        max_HP_EFT, min_HP_EFT = self.ghe.simulate()
        T_excess = self.ghe.cost(max_HP_EFT, min_HP_EFT)

        if self.disp:
            print('Min EFT: {}\nMax EFT: {}'.format(min_HP_EFT, max_HP_EFT))

        return T_excess

    def search(self):

        xL_idx = 0
        xR_idx = len(self.coordinates_domain) - 1
        # Do some initial checks before searching
        # Get the lowest possible excess temperature from minimum height at the
        # smallest location in the domain
        T_0_lower = self.calculate_excess(self.coordinates_domain[xL_idx],
                                          self.sim_params.min_Height)
        T_0_upper = self.calculate_excess(self.coordinates_domain[xL_idx],
                                          self.sim_params.max_Height)
        T_m1 = \
            self.calculate_excess(
                self.coordinates_domain[xR_idx],
                self.sim_params.max_Height)

        self.calculated_temperatures[xL_idx] = T_0_upper
        self.calculated_temperatures[xR_idx] = T_m1

        if check_bracket(sign(T_0_lower), sign(T_0_upper), disp=self.disp):
            # Size between min and max of lower bound in domain
            return self.coordinates_domain[0]
        elif check_bracket(sign(T_0_upper), sign(T_m1), disp=self.disp):
            # Do the integer bisection search routine
            pass
        else:
            # This domain does not bracked the solution
            if T_0_upper < 0.0 and T_m1 < 0.0:
                msg = 'Based on the loads provided, the excess temperatures ' \
                      'for the minimum and maximum number of boreholes falls ' \
                      'below 0. This means that the loads are "miniscule" or ' \
                      'that the lower end of the domain needs to contain ' \
                      'less boreholes.'.format()
                raise ValueError(msg)
            if T_0_upper > 0.0 and T_m1 > 0.0:
                msg = 'Based on the loads provided, the excess temperatures ' \
                      'for the minimum and maximum number of boreholes falls ' \
                      'above 0. This means that the loads are "astronomical" ' \
                      'or that the higher end of the domain needs to contain ' \
                      'more boreholes. Consider increasing the available land' \
                      ' area, or decreasing the minimum allowable borehole ' \
                      'spacing.'
                raise ValueError(msg)
            return None, None

        if self.disp:
            print('Beginning bisection search...')

        xL_sign = sign(T_0_upper)
        xR_sign = sign(T_m1)
        
        i = 0

        while i < self.max_iter:
            c_idx = int(np.ceil((xL_idx + xR_idx) / 2))
            # if the solution is no longer making progress break the while
            if c_idx == xL_idx or c_idx == xR_idx:
                break

            c_T_excess = self.calculate_excess(self.coordinates_domain[c_idx],
                                               self.sim_params.max_Height)
            self.calculated_temperatures[c_idx] = c_T_excess
            c_sign = sign(c_T_excess)

            if c_sign == xL_sign:
                xL_idx = copy.deepcopy(c_idx)
            else:
                xR_idx = copy.deepcopy(c_idx)

            i += 1

        coordinates = self.coordinates_domain[i]

        H = self.sim_params.max_Height

        self.calculate_excess(coordinates, H)
        # Make sure the field being returned pertains to the index which is the
        # closest to 0 but also negative (the maximum of all 0 or negative
        # excess temperatures)
        keys = list(self.calculated_temperatures.keys())
        values = list(self.calculated_temperatures.values())

        negative_excess_values = [values[i] for i in range(len(values))
                                  if values[i] <= 0.0]

        excess_of_interest = max(negative_excess_values)
        idx = values.index(excess_of_interest)
        selection_key = keys[idx]
        selected_coordinates = self.coordinates_domain[selection_key]

        self.initialize_ghe(selected_coordinates, H)

        return selection_key, selected_coordinates


class Bisection2D(Bisection1D):
    def __init__(self, coordinates_domain_nested: list, V_flow_borehole: float,
                 borehole: gt.boreholes.Borehole,
                 bhe_object: PLAT.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: PLAT.media.Pipe,
                 grout: PLAT.media.ThermalProperty, soil: PLAT.media.Soil,
                 sim_params: PLAT.media.SimulationParameters,
                 hourly_extraction_ground_loads: list,
                 max_iter=15, disp=False):
        # Get a coordinates domain for initialization
        coordinates_domain = coordinates_domain_nested[0]
        Bisection1D.__init__(
            self, coordinates_domain, V_flow_borehole, borehole, bhe_object,
            fluid, pipe, grout, soil, sim_params,
            hourly_extraction_ground_loads, max_iter=max_iter, disp=disp,
            search=False)

        self.coordinates_domain_nested = []
        self.calculated_temperatures_nested = []
        # Tack on one borehole at the beginning to provide a high excess
        # temperature
        outer_domain = [coordinates_domain_nested[0][0]]
        for i in range(len(coordinates_domain_nested)):
            outer_domain.append(coordinates_domain_nested[i][-1])

        self.coordinates_domain = outer_domain

        selection_key, selected_coordinates = self.search()

        self.calculated_temperatures_nested.append(self.calculated_temperatures)

        # We tacked on one borehole to the beginning, so we need to subtract 1
        # on the index
        inner_domain = coordinates_domain_nested[selection_key-1]
        self.coordinates_domain = inner_domain

        self.selection_key, self.selected_coordinates = self.search()


# def bi_rectangular_search(length_x, length_y, B_min, B_max_x, B_max_y):
#     # Make this work for the transpose
#     if length_x >= length_y:
#         length_1 = length_x
#         length_2 = length_y
#         B_max_1 = B_max_x
#         B_max_2 = B_max_y
#     else:
#         length_1 = length_y
#         length_2 = length_x
#         B_max_1 = B_max_y
#         B_max_2 = B_max_x
#
#     def func(B, length, n):
#         _n = (length / B) + 1
#         return n - _n
#
#     # find the maximum number of boreholes as a float
#     n_2_max = (length_2 / B_min) + 1
#     n_2_min = (length_2 / B_max_2) + 1
#
#     N_min = int(np.ceil(n_2_min).tolist())
#     N_max = int(np.floor(n_2_max).tolist())
#
#     bi_rectangle_nested_domain = []
#
#     for n_2 in range(N_min, N_max+1):
#         b_2 = length_2 / (n_2 - 1)
#         bi_rectangle_domain = ghedt.domains.bi_rectangular(
#             length_1, length_2, B_min, B_max_1, b_2)
#         bi_rectangle_nested_domain.append(bi_rectangle_domain)
#
#     return bi_rectangle_nested_domain
