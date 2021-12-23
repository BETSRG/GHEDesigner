# Jack C. Cook
# Wednesday, October 27, 2021

import ghedt as dt
import ghedt.pygfunction as gt
import ghedt.peak_load_analysis_tool as plat
from ghedt.utilities import sign, check_bracket
import numpy as np
import copy


class Bisection1D:
    def __init__(self, coordinates_domain: list, V_flow_borehole: float,
                 borehole: gt.boreholes.Borehole,
                 bhe_object: plat.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: plat.media.Pipe,
                 grout: plat.media.ThermalProperty, soil: plat.media.Soil,
                 sim_params: plat.media.SimulationParameters,
                 hourly_extraction_ground_loads: list,
                 max_iter=15, disp=False, search=True):

        # Take the lowest part of the coordinates domain to be used for the
        # initial setup
        coordinates = coordinates_domain[0]

        V_flow_system = V_flow_borehole * float(len(coordinates))
        # Total fluid mass flow rate per borehole (kg/s)
        m_flow_borehole = V_flow_borehole / 1000. * fluid.rho

        self.log_time = dt.utilities.Eskilson_log_times()
        self.bhe_object = bhe_object
        self.sim_params = sim_params
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.coordinates_domain = coordinates_domain
        self.max_iter = max_iter
        self.disp = disp

        B = dt.utilities.borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = dt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            self.bhe_object, self.log_time, coordinates, fluid, pipe, grout,
            soil)

        # Initialize the GHE object
        self.ghe = dt.ground_heat_exchangers.GHE(
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

        B = dt.utilities.borehole_spacing(borehole, coordinates)

        # Calculate a g-function for uniform inlet fluid temperature with
        # 8 unequal segments using the equivalent solver
        g_function = dt.gfunction.compute_live_g_function(
            B, [borehole.H], [borehole.r_b], [borehole.D], m_flow_borehole,
            self.bhe_object, self.log_time, coordinates, fluid, pipe, grout,
            soil)

        # Initialize the GHE object
        self.ghe = dt.ground_heat_exchangers.GHE(
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
            self.initialize_ghe(self.coordinates_domain[0],
                                self.sim_params.max_Height)
            return 0, self.coordinates_domain[0]
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
                 bhe_object: plat.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: plat.media.Pipe,
                 grout: plat.media.ThermalProperty, soil: plat.media.Soil,
                 sim_params: plat.media.SimulationParameters,
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

        self.calculated_temperatures_nested.append(
            copy.deepcopy(self.calculated_temperatures))

        # We tacked on one borehole to the beginning, so we need to subtract 1
        # on the index
        inner_domain = coordinates_domain_nested[selection_key-1]
        self.coordinates_domain = inner_domain

        # Reset calculated temperatures
        self.calculated_temperatures = {}

        self.selection_key, self.selected_coordinates = self.search()


class BisectionZD(Bisection1D):
    def __init__(self, coordinates_domain_nested: list, V_flow_borehole: float,
                 borehole: gt.boreholes.Borehole,
                 bhe_object: plat.borehole_heat_exchangers,
                 fluid: gt.media.Fluid, pipe: plat.media.Pipe,
                 grout: plat.media.ThermalProperty, soil: plat.media.Soil,
                 sim_params: plat.media.SimulationParameters,
                 hourly_extraction_ground_loads: list,
                 max_iter=15, disp=False):
        # Get a coordinates domain for initialization
        coordinates_domain = coordinates_domain_nested[0]
        Bisection1D.__init__(
            self, coordinates_domain, V_flow_borehole, borehole, bhe_object,
            fluid, pipe, grout, soil, sim_params,
            hourly_extraction_ground_loads, max_iter=max_iter, disp=disp,
            search=False)

        self.coordinates_domain_nested = coordinates_domain_nested
        self.calculated_temperatures_nested = {}
        # Tack on one borehole at the beginning to provide a high excess
        # temperature
        outer_domain = [coordinates_domain_nested[0][0]]
        for i in range(len(coordinates_domain_nested)):
            outer_domain.append(coordinates_domain_nested[i][-1])

        self.coordinates_domain = outer_domain

        self.selection_key_outer, self.selected_coordinates_outer = \
            self.search()

        if self.selection_key_outer > 0:
            self.selection_key_outer -= 1
        self.calculated_heights = {}

        self.selection_key, self.selected_coordinates = self.search_successive()

    def search_successive(self, max_iter=None):
        if max_iter is None:
            max_iter = self.selection_key_outer + 7

        i = self.selection_key_outer

        old_height = 99999

        while i < len(self.coordinates_domain_nested) and i < max_iter:

            self.coordinates_domain = self.coordinates_domain_nested[i]
            self.calculated_temperatures = {}
            try:
                selection_key, selected_coordinates = self.search()
            except ValueError:
                break
            self.calculated_temperatures_nested[i] = \
                copy.deepcopy(self.calculated_temperatures)

            self.ghe.compute_g_functions()
            self.ghe.size(method='hybrid')

            nbh = len(selected_coordinates)
            total_drilling = float(nbh) * self.ghe.bhe.b.H
            self.calculated_heights[i] = total_drilling

            if old_height < total_drilling:
                break
            else:
                old_height = copy.deepcopy(total_drilling)

            i += 1

        keys = list(self.calculated_heights.keys())
        values = list(self.calculated_heights.values())

        minimum_total_drilling = min(values)
        idx = values.index(minimum_total_drilling)
        selection_key_outer = keys[idx]
        self.calculated_temperatures = \
            copy.deepcopy(self.calculated_temperatures_nested[
                              selection_key_outer])

        keys = list(self.calculated_temperatures.keys())
        values = list(self.calculated_temperatures.values())

        negative_excess_values = [values[i] for i in range(len(values))
                                  if values[i] <= 0.0]

        excess_of_interest = max(negative_excess_values)
        idx = values.index(excess_of_interest)
        selection_key = keys[idx]
        selected_coordinates = \
            self.coordinates_domain_nested[selection_key_outer][selection_key]

        self.initialize_ghe(selected_coordinates, self.sim_params.max_Height)
        self.ghe.compute_g_functions()
        self.ghe.size(method='hybrid')

        return selection_key, selected_coordinates


# The following functions are utility functions specific to search_routines.py
# ------------------------------------------------------------------------------
def oak_ridge_export(bisection_search, file_name='ghedt_output'):
    # Dictionary for export
    d = {}
    d['number_of_boreholes'] = len(bisection_search.selected_coordinates)
    d['g_function_pairs'] = []
    d['single_u_tube'] = {}

    # create a local single U-tube object
    bhe_eq = bisection_search.ghe.bhe_eq
    d['single_u_tube']['r_b'] = bhe_eq.borehole.r_b  # Borehole radius
    d['single_u_tube']['r_in'] = bhe_eq.r_in  # Inner pipe radius
    d['single_u_tube']['r_out'] = bhe_eq.r_out  # Outer pipe radius
    # Note: Shank spacing or center pipe positions could be used
    d['single_u_tube']['s'] = bhe_eq.pipe.s  # Shank spacing (tube-to-tube)
    d['single_u_tube']['pos'] = bhe_eq.pos  # Center of the pipes
    d['single_u_tube']['m_flow_borehole'] = \
        bhe_eq.m_flow_borehole  # mass flow rate of the borehole
    d['single_u_tube']['k_g'] = bhe_eq.grout.k  # Grout thermal conductivity
    d['single_u_tube']['k_s'] = bhe_eq.soil.k  # Soil thermal conductivity
    d['single_u_tube']['k_p'] = bhe_eq.pipe.k  # Pipe thermal conductivity

    # create a local ghe object
    ghe = bisection_search.ghe
    H = ghe.bhe.b.H
    B_over_H = ghe.B_spacing / H
    g = ghe.grab_g_function(B_over_H)

    lntts = []
    g_values = []
    for i in range(len(g.y)):
        lntts.append(g.x[i].tolist())
        g_values.append(g.y[i].tolist())

    for i in range(len(lntts)):
        d['g_function_pairs'].append({'ln_tts': lntts[i],
                                      'g_value': g_values[i]})

    dt.utilities.js_dump(file_name, d, indent=4)

    return
