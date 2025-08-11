from math import ceil, pi

import numpy as np
from pygfunction.boreholes import Borehole

from ghedesigner.enums import FlowConfigType, TimestepType
from ghedesigner.ghe.gfunction import calc_g_func_for_multiple_lengths
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import GHEFluid, Grout, Soil
from ghedesigner.utilities import borehole_spacing, check_bracket, eskilson_log_times, sign


class Bisection1DTiltDrillPad:
    def __init__(
        self,
        coordinates_domain: tuple[list[tuple[float, float]], list[float], list[float]],
        field_descriptors: list[str],
        v_flow: float,
        borehole: Borehole,
        fluid: GHEFluid,
        pipe: Pipe,
        grout: Grout,
        soil: Soil,
        max_boreholes: int | None,
        min_height: float,
        max_height: float,
        ndp_min: int,
        ndp_max: int,
        continue_if_design_unmet: bool,
        start_month: int,
        end_month: int,
        min_eft: float,
        max_eft: float,
        hourly_extraction_ground_loads: list[float],
        method: TimestepType,
        flow_type: FlowConfigType = FlowConfigType.BOREHOLE,
        max_iter=15,
        disp=False,
        search=True,
        field_type="DRILLPAD",
        load_years=None,
    ) -> None:
        # defaults for load years
        if load_years is None:
            load_years = [2019]
        self.load_years = load_years
        self.searchTracker: list[list] = []
        # store parameters
        self.coordinates_domain = coordinates_domain
        self.fieldDescriptors = field_descriptors
        self.v_flow = v_flow
        self.borehole = borehole
        self.fluid = fluid
        self.pipe = pipe
        self.grout = grout
        self.soil = soil
        self.max_boreholes = max_boreholes
        self.min_height = min_height
        self.max_height = max_height
        self.ndp_min = ndp_min
        self.ndp_max = ndp_max
        self.continue_if_design_unmet = continue_if_design_unmet
        self.start_month = start_month
        self.end_month = end_month
        self.min_eft = min_eft
        self.max_eft = max_eft
        self.hourly_extraction_ground_loads = hourly_extraction_ground_loads
        self.method = method
        self.flow_type = flow_type
        self.field_type = field_type
        self.max_iter = max_iter
        self.disp = disp

        # take first layout as initial
        coords = self.coordinates_domain[0][0]
        tilts = self.coordinates_domain[0][1]
        orients = self.coordinates_domain[0][2]

        # compute flow per borehole
        v_flow_system, m_flow_borehole = self.retrieve_flow(coords, fluid.rho)

        # initial g-function
        b = borehole_spacing(borehole, coords)
        g_function = calc_g_func_for_multiple_lengths(
            b,
            [borehole.H],
            borehole.r_b,
            borehole.D,
            m_flow_borehole,
            pipe.type,
            eskilson_log_times(),
            coords,
            fluid,
            pipe,
            grout,
            soil,
            tilts=tilts,
            orientations=orients,
            solver="similarities",
        )

        # initialize GHE
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
            start_month,
            end_month,
            hourly_extraction_ground_loads,
            field_specifier=field_descriptors,
            field_type=field_type,
        )

        self.g_function = None
        self.calculated_temperatures: dict[int, np.float64] = {}

        if search:
            self.selection_key, self.selected_coordinates = self.search()

    def retrieve_flow(self, coordinates, rho):
        if self.flow_type == FlowConfigType.BOREHOLE:
            v_flow_system = self.v_flow * len(coordinates)
            m_flow_borehole = self.v_flow / 1000.0 * rho
        else:
            v_flow_system = self.v_flow
            m_flow_borehole = (self.v_flow / len(coordinates)) / 1000.0 * rho
        return v_flow_system, m_flow_borehole

    def initialize_g_function(self, coords, h, tilts, orientations):

        self.ghe.bhe.b.H = h
        borehole = self.ghe.bhe.b
        fluid = self.ghe.bhe.fluid
        pipe = self.ghe.bhe.pipe
        grout = self.ghe.bhe.grout
        soil = self.ghe.bhe.soil

        v_flow_system, m_flow_borehole = self.retrieve_flow(coords, fluid.rho)
        b = borehole_spacing(borehole, coords)
        solver = "similarities" if (tilts and orientations) else "equivalent"

        g_function = calc_g_func_for_multiple_lengths(
            b,
            [borehole.H],
            borehole.r_b,
            borehole.D,
            m_flow_borehole,
            pipe.type,
            eskilson_log_times(),
            coords,
            fluid,
            pipe,
            grout,
            soil,
            tilts=tilts,
            orientations=orientations,
            solver=solver,
        )

        self.g_function = g_function

    def initialize_ghe(self, coords, h, field_specifier, scaled_loads):
        # update borehole depth
        self.ghe.bhe.b.H = h
        borehole = self.ghe.bhe.b
        fluid = self.ghe.bhe.fluid
        pipe = self.ghe.bhe.pipe
        grout = self.ghe.bhe.grout
        soil = self.ghe.bhe.soil
        v_flow_system, m_flow_borehole = self.retrieve_flow(coords, fluid.rho)
        b = borehole_spacing(borehole, coords)
        # re-initialize GHE
        self.ghe = GHE(
            v_flow_system,
            b,
            pipe.type,
            fluid,
            borehole,
            pipe,
            grout,
            soil,
            self.g_function,
            self.start_month,
            self.end_month,
            scaled_loads,
            field_specifier=field_specifier,
            field_type=self.field_type,
        )

    def calculate_excess(self, coords, h, field_specifier, scaled_loads):
        self.initialize_ghe(coords, h, field_specifier, scaled_loads)
        max_hp_eft, min_hp_eft = self.ghe.simulate(method=self.method)
        t_excess = self.ghe.cost(max_hp_eft, min_hp_eft, self.max_eft, self.min_eft)
        self.searchTracker.append([field_specifier, t_excess, max_hp_eft, min_hp_eft])
        return t_excess

    def search(self):
        # bracket on pad‐count index
        x_l = self.ndp_min
        x_r = self.ndp_max

        base_loads = np.array(self.hourly_extraction_ground_loads, dtype=float) * pi
        scaled_loads_l = (1 / x_l) * base_loads
        scaled_loads_r = (1 / x_r) * base_loads


        # evaluate at smallest pad count

        self.initialize_g_function(self.coordinates_domain[0][0], self.max_height , self.coordinates_domain[0][1], self.coordinates_domain[0][2])
        t_l = self.calculate_excess(self.coordinates_domain[0][0], self.max_height, self.fieldDescriptors[x_l-1], scaled_loads_l)
        t_r = self.calculate_excess(self.coordinates_domain[0][0], self.max_height, self.fieldDescriptors[x_r-1], scaled_loads_r)

        self.calculated_temperatures[x_l-1] = t_l
        self.calculated_temperatures[x_r-1] = t_r

        # check for valid bracket
        if check_bracket(sign(t_l), sign(t_r)):
            if t_r > 0:  # undersize even at max pads
                if self.continue_if_design_unmet:
                    return x_r, self.coordinates_domain
                raise ValueError("Search failed: not enough pads available.")

        # bisection on index
        i = 0
        valid = x_r - x_l
        last_valid = None
        prev_x_c = 0
        while i < self.max_iter and valid > 0:
            x_c = ceil((x_l + x_r) / 2)
            if x_c == prev_x_c:
                break
            prev_x_c = x_c
            scaled_loads_c = (1 / x_c) * base_loads
            t_c = self.calculate_excess(self.coordinates_domain[0][0], self.max_height, self.fieldDescriptors[x_c-1], scaled_loads_c)
            self.calculated_temperatures[x_c-1] = t_c
            if t_c < 0:
                last_valid = x_c
                x_r, t_r = x_c, t_c
            elif t_c >= 0:
                x_l, t_l = x_c, t_c
            i += 1
            valid = x_r - x_l
        if last_valid is not None:
            selection_key = last_valid
        else:
            if self.continue_if_design_unmet:
                selection_key = x_r
            else:
                raise ValueError("Search failed: not enough pads available.")

        return selection_key, self.coordinates_domain[0]
