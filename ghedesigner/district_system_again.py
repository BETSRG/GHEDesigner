import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pygfunction.boreholes import Borehole

from ghedesigner.enums import BHType
from ghedesigner.ghe.boreholes.factory import get_bhe_object
from ghedesigner.ghe.gfunction import GFunction, calc_g_func_for_multiple_lengths
from ghedesigner.ghe.pipe import Pipe as MediaPipe
from ghedesigner.media import Grout, Soil, Fluid
from ghedesigner.utilities import combine_sts_lts
from ghedesigner.utilities import eskilson_log_times

start = time.time()


class GHX:
    def __init__(self):
        self.ID = None
        self.type = "GHX"
        self.nodeID = None
        self.input = None
        self.n_rows = None
        self.n_cols = None
        self.row_spacing = None
        self.col_spacing = None
        self.nbh = None
        self.height = None
        self.upstream_device = None
        self.downstream_device = None
        self.matrix_line = None
        self.height = None
        self.row_index = None

        # Parameters to be assigned later
        self.m_dot_total = None

        # Thermal object references (to be set during setup)
        self.pipe = Pipe
        self.soil = Soil
        self.grout = Grout
        self.borehole = Borehole
        self.g_function: GFunction
        self.fluid = None
        self.bhe_type = BHType.SINGLEUTUBE
        self.split_ratio = None

        # Computed properties
        self.bhe = None
        self.r_b = None
        self.gFunction = None
        self.mass_flow_ghe = None
        self.mass_flow_ghe_borehole = None
        self.depth = None

        self.mass_flow_ghe_design = None
        self.mass_flow_ghe_borehole_design = None
        self.H_n_ghe = None
        self.total_values_ghe = None

        # for output
        self.t_eft = None
        self.t_mft = None
        self.q_ghe = None
        self.t_exft = None
        self.t_bw = None
        self.t_combining_node = None
        self.sim_params = None

        # for initializing gFunction object
        self.bore_locations = None
        self.log_time = None

        # All dummy values are used, because the goal is only to generate
        # self.gFunction as an object of class GFunction so that I can
        # initialize self.gFunction in "initialize_gFunction_object" method
        self.gFunction = GFunction(b=0.0, d=0.0, r_b_values={}, g_lts={}, log_time=[], bore_locations=[])

    def initialize_gFunction_object(self):
        self.gFunction.bore_locations = [(i * self.row_spacing, j * self.row_spacing) for i in range(int(self.n_rows))
                                         for j in range(int(self.n_cols))]
        self.gFunction.log_time = eskilson_log_times()

    def compute_g_functions(self):
        # Compute g-functions for a bracketed solution, based on min and max
        # height
        min_height = 100  # TODO: fix this
        max_height = 100
        avg_height = (min_height + max_height) / 2.0
        h_values = [min_height, avg_height, max_height]

        coordinates = self.gFunction.bore_locations
        log_time = self.gFunction.log_time

        g_function = calc_g_func_for_multiple_lengths(
            self.row_spacing,
            h_values,
            0.07,  # TODO: fix this, borehole radius
            2,  # TODO: fix this, depth of top of borehole
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

    def grab_g_function(self):
        """
        Interpolates g-function values using self.gFunction and self.bhe,
        and returns g and g_bhw arrays.
        """

        # Interpolate LTS g-function
        g_function, rb_value, _, _ = self.gFunction.g_function_interpolation(self.row_spacing / self.height)

        # Correct the g-function for borehole radius
        g_function_corrected = self.gFunction.borehole_radius_correction(
            g_function,
            rb_value,
            0.07  # TODO: Fix this, borehole radius
        )

        # Combine STS and LTS g-functions
        g = combine_sts_lts(
            self.gFunction.log_time,
            g_function_corrected,
            self.bhe.lntts.tolist(),
            self.bhe.g.tolist(),
        )

        g_bhw = combine_sts_lts(
            self.gFunction.log_time,
            g_function_corrected,
            self.bhe.lntts.tolist(),
            self.bhe.g_bhw.tolist(),
        )
        return g, g_bhw

    def calculation_of_ghe_constant_c_n(self, g, ts, time_array, n_timesteps, bhe_effective_resist):
        """
        Calculate C_n values for three GHEs based on their g-functions.

        Cn = 1 / (2 * pi * K_s) * g((tn - tn-1) / t_s) + R_b
        """

        two_pi_k = 2 * np.pi * self.soil.k
        c_n = np.zeros(n_timesteps, dtype=float)

        for i in range(1, n_timesteps):
            delta_log_time = np.log((time_array[i] - time_array[i - 1]) / (ts / 3600.0))
            g_val = g(delta_log_time)
            c_n[i] = (1 / two_pi_k * g_val) + bhe_effective_resist

        return c_n

    def compute_history_term(self, i, time_array, ts, two_pi_k, g, tg, H_n_ghe, total_values_ghe, q_ghe):
        """
        Computes the history term H_n for this GHX at time index `i`.
        Updates self.total_values_ghe and self.H_n_ghe in place.
        """
        if i == 0:
            H_n_ghe[i] = tg
            total_values_ghe[i] = 0
            return

        time_n = time_array[i]

        # Compute dimensionless time for all indices from 1 to i-1
        indices = np.arange(1, i)
        dim_less_time = np.log((time_n - time_array[indices - 1]) / (ts / 3600.0))

        # Compute contributions from all previous steps
        delta_q_ghe = (q_ghe[indices] - q_ghe[indices - 1]) / two_pi_k
        values = np.sum(delta_q_ghe * g(dim_less_time))

        total_values_ghe[i] = values

        # Contribution from the last time step only
        dim1_less_time = np.log((time_n - time_array[i - 1]) / (ts / 3600.0))

        H_n_ghe[i] = tg + total_values_ghe[i] - (
                q_ghe[i - 1] / two_pi_k * g(dim1_less_time)
        )
        return H_n_ghe[i]

    def generate_GHX_matrix_row(self, matrix_size, c_n, i, GHX_inlet_index, mass_flow_ghe, cp, m_loop_ghe, H_n_ghe,
                                m_loop, configuration):
        row1 = np.zeros(matrix_size)
        row2 = np.zeros(matrix_size)
        row3 = np.zeros(matrix_size)
        row4 = np.zeros(matrix_size)

        row_index = self.row_index
        neighbour_index = self.downstream_device.row_index

        if configuration == "1-pipe":
            row1[row_index] = (m_loop - mass_flow_ghe) * cp
            row1[row_index + 3] = mass_flow_ghe * cp
            row1[neighbour_index] = - m_loop * cp

            row2[row_index + 1] = 1
            row2[row_index + 2] = -c_n[i]

            row3[row_index] = -1
            row3[row_index + 1] = 2
            row3[row_index + 3] = -1

            row4[row_index] = mass_flow_ghe * cp
            row4[row_index + 2] = -self.height * (self.n_rows * self.n_cols)
            row4[row_index + 3] = - mass_flow_ghe * cp

            rhs1, rhs2, rhs3, rhs4 = 0, H_n_ghe, 0, 0

        elif configuration == "2-pipe":
            row1[row_index + 1] = 1
            row1[row_index + 2] = -c_n[i]

            row2[row_index + 1] = 2
            row2[GHX_inlet_index] = -1
            row2[row_index + 3] = -1

            row3[GHX_inlet_index] = mass_flow_ghe * cp
            row3[row_index + 3] = -mass_flow_ghe * cp
            row3[row_index + 2] = -self.height * (self.n_rows * self.n_cols)

            row4[row_index] = (m_loop_ghe - mass_flow_ghe) * cp
            row4[row_index + 3] = mass_flow_ghe * cp
            row4[neighbour_index] = -m_loop_ghe * cp

            rhs1, rhs2, rhs3, rhs4 = H_n_ghe, 0, 0, 0

        else:
            raise ValueError(f"Invalid configuration type: {configuration}")

        rows = [row1, row2, row3, row4]
        rhs = [rhs1, rhs2, rhs3, rhs4]

        return rows, rhs


class Building:
    def __init__(self):
        self.name = None
        self.ID = None
        self.zoneIDs = []  # list of zone ids
        self.zones = []  # list of zones


class Zone:
    def __init__(self):
        # values read from the file
        self.name = None
        self.ID = None
        self.type = "zone"
        self.connection_downstream = None
        self.nodeID = None
        self.node = None
        self.HPmodel = None
        self.HP = None
        self.loads_file = None
        self.matrix_line = None
        self.row_index = None
        self.index = None
        self.mass_flow_zone = None
        self.df_zone = None
        self.upstream_device = None
        self.downstream_device = None

        self.P_zone_htg = None
        self.P_zone_clg = None
        self.P_zone_cp = None

        self.t_eft = None
        self.t_exft = None
        self.t_combining_node = None

        self.h = None
        self.c = None

    def q_net_clg(self):
        """
        Calculate net heat extracted/rejected each hour for the zone.
        If either column is missing, default to zeros.
        """
        if "HPHtgLd_W" in self.df_zone.columns:
            self.h = np.array(self.df_zone["HPHtgLd_W"])
        else:
            self.h = np.zeros(len(self.df_zone))

        if "HPClgLd_W" in self.df_zone.columns:
            self.c = np.array(self.df_zone["HPClgLd_W"])
        else:
            self.c = np.zeros(len(self.df_zone))

        return self.c - self.h

    def zone_mass_flow_rate(self, t_eft, i):
        hp = self.HP
        cap_htg = hp.c1_htg * t_eft ** 2 + hp.c2_htg * t_eft + hp.c3_htg
        cap_clg = hp.c1_clg * t_eft ** 2 + hp.c2_clg * t_eft + hp.c3_clg

        if cap_clg == 0:
            rtf = abs(self.h[i] / cap_htg)
        else:
            rtf = abs(self.h[i] / cap_htg) + abs(self.c[i] / cap_clg)

        m_single_hp = hp.m_single_hp
        self.mass_flow_zone = rtf * m_single_hp

        return self.mass_flow_zone

    def calculate_r1_r2(self, t_eft, hour_index):
        """
        Calculate r1 and r2 for this zone based on entering fluid temperature and HP coefficients.
        """

        # Extract loads
        h = self.df_zone["HPHtgLd_W"].iloc[hour_index] if "HPHtgLd_W" in self.df_zone.columns else 0.0
        c = self.df_zone["HPClgLd_W"].iloc[hour_index] if "HPClgLd_W" in self.df_zone.columns else 0.0

        # Extract HP coefficients
        a_htg = self.HP.a_htg
        b_htg = self.HP.b_htg
        c_htg = self.HP.c_htg

        a_clg = self.HP.a_clg
        b_clg = self.HP.b_clg
        c_clg = self.HP.c_clg

        # Heating calculations
        slope_htg = 2 * a_htg * t_eft + b_htg
        ratio_htg = a_htg * t_eft ** 2 + b_htg * t_eft + c_htg
        u = ratio_htg - slope_htg * t_eft
        v = slope_htg

        # Cooling calculations
        slope_clg = 2 * a_clg * t_eft + b_clg
        ratio_clg = a_clg * t_eft ** 2 + b_clg * t_eft + c_clg
        a = ratio_clg - slope_clg * t_eft
        b = slope_clg

        # Final arrays
        r1 = b * c - v * h
        r2 = a * c - u * h

        return r1, r2

    def generate_zone_matrix_row(self, matrix_size, zone_inlet_index, r1, mass_flow_zone, cp, m_loop_zone, m_loop, r2,
                                 configuration):
        row1 = np.zeros(matrix_size)
        row2 = np.zeros(matrix_size)

        row_index = self.row_index
        neighbour_index = self.downstream_device.row_index

        if configuration == "1-pipe":
            row = np.zeros(matrix_size)
            row[self.row_index] = 1 + r1 / (m_loop * cp)
            if self.downstream_device.type in ("zone", "GHX"):
                row[neighbour_index] = -1
            else:
                row[neighbour_index + 2] = -1
            rhs = -r2 / (m_loop * cp)

            rows = [row]
            rhs_list = [rhs]

        elif configuration == "2-pipe":

            if mass_flow_zone == 0:
                row1[zone_inlet_index] = 1
                row1[row_index + 1] = -1
            else:
                row1[zone_inlet_index] = r1 + mass_flow_zone * cp
                row1[row_index + 1] = -mass_flow_zone * cp

            row2[row_index] = (m_loop_zone - mass_flow_zone) * cp
            row2[row_index + 1] = mass_flow_zone * cp
            row2[neighbour_index] = - m_loop_zone * cp

            rhs1, rhs2 = -r2, 0

            rows = [row1, row2]
            rhs_list = [rhs1, rhs2]

        else:
            raise ValueError(f"Invalid configuration type: {configuration}")

        return rows, rhs_list

    def zone_energy_consumption(self, t_eft, i, m_flow_zone, density, cp_efficiency, beta_HP_delta_P, delta_P_HP):

        # Extract loads
        htg_load = self.df_zone["HPHtgLd_W"].iloc[i] if "HPHtgLd_W" in self.df_zone.columns else 0.0
        clg_load = self.df_zone["HPClgLd_W"].iloc[i] if "HPClgLd_W" in self.df_zone.columns else 0.0

        # Extract HP coefficients
        a_htg = self.HP.a_htg
        b_htg = self.HP.b_htg
        c_htg = self.HP.c_htg

        a_clg = self.HP.a_clg
        b_clg = self.HP.b_clg
        c_clg = self.HP.c_clg

        ratio_htg = a_htg * t_eft ** 2 + b_htg * t_eft + c_htg
        ratio_clg = a_clg * t_eft ** 2 + b_clg * t_eft + c_clg

        # zone (HP) power consumed
        Power_zone_htg = htg_load * (1 - ratio_htg)
        Power_zone_clg = clg_load * (ratio_clg - 1)

        # power consumed by circulating pump
        Power_zone_cp = m_flow_zone / (density * cp_efficiency) * beta_HP_delta_P * delta_P_HP

        return Power_zone_htg, Power_zone_clg, Power_zone_cp


class Node:
    def __init__(self):
        self.ID = None
        self.type = None
        self.x = None
        self.y = None
        self.z = None
        self.input = None
        self.output = None
        self.diversion = None
        self.merger = None


class Pipe:
    def __init__(self):
        self.ID = None
        self.node_in_name = None
        self.node_out_name = None
        self.input = None
        self.output = None
        self.length = None
        self.type = None


class HPmodel:
    def __init__(self):
        self.name = None
        self.ID = None
        self.a_htg, self.b_htg, self.c_htg = None, None, None
        self.a_clg, self.b_clg, self.c_clg = None, None, None
        self.c1_htg, self.c2_htg, self.c3_htg = None, None, None
        self.c1_clg, self.c2_clg, self.c3_clg = None, None, None
        self.m_single_hp = None
        self.design_htg_cap = None
        self.design_clg_cap = None
        self.delta_P_HP = None


class IsolationHX:
    def __init__(self):
        self.name = None
        self.type = "ISHX"
        self.ID = None
        self.node_network_inlet_ID = None
        self.node_HP_inlet_ID = None
        self.node_HP_outlet_ID = None
        self.beta_ISHX = None
        self.input = None
        self.HP_output = None
        self.HP_input = None
        self.upstream_device = None
        self.downstream_device = None
        self.upstream_device_HP = None
        self.downstream_device_HP = None
        self.row_index = None

        self.zoneIDs = []  # list of zone ids
        self.zones = []  # list of zones

        self.m_loop_n = None
        self.m_loop_hp = None

    def generate_ISHX_matrix_row(self, matrix_size, C_n, C_hp, effec, C_min, m_loop, cp, m_loop_n):
        row1 = np.zeros(matrix_size)
        row2 = np.zeros(matrix_size)
        row3 = np.zeros(matrix_size)

        row_index = self.row_index
        neighbour_index_loop_side = self.downstream_device.row_index
        neighbour_index_HP_side = self.downstream_device_HP.row_index

        row1[row_index] = effec * C_min - C_n
        row1[row_index + 1] = C_n
        row1[row_index + 2] = -effec * C_min

        row2[row_index] = -(effec * C_min)
        row2[row_index + 2] = effec * C_min - C_hp
        row2[neighbour_index_HP_side] = C_hp

        row3[row_index] = (m_loop - m_loop_n) * cp
        row3[row_index + 1] = m_loop_n * cp
        row3[neighbour_index_loop_side] = - (m_loop * cp)

        rhs1, rhs2, rhs3 = 0, 0, 0

        rows = [row1, row2, row3]
        rhs = [rhs1, rhs2, rhs3]

        return rows, rhs


class GHEHPSystem:
    def __init__(self):
        self.title = None
        self.configuration = None
        self.GHXs = []
        self.buildings = []
        self.zones = []
        self.nodes = []
        self.pipes = []
        self.HPmodels = []
        self.ISHXs = []
        self.current_row = 0
        self.m_loop = None
        self.bhe = None
        self.g_value = {}
        self.c_n = {}
        self.time_array = None
        self.time_array_size = None

        # Thermal object references (to be set during setup)
        self.pipe = None
        self.soil = None
        self.grout = None
        self.borehole = None
        self.fluid = None
        self.mass_flow_ghe_borehole = None
        self.nbh_total = None
        self.gFunction = None
        self.g = None
        self.g_bhw = None
        self.log_time = None
        self.mass_flow_ghe = None
        self.bhe_eq = None
        self.c_n = None
        self.m_loop = None
        self.beta_CL_flow = None
        self.beta_ISHX_loop = None
        self.beta_cl_cp_delta_P = None

        self.df = None
        self.df1 = None
        self.current_frame = 0
        self.data = None

        # for energy consumption calculations
        self.HP_cp_efficiency = None
        self.ISHX_cp_efficiency = None
        self.GHE_cp_efficiency = None
        self.beta_HP_delta_P = None
        self.P_cl_cp = None
        self.CL_P_per_m = None

    def read_GHEHPSystem_data(self, input_file_path: Path):

        data = input_file_path.read_text().split("\n")  # read the entire file as a list of strings
        in_file_dir = input_file_path.parent.resolve()

        next_matrix_line = 0
        for line in data:  # loop over all the lines
            cells = [c.strip() for c in line.strip().split(',')]
            keyword = cells[0].lower()

            if keyword == "configuration":
                self.configuration = cells[1].replace("'", "")

            if keyword == 'title':
                self.title = cells[1].replace("'", "")

            if keyword == 'ghx':
                thisghx = GHX()
                thisghx.ID = str(cells[1])
                thisghx.inlet_nodeID = str(cells[2])
                thisghx.outlet_nodeID = str(cells[3])
                thisghx.n_rows = float(cells[4])
                thisghx.n_cols = float(cells[5])
                thisghx.row_spacing = float(cells[6])
                thisghx.col_spacing = float(cells[7])
                thisghx.ghe_height = float(cells[8])
                thisghx.mass_flow_ghe_design = float(cells[9])
                thisghx.matrix_line = next_matrix_line
                next_matrix_line += 4
                self.GHXs.append(thisghx)

            if keyword == 'building':
                thisbuilding = Building()
                thisbuilding.name = str(cells[1])
                thisbuilding.ID = str(cells[2])
                thisbuilding.zoneIDs = ([zones.strip() for zones in cells[3:]])
                self.buildings.append(thisbuilding)

            if keyword == 'zone':
                df = pd.read_csv(in_file_dir / cells[7])
                self.time_array = df['Hours'].values
                self.time_array_size = len(self.time_array)

                thiszone = Zone()
                thiszone.name = str(cells[1])
                thiszone.ISHX_ID = str(cells[2])
                thiszone.ID = str(cells[3])
                thiszone.inlet_nodeID = str(cells[4])
                thiszone.outlet_nodeID = str(cells[5])
                thiszone.HPmodel = str(cells[6])
                thiszone.loads_file = pd.read_csv(in_file_dir / cells[7])
                thiszone.matrix_line = next_matrix_line
                next_matrix_line += 1
                self.zones.append(thiszone)

            if keyword == 'ishx':
                thisishx = IsolationHX()
                thisishx.name = str(cells[1])
                thisishx.ID = str(cells[2])
                thisishx.node_network_inlet_ID = str(cells[3])
                thisishx.node_HP_inlet_ID = str(cells[4])
                thisishx.node_HP_outlet_ID = str(cells[5])
                thisishx.beta_ISHX = float(cells[6])
                thisishx.effectiveness = float(cells[7])
                thisishx.zoneIDs = ([zones.strip() for zones in cells[8:]])
                self.ISHXs.append(thisishx)

            if keyword == 'node':
                thisnode = Node()
                thisnode.ID = str(cells[1])
                thisnode.type = str(cells[2])
                thisnode.x = float(cells[3])
                thisnode.y = float(cells[4])
                thisnode.z = float(cells[5])
                self.nodes.append(thisnode)

            if keyword == 'pipe':
                thispipe = Pipe()
                thispipe.ID = str(cells[1])
                thispipe.type = str(cells[2])
                thispipe.node_in_name = str(cells[3])
                thispipe.node_out_name = str(cells[4])
                thispipe.length = float(cells[5])
                self.pipes.append(thispipe)

            if keyword == 'hpmodel':
                thishpmodel = HPmodel()
                thishpmodel.name = str(cells[1])
                thishpmodel.ID = str(cells[2])
                thishpmodel.a_htg, thishpmodel.b_htg, thishpmodel.c_htg = (float(cells[3]), float(cells[4]),
                                                                           float(cells[5]))
                thishpmodel.a_clg, thishpmodel.b_clg, thishpmodel.c_clg = (float(cells[6]), float(cells[7]),
                                                                           float(cells[8]))
                thishpmodel.c1_htg, thishpmodel.c2_htg, thishpmodel.c3_htg = (float(cells[9]), float(cells[10]),
                                                                              float(cells[11]))
                thishpmodel.c1_clg, thishpmodel.c2_clg, thishpmodel.c3_clg = (float(cells[12]), float(cells[13]),
                                                                              float(cells[14]))
                thishpmodel.m_single_hp = float(cells[15])
                thishpmodel.delta_P_HP = float(cells[16])
                self.HPmodels.append(thishpmodel)

            if keyword == "pressure_drop":
                self.CL_P_per_m = float(cells[1])
                self.delta_P_ref_ISHX = float(cells[2])

            if keyword == "beta":
                self.beta_CL_flow = float(cells[1])
                self.beta_ISHX_HP_flow = float(cells[2])
                self.beta_ISHX_N_flow = float(cells[3])
                self.beta_HP_delta_P = float(cells[4])
                self.beta_GHE_delta_P = float(cells[5])
                self.beta_ISHX_delta_P = float(cells[6])

            if keyword == "efficiency":
                self.HP_cp_efficiency = float(cells[1])
                self.ISHX_cp_efficiency = float(cells[2])
                self.GHE_cp_efficiency = float(cells[3])
                self.CL_efficiency = float(cells[4])

            if keyword == "length":
                self.length_CL = float(cells[1])

        # end for line
        self.UpdateConnections()

    def read_data_from_json_file(self, json_data: dict):

        self.data = json_data

        # Extract input values
        fluid_data = json_data["fluid"]
        soil_data = json_data["ground_heat_exchanger"]["ghe1"]["soil"]
        grout_data = json_data["ground_heat_exchanger"]["ghe1"]["grout"]
        pipe_data = json_data["ground_heat_exchanger"]["ghe1"]["pipe"]
        borehole_data = json_data["ground_heat_exchanger"]["ghe1"]["borehole"]
        geometric_data = json_data["ground_heat_exchanger"]["ghe1"]["geometric_constraints"]
        design_data = json_data["ground_heat_exchanger"]["ghe1"]["design"]

        # Construct objects
        fluid = (
            Fluid(
                fluid_data["fluid_name"],
                fluid_data["concentration_percent"],
                fluid_data["temperature"]
            ))

        # Pipe object (Single U-tube)
        pipe = MediaPipe.init_single_u_tube(
            pipe_data["conductivity"],
            pipe_data["rho_cp"],
            pipe_data["inner_diameter"],
            pipe_data["outer_diameter"],
            pipe_data["shank_spacing"],
            pipe_data["roughness"],
        )

        soil = Soil(soil_data["conductivity"], soil_data["rho_cp"], soil_data["undisturbed_temp"])
        grout = Grout(grout_data["conductivity"], grout_data["rho_cp"])
        borehole = Borehole(100.0, borehole_data["buried_depth"], borehole_data["diameter"] / 2.0, 0.0, 0.0)

        # Simulation parameters
        # self.sim_params = SimulationParameters(num_months=12)
        # self.sim_params.set_design_heights(geometric_data["max_height"], geometric_data["min_height"])
        # self.sim_params.set_design_temps(design_data["max_eft"], design_data["min_eft"])

        return fluid, pipe, grout, soil, borehole

    def solveSystem(self, fluid, pipe, grout, soil, borehole):
        # precompute all time invariant constants

        time_array = self.time_array
        n_timesteps = self.time_array_size
        configuration = self.configuration

        if configuration == "1-pipe":
            matrix_size = len(self.zones) + 4 * len(self.GHXs) + 3 * len(self.ISHXs)
        elif configuration == "2-pipe":
            matrix_size = 2 * len(self.zones) + 4 * len(self.GHXs) + 3 * len(self.ISHXs)
        else:
            raise ValueError(f"Invalid configuration type: {configuration}")

        for GHX in self.GHXs:
            GHX.fluid = fluid
            GHX.pipe = pipe
            GHX.grout = grout
            GHX.soil = soil
            GHX.borehole = borehole
            GHX.initialize_gFunction_object()

        # for getting g_functions and bhe object
        for GHX in self.GHXs:
            GHX.borehole = borehole
            GHX.height = GHX.borehole.H
            GHX.nbh = len(GHX.gFunction.bore_locations)
            GHX.mass_flow_ghe_borehole_design = GHX.mass_flow_ghe_design / GHX.nbh
            GHX.bhe = get_bhe_object(GHX.bhe_type, GHX.mass_flow_ghe_borehole_design, GHX.fluid, GHX.borehole,
                                     GHX.pipe, GHX.grout, GHX.soil)
            GHX.bhe_eq = GHX.bhe.to_single()
            GHX.bhe_eq.calc_sts_g_functions()
            log_time = eskilson_log_times()
            self.gFunction = GHX.compute_g_functions()
            ts = GHX.bhe_eq.t_s
            self.log_time = eskilson_log_times()
            cp = 3857.004010495233  # TODO: fix this
            tg = GHX.bhe.soil.ugt
            borehole.H = GHX.height

            self.g, _ = GHX.grab_g_function()
            # self.bhe_effective_resist = GHX.bhe.calc_effective_borehole_resistance()
            self.bhe_effective_resist = 0.15690883427464597  # TODO: Fix this
            GHX.c_n = GHX.calculation_of_ghe_constant_c_n(self.g, ts, time_array, n_timesteps,
                                                          self.bhe_effective_resist)

        # Initializing the values
        for GHX in self.GHXs:
            GHX.H_n_ghe, GHX.total_values_ghe, GHX.q_ghe = np.full(n_timesteps, tg), np.zeros(
                n_timesteps), np.zeros(n_timesteps)

        # Initializing t_eft, t_mean, q_ghe, t_exit
        for zone in self.zones:
            zone.t_eft = np.full(n_timesteps, tg)
            zone.t_exft = np.full(n_timesteps, tg)
            zone.t_combining_node = np.full(n_timesteps, tg)

        for GHX in self.GHXs:
            GHX.t_eft = np.full(n_timesteps, tg)
            GHX.t_mft = np.full(n_timesteps, tg)
            GHX.t_bhw = np.full(n_timesteps, tg)
            GHX.q_ghe = np.zeros(n_timesteps)
            GHX.t_exft = np.full(n_timesteps, tg)
            GHX.t_combining_node = np.full(n_timesteps, tg)

        for ISHX in self.ISHXs:
            ISHX.t_n_eft = np.full(n_timesteps, tg)
            ISHX.t_n_exft = np.full(n_timesteps, tg)
            ISHX.t_hp_eft = np.full(n_timesteps, tg)

        # Assigning row_indices
        if configuration == "1-pipe":
            for k, zone in enumerate(self.zones):
                zone.row_index = k
            for k, GHX in enumerate(self.GHXs):
                GHX.row_index = len(self.zones) + k * 4
            for k, ISHX in enumerate(self.ISHXs):
                ISHX.row_index = len(self.zones) + len(self.GHXs) * 4

        elif configuration == "2-pipe":
            for k, zone in enumerate(self.zones):
                zone.row_index = k * 2
            for k, GHX in enumerate(self.GHXs):
                GHX.row_index = 2 * len(self.zones) + k * 4
            for k, ISHX in enumerate(self.ISHXs):
                ISHX.row_index = len(self.zones) + len(self.GHXs) * 4

        else:
            raise ValueError(f"Invalid configuration type: {configuration}")

        # Initializing
        m_loop_array = np.zeros(n_timesteps)
        for ISHX in self.ISHXs:
            ISHX.m_loop_ISHX_array = np.zeros(n_timesteps)

        for zone in self.zones:
            zone.P_zone_htg = np.zeros(n_timesteps)
            zone.P_zone_clg = np.zeros(n_timesteps)
            zone.P_zone_cp = np.zeros(n_timesteps)
        self.P_cl_cp = np.zeros(n_timesteps)
        for GHX in self.GHXs:
            GHX.P_ghe_cp = np.zeros(n_timesteps)
        for ISHX in self.ISHXs:
            ISHX.P_ishx_cp = np.zeros(n_timesteps)

        for i in range(1, n_timesteps):  # loop over all timestep
            matrix_rows = []
            matrix_rhs = []
            # Calculating total hp flows in each ISHX

            for ISHX in self.ISHXs:
                total_hp_flow_ISHX = 0
                for zone in self.zones:
                    if zone in ISHX.zones:
                        t_eft = zone.t_eft[i - 1]
                        zone.df_zone = zone.loads_file
                        q_net_clg = zone.q_net_clg()
                        m_zone = zone.zone_mass_flow_rate(t_eft, i)
                        total_hp_flow_ISHX += m_zone
                        ISHX.m_loop_hp = total_hp_flow_ISHX * self.beta_ISHX_HP_flow

            # Calculating total hp flows in heat pumps connected directly to loop
            total_hp_flow = 0
            for zone in self.zones:
                if zone.ISHX_ID == "None":
                    t_eft = zone.t_eft[i - 1]
                    zone.df_zone = zone.loads_file
                    q_net_clg = zone.q_net_clg()
                    m_zone = zone.zone_mass_flow_rate(t_eft, i)
                    total_hp_flow += m_zone

            total_m_loop_n = 0
            for ISHX in self.ISHXs:
                ISHX.m_loop_n = ISHX.m_loop_hp * self.beta_ISHX_N_flow
                total_m_loop_n += ISHX.m_loop_n
                ISHX.m_loop_ISHX_array[i] = total_m_loop_n

            m_loop = (total_m_loop_n + total_hp_flow) * self.beta_CL_flow
            m_loop_array[i] = m_loop

            zone_inlet_index = self.zones[0].row_index

            # Generating matrix for zones connected to ISHX
            m_loop_zone = 0
            for ISHX in self.ISHXs:
                for zone in self.zones:
                    if zone in ISHX.zones:
                        m_loop = ISHX.m_loop_hp
                        t_eft = zone.t_eft[i - 1]
                        r1, r2 = zone.calculate_r1_r2(t_eft, i)
                        q_net_clg = zone.q_net_clg()
                        mass_flow_zone = zone.zone_mass_flow_rate(t_eft, i)
                        m_loop_zone += mass_flow_zone
                        this_zone_row, rhs = zone.generate_zone_matrix_row(matrix_size, zone_inlet_index, r1,
                                                                           mass_flow_zone, cp, m_loop_zone, m_loop,
                                                                           r2, configuration)
                        for row, rhs in zip(this_zone_row, rhs):
                            matrix_rows.append(row)
                            matrix_rhs.append(rhs)

            # Generating matrix for zones not connected to ISHXs
            m_loop_zone = 0
            zone_inlet_index = self.zones[0].row_index
            for zone in self.zones:
                if zone.ISHX_ID == "None":
                    t_eft = zone.t_eft[i - 1]
                    r1, r2 = zone.calculate_r1_r2(t_eft, i)
                    mass_flow_zone = zone.zone_mass_flow_rate(t_eft, i)
                    m_loop = (total_m_loop_n + total_hp_flow) * self.beta_CL_flow
                    m_loop_zone += mass_flow_zone
                    this_zone_row, rhs = zone.generate_zone_matrix_row(matrix_size, zone_inlet_index, r1,
                                                                       mass_flow_zone, cp, m_loop_zone, m_loop, r2,
                                                                       configuration)

                    for row, rhs in zip(this_zone_row, rhs):
                        matrix_rows.append(row)
                        matrix_rhs.append(rhs)

            # Generating matrix for ground heat exchangers
            m_loop_ghe = 0
            nbh_total = sum(GHX.nbh for GHX in self.GHXs)
            GHX_inlet_index = self.GHXs[0].row_index
            for j, GHX in enumerate(self.GHXs):
                q_ghe = GHX.q_ghe[:i]  # <--- FIXED: slice of all past values, it is an array
                two_pi_k = 2 * np.pi * GHX.soil.k
                GHX.nbh = len(GHX.gFunction.bore_locations)
                split_ratio = GHX.nbh / nbh_total
                mass_flow_ghe = m_loop * split_ratio
                c_n = GHX.c_n  # this is array, while using this in matrix we pick c_n[i], a single float number
                H_n_ghe = GHX.compute_history_term(i, time_array, ts, two_pi_k, self.g, tg, GHX.H_n_ghe,
                                                   GHX.total_values_ghe, q_ghe)
                m_loop_ghe += mass_flow_ghe

                rows, rhs_values = GHX.generate_GHX_matrix_row(matrix_size, c_n, i, GHX_inlet_index, mass_flow_ghe, cp,
                                                               m_loop_ghe, H_n_ghe, m_loop, configuration)

                for row, rhs in zip(rows, rhs_values):
                    matrix_rows.append(row)
                    matrix_rhs.append(rhs)

            # Generating matrix for isolation heat exchanger
            for ISHX in self.ISHXs:
                effec = ISHX.effectiveness
                C_n = ISHX.m_loop_n * cp
                C_hp = ISHX.m_loop_hp * cp
                C_min = min(C_n, C_hp)
                m_loop_n = ISHX.m_loop_n
                m_loop = (total_m_loop_n + total_hp_flow) * self.beta_CL_flow
                rows, rhs_values = ISHX.generate_ISHX_matrix_row(matrix_size, C_n, C_hp, effec, C_min, m_loop, cp,
                                                                 m_loop_n)
                for row, rhs in zip(rows, rhs_values):
                    matrix_rows.append(row)
                    matrix_rhs.append(rhs)

            # Solve the matrix
            A = np.array(matrix_rows, dtype=float)
            B = np.array(matrix_rhs, dtype=float)

            X = np.linalg.solve(A, B)

            # for getting values for one-pipe system
            if self.configuration == "1-pipe":
                for zone in self.zones:
                    zone.t_eft[i] = X[zone.row_index]

                for GHX in self.GHXs:
                    GHX.t_eft[i] = X[GHX.row_index]
                    GHX.t_mft[i] = X[GHX.row_index + 1]
                    GHX.q_ghe[i] = X[GHX.row_index + 2]
                    GHX.t_exft[i] = X[GHX.row_index + 3]

                for ISHX in self.ISHXs:
                    ISHX.t_n_eft[i] = X[ISHX.row_index]
                    ISHX.t_n_exft[i] = X[ISHX.row_index + 1]
                    ISHX.t_hp_eft[i] = X[ISHX.row_index + 2]

            # for getting values for 2-pipe system
            if self.configuration == "2-pipe":
                for zone in self.zones:
                    zone.t_eft[i] = X[zone_inlet_index]
                    zone.t_exft[i] = X[zone.row_index + 1]
                    zone.t_combining_node[i] = X[zone.downstream_device.row_index]

                for GHX in self.GHXs:
                    GHX.t_eft[i] = X[GHX_inlet_index]
                    GHX.t_mft[i] = X[GHX.row_index + 1]
                    GHX.q_ghe[i] = X[GHX.row_index + 2]
                    GHX.t_exft[i] = X[GHX.row_index + 3]
                    GHX.t_combining_node[i] = X[GHX.downstream_device.row_index]

            # zone energy consumption
            for zone in self.zones:
                q_net_clg = zone.q_net_clg()
                t_eft = zone.t_eft[i - 1]
                m_flow_zone = zone.zone_mass_flow_rate(t_eft, i)
                cp_efficiency = self.HP_cp_efficiency
                delta_P_HP = zone.HP.delta_P_HP
                density = 1023.7849656966806  # TODO: fix this
                zone.P_zone_htg[i], zone.P_zone_clg[i], zone.P_zone_cp[i] = zone.zone_energy_consumption(t_eft, i,
                                                                                                         m_flow_zone,
                                                                                                         density,
                                                                                                         cp_efficiency,
                                                                                                         self.beta_HP_delta_P,
                                                                                                         delta_P_HP)

            # ground heat exchanger energy consumption
            for GHX in self.GHXs:
                nbh = len(GHX.gFunction.bore_locations)
                length_ghe = 2 * GHX.height
                split_ratio = nbh / nbh_total
                mass_flow_ghe = m_loop * split_ratio
                pipe_dia = 2 * pipe.r_in
                roughness = 0.000001  # check this and all values
                velocity = (mass_flow_ghe / nbh) / (density * np.pi * pipe.r_in ** 2)
                Re_n = velocity * pipe.r_in * 2 / (
                        0.0029649746891525514 / density)  # TODO: fix this: kinematic viscosity
                A = 2.457 * np.log((7 / Re_n) ** 0.9 + 0.27 * (roughness / pipe_dia)) ** 16
                B = (37530 / Re_n) ** 16
                friction_factor = 8 * ((8 / Re_n) ** 12 + (A + B) ** -1.5) ** (1 / 12)
                delta_P_GHE = friction_factor * length_ghe * density * velocity ** 2 / (2 * pipe_dia)
                GHX.P_ghe_cp[i] = mass_flow_ghe / (
                        density * self.GHE_cp_efficiency) * delta_P_GHE * self.beta_GHE_delta_P

        # central loop energy consumption
        m_ref_loop = max(m_loop_array)
        CL_delta_P = self.CL_P_per_m * self.length_CL
        for i in range(1, n_timesteps):
            delta_P_loop = (CL_delta_P / m_ref_loop ** 2) * m_loop_array[i] ** 2
            self.P_cl_cp[i] = m_loop_array[i] / (density * self.CL_efficiency) * delta_P_loop

        # ISHX loop energy consumption
        for i in range(1, n_timesteps):
            for ISHX in self.ISHXs:
                m_ref_ISHX = max(ISHX.m_loop_ISHX_array)
                delta_P_ISHX = (self.delta_P_ref_ISHX / m_ref_ISHX ** 2) * ISHX.m_loop_ISHX_array[i] ** 2
                ISHX.P_ishx_cp[i] = ISHX.m_loop_ISHX_array[i] / (
                        density * self.ISHX_cp_efficiency) * delta_P_ISHX * self.beta_ISHX_delta_P

    def createOutput(self, output_path: Path):
        if self.configuration == "1-pipe":
            # Step 1: create csv files
            n_timesteps = self.time_array_size
            data_rows = []

            for i in range(n_timesteps):
                row = []
                for zone in self.zones:
                    row.append(zone.t_eft[i])

                for GHX in self.GHXs:
                    row.append(GHX.t_eft[i])
                    row.append(GHX.t_mft[i])
                    row.append(GHX.q_ghe[i])
                    row.append(GHX.t_exft[i])

                for ISHX in self.ISHXs:
                    row.append(ISHX.t_n_eft[i])
                    row.append(ISHX.t_n_exft[i])
                    row.append(ISHX.t_hp_eft[i])

                data_rows.append(row)

            # Step 2: Create column labels
            column_names = []

            for j, zone in enumerate(self.zones):
                column_names.append(f"Zone{j}_EFT[C]")

            for j, GHX in enumerate(self.GHXs):
                column_names += [
                    f"GHX{j}_EFT[C]",
                    f"GHX{j}_MFT[C]",
                    f"GHX{j}_q_ghe[W/m]",
                    f"GHX{j}_ExFT[C]"
                ]

            for j, ISHX in enumerate(self.ISHXs):
                column_names += [
                    f"ISHX{j}_N_EFT[C]",
                    f"ISHX{j}_N_ExFT[C]",
                    f"ISHX{j}_HP_EFT[C]"
                ]
        elif self.configuration == "2-pipe":
            # Step 1: create csv files
            n_timesteps = self.time_array_size
            data_rows = []

            for i in range(n_timesteps):
                row = []
                for zone in self.zones:
                    row.append(zone.t_eft[i])
                    row.append(zone.t_exft[i])
                    row.append(zone.t_combining_node[i])

                for GHX in self.GHXs:
                    row.append(GHX.t_eft[i])
                    row.append(GHX.t_mft[i])
                    row.append(GHX.q_ghe[i])
                    row.append(GHX.t_exft[i])
                    row.append(GHX.t_combining_node[i])

                for ISHX in self.ISHXs:
                    row.append(ISHX.t_n_eft[i])
                    row.append(ISHX.t_n_exft[i])
                    row.append(ISHX.t_hp_eft[i])

                data_rows.append(row)

            # Step 2: Create column labels
            column_names = []

            for j, zone in enumerate(self.zones):
                column_names.append(f"Zone{j}_EFT[C]")
                column_names.append(f"Zone{j}_ExFT[C]")
                column_names.append(f"Zone{j}_CNT[C]")

            for j, GHX in enumerate(self.GHXs):
                column_names += [
                    f"GHX{j}_EFT[C]",
                    f"GHX{j}_MFT[C]",
                    f"GHX{j}_q_ghe[W/m]",
                    f"GHX{j}_ExFT[C]",
                    f"GHX{j}_CNT[C]"
                ]

            for j, ISHX in enumerate(self.ISHXs):
                column_names += [
                    f"ISHX{j}_N_EFT",
                    f"ISHX{j}_N_ExFT",
                    f"ISHX{j}_HP_EFT"
                ]
        else:
            raise ValueError(f"Invalid configuration type: {self.configuration}")

        # Step 3: Create and save DataFrame
        self.df = pd.DataFrame(data_rows, columns=column_names)
        self.df.index.name = "Hour"

        # Drop timestep 0 and reindex starting from 1
        self.df = self.df.iloc[1:]
        self.df.index = range(1, len(self.df) + 1)

        # Save to CSV
        self.df.to_csv(output_path, float_format="%.6f")

    def output_file_energy_consumption(self, output_path: Path):
        # create csv files
        n_timesteps = self.time_array_size
        data_rows = []

        for i in range(n_timesteps):
            row = []
            for zone in self.zones:
                row.append(zone.P_zone_htg[i])
                row.append(zone.P_zone_clg[i])
                row.append(zone.P_zone_cp[i])

            row.append(self.P_cl_cp[i])

            for GHX in self.GHXs:
                row.append(GHX.P_ghe_cp[i])

            for ISHX in self.ISHXs:
                row.append(ISHX.P_ishx_cp[i])

            data_rows.append(row)

        # Step 2: Create column labels
        column_names = []

        for j, zone in enumerate(self.zones):
            column_names.append(f"Zone{j}_P_htg")
            column_names.append(f"Zone{j}_P_clg")
            column_names.append(f"Zone{j}_P_cp")

        column_names.append(f"central_loop_P_cp")

        for j, GHX in enumerate(self.GHXs):
            column_names.append(f"GHX{j}_P_cp")

        for j, ISHX in enumerate(self.ISHXs):
            column_names.append(f"ISHX{j}_P_cp")

        # Step 3: Create and save DataFrame
        self.df1 = pd.DataFrame(data_rows, columns=column_names)
        self.df1.index.name = "Hour"

        # Drop timestep 0 and reindex starting from 1
        self.df1 = self.df1.iloc[1:]
        self.df1.index = range(1, len(self.df) + 1)

        # Save to CSV
        self.df1.to_csv(output_path, float_format="%.6f")

    def UpdateConnections(self):

        for pipe in self.pipes:
            pipe.input = FindItemByID(pipe.node_in_name, self.nodes)
            pipe.output = FindItemByID(pipe.node_out_name, self.nodes)
            if pipe.type == "1way":
                pipe.input.output = pipe
                pipe.output.input = pipe
            elif pipe.type == "branch":
                pipe.input.diversion = pipe
                pipe.output.input = pipe
            else:
                pipe.output.merger = pipe
                pipe.input.output = pipe

        for zone in self.zones:
            zone.HP = FindItemByID(zone.HPmodel, self.HPmodels)
            zone.input = FindItemByID(zone.inlet_nodeID, self.nodes)
            zone.input.output = zone
            if self.configuration == "2-pipe":
                zone.output = FindItemByID(zone.outlet_nodeID, self.nodes)
                zone.output.input = zone

        for building in self.buildings:
            for zoneID in building.zoneIDs:
                zone = FindItemByID(zoneID, self.zones)
                building.zones.append(zone)

        for GHX in self.GHXs:
            GHX.input = FindItemByID(GHX.inlet_nodeID, self.nodes)
            GHX.input.output = GHX
            if self.configuration == "2-pipe":
                GHX.output = FindItemByID(GHX.outlet_nodeID, self.nodes)
                GHX.output.input = GHX

        for ISHX in self.ISHXs:
            ISHX.input = FindItemByID(ISHX.node_network_inlet_ID, self.nodes)
            ISHX.HP_input = FindItemByID(ISHX.node_HP_inlet_ID, self.nodes)
            ISHX.HP_output = FindItemByID(ISHX.node_HP_outlet_ID, self.nodes)

            ISHX.input.output = ISHX
            ISHX.HP_output.input = ISHX
            ISHX.HP_input.output = ISHX

        for ISHX in self.ISHXs:
            for zoneID in ISHX.zoneIDs:
                zone = FindItemByID(zoneID, self.zones)
                ISHX.zones.append(zone)

        # finding upstream and downstream device for GHX

        for GHX in self.GHXs:

            # find the first upstream mixing node
            device = GHX.input
            while device.type != "mixing":
                device = device.input

            # find the second upstream mixing node
            device = device.input
            while device.type != "mixing" and device.type != "combining":
                device = device.input

            # find the upstream device
            if device.type == "mixing":
                device = device.diversion
                while device.type != "GHX" and device.type != "zone":
                    device = device.output
            else:
                device = device.merger
                while device.type != "GHX" and device.type != "zone":
                    device = device.input

            GHX.upstream_device = device
            device.downstream_device = GHX

        # finding upstream and donwstream device for zones

        for zone in self.zones:
            # find the first upstream mixing node
            device = zone.input
            while device.type != "mixing":
                device = device.input

            # find the second upstream mixing node or upstream device, if it is connected to ISHX
            device = device.input
            while device.type != "mixing" and device.type != "device" and device.type != "combining":
                device = device.input

            # find the upstream device
            if device.type == "mixing":
                device = device.diversion
                while device.type != "GHX" and device.type != "zone" and device.type != "ISHX":
                    device = device.output
            elif device.type == "combining":
                device = device.merger
                while device.type != "GHX":
                    device = device.input

            else:
                device = device.input

            zone.upstream_device = device
            device.downstream_device = zone

        # finding upstream and donwstream device for ISHX
        for ISHX in self.ISHXs:
            # find first upstream node
            device = ISHX.input
            while device.type != "mixing":
                device = device.input

            # find the second upstream mixing node
            device = device.input
            while device.type != "mixing":
                device = device.input

            # finding upstream device
            device = device.diversion
            while device.type != "GHX" and device.type != "zone":
                device = device.output

            ISHX.upstream_device = device
            device.downstream_device = ISHX

        # finding ISHX upstream device in HP side

        for ISHX in self.ISHXs:
            device = ISHX.HP_input
            # finding first upstream mixing node
            while device.type != "mixing":
                device = device.input

            # finding upstream device
            device = device.diversion
            while device.type != "zone":
                device = device.output

            ISHX.upstream_device_HP = device
            device.downstream_device = ISHX

        # finding ISHX downstream device in HP side

        for ISHX in self.ISHXs:
            device = ISHX.HP_output

            # finding first mixing node
            while device.type != "mixing":
                device = device.output

            # finding downstream device
            device = device.diversion
            while device.type != "zone":
                device = device.output

            ISHX.downstream_device_HP = device
            device.upstream_device = ISHX


def FindItemByID(ID, objectlist):
    # search a list of objects to find one with a particular name
    # of course, the objects must have a "name" member
    for item in objectlist:  # all objects in the list
        if item.ID == ID:  # does it have the ID I am seeking?
            return item  # then return this one
    # next item
    return None  # couldn't find it


def main():
    f1 = open("tests/test_data/1-pipe_3ghe-6hp_system_wo_ISHX_input.txt", 'r')
    data = f1.readlines()  # read the entire file as a list of strings
    f1.close()  # close the file  ... very important

    f2 = open("/home/mitchute/Projects/GHEDesigner/demos/find_design_bi_rectangle_single_u_tube.json", 'r')
    json_data = json.load(f2)

    System = GHEHPSystem()
    System.read_GHEHPSystem_data(data)
    System.read_data_from_json_file(json_data)

    fluid, pipe, grout, soil, borehole = System.read_data_from_json_file(json_data)
    System.solveSystem(fluid, pipe, grout, soil, borehole)
    System.createOutput()
    System.output_file_energy_consumption()


if __name__ == "__main__":
    main()

end = time.time()
print(f"Execution time: {end - start:.4f} seconds")
