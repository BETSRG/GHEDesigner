import json
from pathlib import Path

import numpy as np
import pandas as pd

from ghedesigner.constants import LPS_TO_M3S, TWO_PI
from ghedesigner.enums import BHType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.boreholes.factory import get_bhe_object
from ghedesigner.ghe.gfunction import calc_g_func_for_multiple_lengths
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.utilities import combine_sts_lts, eskilson_log_times


class HPmodel:
    def __init__(self, hp_id: str, hp_data: dict):
        self.name = hp_id
        self.ID = hp_id.upper()

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
        self.delta_P_HP = hp_data["design_pressure_loss"]

        self.design_htg_cap_single_hp = hp_data["heating_performance"]["design_cap"]
        self.design_clg_cap_single_hp = hp_data["cooling_performance"]["design_cap"]


class GHX:
    MATRIX_ROWS = 4

    def __init__(self, ghx_id: dict, ghx_data: dict, fluid: Fluid):
        self.pipe = Pipe.init_single_u_tube(
            inner_diameter=ghx_data["pipe"]["inner_diameter"],
            outer_diameter=ghx_data["pipe"]["outer_diameter"],
            shank_spacing=ghx_data["pipe"]["shank_spacing"],
            roughness=ghx_data["pipe"]["roughness"],
            conductivity=ghx_data["pipe"]["conductivity"],
            rho_cp=ghx_data["pipe"]["rho_cp"],
        )

        self.soil = Soil(
            k=ghx_data["soil"]["conductivity"],
            rho_cp=ghx_data["soil"]["rho_cp"],
            ugt=ghx_data["soil"]["undisturbed_temp"],
        )

        self.grout = Grout(k=ghx_data["grout"]["conductivity"], rho_cp=ghx_data["grout"]["rho_cp"])

        self.borehole = Borehole(
            burial_depth=ghx_data["borehole"]["buried_depth"], borehole_radius=ghx_data["borehole"]["diameter"] / 2.0
        )

        self.fluid = fluid

        self.n_rows = ghx_data["pre_designed"]["boreholes_in_x_dimension"]
        self.n_cols = ghx_data["pre_designed"]["boreholes_in_y_dimension"]
        self.row_spacing = ghx_data["pre_designed"]["spacing_in_x_dimension"]
        self.col_spacing = ghx_data["pre_designed"]["spacing_in_y_dimension"]
        self.height = ghx_data["pre_designed"]["H"]
        self.nbh = self.n_rows * self.n_cols
        self.mass_flow_per_borehole = ghx_data["flow_rate"] * LPS_TO_M3S * self.fluid.rho
        self.mass_flow_ghe = self.mass_flow_per_borehole * self.nbh
        self.two_pi_k = TWO_PI * self.soil.k
        self.split_ratio = None
        self.c_n = None
        self.matrix_size = None

        self.ID = ghx_id
        self.type = "GHX"

        self.input = None
        self.upstream_device = None
        self.downstream_device = None
        self.matrix_line = None
        self.row_index = None

        # Parameters to be assigned later
        self.m_dot_total = None

        self.bhe_type = BHType.SINGLEUTUBE
        self.split_ratio = None

        # Computed properties
        self.mass_flow_ghe = None

        self.H_n_ghe = None
        self.total_values_ghe = None

        # for output
        self.t_eft = None
        self.t_mft = None
        self.q_ghe = None
        self.t_exft = None
        self.t_bw = None
        self.t_combining_node = None

        self.bhe = get_bhe_object(
            self.bhe_type,
            self.mass_flow_per_borehole,
            self.fluid,
            self.borehole,
            self.pipe,
            self.grout,
            self.soil,
        )

        self.g_fn = self.compute_g_functions()

    def compute_g_functions(self):
        self.bhe.calc_sts_g_functions()

        bore_locations = [
            (i * self.row_spacing, j * self.row_spacing)
            for i in range(int(self.n_rows))
            for j in range(int(self.n_cols))
        ]

        g_lts = calc_g_func_for_multiple_lengths(
            self.row_spacing,
            [self.height],
            self.borehole.r_b,
            self.borehole.D,
            self.bhe.m_flow_borehole,
            self.bhe_type,
            eskilson_log_times(),
            bore_locations,
            self.bhe.fluid,
            self.bhe.pipe,
            self.bhe.grout,
            self.bhe.soil,
        )

        # Interpolate LTS g-function
        g_lts_vals, rb_value, _, _ = g_lts.g_function_interpolation(self.row_spacing / self.height)

        # Correct the g-function for borehole radius
        g_lts_corrected = g_lts.borehole_radius_correction(g_lts_vals, rb_value, self.borehole.r_b)

        # Combine STS and LTS g-functions
        g_fn = combine_sts_lts(
            eskilson_log_times(),
            g_lts_corrected,
            self.bhe.lntts.tolist(),
            self.bhe.g.tolist(),
        )

        return g_fn

    def set_cn_const_array(self, time_array, n_timesteps, bhe_effective_resist):
        """
        Calculate C_n values for three GHEs based on their g-functions.

        Cn = 1 / (2 * pi * K_s) * g((tn - tn-1) / t_s) + R_b
        """

        c_n = np.zeros(n_timesteps, dtype=float)
        ts = self.bhe.t_s
        ts_hr = ts / 3600

        for i in range(1, n_timesteps):
            delta_log_time = np.log((time_array[i] - time_array[i - 1]) / ts_hr)
            g_val = self.g_fn(delta_log_time)
            c_n[i] = (1 / self.two_pi_k * g_val) + bhe_effective_resist

        self.c_n = c_n

    def calc_history_term(self, i, time_array, tg, H_n_ghe, total_values_ghe, q_ghe):
        """
        Computes the history term H_n for this GHX at time index `i`.
        Updates self.total_values_ghe and self.H_n_ghe in place.
        """
        if i == 0:
            raise IndexError("Timestep index error")

        ts = self.bhe.t_s
        ts_hr = ts / 3600
        time_n = time_array[i]

        # Compute dimensionless time for all indices from 1 to i-1
        indices = np.arange(1, i)
        dim_less_time = np.log((time_n - time_array[indices - 1]) / ts_hr)

        # Compute contributions from all previous steps
        delta_q_ghe = (q_ghe[indices] - q_ghe[indices - 1]) / self.two_pi_k
        values = np.sum(delta_q_ghe * self.g_fn(dim_less_time))

        total_values_ghe[i] = values

        # Contribution from the last time step only
        dim1_less_time = np.log((time_n - time_array[i - 1]) / ts_hr)

        H_n_ghe[i] = tg + total_values_ghe[i] - (q_ghe[i - 1] / self.two_pi_k * self.g_fn(dim1_less_time))
        return H_n_ghe[i]

    def generate_GHX_matrix_row(
        self, i, GHX_inlet_index, mass_flow_ghe, cp, m_loop_ghe, H_n_ghe, m_loop, configuration
    ):
        row1 = np.zeros(self.matrix_size)
        row2 = np.zeros(self.matrix_size)
        row3 = np.zeros(self.matrix_size)
        row4 = np.zeros(self.matrix_size)

        row_index = self.row_index
        neighbour_index = self.downstream_device.row_index

        if configuration == "1-pipe":
            row1[row_index] = (m_loop - mass_flow_ghe) * cp
            row1[row_index + 3] = mass_flow_ghe * cp
            row1[neighbour_index] = -m_loop * cp

            row2[row_index + 1] = 1
            row2[row_index + 2] = -self.c_n[i]

            row3[row_index] = -1
            row3[row_index + 1] = 2
            row3[row_index + 3] = -1

            row4[row_index] = mass_flow_ghe * cp
            row4[row_index + 2] = -self.height * (self.n_rows * self.n_cols)
            row4[row_index + 3] = -mass_flow_ghe * cp

            rhs1, rhs2, rhs3, rhs4 = 0, H_n_ghe, 0, 0

        elif configuration == "2-pipe":
            row1[row_index + 1] = 1
            row1[row_index + 2] = -self.c_n[i]

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


class Zone:
    MATRIX_ROWS_1_PIPE = 1
    MATRIX_ROWS_2_PIPE = 2

    def __init__(self):
        # values read from the file
        self.name = None
        self.ID = None
        self.type = "zone"
        self.connection_downstream = None
        self.node = None
        self.HPmodel = None
        self.HP = None
        self.matrix_size = None
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

    def get_zone_loads(self):
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

    def zone_mass_flow_rate(self, t_eft, i):
        cap_htg = self.HP.c1_htg * t_eft**2 + self.HP.c2_htg * t_eft + self.HP.c3_htg
        cap_clg = self.HP.c1_clg * t_eft**2 + self.HP.c2_clg * t_eft + self.HP.c3_clg

        if cap_clg == 0:
            rtf = abs(self.h[i] / cap_htg)
        else:
            rtf = abs(self.h[i] / cap_htg) + abs(self.c[i] / cap_clg)

        m_flow_single_hp = self.HP.m_flow_single_hp
        self.mass_flow_zone = rtf * m_flow_single_hp

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
        ratio_htg = a_htg * t_eft**2 + b_htg * t_eft + c_htg
        u = ratio_htg - slope_htg * t_eft
        v = slope_htg

        # Cooling calculations
        slope_clg = 2 * a_clg * t_eft + b_clg
        ratio_clg = a_clg * t_eft**2 + b_clg * t_eft + c_clg
        a = ratio_clg - slope_clg * t_eft
        b = slope_clg

        # Final arrays
        r1 = b * c - v * h
        r2 = a * c - u * h

        return r1, r2

    def generate_zone_matrix_row(
        self, zone_inlet_index, r1, mass_flow_zone, cp, m_loop_zone, m_loop, r2, configuration
    ):
        row1 = np.zeros(self.matrix_size)
        row2 = np.zeros(self.matrix_size)

        row_index = self.row_index
        neighbour_index = self.downstream_device.row_index

        if configuration == "1-pipe":
            row = np.zeros(self.matrix_size)
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
            row2[neighbour_index] = -m_loop_zone * cp

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

        ratio_htg = a_htg * t_eft**2 + b_htg * t_eft + c_htg
        ratio_clg = a_clg * t_eft**2 + b_clg * t_eft + c_clg

        # zone (HP) power consumed
        Power_zone_htg = htg_load * (1 - ratio_htg)
        Power_zone_clg = clg_load * (ratio_clg - 1)

        # power consumed by circulating pump
        Power_zone_cp = m_flow_zone / (density * cp_efficiency) * beta_HP_delta_P * delta_P_HP

        return Power_zone_htg, Power_zone_clg, Power_zone_cp


class Node:
    def __init__(self):
        self.ID = None
        self.input = None
        self.output = None
        self.diversion = None
        self.merger = None


class LocalPipe:
    def __init__(self):
        self.ID = None
        self.node_in_name = None
        self.node_out_name = None
        self.input = None
        self.output = None


class IsolationHX:
    MATRIX_ROWS = 3

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
        self.matrix_size = None

        self.zoneIDs = []  # list of zone ids
        self.zones = []  # list of zones

        self.effectiveness = None
        self.m_loop_n = None
        self.m_loop_hp = None

    def generate_ISHX_matrix_row(self, m_loop, cp):
        effec = self.effectiveness
        m_loop_n = self.m_loop_n

        C_n = self.m_loop_n * cp
        C_hp = self.m_loop_hp * cp
        C_min = min(C_n, C_hp)

        row1 = np.zeros(self.matrix_size)
        row2 = np.zeros(self.matrix_size)
        row3 = np.zeros(self.matrix_size)

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
        row3[neighbour_index_loop_side] = -(m_loop * cp)

        rhs1, rhs2, rhs3 = 0, 0, 0

        rows = [row1, row2, row3]
        rhs = [rhs1, rhs2, rhs3]

        return rows, rhs


class GHEHPSystem:
    def __init__(self, f_path_json: Path):
        json_data = json.loads(f_path_json.read_text())

        fluid_data = json_data["fluid"]
        self.fluid = Fluid(
            fluid_name=fluid_data["fluid_name"],
            temperature=fluid_data["temperature"],
            percent=fluid_data.get("concentration_percent", 0),
        )

        self.HPmodels = []
        all_hp_data = json_data["heat_pump"]
        for hp_name, hp_data in all_hp_data.items():
            self.HPmodels.append(HPmodel(hp_name, hp_data))

        self.GHXs = []
        nbh_tot = 0
        all_ghx_data = json_data["ground_heat_exchanger"]
        for ghx_name, ghx_data in all_ghx_data.items():
            self.GHXs.append(GHX(ghx_name, ghx_data, self.fluid))
            nbh_tot += self.GHXs[-1].nbh

        for idx, ghx in enumerate(self.GHXs):
            self.GHXs[idx].split_ratio = ghx.nbh / nbh_tot

        self.initial_temp = self.GHXs[0].soil.ugt

        self.configuration = None
        self.matrix_size = None

        self.zones = []
        self.nodes = []
        self.pipes = []
        self.ISHXs = []
        self.current_row = 0
        self.m_loop = None
        self.time_array = None
        self.num_hours = None

        # Thermal object references (to be set during setup)
        self.nbh_total = None
        self.mass_flow_ghe = None
        self.m_loop = None
        self.beta_CL_flow = None
        self.beta_ISHX_loop = None
        self.beta_cl_cp_delta_P = None

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
            cells = [c.strip() for c in line.strip().split(",")]
            keyword = cells[0].lower()

            if keyword == "configuration":
                self.configuration = cells[1].replace("'", "")

            if keyword == "ghx":
                for ghx in self.GHXs:
                    if str(cells[1]) == ghx.ID:
                        ghx.inlet_nodeID = str(cells[2])
                        ghx.outlet_nodeID = str(cells[3])
                        ghx.matrix_line = next_matrix_line
                        next_matrix_line += GHX.MATRIX_ROWS

            if keyword == "zone":
                df = pd.read_csv(in_file_dir / cells[7])
                self.time_array = df["Hours"].values
                self.num_hours = len(self.time_array)

                thiszone = Zone()
                thiszone.name = str(cells[1])
                thiszone.ISHX_ID = str(cells[2])
                thiszone.ID = str(cells[3])
                thiszone.inlet_nodeID = str(cells[4])
                thiszone.outlet_nodeID = str(cells[5])
                thiszone.HPmodel = str(cells[6])
                thiszone.df_zone = pd.read_csv(in_file_dir / cells[7])
                thiszone.matrix_line = next_matrix_line
                next_matrix_line += 1
                self.zones.append(thiszone)

            if keyword == "ishx":
                thisishx = IsolationHX()
                thisishx.name = str(cells[1])
                thisishx.ID = str(cells[2])
                thisishx.node_network_inlet_ID = str(cells[3])
                thisishx.node_HP_inlet_ID = str(cells[4])
                thisishx.node_HP_outlet_ID = str(cells[5])
                thisishx.beta_ISHX = float(cells[6])
                thisishx.effectiveness = float(cells[7])
                thisishx.zoneIDs = [zones.strip() for zones in cells[8:]]
                self.ISHXs.append(thisishx)

            if keyword == "node":
                thisnode = Node()
                thisnode.ID = str(cells[1])
                thisnode.type = str(cells[2])
                self.nodes.append(thisnode)

            if keyword == "pipe":
                thispipe = LocalPipe()
                thispipe.ID = str(cells[1])
                thispipe.type = str(cells[2])
                thispipe.node_in_name = str(cells[3])
                thispipe.node_out_name = str(cells[4])
                self.pipes.append(thispipe)

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

        for this_zone in self.zones:
            this_zone.get_zone_loads()

        if self.configuration == "1-pipe":
            self.matrix_size = np.dot(
                [len(self.zones), len(self.GHXs), len(self.ISHXs)],
                [Zone.MATRIX_ROWS_1_PIPE, GHX.MATRIX_ROWS, IsolationHX.MATRIX_ROWS],
            )
        elif self.configuration == "2-pipe":
            self.matrix_size = np.dot(
                [len(self.zones), len(self.GHXs), len(self.ISHXs)],
                [Zone.MATRIX_ROWS_2_PIPE, GHX.MATRIX_ROWS, IsolationHX.MATRIX_ROWS],
            )
        else:
            raise ValueError(f"Invalid configuration type: {self.configuration}")

        for this_ghx in self.GHXs:
            this_ghx.matrix_size = self.matrix_size
        for this_zone in self.zones:
            this_zone.matrix_size = self.matrix_size
        for this_isx in self.ISHXs:
            this_isx.matrix_size = self.matrix_size

        # for getting g_functions and bhe object
        for this_ghx in self.GHXs:
            # self.bhe_effective_resist = this_ghx.bhe.calc_effective_borehole_resistance()
            self.bhe_effective_resist = 0.15690883427464597  # TODO: Fix this
            this_ghx.set_cn_const_array(self.time_array, self.num_hours, self.bhe_effective_resist)

        # Initializing the values
        for this_ghx in self.GHXs:
            this_ghx.H_n_ghe = np.full(self.num_hours, self.initial_temp)
            this_ghx.total_values_ghe = np.zeros(self.num_hours)
            this_ghx.q_ghe = np.zeros(self.num_hours)

        # Initializing t_eft, t_mean, q_ghe, t_exit
        for this_zone in self.zones:
            this_zone.t_eft = np.full(self.num_hours, self.initial_temp)
            this_zone.t_exft = np.full(self.num_hours, self.initial_temp)
            this_zone.t_combining_node = np.full(self.num_hours, self.initial_temp)

        for this_ghx in self.GHXs:
            this_ghx.t_eft = np.full(self.num_hours, self.initial_temp)
            this_ghx.t_mft = np.full(self.num_hours, self.initial_temp)
            this_ghx.t_bhw = np.full(self.num_hours, self.initial_temp)
            this_ghx.q_ghe = np.zeros(self.num_hours)
            this_ghx.t_exft = np.full(self.num_hours, self.initial_temp)
            this_ghx.t_combining_node = np.full(self.num_hours, self.initial_temp)

        for this_ihx in self.ISHXs:
            this_ihx.t_n_eft = np.full(self.num_hours, self.initial_temp)
            this_ihx.t_n_exft = np.full(self.num_hours, self.initial_temp)
            this_ihx.t_hp_eft = np.full(self.num_hours, self.initial_temp)

        # end for line
        self.UpdateConnections()

        # Assigning row_indices
        if self.configuration == "1-pipe":
            for k, zone in enumerate(self.zones):
                zone.row_index = k
            for k, this_ghx in enumerate(self.GHXs):
                this_ghx.row_index = len(self.zones) + k * GHX.MATRIX_ROWS
            for k, this_ihx in enumerate(self.ISHXs):
                this_ihx.row_index = len(self.zones) + len(self.GHXs) * GHX.MATRIX_ROWS

        elif self.configuration == "2-pipe":
            for k, zone in enumerate(self.zones):
                zone.row_index = k * Zone.MATRIX_ROWS_2_PIPE
            for k, this_ghx in enumerate(self.GHXs):
                this_ghx.row_index = Zone.MATRIX_ROWS_2_PIPE * len(self.zones) + k * GHX.MATRIX_ROWS
            for k, this_ihx in enumerate(self.ISHXs):
                this_ihx.row_index = len(self.zones) + len(self.GHXs) * GHX.MATRIX_ROWS

        else:
            raise ValueError(f"Invalid configuration type: {self.configuration}")

        print("\nDevice,RowIDX,Downstream Device,Downstream RowIDX,Downstream HP Device,Downstream HP RowIDX")
        for idx, this_zone in enumerate(self.zones):
            print(f"{this_zone.type}-{this_zone.ID},{this_zone.row_index},{this_zone.downstream_device.type}-{this_zone.downstream_device.ID},{this_zone.downstream_device.row_index}")

        for idx, this_ghx in enumerate(self.GHXs):
            print(f"{this_ghx.type}-{this_ghx.ID},{this_ghx.row_index},{this_ghx.downstream_device.type}-{this_ghx.downstream_device.ID},{this_ghx.downstream_device.row_index}")

        for idx, this_isx in enumerate(self.ISHXs):
            print(f"{this_isx.type}-{this_isx.ID},{this_isx.row_index},{this_isx.downstream_device.type}-{this_isx.downstream_device.ID},{this_isx.downstream_device.row_index},{this_isx.downstream_device_HP.type}-{this_isx.downstream_device_HP.ID},{this_isx.downstream_device_HP.row_index}")

    def solve_system(self):
        # precompute all time invariant constants

        time_array = self.time_array
        configuration = self.configuration

        # Initializing
        m_loop_array = np.zeros(self.num_hours)
        for this_ihx in self.ISHXs:
            this_ihx.m_loop_ISHX_array = np.zeros(self.num_hours)

        for zone in self.zones:
            zone.P_zone_htg = np.zeros(self.num_hours)
            zone.P_zone_clg = np.zeros(self.num_hours)
            zone.P_zone_cp = np.zeros(self.num_hours)

        self.P_cl_cp = np.zeros(self.num_hours)

        for this_ghx in self.GHXs:
            this_ghx.P_ghe_cp = np.zeros(self.num_hours)

        for this_ihx in self.ISHXs:
            this_ihx.P_ishx_cp = np.zeros(self.num_hours)

        for i in range(1, self.num_hours):  # loop over all timestep
            matrix_rows = []
            matrix_rhs = []
            # Calculating total hp flows in each ISHX

            for this_ihx in self.ISHXs:
                total_hp_flow_ISHX = 0
                for zone in self.zones:
                    if zone in this_ihx.zones:
                        t_eft = zone.t_eft[i - 1]
                        m_zone = zone.zone_mass_flow_rate(t_eft, i)
                        total_hp_flow_ISHX += m_zone
                        this_ihx.m_loop_hp = total_hp_flow_ISHX * self.beta_ISHX_HP_flow

            # Calculating total hp flows in heat pumps connected directly to loop
            total_hp_flow = 0
            for zone in self.zones:
                if zone.ISHX_ID == "None":
                    t_eft = zone.t_eft[i - 1]
                    m_zone = zone.zone_mass_flow_rate(t_eft, i)
                    total_hp_flow += m_zone

            total_m_loop_n = 0
            for this_ihx in self.ISHXs:
                this_ihx.m_loop_n = this_ihx.m_loop_hp * self.beta_ISHX_N_flow
                total_m_loop_n += this_ihx.m_loop_n
                this_ihx.m_loop_ISHX_array[i] = total_m_loop_n

            m_loop = (total_m_loop_n + total_hp_flow) * self.beta_CL_flow
            m_loop_array[i] = m_loop

            zone_inlet_index = self.zones[0].row_index

            # Generating matrix for zones connected to ISHX
            m_loop_zone = 0
            for this_ihx in self.ISHXs:
                for zone in self.zones:
                    if zone in this_ihx.zones:
                        m_loop = this_ihx.m_loop_hp
                        t_eft = zone.t_eft[i - 1]
                        r1, r2 = zone.calculate_r1_r2(t_eft, i)
                        mass_flow_zone = zone.zone_mass_flow_rate(t_eft, i)
                        m_loop_zone += mass_flow_zone
                        this_zone_row, rhs = zone.generate_zone_matrix_row(
                            zone_inlet_index,
                            r1,
                            mass_flow_zone,
                            self.fluid.cp,
                            m_loop_zone,
                            m_loop,
                            r2,
                            configuration,
                        )
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
                    this_zone_row, rhs = zone.generate_zone_matrix_row(
                        zone_inlet_index, r1, mass_flow_zone, self.fluid.cp, m_loop_zone, m_loop, r2, configuration
                    )

                    for row, rhs in zip(this_zone_row, rhs):
                        matrix_rows.append(row)
                        matrix_rhs.append(rhs)

            # Generating matrix for ground heat exchangers
            m_loop_ghe = 0
            GHX_inlet_index = self.GHXs[0].row_index
            for j, this_ghx in enumerate(self.GHXs):
                q_ghe = this_ghx.q_ghe[:i]  # <--- FIXED: slice of all past values, it is an array
                mass_flow_ghe = m_loop * this_ghx.split_ratio
                H_n_ghe = this_ghx.calc_history_term(
                    i, time_array, self.initial_temp, this_ghx.H_n_ghe, this_ghx.total_values_ghe, q_ghe
                )
                m_loop_ghe += mass_flow_ghe

                rows, rhs_values = this_ghx.generate_GHX_matrix_row(
                    i, GHX_inlet_index, mass_flow_ghe, self.fluid.cp, m_loop_ghe, H_n_ghe, m_loop, configuration
                )

                for row, rhs in zip(rows, rhs_values):
                    matrix_rows.append(row)
                    matrix_rhs.append(rhs)

            # Generating matrix for isolation heat exchanger
            for this_ihx in self.ISHXs:
                m_loop = (total_m_loop_n + total_hp_flow) * self.beta_CL_flow
                rows, rhs_values = this_ihx.generate_ISHX_matrix_row(m_loop, self.fluid.cp)
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

                for this_ghx in self.GHXs:
                    this_ghx.t_eft[i] = X[this_ghx.row_index]
                    this_ghx.t_mft[i] = X[this_ghx.row_index + 1]
                    this_ghx.q_ghe[i] = X[this_ghx.row_index + 2]
                    this_ghx.t_exft[i] = X[this_ghx.row_index + 3]

                for this_ihx in self.ISHXs:
                    this_ihx.t_n_eft[i] = X[this_ihx.row_index]
                    this_ihx.t_n_exft[i] = X[this_ihx.row_index + 1]
                    this_ihx.t_hp_eft[i] = X[this_ihx.row_index + 2]

            # for getting values for 2-pipe system
            elif self.configuration == "2-pipe":
                for zone in self.zones:
                    zone.t_eft[i] = X[zone_inlet_index]
                    zone.t_exft[i] = X[zone.row_index + 1]
                    zone.t_combining_node[i] = X[zone.downstream_device.row_index]

                for this_ghx in self.GHXs:
                    this_ghx.t_eft[i] = X[GHX_inlet_index]
                    this_ghx.t_mft[i] = X[this_ghx.row_index + 1]
                    this_ghx.q_ghe[i] = X[this_ghx.row_index + 2]
                    this_ghx.t_exft[i] = X[this_ghx.row_index + 3]
                    this_ghx.t_combining_node[i] = X[this_ghx.downstream_device.row_index]

            # zone energy consumption
            for zone in self.zones:
                t_eft = zone.t_eft[i - 1]
                m_flow_zone = zone.zone_mass_flow_rate(t_eft, i)
                cp_efficiency = self.HP_cp_efficiency
                delta_P_HP = zone.HP.delta_P_HP
                density = self.fluid.rho
                zone.P_zone_htg[i], zone.P_zone_clg[i], zone.P_zone_cp[i] = zone.zone_energy_consumption(
                    t_eft, i, m_flow_zone, density, cp_efficiency, self.beta_HP_delta_P, delta_P_HP
                )

            # ground heat exchanger energy consumption
            for this_ghx in self.GHXs:
                length_ghe = 2 * this_ghx.height
                mass_flow_ghe = m_loop * this_ghx.split_ratio
                pipe_dia = 2 * this_ghx.pipe.r_in
                roughness = this_ghx.pipe.roughness
                dynamic_viscosity = self.fluid.mu
                density = self.fluid.rho
                velocity = (mass_flow_ghe / this_ghx.nbh) / (density * np.pi * this_ghx.pipe.r_in**2)
                Re_n = velocity * this_ghx.pipe.r_in * 2 / (dynamic_viscosity / density)
                A = 2.457 * np.log((7 / Re_n) ** 0.9 + 0.27 * (roughness / pipe_dia)) ** 16
                B = (37530 / Re_n) ** 16
                friction_factor = 8 * ((8 / Re_n) ** 12 + (A + B) ** -1.5) ** (1 / 12)
                delta_P_GHE = friction_factor * length_ghe * density * velocity**2 / (2 * pipe_dia)
                this_ghx.P_ghe_cp[i] = (
                    mass_flow_ghe / (density * self.GHE_cp_efficiency) * delta_P_GHE * self.beta_GHE_delta_P
                )

        # central loop energy consumption
        m_ref_loop = max(m_loop_array)
        CL_delta_P = self.CL_P_per_m * self.length_CL
        for i in range(1, self.num_hours):
            delta_P_loop = (CL_delta_P / m_ref_loop**2) * m_loop_array[i] ** 2
            density = self.fluid.rho
            self.P_cl_cp[i] = m_loop_array[i] / (density * self.CL_efficiency) * delta_P_loop

        # ISHX loop energy consumption
        for i in range(1, self.num_hours):
            for this_ihx in self.ISHXs:
                density = self.fluid.rho
                m_ref_ISHX = max(this_ihx.m_loop_ISHX_array)
                delta_P_ISHX = (self.delta_P_ref_ISHX / m_ref_ISHX**2) * this_ihx.m_loop_ISHX_array[i] ** 2
                this_ihx.P_ishx_cp[i] = (
                    this_ihx.m_loop_ISHX_array[i]
                    / (density * self.ISHX_cp_efficiency)
                    * delta_P_ISHX
                    * self.beta_ISHX_delta_P
                )

    def write_state_outputs(self, output_path: Path):
        output_data = pd.DataFrame()
        output_data.index.name = "Hour"

        if self.configuration == "1-pipe":
            for zone_num, this_zone in enumerate(self.zones):
                output_data[f"Zone{zone_num}_EFT[C]"] = this_zone.t_eft

            for ghx_num, this_ghx in enumerate(self.GHXs):
                output_data[f"GHX{ghx_num}_EFT[C]"] = this_ghx.t_eft
                output_data[f"GHX{ghx_num}_MFT[C]"] = this_ghx.t_mft
                output_data[f"GHX{ghx_num}_q_ghe[W/m]"] = this_ghx.q_ghe
                output_data[f"GHX{ghx_num}_ExFT[C]"] = this_ghx.t_exft

            for hx_num, this_hx in enumerate(self.ISHXs):
                output_data[f"ISHX{hx_num}_N_EFT[C]"] = this_hx.t_n_eft
                output_data[f"ISHX{hx_num}_N_ExFT[C]"] = this_hx.t_n_exft
                output_data[f"ISHX{hx_num}_HP_EFT[C]"] = this_hx.t_hp_eft

        elif self.configuration == "2-pipe":
            for zone_num, this_zone in enumerate(self.zones):
                output_data[f"Zone{zone_num}_EFT[C]"] = this_zone.t_eft
                output_data[f"Zone{zone_num}_ExFT[C]"] = this_zone.t_exft
                output_data[f"Zone{zone_num}_CNT[C]"] = this_zone.t_combining_node

            for ghx_num, this_ghx in enumerate(self.GHXs):
                output_data[f"GHX{ghx_num}_EFT[C]"] = this_ghx.t_eft
                output_data[f"GHX{ghx_num}_MFT[C]"] = this_ghx.t_mft
                output_data[f"GHX{ghx_num}_q_ghe[W/m]"] = this_ghx.q_ghe
                output_data[f"GHX{ghx_num}_ExFT[C]"] = this_ghx.t_exft
                output_data[f"GHX{ghx_num}_CNT[C]"] = this_ghx.t_combining_node

            for hx_num, this_hx in enumerate(self.ISHXs):
                output_data[f"ISHX{hx_num}_N_EFT[C]"] = this_hx.t_n_eft
                output_data[f"ISHX{hx_num}_N_ExFT[C]"] = this_hx.t_n_exft
                output_data[f"ISHX{hx_num}_HP_EFT[C]"] = this_hx.t_hp_eft

        else:
            raise ValueError(f"Invalid configuration type: {self.configuration}")

        output_data = output_data.iloc[1:]
        output_data.to_csv(output_path, float_format="%.6f")

    def write_energy_outputs(self, output_path: Path):
        output_data = pd.DataFrame()
        output_data.index.name = "Hour"

        for zone_num, this_zone in enumerate(self.zones):
            output_data[f"Zone{zone_num}_P_htg"] = this_zone.P_zone_htg
            output_data[f"Zone{zone_num}_P_clg"] = this_zone.P_zone_clg
            output_data[f"Zone{zone_num}_P_cp"] = this_zone.P_zone_cp

        output_data["central_loop_P_cp"] = self.P_cl_cp

        for ghx_num, this_hx in enumerate(self.GHXs):
            output_data[f"GHX{ghx_num}_P_cp"] = this_hx.P_ghe_cp

        for hx_num, this_hx in enumerate(self.ISHXs):
            output_data[f"ISHX{hx_num}_P_cp"] = this_hx.P_ishx_cp

        output_data = output_data.iloc[1:]
        output_data.to_csv(output_path, float_format="%.6f")

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

        # finding upstream and downstream device for zones

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

        # finding upstream and downstream device for ISHX
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
