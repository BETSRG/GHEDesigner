import json
from pathlib import Path

import numpy as np
import pandas as pd

from ghedesigner.constants import TWO_PI
from ghedesigner.enums import PipeType, SimCompType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.boreholes.factory import get_bhe_object
from ghedesigner.ghe.gfunction import calc_g_func_for_multiple_lengths
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import GHEFluid, Grout, Soil
from ghedesigner.utilities import combine_sts_lts

N_TIMESTEPS = 8760


class GHX:
    def __init__(self, ghe_id: str, ghe_data: dict, fluid: GHEFluid):
        self.type = "GHX"
        self.new_type = SimCompType.GROUND_HEAT_EXCHANGER
        self.input = None
        self.downstream_device = None
        self.height = None
        self.row_index = None

        # Parameters to be assigned later
        self.m_dot_total = None

        # Thermal object references (to be set during setup)

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
            burial_depth=ghe_data["borehole"]["buried_depth"], borehole_radius=ghe_data["borehole"]["diameter"] / 2.0
        )

        self.fluid = fluid
        self.bhe_type = PipeType.SINGLEUTUBE
        self.split_ratio = None

        self.two_pi_k = TWO_PI * self.soil.k

        # Computed properties
        self.bhe = None
        self.r_b = None
        self.gFunction = None
        self.depth = None

        # self.mass_flow_ghe_design = None
        self.mass_flow_ghe_borehole_design = None
        self.history_terms = None
        self.total_values_ghe = None

        # for output
        self.t_eft = None
        self.t_mean = None
        self.q_ghe = None
        self.t_exit = None

        self.ID = ghe_id
        self.nodeID = ghe_data["node_id"]
        self.n_rows = ghe_data["pre_designed"]["boreholes_in_x_dimension"]
        self.n_cols = ghe_data["pre_designed"]["boreholes_in_y_dimension"]
        self.row_spacing = ghe_data["pre_designed"]["spacing_in_x_dimension"]
        self.col_spacing = ghe_data["pre_designed"]["spacing_in_y_dimension"]
        self.ghe_height = ghe_data["pre_designed"]["H"]
        self.nbh = self.n_rows * self.n_cols
        self.mass_flow_ghe_design = ghe_data["flow_rate"] * self.nbh
        self.matrix_size = None

        self.history_terms, self.total_values_ghe, self.q_ghe = (
            np.full(N_TIMESTEPS, self.soil.ugt),
            np.zeros(N_TIMESTEPS),
            np.zeros(N_TIMESTEPS),
        )

        self.t_eft = np.full(N_TIMESTEPS, self.soil.ugt)
        self.t_mean = np.full(N_TIMESTEPS, self.soil.ugt)
        self.q_ghe = np.zeros(N_TIMESTEPS)
        self.t_exit = np.full(N_TIMESTEPS, self.soil.ugt)
        self.time_array = np.arange(1, N_TIMESTEPS + 1)
        self.gFunction = None
        self.g = None
        self.c_n = None

        self.height = self.borehole.H
        self.nbh = self.n_rows * self.n_cols
        self.mass_flow_ghe_borehole_design = self.mass_flow_ghe_design / self.nbh
        self.bhe = get_bhe_object(
            self.bhe_type,
            self.mass_flow_ghe_borehole_design,
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
        self.c_n = self.calculation_of_ghe_constant_c_n()

    def generate_g_function_object(self):
        self.r_b = self.bhe.calc_effective_borehole_resistance()
        self.depth = self.bhe.borehole.D
        self.mass_flow_ghe_borehole_design = self.mass_flow_ghe_design / self.nbh
        h_values = [self.height]
        coordinates_ghe = [
            (i * self.row_spacing, j * self.row_spacing) for i in range(self.n_rows) for j in range(self.n_cols)
        ]
        self.gFunction = calc_g_func_for_multiple_lengths(
            self.row_spacing,
            h_values,
            self.r_b,
            self.depth,
            self.mass_flow_ghe_borehole_design,
            self.bhe_type,
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

    def calculation_of_ghe_constant_c_n(self):
        """
        Calculate C_n values for three GHEs based on their g-functions.

        Cn = 1 / (2 * pi * K_s) * g((tn - tn-1) / t_s) + R_b
        """

        c_n = np.zeros(N_TIMESTEPS, dtype=float)

        for i in range(1, N_TIMESTEPS):
            delta_log_time = np.log((self.time_array[i] - self.time_array[i - 1]) / (self.ts / 3600))
            g_val = self.g(delta_log_time)

            c_n[i] = (1 / self.two_pi_k * g_val) + self.r_b

        return c_n

    def compute_history_term(self, idx_timestep, history_terms, total_values_ghe):
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
        dim_less_time = np.log((time_n - self.time_array[indices - 1]) / (self.ts / 3600))

        # Compute contributions from all previous steps
        delta_q_ghe = (q_ghe[indices] - q_ghe[indices - 1]) / self.two_pi_k
        values = np.sum(delta_q_ghe * self.g(dim_less_time))

        total_values_ghe[idx_timestep] = values

        # Contribution from the last time step only
        dim1_less_time = np.log((time_n - self.time_array[idx_timestep - 1]) / (self.ts / 3600))
        history_terms[idx_timestep] = (
            self.soil.ugt
            - total_values_ghe[idx_timestep]
            + (q_ghe[idx_timestep - 1] / self.two_pi_k * self.g(dim1_less_time))
        )

        return history_terms, total_values_ghe

    def generate_matrix_row(self, m_loop, idx_timestep):
        self.history_terms, self.total_values_ghe = self.compute_history_term(
            idx_timestep, self.history_terms, self.total_values_ghe
        )

        row1 = np.zeros(self.matrix_size)
        row2 = np.zeros(self.matrix_size)
        row3 = np.zeros(self.matrix_size)
        row4 = np.zeros(self.matrix_size)

        row_index = self.row_index
        neighbour_index = self.downstream_device.row_index
        mass_flow_ghe = m_loop * self.split_ratio

        row1[row_index] = (m_loop - mass_flow_ghe) * self.cp
        row1[row_index + 3] = mass_flow_ghe * self.cp
        row1[neighbour_index] = -m_loop * self.cp

        row2[row_index + 1] = 1
        row2[row_index + 2] = self.c_n[idx_timestep]

        row3[row_index] = -1
        row3[row_index + 1] = 2
        row3[row_index + 3] = -1

        row4[row_index] = mass_flow_ghe * self.cp
        row4[row_index + 2] = self.height * self.nbh
        row4[row_index + 3] = -mass_flow_ghe * self.cp

        rhs1, rhs2, rhs3, rhs4 = 0, self.history_terms[idx_timestep], 0, 0

        rows = [row1, row2, row3, row4]
        rhs = [rhs1, rhs2, rhs3, rhs4]
        return rows, rhs


class Building:
    def __init__(self, bldg_id: str, bldg_data: dict, hp_data: dict, parent_dir: Path, tg):
        # values read from the file
        self.type = "building"
        self.new_type = SimCompType.BUILDING
        self.node = None
        self.hp = None
        self.row_index = None
        self.df_bldg = None
        self.t_eft = None
        self.downstream_device = None
        self.matrix_size = None
        self.cp = None

        self.ID = bldg_id
        self.nodeID = bldg_data["node_id"]

        self.heating_exists = bool("heating" in bldg_data)
        self.cooling_exists = bool("cooling" in bldg_data)

        self.htg_vals = np.zeros(N_TIMESTEPS, dtype=float)
        self.clg_vals = np.zeros(N_TIMESTEPS, dtype=float)

        if self.heating_exists:
            hp_htg_name = bldg_data["heating"]["heat_pump"]
            hp_htg_data = hp_data[hp_htg_name]
            self.hp_htg = HPmodel(hp_htg_name, hp_htg_data)
            htg_loads_path = parent_dir / bldg_data["heating"]["loads"]["file_path"]
            htg_loads_path = htg_loads_path.resolve()
            htg_col = bldg_data["heating"]["loads"]["column"]
            df_htg = pd.read_csv(htg_loads_path, usecols=[htg_col])
            self.htg_vals = df_htg[htg_col].to_numpy()

        if self.cooling_exists:
            hp_clg_name = bldg_data["cooling"]["heat_pump"]
            hp_clg_data = hp_data[hp_clg_name]
            self.hp_clg = HPmodel(hp_clg_name, hp_clg_data)
            clg_loads_path = parent_dir / bldg_data["cooling"]["loads"]["file_path"]
            clg_loads_path = clg_loads_path.resolve()
            clg_col = bldg_data["cooling"]["loads"]["column"]
            df_clg = pd.read_csv(clg_loads_path, usecols=[clg_col])
            self.clg_vals = df_clg[clg_col].to_numpy()

        self.q_net_htg = self.htg_vals - self.clg_vals
        self.t_eft = np.full(N_TIMESTEPS, tg)

    def calc_bldg_mass_flow_rate(self, t_eft, i):
        if self.heating_exists:
            cap_htg = self.hp_htg.c1_htg * t_eft**2 + self.hp_htg.c2_htg * t_eft + self.hp_htg.c3_htg
            m_single_hp_htg = self.hp_htg.m_single_hp
        else:
            cap_htg = 0.0
            m_single_hp_htg = 0.0

        if self.cooling_exists:
            cap_clg = self.hp_clg.c1_clg * t_eft**2 + self.hp_clg.c2_clg * t_eft + self.hp_clg.c3_clg
            m_single_hp_clg = self.hp_clg.m_single_hp
        else:
            cap_clg = 0.0
            m_single_hp_clg = 0.0

        m_single_hp = max(m_single_hp_htg, m_single_hp_clg)

        q_i = self.q_net_htg[i]
        hp_capacity = cap_htg if q_i > 0 else cap_clg

        # compute mass flow rates
        mass_flow_bldg = np.abs(q_i) / hp_capacity * m_single_hp

        return mass_flow_bldg

    def calc_r1_r2(self, t_eft, hour_index):
        """
        Calculate r1 and r2 for this building based on entering fluid temperature and HP coefficients.
        """

        a = 0
        b = 0
        u = 0
        v = 0

        # Extract loads
        h = self.htg_vals[hour_index]
        c = self.clg_vals[hour_index]

        # Heating calculations
        if self.heating_exists:
            slope_htg = 2 * self.hp_htg.a_htg * t_eft + self.hp_htg.b_htg
            ratio_htg = self.hp_htg.a_htg * t_eft**2 + self.hp_htg.b_htg * t_eft + self.hp_htg.c_htg
            u = ratio_htg - slope_htg * t_eft
            v = slope_htg

        # Cooling calculations
        if self.cooling_exists:
            slope_clg = 2 * self.hp_clg.a_clg * t_eft + self.hp_clg.b_clg
            ratio_clg = self.hp_clg.a_clg * t_eft**2 + self.hp_clg.b_clg * t_eft + self.hp_clg.c_clg
            a = ratio_clg - slope_clg * t_eft
            b = slope_clg

        # Final arrays
        r1 = v * h - b * c
        r2 = u * h - a * c

        return r1, r2

    def generate_matrix_row(self, m_loop, idx_timestep):
        t_eft = self.t_eft[idx_timestep - 1]
        r1, r2 = self.calc_r1_r2(t_eft, idx_timestep)
        neighbour_index = self.downstream_device.row_index
        row = np.zeros(self.matrix_size)
        row[self.row_index] = 1 - r1 / (m_loop * self.cp)
        row[neighbour_index] = -1
        rhs = r2 / (m_loop * self.cp)
        return [row], [rhs]


class Node:
    def __init__(self, cells):
        self.input = None
        self.output = None
        self.diversion = None
        self.ID = str(cells[1])
        self.type = str(cells[2])


class DistPipe:
    def __init__(self, cells):
        self.node_in_name = None
        self.node_out_name = None
        self.input = None
        self.output = None
        self.ID = str(cells[1])
        self.type = str(cells[2])
        self.node_in_name = str(cells[3])
        self.node_out_name = str(cells[4])


class HPmodel:
    def __init__(self, hp_id: str, hp_data: dict):
        self.ID = hp_id

        self.a_htg = hp_data["heating"]["a"]
        self.b_htg = hp_data["heating"]["b"]
        self.c_htg = hp_data["heating"]["c"]

        self.a_clg = hp_data["cooling"]["a"]
        self.b_clg = hp_data["cooling"]["b"]
        self.c_clg = hp_data["cooling"]["c"]

        self.c1_htg = hp_data["heating"]["c1"]
        self.c2_htg = hp_data["heating"]["c2"]
        self.c3_htg = hp_data["heating"]["c3"]

        self.c1_clg = hp_data["cooling"]["c1"]
        self.c2_clg = hp_data["cooling"]["c2"]
        self.c3_clg = hp_data["cooling"]["c3"]

        self.m_single_hp = hp_data["design_flow"]
        self.m_design_htg_cap = hp_data["heating"]["design_cap"]
        self.m_design_clg_cap = hp_data["cooling"]["design_cap"]


class GHEHPSystem:
    def __init__(self, f_path_txt: Path, f_path_json: Path):
        self.GHXs = []
        self.buildings = []
        self.components = []
        self.nodes = []
        self.pipes = []
        self.nbh_total = None
        self.beta = 1.5
        self.matrix_size = 0

        with open(f_path_txt) as f1:
            txt_data = f1.readlines()

        json_data = json.loads(f_path_json.read_text())
        input_dir = f_path_json.resolve().parent

        fluid_data = json_data["fluid"]
        self.fluid = GHEFluid(
            fluid_str=fluid_data["fluid_name"],
            percent=fluid_data["concentration_percent"],
            temperature=fluid_data["temperature"],
        )

        topology_data = json_data["topology"]
        tg = json_data["ground_heat_exchanger"]["ghe1"]["soil"]["undisturbed_temp"]  # TODO: fix this

        heat_pump_data = json_data["heat_pump"]
        building_data = json_data["building"]
        for this_building_id, this_bldg_data in building_data.items():
            this_bldg = Building(this_building_id, this_bldg_data, heat_pump_data, input_dir, tg)
            self.buildings.append(this_bldg)
            self.components.append(this_bldg)

        ghe_data = json_data["ground_heat_exchanger"]
        for ghe_id, ghe_data in ghe_data.items():
            this_ghx = GHX(ghe_id, ghe_data, self.fluid)
            self.GHXs.append(this_ghx)
            self.components.append(this_ghx)

        for line in txt_data:  # loop over all the lines
            cells = [c.strip() for c in line.strip().split(",")]
            keyword = cells[0].lower()

            if keyword == "node":
                this_node = Node(cells)
                self.nodes.append(this_node)

            if keyword == "pipe":
                this_pipe = DistPipe(cells)
                self.pipes.append(this_pipe)

        self.update_connections()
        self.nbh_total = sum(this_ghx.n_rows * this_ghx.n_cols for this_ghx in self.GHXs)
        self.matrix_size = 4 * len(self.GHXs) + len(self.buildings)

        for this_ghx in self.GHXs:
            this_ghx.matrix_size = self.matrix_size
            this_ghx.split_ratio = this_ghx.nbh / self.nbh_total

        for this_bldg in self.buildings:
            this_bldg.matrix_size = self.matrix_size
            this_bldg.cp = self.GHXs[0].bhe.fluid.cp  # TODO: fix this

    def solve_system(self):
        # Assigning row_indices
        for k, this_bldg in enumerate(self.buildings):
            this_bldg.row_index = k

        for k, this_ghx in enumerate(self.GHXs):
            this_ghx.row_index = len(self.buildings) + k * 4

        for idx_timestep in range(1, N_TIMESTEPS):  # loop over all timestep
            matrix_rows = []
            matrix_rhs = []
            total_hp_flow = 0

            for this_bldg in self.buildings:
                t_eft = this_bldg.t_eft[idx_timestep - 1]
                m_bldg = this_bldg.calc_bldg_mass_flow_rate(t_eft, idx_timestep)
                total_hp_flow += m_bldg

            m_loop = total_hp_flow * self.beta

            for this_bldg in self.buildings:
                bldg_row, rhs = this_bldg.generate_matrix_row(m_loop, idx_timestep)
                matrix_rows.extend(bldg_row)
                matrix_rhs.extend(rhs)

            for idx_ghx, this_ghx in enumerate(self.GHXs):
                ghx_rows, rhs_values = this_ghx.generate_matrix_row(m_loop, idx_timestep)
                matrix_rows.extend(ghx_rows)
                matrix_rhs.extend(rhs_values)

            # Solve the system = A * X = B
            a_matrix = np.array(matrix_rows, dtype=float)
            b_vector = np.array(matrix_rhs, dtype=float)

            x_vector = np.linalg.solve(a_matrix, b_vector)

            for idx_ghx, this_bldg in enumerate(self.buildings):
                this_bldg.t_eft[idx_timestep] = x_vector[idx_ghx]

            x_ghe_vector = x_vector[len(self.buildings) :]

            for idx_ghx, this_ghx in enumerate(self.GHXs):
                base = 4 * idx_ghx
                this_ghx.t_eft[idx_timestep] = x_ghe_vector[base]
                this_ghx.t_mean[idx_timestep] = x_ghe_vector[base + 1]
                this_ghx.q_ghe[idx_timestep] = x_ghe_vector[base + 2]
                this_ghx.t_exit[idx_timestep] = x_ghe_vector[base + 3]

    def create_output(self, output_dir: Path):
        data_rows = []

        for i in range(N_TIMESTEPS):
            row = []
            for this_bldg in self.buildings:
                row.append(this_bldg.t_eft[i])

            for this_ghx in self.GHXs:
                row.append(this_ghx.t_eft[i])
                row.append(this_ghx.t_mean[i])
                row.append(this_ghx.q_ghe[i])
                row.append(this_ghx.t_exit[i])

            data_rows.append(row)

        # Step 2: Create column labels
        column_names = []

        for j, _ in enumerate(self.buildings):
            column_names.append(f"Building{j}_t_eft")

        for j, _ in enumerate(self.GHXs):
            column_names += [f"GHX{j}_t_eft", f"GHX{j}_t_mean", f"GHX{j}_q_ghe", f"GHX{j}_t_exit"]

        # Step 3: Create and save DataFrame
        output_data = pd.DataFrame(data_rows, columns=column_names)
        output_data.index.name = "Hour"
        output_data.to_csv(output_dir / "output_results.csv", float_format="%0.8f")

    def update_connections(self):
        for this_pipe in self.pipes:
            this_pipe.input = find_item_by_id(this_pipe.node_in_name, self.nodes)
            this_pipe.output = find_item_by_id(this_pipe.node_out_name, self.nodes)
            if this_pipe.type == "1way":
                this_pipe.input.output = this_pipe
                this_pipe.output.input = this_pipe
            else:
                this_pipe.input.diversion = this_pipe
                this_pipe.output.input = this_pipe

        for this_comp in self.buildings:
            this_comp.input = find_item_by_id(this_comp.nodeID, self.nodes)
            this_comp.input.output = this_comp

        for this_comp in self.GHXs:
            this_comp.input = find_item_by_id(this_comp.nodeID, self.nodes)
            this_comp.input.output = this_comp

        for this_comp in self.components:
            this_comp.input = find_item_by_id(this_comp.nodeID, self.nodes)
            this_comp.input.output = this_comp

        for this_comp in self.GHXs:
            # find the upstream device

            # find the first upstream mixing node
            device = this_comp.input
            while device.type != "mixing":
                device = device.input

            # find the second upstream mixing node
            device = device.input
            while device.type != "mixing":
                device = device.input

            # find the upstream device
            device = device.diversion
            while device.type != "GHX" and device.type != "building":
                device = device.output

            device.downstream_device = this_comp

        # find the upstream device

        for this_comp in self.buildings:
            # find the first upstream mixing node
            device = this_comp.input
            while device.type != "mixing":
                device = device.input

            # find the second upstream mixing node
            device = device.input
            while device.type != "mixing":
                device = device.input

            # find the upstream device
            device = device.diversion
            while device.type != "GHX" and device.type != "building":
                device = device.output

            device.downstream_device = this_comp

        for this_comp in self.components:
            # find the first upstream mixing node
            device = this_comp.input
            while device.type != "mixing":
                device = device.input

            # find the second upstream mixing node
            device = device.input
            while device.type != "mixing":
                device = device.input

            # find the upstream device
            device = device.diversion
            while device.type != "GHX" and device.type != "building":
                device = device.output

            device.downstream_device = this_comp


def find_item_by_id(obj_id, objectlist):
    # search a list of objects to find one with a particular ID
    # of course, the objects must have an "ID" member
    for item in objectlist:  # all objects in the list
        if obj_id.lower() == item.ID.lower():  # does it have the ID I am seeking?
            return item  # then return this one
    # next item
    return None  # couldn't find it
