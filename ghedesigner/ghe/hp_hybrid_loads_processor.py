from copy import deepcopy

import numpy as np
import pandas as pd

from ghedesigner.enums import BHType, SimCompType
from ghedesigner.ghe.boreholes.core import Borehole
from ghedesigner.ghe.boreholes.factory import get_bhe_object
from ghedesigner.ghe.ground_loads import HybridLoad
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Grout, Soil
from ghedesigner.utilities import HPmodel, get_loads


class Zone:
    def __init__(self):
        self.q_htg = None
        self.q_clg = None
        self.q_ext = None
        self.q_rej = None

        self.q_htg_hybrid = None
        self.q_clg_hybrid = None
        self.q_ext_hybrid = None
        self.q_rej_hybrid = None

        self.q_ext_common = None
        self.q_rej_common = None

        self.q_ext_hybrid_time_array = None
        self.q_rej_hybrid_time_array = None
        self.hybrid_time_array = None  # this is heat pump common time array and is same for all zones (heat pumps)

        self.COP_clg = None
        self.COP_htg = None
        self.q_htg_1yr = None
        self.q_clg_1yr = None
        self.name = None

    def initialize_load_arrays(self, n_years):
        h_full = np.tile(self.q_htg_1yr, n_years)
        c_full = np.tile(self.q_clg_1yr, n_years)

        self.q_htg = np.insert(h_full, 0, 0.0)
        self.q_clg = np.insert(c_full, 0, 0.0)

        return self.q_clg, self.q_htg

    def convert_hp_loads_to_ground_loads(self):
        self.q_rej = self.q_clg * (1 + 1 / self.COP_clg)
        self.q_ext = self.q_htg * (1 - 1 / self.COP_htg)

        return self.q_rej, self.q_ext

    def generate_hybrid_loads(self, bhe, radial_numerical, start_month, end_month):
        ext_obj = HybridLoad(
            raw_loads=self.q_ext[1:],
            bhe=bhe,
            radial_numerical=radial_numerical,
            start_month=start_month,
            end_month=end_month,
            peaks_every_year=False,
        )

        rej_obj = HybridLoad(
            raw_loads=-self.q_rej[1:],
            bhe=bhe,
            radial_numerical=radial_numerical,
            start_month=start_month,
            end_month=end_month,
            peaks_every_year=False,
        )

        self.q_ext_hybrid = ext_obj.load[2:] * 1000
        self.q_ext_hybrid_time_array = ext_obj.hour[2:]
        self.q_rej_hybrid = rej_obj.load[2:] * 1000
        self.q_rej_hybrid_time_array = rej_obj.hour[2:]

    def map_loads_to_common_time(self, common_time):

        q_ext_time = self.q_ext_hybrid_time_array
        q_rej_time = self.q_rej_hybrid_time_array

        idx_ext = np.searchsorted(q_ext_time, common_time, side="left")
        idx_ext = np.clip(idx_ext, 0, len(self.q_ext_hybrid) - 1)
        self.q_ext_common = self.q_ext_hybrid[idx_ext]

        idx_rej = np.searchsorted(q_rej_time, common_time, side="left")
        idx_rej = np.clip(idx_rej, 0, len(self.q_rej_hybrid) - 1)
        self.q_rej_common = self.q_rej_hybrid[idx_rej]

        if len(common_time) > 0 and common_time[0] == 0.0:
            self.q_ext_common[0] = 0.0
            self.q_rej_common[0] = 0.0

        return self.q_rej_common, self.q_ext_common

    def convert_ground_hybrid_loads_to_hp_loads(self, common_time):
        self.q_htg_hybrid = self.q_ext_common / (1 - 1 / self.COP_htg) * (-1)
        self.q_clg_hybrid = self.q_rej_common / (1 + 1 / self.COP_clg)
        self.hybrid_time_array = common_time


class ProcessLoads:
    def __init__(self):
        self.n_years = None
        self.zones = []

        self.fluid = None
        self.pipe = None
        self.grout = None
        self.soil = None
        self.borehole = None
        self.start_month = None
        self.end_month = None
        self.hp_data = None

        self.bhe = None
        self.bhe_eq = None

        self.mass_flow_rate = None
        self.flow_type = None

        self.common_time = None

    def read_data_from_json_file(self, json_data):
        self.data = json_data

        # Extract input values
        fluid_data = json_data["fluid"]
        first_ghe_key = next(iter(json_data["ground_heat_exchanger"]))
        soil_data = json_data["ground_heat_exchanger"][first_ghe_key]["soil"]
        grout_data = json_data["ground_heat_exchanger"][first_ghe_key]["grout"]
        pipe_data = json_data["ground_heat_exchanger"][first_ghe_key]["pipe"]
        borehole_data = json_data["ground_heat_exchanger"][first_ghe_key]["borehole"]
        ghe_data = json_data["ground_heat_exchanger"][first_ghe_key]
        sim_data = json_data["simulation_control"]
        self.hp_data = json_data.get("heat_pump", {})

        self.n_years = sim_data["simulation_years"]
        self.start_month = 1
        self.end_month = 12 * self.n_years

        # Construct objects
        self.fluid = Fluid(fluid_data["fluid_name"], fluid_data["concentration_percent"], fluid_data["temperature"])

        self.pipe = Pipe.init_single_u_tube(
            inner_diameter=pipe_data["inner_diameter"],
            outer_diameter=pipe_data["outer_diameter"],
            shank_spacing=pipe_data["shank_spacing"],
            roughness=pipe_data["roughness"],
            conductivity=pipe_data["conductivity"],
            rho_cp=pipe_data["rho_cp"],
        )
        self.soil = Soil(soil_data["conductivity"], soil_data["rho_cp"], soil_data["undisturbed_temp"])
        self.grout = Grout(grout_data["conductivity"], grout_data["rho_cp"])
        if "pre_designed" in ghe_data:
            self.borehole = Borehole(
                burial_depth=borehole_data["buried_depth"],
                borehole_radius=borehole_data["diameter"] / 2.0,
                borehole_height=ghe_data["pre_designed"]["H"],
            )
        elif "design" in ghe_data:
            self.borehole = Borehole(
                burial_depth=borehole_data["buried_depth"],
                borehole_radius=borehole_data["diameter"] / 2.0,
                borehole_height=ghe_data["design"]["max_height"],
            )
        else:
            raise ValueError("First GHE contains neither a pre-designed GHE or the definition to design one.")
        # mass flow rate
        self.mass_flow_rate = ghe_data["flow_rate"]
        self.flow_type = ghe_data["flow_type"]

        return self.fluid, self.pipe, self.grout, self.soil, self.borehole

    def read_hp_load_from_json(self, json_data, beta=0.5):
        building_data = json_data["building"]

        self.zones = []

        for bldg_id, bldg_data in building_data.items():
            zone = Zone()
            zone.name = bldg_id

            if "heating_load" in bldg_data:
                heating_load = bldg_data["heating_load"]
                zone.q_htg_1yr = np.array(
                    get_loads(
                        heating_load.get("heat_pump_name", f"{bldg_id}_heating"),
                        SimCompType.HEAT_PUMP.name,
                        heating_load,
                    ),
                    dtype=float,
                )

            if "cooling_load" in bldg_data:
                cooling_load = bldg_data["cooling_load"]
                zone.q_clg_1yr = np.array(
                    get_loads(
                        cooling_load.get("heat_pump_name", f"{bldg_id}_cooling"),
                        SimCompType.HEAT_PUMP.name,
                        cooling_load,
                    ),
                    dtype=float,
                )

            if zone.q_htg_1yr is None and zone.q_clg_1yr is None:
                raise ValueError(f"Building '{bldg_id}' has no heating or cooling load.")

            if zone.q_htg_1yr is None:
                zone.q_htg_1yr = np.zeros_like(zone.q_clg_1yr)

            if zone.q_clg_1yr is None:
                zone.q_clg_1yr = np.zeros_like(zone.q_htg_1yr)

            if "heating_load" in bldg_data and "heat_pump_cop" in bldg_data["heating_load"]:
                zone.COP_htg = float(bldg_data["heating_load"]["heat_pump_cop"])
            elif "heating_load" in bldg_data and "heat_pump_name" in bldg_data["heating_load"]:
                hp_htg_name = bldg_data["heating_load"]["heat_pump_name"]
                hp_htg_data = self.hp_data[hp_htg_name]
                hp_htg = HPmodel(hp_htg_name, hp_htg_data, self.soil.ugt)
                if "min_eft" in bldg_data:
                    heating_temp = (1 - beta) * bldg_data["min_eft"] + beta * self.soil.ugt
                else:
                    heating_temp = self.soil.ugt
                q_extr_ratio = hp_htg.a_htg * heating_temp * heating_temp + hp_htg.b_htg * heating_temp + hp_htg.c_htg
                zone.COP_htg = 1.0 / (1.0 - q_extr_ratio)
            elif np.any(zone.q_htg_1yr != 0):
                raise ValueError(f"Building '{bldg_id}' is missing 'heating_cop'.")
            else:
                zone.COP_htg = 1.0  # assigning harmless value

            if "cooling_load" in bldg_data and "heat_pump_cop" in bldg_data["cooling_load"]:
                zone.COP_clg = float(bldg_data["cooling_load"]["heat_pump_cop"])
            elif "cooling_load" in bldg_data and "heat_pump_name" in bldg_data["cooling_load"]:
                hp_clg_name = bldg_data["cooling_load"]["heat_pump_name"]
                hp_clg_data = self.hp_data[hp_clg_name]
                hp_clg = HPmodel(hp_clg_name, hp_clg_data, self.soil.ugt)
                if "max_eft" in bldg_data:
                    cooling_temp = (1 - beta) * bldg_data["max_eft"] + beta * self.soil.ugt
                else:
                    cooling_temp = self.soil.ugt
                q_rej_ratio = hp_clg.a_clg * cooling_temp * cooling_temp + hp_clg.b_clg * cooling_temp + hp_clg.c_clg
                zone.COP_clg = 1.0 / (q_rej_ratio - 1.0)
            elif np.any(zone.q_clg_1yr != 0):
                raise ValueError(f"Building '{bldg_id}' is missing 'cooling_cop'.")
            else:
                zone.COP_clg = 1.0  # assigning harmless value

            self.zones.append(zone)

    def prepare_bhe_for_hybrid(self):
        # example: this assumes these objects are already assigned
        borehole = deepcopy(self.borehole)

        bhe_type = BHType.SINGLEUTUBE
        mass_flow_borehole = self.mass_flow_rate

        self.bhe = get_bhe_object(
            bhe_type,
            mass_flow_borehole,
            self.fluid,
            borehole,
            self.pipe,
            self.grout,
            self.soil,
        )

        self.bhe_eq = self.bhe.to_single()
        self.bhe_eq.calc_sts_g_functions()

    def generate_hybrid_ground_loads(self):
        for zone in self.zones:
            zone.q_clg, zone.q_htg = zone.initialize_load_arrays(self.n_years)
            zone.q_rej, zone.q_ext = zone.convert_hp_loads_to_ground_loads()
            zone.generate_hybrid_loads(
                bhe=self.bhe_eq,
                radial_numerical=self.bhe_eq,
                start_month=self.start_month,
                end_month=self.end_month,
            )

    def generate_common_timegrid(self):
        all_times = np.concatenate(
            [zone.q_ext_hybrid_time_array for zone in self.zones]
            + [zone.q_rej_hybrid_time_array for zone in self.zones]
        )
        self.common_time = np.unique(all_times)
        self.common_time.sort()

        if len(self.common_time) == 0 or self.common_time[0] != 0.0:
            self.common_time = np.insert(self.common_time, 0, 0.0)

        return self.common_time

    def map_all_zones(self):
        for zone in self.zones:
            zone.map_loads_to_common_time(self.common_time)

    def create_hp_hybrid_loads(self):
        for zone in self.zones:
            zone.convert_ground_hybrid_loads_to_hp_loads(self.common_time)

    def get_hybrid_loads_for_district(self):
        data = {}
        for zone in self.zones:
            data[zone.name] = {
                "time": zone.hybrid_time_array,
                "q_htg": zone.q_htg_hybrid,
                "q_clg": zone.q_clg_hybrid,
            }
        return data

    def create_output_dataframe(self):
        df = pd.DataFrame()

        for zone in self.zones:
            df[f"{zone.name}_Time"] = zone.hybrid_time_array
            df[f"{zone.name}_q_htg"] = zone.q_htg_hybrid
            df[f"{zone.name}_q_clg"] = zone.q_clg_hybrid

        self.output_df = df
        return df

    def write_hybrid_output_csv(self, output_file="..\\ghedesigner\\ghe\\nbast_results\\hybrid_timesteps.csv"):
        if not hasattr(self, "output_df") or self.output_df is None:
            self.create_output_dataframe()

        df = self.output_df.copy()
        step_df = pd.DataFrame()

        for col in df.columns:
            values = df[col].to_numpy()

            repeated_values = []

            if "Time" in col:
                # time: [t0, t0, t1, t1, ...]
                for i in range(len(values)):
                    repeated_values.append(values[i])
                    repeated_values.append(values[i])

            else:
                # loads: [q0, q1, q1, q2, q2, q3, ...]
                for i in range(len(values) - 1):
                    repeated_values.append(values[i])
                    repeated_values.append(values[i + 1])

                # handle last value → repeat it
                repeated_values.append(values[-1])
                repeated_values.append(values[-1])

            step_df[col] = repeated_values

        step_df.to_csv(output_file, index=False)

    def run_hybrid_pipeline(self):
        self.prepare_bhe_for_hybrid()
        self.generate_hybrid_ground_loads()
        self.generate_common_timegrid()
        self.map_all_zones()
        self.create_hp_hybrid_loads()
        self.create_output_dataframe()
        # self.write_hybrid_output_csv()
        return self.get_hybrid_loads_for_district()
