import csv
import os
import re
import warnings
from datetime import datetime
from json import dumps
from math import floor
from pathlib import Path

from ghedesigner.borehole_heat_exchangers import CoaxialPipe, GHEDesignerBoreholeBase
from ghedesigner.design import AnyBisectionType
from ghedesigner.enums import TimestepType


class OutputManager:

    def __init__(self,
                 design: AnyBisectionType,
                 time: float,
                 project_name: str,
                 notes: str,
                 author: str,
                 model_name: str,
                 load_method: TimestepType,
                 allocated_width=100,
                 ):

        # this constructor should take all the args to build out a full output manager
        # then the client code can decide what to do -- just access data through functions?
        # write all the data to files in a directory?
        # make individual hidden worker functions to build out each part
        # but then add individual public functions to write specific files
        # have one routine to write all of them
        self.text_summary = self.get_summary_text(allocated_width, project_name, model_name, notes, author, time,
                                                  design, load_method)
        self.loading_data_rows = self.get_loading_data(design)
        self.borehole_location_data_rows = self.get_borehole_location_data(design)
        self.hourly_loading_data_rows = self.get_hourly_loading_data(design)
        self.g_function_data_rows = self.get_g_function_data(design)
        self.output_dict = self.get_summary_object(design, time, project_name, notes, author, model_name, load_method)

    def write_all_output_files(self, output_directory: Path, file_suffix: str = ""):
        output_directory.mkdir(exist_ok=True)
        (output_directory / f"SimulationSummary{file_suffix}.txt").write_text(self.text_summary)
        with open(os.path.join(output_directory, f"TimeDependentValues{file_suffix}.csv"), "w", newline="") as csv1OF:
            csv.writer(csv1OF).writerows(self.loading_data_rows)
        with open(os.path.join(output_directory, f"BoreFieldData{file_suffix}.csv"), "w", newline="") as f_csv:
            csv.writer(f_csv).writerows(self.borehole_location_data_rows)
        with open(os.path.join(output_directory, f"Loadings{file_suffix}.csv"), "w", newline="") as f_csv:
            csv.writer(f_csv).writerows(self.hourly_loading_data_rows)
        with open(os.path.join(output_directory, f"Gfunction{file_suffix}.csv"), "w", newline="") as f_csv:
            csv.writer(f_csv).writerows(self.g_function_data_rows)
        with open(str(output_directory / f"SimulationSummary{file_suffix}.json"), "w", newline="") as f_json:
            f_json.write(dumps(self.output_dict, indent=2))

    def get_loading_data(self, design):
        csv_array = [
            [
                "Time (hr)",
                "Time (month)",
                "Q (Rejection) (w) (before time)",
                "Q (Rejection) (W/m) (before time)",
                "Tb (C)",
                "GHE ExFT (C)",
            ]
        ]
        loading_values = design.ghe.loading
        for i, (tv, d_tb, lv) in enumerate(zip(design.ghe.times, design.ghe.dTb, loading_values)):
            if i + 1 < len(design.ghe.times):
                current_time = tv
                loading = loading_values[i + 1]
                current_month = self.hours_to_month(tv)
                normalized_loading = loading / (design.ghe.bhe.b.H * design.ghe.nbh)
                wall_temperature = design.ghe.bhe.soil.ugt + d_tb
                hp_eft_val = design.ghe.hp_eft[i]
                csv_row = list()
                csv_row.append(tv)
                csv_row.append(self.hours_to_month(tv))
                if i > 1:
                    csv_row.append(lv)
                    csv_row.append(lv / (design.ghe.bhe.b.H * design.ghe.nbh))
                else:
                    csv_row.append(0)
                    csv_row.append(0)
                csv_row.append(design.ghe.bhe.soil.ugt + design.ghe.dTb[i - 1])
                csv_row.append(design.ghe.hp_eft[i - 1])
                csv_array.append(csv_row)
            else:
                csv_row = list()
                csv_row.append(tv)
                csv_row.append(self.hours_to_month(tv))
                if i > 1:
                    csv_row.append(lv)
                    csv_row.append(lv / (design.ghe.bhe.b.H * design.ghe.nbh))
                else:
                    csv_row.append(0)
                    csv_row.append(0)
                csv_row.append(design.ghe.bhe.soil.ugt + design.ghe.dTb[i - 1])
                csv_row.append(design.ghe.hp_eft[i - 1])
                csv_array.append(csv_row)

                current_time = tv
                loading = 0
                current_month = self.hours_to_month(tv)
                normalized_loading = loading / (design.ghe.bhe.b.H * design.ghe.nbh)
                wall_temperature = design.ghe.bhe.soil.ugt + d_tb
                hp_eft_val = design.ghe.hp_eft[i]
            csv_row = list()
            csv_row.append(current_time)
            csv_row.append(current_month)
            csv_row.append(loading)
            csv_row.append(normalized_loading)
            csv_row.append(wall_temperature)
            csv_row.append(hp_eft_val)
            csv_array.append(csv_row)
        return csv_array

    @staticmethod
    def get_borehole_location_data(design):
        csv_array = list()
        csv_array.append(["x", "y"])
        for bL in design.ghe.gFunction.bore_locations:
            csv_array.append([bL[0], bL[1]])
        return csv_array

    def get_hourly_loading_data(self, design):
        hourly_loadings = design.ghe.hourly_extraction_ground_loads
        csv_array = list()
        csv_array.append(
            ["Month", "Day", "Hour", "Time (Hours)", "Loading (W) (Extraction)"]
        )
        for hour, hour_load in enumerate(hourly_loadings):
            month, day_in_month, hour_in_day = self.ghe_time_convert(hour)
            csv_array.append([month, day_in_month, hour_in_day, hour, hour_load])
        return csv_array

    @staticmethod
    def get_g_function_data(design):
        csv_array = [["ln(t/ts)", f"H:{design.ghe.bhe.b.H:0.2f}"]]
        ghe_gf_adjusted = design.ghe.grab_g_function(design.ghe.B_spacing / float(design.ghe.bhe.b.H))
        gfunction_log_vals = ghe_gf_adjusted.x
        gfunction_g_vals = ghe_gf_adjusted.y
        for log_val, g_val in zip(gfunction_log_vals, gfunction_g_vals):
            csv_array.append([log_val, g_val])
        return csv_array

    @staticmethod
    def get_timestep_str(load_method: TimestepType):
        if load_method == TimestepType.HYBRID:
            return TimestepType.HYBRID.name
        if load_method == TimestepType.HOURLY:
            return TimestepType.HOURLY.name
        warnings.warn("Load method not implemented")
        return ""

    def get_summary_object(self,
                           design: AnyBisectionType,
                           time: float,
                           project_name: str,
                           notes: str,
                           author: str,
                           model_name: str,
                           load_method: TimestepType) -> dict:
        # gFunction LTS Table
        g_function_col_titles = ["ln(t/ts)"]
        for g_function_name in list(design.ghe.gFunction.g_lts):
            g_function_col_titles.append("H:" + str(round(g_function_name, 0)) + "m")
        g_function_col_titles.append("H:" + str(round(design.ghe.bhe.b.H, 2)) + "m")
        g_function_data = []
        ghe_gf = design.ghe.gFunction.g_function_interpolation(float(design.ghe.B_spacing) / design.ghe.bhe.b.H)[0]
        for i in range(len(design.ghe.gFunction.log_time)):
            gf_row = list()
            gf_row.append(design.ghe.gFunction.log_time[i])
            for g_function_name in list(design.ghe.gFunction.g_lts):
                gf_row.append(design.ghe.gFunction.g_lts[g_function_name][i])
            gf_row.append(ghe_gf[i])
            g_function_data.append(gf_row)

        def add_with_units(val, units):
            return {'units': units, 'value': val}

        # these are dependent on the # pipes in each borehole, so precalculate
        if isinstance(design.ghe.bhe.pipe.r_out, float):
            pipe_geometry = {'pipe_outer_diameter': add_with_units(design.ghe.bhe.pipe.r_out * 2.0, 'm'),
                             'pipe_inner_diameter': add_with_units(design.ghe.bhe.pipe.r_in * 2.0, 'm')}
            reynolds = GHEDesignerBoreholeBase.compute_reynolds(design.ghe.bhe.m_flow_borehole,
                                                                design.ghe.bhe.pipe.r_in,
                                                                design.ghe.bhe.fluid)
        else:
            pipe_geometry = {
                'inner_pipe_inner_diameter': add_with_units(design.ghe.bhe.pipe.r_in[0] * 2.0, 'm'),
                'inner_pipe_outer_diameter': add_with_units(design.ghe.bhe.pipe.r_in[1] * 2.0, 'm'),
                'outer_pipe_inner_diameter': add_with_units(design.ghe.bhe.pipe.r_out[0] * 2.0, 'm'),
                'outer_pipe_outer_diameter': add_with_units(design.ghe.bhe.pipe.r_out[1] * 2.0, 'm'),
            }
            reynolds = CoaxialPipe.compute_reynolds_concentric(design.ghe.bhe.m_flow_borehole,
                                                               design.ghe.bhe.r_in_out,
                                                               design.ghe.bhe.r_out_in,
                                                               design.ghe.bhe.fluid)
        # build out the actual output dictionary
        output_dict = {
            'project_name': project_name,
            'notes': notes,
            'model_name': model_name,
            'simulation_time_stamp': datetime.now().strftime("%m/%d/%Y %H:%M:%S %p"),
            'simulation_author': author,
            'simulation_runtime': add_with_units(time, 's'),
            'design_selection_search_log': {
                'titles': ["Field", "Excess Temperature", "Max Temperature", "Min Temperature"],
                'units': [" ", "(C)", "(C)", "(C)"],
                'data': design.searchTracker
            },
            'ghe_system': {
                'search_log': {
                    'titles': g_function_col_titles,
                    'units': None,
                    'data': g_function_data
                },
                'active_borehole_length': add_with_units(design.ghe.bhe.b.H, 'm'),
                'borehole_diameter': add_with_units(design.ghe.bhe.b.r_b * 2.0, 'm'),
                'borehole_spacing': add_with_units(design.ghe.B_spacing, 'm'),
                'total_drilling': add_with_units(design.ghe.bhe.b.H * len(design.ghe.gFunction.bore_locations), 'm'),
                'field_type': design.ghe.fieldType,
                'field_specifier': design.ghe.fieldSpecifier,
                'number_of_boreholes': len(design.ghe.gFunction.bore_locations),
                'shank_spacing': add_with_units(design.ghe.bhe.pipe.s, 'm'),
                'pipe_geometry': pipe_geometry,
                'pipe_roughness': add_with_units(design.ghe.bhe.pipe.roughness, 'm'),
                'grout_thermal_conductivity': add_with_units(design.ghe.bhe.grout.k, 'W/mK'),
                'grout_volumetric_heat_capacity': add_with_units(design.ghe.bhe.grout.rhoCp, 'kJ/m3-K'),
                # TODO: Corrected arg to .rhoCp - verify, should be / 1000?
                'reynolds_number': reynolds,
                'effective_borehole_resistance': add_with_units(design.ghe.bhe.calc_effective_borehole_resistance(),
                                                                'W/m-K'),
                # TODO: are the units right here?
                'soil_thermal_conductivity': add_with_units(design.ghe.bhe.soil.k, 'W/m-K'),
                'soil_volumetric_heat_capacity': add_with_units(design.ghe.bhe.soil.rhoCp, 'kJ/m3-K'),
                # TODO: Should be / 1000?
                'soil_undisturbed_ground_temp': add_with_units(design.ghe.bhe.soil.ugt, 'C'),
                'fluid_volumetric_heat_capacity': add_with_units(design.ghe.bhe.fluid.rhoCp / 1000, 'kJ/m3-K'),
                'fluid_thermal_conductivity': add_with_units(design.ghe.bhe.fluid.k, 'W/mK'),
                'fluid_mixture': design.ghe.bhe.fluid.fluid.fluid_name,  # TODO: Is this the right lookup!?!?!? :)
                'fluid_density': add_with_units(design.ghe.bhe.fluid.rho, 'kg/m3'),
                'fluid_mass_flow_rate_per_borehole': add_with_units(design.ghe.bhe.m_flow_borehole, 'kg/s'),
            },
            'simulation_parameters': {
                'start_month': design.ghe.sim_params.start_month,
                'end_month': design.ghe.sim_params.end_month,
                'maximum_allowable_hp_eft': add_with_units(design.ghe.sim_params.max_EFT_allowable, 'C'),
                'minimum_allowable_hp_eft': add_with_units(design.ghe.sim_params.min_EFT_allowable, 'C'),
                'maximum_allowable_height': add_with_units(design.ghe.sim_params.max_height, 'm'),
                'minimum_allowable_height': add_with_units(design.ghe.sim_params.min_height, 'm'),
                'simulation_time': add_with_units(int(design.ghe.sim_params.end_month / 12), 'years'),
                'simulation_load_method': self.get_timestep_str(load_method)
            },
            'simulation_results': {

            }

        }

        # potentially add convection coefficient -- not sure why we wouldn't do it
        if hasattr(design.ghe.bhe, "h_f"):
            # TODO: Should be W/m2-K?
            output_dict['ghe_system']['fluid_convection_coefficient'] = add_with_units(design.ghe.bhe.h_f, 'W/m-K')

        # add monthly load summary
        monthly_load_values = []
        n_months = len(design.ghe.hybrid_load.monthly_cl) - 1
        n_years = int(n_months / 12)
        months = n_years * [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        start_ind = 1
        stop_ind = n_months
        for i in range(start_ind, stop_ind + 1):
            monthly_load_values.append(
                [
                    months[i - 1],
                    design.ghe.hybrid_load.monthly_hl[i],
                    design.ghe.hybrid_load.monthly_cl[i],
                    design.ghe.hybrid_load.monthly_peak_hl[i],
                    design.ghe.hybrid_load.monthly_peak_hl_duration[i],
                    design.ghe.hybrid_load.monthly_peak_cl[i],
                    design.ghe.hybrid_load.monthly_peak_cl_duration[i]]
            )
        output_dict['ghe_system']['glhe_monthly_loads'] = {
            'titles': [
                "Month",
                "Total Heating",
                "Total Cooling",
                "Peak Heating",
                "PH Duration",
                "Peak Cooling",
                "PC Duration",
            ],
            'units': ["", "kWh", "kWh", "kW", "hr", "kW", "hr"],
            'data': monthly_load_values
        }

        # add simulation results stuff
        n_years = 0
        out_array = []
        last_month = -1
        month_tb_vals = []
        month_eft_vals = []
        for tv, d_tb, eft in zip(design.ghe.times, design.ghe.dTb, design.ghe.hp_eft):
            current_month = floor(self.hours_to_month(tv))
            if current_month == last_month:
                month_tb_vals.append(d_tb)
                month_eft_vals.append(eft)
            elif current_month != last_month:
                if len(month_tb_vals) > 0:
                    if len(out_array) == 0:
                        previous_temp = design.ghe.bhe.soil.ugt
                    else:
                        previous_temp = design.ghe.bhe.soil.ugt
                    out_array.append(
                        [
                            current_month,
                            previous_temp + month_tb_vals[-1],
                            max(month_eft_vals),
                            min(month_eft_vals),
                        ]
                    )
                last_month = current_month
                month_tb_vals = [d_tb]
                month_eft_vals = [eft]
            if current_month % 11 == 0:
                n_years += 1
        max_eft = max(design.ghe.hp_eft)
        min_eft = min(design.ghe.hp_eft)
        max_eft_time = design.ghe.times[design.ghe.hp_eft.index(max(design.ghe.hp_eft))]
        min_eft_time = design.ghe.times[design.ghe.hp_eft.index(min(design.ghe.hp_eft))]
        max_eft_time = self.hours_to_month(max_eft_time)
        min_eft_time = self.hours_to_month(min_eft_time)
        output_dict['simulation_results'] = {
            'max_hp_eft': add_with_units(max_eft, 'C'),
            'max_hp_eft_time': add_with_units(max_eft_time, 'months'),
            'min_hp_eft': add_with_units(min_eft, 'C'),
            'min_hp_eft_time': add_with_units(min_eft_time, 'months'),
            'monthly_temp_summary': {
                'titles': ["Time", "Tbw", "Max hp_eft", "Min hp_eft"],
                'units': ["(months)", "(C)", "(C)", "(C)"],
                'data': out_array
            }
        }

        return output_dict

    def get_summary_text(self, width, project_name, model_name, notes, author, time, design, load_method):

        f_int = ".0f"
        f_1f = ".1f"
        f_2f = ".2f"
        f_3f = ".3f"
        f_4f = ".4f"
        f_str = "s"
        f_sci = ".3e"

        blank_line = self.create_line(width)
        empty_line = self.create_line(width, character=" ")

        o = blank_line
        o += self.d_row(width, "Project Name:", project_name, f_str)
        o += blank_line
        o += "Notes:\n\n" + notes + "\n"
        o += blank_line
        o += self.d_row(width, "File/Model Name:", model_name, f_str)
        now = datetime.now()
        time_string = now.strftime("%m/%d/%Y %H:%M:%S %p")
        o += self.d_row(width, "Simulated On:", time_string, f_str)
        o += self.d_row(width, "Simulated By:", author, f_str)
        o += self.d_row(width, "Calculation Time, s:", time, f_3f)
        o += empty_line
        o += self.create_title(width, "Design Selection", filler_symbol="-")

        design_header = [
            ["Field", "Excess Temperature", "Max Temperature", "Min Temperature"],
            [" ", "(C)", "(C)", "(C)"],
        ]
        try:
            design_values = design.searchTracker
        except:
            design_values = ""
        design_formats = [f_str, f_2f, f_2f, f_2f]

        o += self.create_table("Field Search Log", design_header, design_values, width, design_formats,
                               filler_symbol="-", centering="^")

        o += empty_line
        o += self.create_title(width, "GHE System", filler_symbol="-")

        # gFunction LTS Table
        g_function_table_formats = [f_3f]
        gf_table_ff = [f_3f] * (len(design.ghe.gFunction.g_lts) + 1)
        g_function_table_formats.extend(gf_table_ff)
        g_function_col_titles = ["ln(t/ts)"]

        for g_function_name in list(design.ghe.gFunction.g_lts):
            g_function_col_titles.append("H:" + str(round(g_function_name, 0)) + "m")
        g_function_col_titles.append("H:" + str(round(design.ghe.bhe.b.H, 2)) + "m")

        g_function_data = []
        ghe_gf = design.ghe.gFunction.g_function_interpolation(float(design.ghe.B_spacing) / design.ghe.bhe.b.H)[0]
        for i in range(len(design.ghe.gFunction.log_time)):
            gf_row = list()
            gf_row.append(design.ghe.gFunction.log_time[i])
            for g_function_name in list(design.ghe.gFunction.g_lts):
                gf_row.append(design.ghe.gFunction.g_lts[g_function_name][i])
            gf_row.append(ghe_gf[i])
            g_function_data.append(gf_row)

        o += self.create_table("gFunction LTS Values", [g_function_col_titles], g_function_data, width,
                               g_function_table_formats, filler_symbol="-", centering="^")
        o += empty_line

        o += self.create_title(width, "System Parameters", filler_symbol="-")
        o += self.d_row(width, "Active Borehole Length, m:", design.ghe.bhe.b.H, f_int)
        o += self.d_row(width, "Borehole Diameter, mm:", design.ghe.bhe.b.r_b * 1000 * 2.0, f_2f)
        o += self.d_row(width, "Borehole Spacing, m:", design.ghe.B_spacing, f_3f)
        o += self.d_row(width, "Total Drilling, m:", design.ghe.bhe.b.H * len(design.ghe.gFunction.bore_locations),
                        f_int)

        o += "Field Geometry: " + "\n"
        o += self.d_row(width, "Field Type:", design.ghe.fieldType, f_str, n_tabs=1)
        o += self.d_row(width, "Field Specifier:", design.ghe.fieldSpecifier, f_str, n_tabs=1)
        o += self.d_row(width, "NBH:", len(design.ghe.gFunction.bore_locations), f_int, n_tabs=1)

        o += "Borehole Information: " + "\n"

        if isinstance(design.ghe.bhe.pipe.r_out, float):
            o += self.d_row(width, "Pipe Outer Diameter, mm:", design.ghe.bhe.pipe.r_out * 1000 * 2.0, f_2f, n_tabs=1)
            o += self.d_row(width, "Pipe Inner Diameter, mm:", design.ghe.bhe.pipe.r_in * 1000 * 2.0, f_2f, n_tabs=1)
        else:
            o += self.d_row(width, "Outer Pipe Outer Diameter, mm:", design.ghe.bhe.pipe.r_out[1] * 1000 * 2.0, f_2f,
                            n_tabs=1)
            o += self.d_row(width, "Outer Pipe Inner Diameter, mm:", design.ghe.bhe.pipe.r_out[0] * 1000 * 2.0, f_2f,
                            n_tabs=1)
            o += self.d_row(width, "Inner Pipe Outer Diameter, mm:", design.ghe.bhe.pipe.r_in[1] * 1000 * 2.0, f_2f,
                            n_tabs=1)
            o += self.d_row(width, "Inner Pipe Inner Diameter, mm:", design.ghe.bhe.pipe.r_in[0] * 1000 * 2.0, f_2f,
                            n_tabs=1)

        o += self.d_row(width, "Pipe Roughness, m:", design.ghe.bhe.pipe.roughness, f_sci, n_tabs=1)
        o += self.d_row(width, "Shank Spacing, mm:", design.ghe.bhe.pipe.s * 1000, f_2f, n_tabs=1)
        o += self.d_row(width, "Grout Thermal Conductivity, W/(m-K):", design.ghe.bhe.grout.k, f_3f, n_tabs=1)
        o += self.d_row(width, "Grout Volumetric Heat Capacity, kJ/(K-m^3):", design.ghe.bhe.grout.rhoCp / 1000, f_2f,
                        n_tabs=1)
        if isinstance(design.ghe.bhe.pipe.r_out, float):
            o += self.d_row(width, "Reynold's Number:",
                            GHEDesignerBoreholeBase.compute_reynolds(design.ghe.bhe.m_flow_borehole,
                                                                     design.ghe.bhe.pipe.r_in,
                                                                     design.ghe.bhe.fluid),
                            f_int, n_tabs=1)
        else:

            o += self.d_row(width, "Reynold's Number:",
                            CoaxialPipe.compute_reynolds_concentric(design.ghe.bhe.m_flow_borehole,
                                                                    design.ghe.bhe.r_in_out,
                                                                    design.ghe.bhe.r_out_in,
                                                                    design.ghe.bhe.fluid),
                            f_int, n_tabs=1)

        o += self.d_row(width, "Effective Borehole Resistance, W/(m-K):",
                        design.ghe.bhe.calc_effective_borehole_resistance(),
                        f_4f, n_tabs=1)
        # Shank Spacing, Pipe Type, etc.

        o += "Soil Properties: " + "\n"
        o += self.d_row(width, "Thermal Conductivity, W/(m-K):", design.ghe.bhe.soil.k, f_3f, n_tabs=1)
        o += self.d_row(width, "Volumetric Heat Capacity, kJ/(K-m^3):", design.ghe.bhe.soil.rhoCp / 1000, f_2f,
                        n_tabs=1)
        o += self.d_row(width, "Undisturbed Ground Temperature, C:", design.ghe.bhe.soil.ugt, f_2f, n_tabs=1)

        o += "Fluid Properties" + "\n"
        o += self.d_row(width, "Volumetric Heat Capacity, kJ/(K-m^3):", design.ghe.bhe.fluid.rhoCp / 1000, f_2f,
                        n_tabs=1)
        o += self.d_row(width, "Thermal Conductivity, W/(m-K):", design.ghe.bhe.fluid.k, f_2f, n_tabs=1)
        o += self.d_row(width, "Fluid Mix:", design.ghe.bhe.fluid.fluid.fluid_name, f_str, n_tabs=1)
        o += self.d_row(width, "Density, kg/m^3:", design.ghe.bhe.fluid.rho, f_2f, n_tabs=1)
        o += self.d_row(width, "Mass Flow Rate Per Borehole, kg/s:", design.ghe.bhe.m_flow_borehole, f_3f, n_tabs=1)
        if hasattr(design.ghe.bhe, "h_f"):
            o += self.d_row(width, "Fluid Convection Coefficient, W/(m-K):", design.ghe.bhe.h_f, f_int, n_tabs=1)
        o += empty_line

        monthly_load_values = []
        n_months = len(design.ghe.hybrid_load.monthly_cl) - 1
        n_years = int(n_months / 12)
        months = n_years * [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]

        start_ind = 1
        stop_ind = n_months
        for i in range(start_ind, stop_ind + 1):
            monthly_load_values.append(
                [
                    months[i - 1],
                    design.ghe.hybrid_load.monthly_hl[i],
                    design.ghe.hybrid_load.monthly_cl[i],
                    design.ghe.hybrid_load.monthly_peak_hl[i],
                    design.ghe.hybrid_load.monthly_peak_hl_duration[i],
                    design.ghe.hybrid_load.monthly_peak_cl[i],
                    design.ghe.hybrid_load.monthly_peak_cl_duration[i]
                ]
            )
        month_header = [
            ["Month", "Total Heating", "Total Cooling", "Peak Heating", "PH Duration", "Peak Cooling", "PC Duration"],
            ["", "kWh", "kWh", "kW", "hr", "kW", "hr"],
        ]

        month_table_formats = [f_str, f_1f, f_1f, f_1f, f_1f, f_1f, f_1f]

        o += self.create_table("GLHE Monthly Loads", month_header, monthly_load_values, width, month_table_formats,
                               filler_symbol="-", centering="^")

        o += empty_line

        o += self.create_title(width, "Simulation Parameters")
        o += self.d_row(width, "Start Month: ", design.ghe.sim_params.start_month, f_int)
        o += self.d_row(width, "End Month: ", design.ghe.sim_params.end_month, f_int)
        o += self.d_row(width, "Maximum Allowable hp_eft, C: ", design.ghe.sim_params.max_EFT_allowable, f_2f)
        o += self.d_row(width, "Minimum Allowable hp_eft, C: ", design.ghe.sim_params.min_EFT_allowable, f_2f)
        o += self.d_row(width, "Maximum Allowable Height, m: ", design.ghe.sim_params.max_height, f_2f)
        o += self.d_row(width, "Minimum Allowable Height, m: ", design.ghe.sim_params.min_height, f_2f)
        o += self.d_row(width, "Simulation Time, years: ", int(design.ghe.sim_params.end_month / 12), f_int)
        load_method_string = self.get_timestep_str(load_method)
        o += self.d_row(width, "Simulation Loading Type: ", load_method_string, f_str)

        o += empty_line

        # Loading Stuff
        o += self.create_title(width, "Simulation Results")
        o += empty_line

        # Simulation Results
        eft_table_title = "Monthly Temperature Summary"
        n_years = 0
        out_array = []
        last_month = -1
        month_tb_vals = []
        month_eft_vals = []
        for tv, d_tb, eft in zip(design.ghe.times, design.ghe.dTb, design.ghe.hp_eft):
            # currentHourMonth = timeVals[i] - hTotalYear * nYears
            current_month = floor(self.hours_to_month(tv))
            # print(monthEFTVals)
            if current_month == last_month:
                month_tb_vals.append(d_tb)
                month_eft_vals.append(eft)
            elif current_month != last_month:
                if len(month_tb_vals) > 0:
                    if len(out_array) == 0:
                        previous_temp = design.ghe.bhe.soil.ugt
                    else:
                        # print(i)
                        # print(len(outArray))
                        # previousTemp = outArray[-1][1]
                        previous_temp = design.ghe.bhe.soil.ugt
                    out_array.append(
                        [
                            current_month,
                            previous_temp + month_tb_vals[-1],
                            max(month_eft_vals),
                            min(month_eft_vals),
                        ]
                    )
                last_month = current_month
                month_tb_vals = [d_tb]
                month_eft_vals = [eft]
            if current_month % 11 == 0:
                n_years += 1

        header_array = [
            ["Time", "Tbw", "Max hp_eft", "Min hp_eft"],
            ["(months)", "(C)", "(C)", "(C)"],
        ]
        eft_table_formats = [f_int, f_2f, f_2f, f_2f]

        o += self.create_title(width, "Peak Temperature", filler_symbol="-")
        max_eft = max(design.ghe.hp_eft)
        min_eft = min(design.ghe.hp_eft)
        max_eft_time = design.ghe.times[design.ghe.hp_eft.index(max(design.ghe.hp_eft))]
        min_eft_time = design.ghe.times[design.ghe.hp_eft.index(min(design.ghe.hp_eft))]
        max_eft_time = self.hours_to_month(max_eft_time)
        min_eft_time = self.hours_to_month(min_eft_time)
        o += self.d_row(width, "Max hp_eft, C:", max_eft, f_3f)
        o += self.d_row(width, "Max hp_eft Time, Months:", max_eft_time, f_3f)
        o += self.d_row(width, "Min hp_eft, C:", min_eft, f_3f)
        o += self.d_row(width, "Min hp_eft Time, Months:", min_eft_time, f_3f)

        o += self.create_table(eft_table_title, header_array, out_array, width, eft_table_formats, filler_symbol="-",
                               centering="^")

        # strip out all trailing whitespace
        o = re.sub(r"\s+\n", "\n", o)

        return o

    @staticmethod
    def create_title(allocated_width, title, filler_symbol=" "):
        return "{:{fS}^{L}s}\n".format(" " + title + " ", L=allocated_width, fS=filler_symbol)

    @staticmethod
    def create_row(allocated_width, row_data, data_formats, centering=">"):
        r_s = ""
        n_cols = len(row_data)
        col_width = int(allocated_width / n_cols)
        left_over = float(allocated_width) % col_width
        for d_f, data in zip(data_formats, row_data):
            width = col_width
            if left_over > 0:
                width = col_width + 1
                left_over -= 1
            try:
                r_s += "{:{c}{w}{fm}}".format(data, c=centering, w=width, fm=d_f)
            except:
                print("Output Row creation error: ", d_f)
                raise ValueError

        r_s += "\n"
        return r_s

    @staticmethod
    def create_table(title, col_titles, rows, allocated_width, col_formats, filler_symbol=" ", centering=">"):
        n_cols = len(col_titles[0])
        r_s = ""
        r_s += OutputManager.create_title(allocated_width, title, filler_symbol=filler_symbol)
        blank_line = OutputManager.create_line(allocated_width)
        r_s += blank_line
        header_format = ["s"] * n_cols
        for colT in col_titles:
            r_s += OutputManager.create_row(allocated_width, colT, header_format, centering="^")
        r_s += blank_line
        for row in rows:
            r_s += OutputManager.create_row(allocated_width, row, col_formats, centering=centering)
        r_s += blank_line
        return r_s

    @staticmethod
    def d_row(row_allocation: int, entry_1: str, entry_2, d_type: str, n_tabs: int = 0):
        tab_width = 4
        leading_spaces = n_tabs * tab_width

        l_str = f"{' ' * leading_spaces}{entry_1}"
        r_str = f"{entry_2:{d_type}}"

        l_chars_needed = len(l_str)
        r_chars_needed = len(r_str)
        c_spaces = row_allocation - l_chars_needed - r_chars_needed

        if c_spaces < 0:
            warnings.warn("Unable to write output string with specified formatting.")
            c_spaces = 4

        c_str = ' ' * c_spaces
        r_s = f"{l_str}{c_str}{r_str}\n"
        return r_s

    @staticmethod
    def create_line(row_allocation, character="*"):
        return character * row_allocation + "\n"

    @staticmethod
    def hours_to_month(hours):
        days_in_year = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        hours_in_year = [24 * x for x in days_in_year]
        n_years = floor(hours / sum(hours_in_year))
        frac_month = n_years * len(days_in_year)
        month_in_year = 0
        for idx, _ in enumerate(days_in_year):
            hours_left = hours - n_years * sum(hours_in_year)
            if sum(hours_in_year[0: idx + 1]) >= hours_left:
                month_in_year = idx
                break
        frac_month += month_in_year
        h_l = hours - n_years * sum(hours_in_year) - sum(hours_in_year[0:month_in_year])
        frac_month += h_l / (hours_in_year[month_in_year])
        return frac_month

    @staticmethod
    def ghe_time_convert(hours):
        days_in_year = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        hours_in_year = [24 * x for x in days_in_year]
        month_in_year = 0
        year_hour_sum = 0
        for idx, _ in enumerate(days_in_year):
            hours_left = hours
            if year_hour_sum + hours_in_year[idx] - 1 >= hours_left:
                month_in_year = idx
                break
            else:
                year_hour_sum += hours_in_year[idx]
        h_l = hours - sum(hours_in_year[0:month_in_year])
        day_in_month = floor(h_l / 24) + 1
        hour_in_day = h_l % 24 + 1
        return month_in_year + 1, day_in_month, hour_in_day
