import csv
from json import dumps
import os
from datetime import datetime
from math import floor
from pathlib import Path

from ghedesigner.borehole_heat_exchangers import GHEDesignerBoreholeBase
from ghedesigner.design import AnyBisectionType
from ghedesigner.utilities import DesignMethod


def hours_to_month(hours):
    days_in_year = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hours_in_year = 24 * days_in_year
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


def ghe_time_convert(hours):
    days_in_year = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    hours_in_year = 24 * days_in_year
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


def design_summary(
        design: AnyBisectionType,
        time: float,
        project_name: str,
        notes: str,
        author: str,
        model_name: str,
        load_method: DesignMethod) -> dict:

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

    # these are dependent on the # pipes in each borehole, so precalculate
    if isinstance(design.ghe.bhe.pipe.r_out, float):
        pipe_geometry = {'pipe_outer_radius': design.ghe.bhe.pipe.r_out, 'pipe_inner_radius': design.ghe.bhe.pipe.r_in}
        reynolds = GHEDesignerBoreholeBase.compute_reynolds(design.ghe.bhe.m_flow_borehole, design.ghe.bhe.pipe.r_in,
                                                            design.ghe.bhe.fluid)
    else:
        pipe_geometry = {
            'outer_pipe_outer_radius': design.ghe.bhe.pipe.r_out[0],
            'inner_pipe_outer_radius': design.ghe.bhe.pipe.r_out[1],
            'outer_pipe_inner_radius': design.ghe.bhe.pipe.r_in[0],
            'inner_pipe_inner_radius': design.ghe.bhe.pipe.r_in[1],
        }
        reynolds = GHEDesignerBoreholeBase.compute_reynolds_concentric(design.ghe.bhe.m_flow_pipe,
                                                                       design.ghe.bhe.r_in_out, design.ghe.bhe.r_out_in,
                                                                       design.ghe.bhe.fluid)
    # build out the actual output dictionary
    output_dict = {
        'project_name': project_name,
        'notes': notes,
        'model_name': model_name,
        'simulation_time_stamp': datetime.now().strftime("%m/%d/%Y %H:%M:%S %p"),
        'simulation_author': author,
        'simulation_runtime': {'units': 's', 'value': time},
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
            'active_borehole_length': design.ghe.bhe.b.H,
            'borehole_radius': design.ghe.bhe.b.r_b,
            'borehole_spacing': design.ghe.B_spacing,
            'total_drilling': design.ghe.bhe.b.H * len(design.ghe.gFunction.bore_locations),
            'field_type': design.ghe.fieldType,
            'field_specifier': design.ghe.fieldSpecifier,
            'number_of_boreholes': len(design.ghe.gFunction.bore_locations),
            'shank_spacing': design.ghe.bhe.pipe.s,
            'pipe_geometry': pipe_geometry,
            'pipe_roughness': design.ghe.bhe.pipe.roughness,
            'grout_thermal_conductivity': {'units': 'W/mK', 'value': design.ghe.bhe.grout.k},
            'grout_volumetric_heat_capacity': {'units': 'kJ/m3-K', 'value': design.ghe.bhe.grout.rhoCp},
            # TODO: Corrected arg to .rhoCp - verify, should be / 1000?
            'reynolds_number': reynolds,
            'effective_borehole_resistance': {'units': 'W/m-K',
                                              'value': design.ghe.bhe.calc_effective_borehole_resistance()},
            # TODO: are the units right here?
            'soil_thermal_conductivity': {'units': 'W/m-K', 'value': design.ghe.bhe.soil.k},
            'soil_volumetric_heat_capacity': {'units': 'kJ/m3-K', 'value': design.ghe.bhe.soil.rhoCp},
            # TODO: Should be / 1000?
            'soil_undisturbed_ground_temp': {'units': 'C', 'value': design.ghe.bhe.soil.ugt},
            'fluid_volumetric_heat_capacity': {'units': 'kJ/m3-K', 'value': design.ghe.bhe.fluid.rhoCp / 1000},
            'fluid_thermal_conductivity': {'units': 'W/mK', 'value': design.ghe.bhe.fluid.k},
            'fluid_mixture': design.ghe.bhe.fluid.fluid.fluid_name,  # TODO: Is this the right lookup!?!?!? :)
            'fluid_density': {'units': 'kg/m3', 'value': design.ghe.bhe.fluid.rho},
            'fluid_mass_flow_rate_per_borehole': {'units': 'kg/s', 'value': design.ghe.bhe.m_flow_borehole},
        },
        'simulation_parameters': {
            'start_month': design.ghe.sim_params.start_month,
            'end_month': design.ghe.sim_params.end_month,
            'maximum_allowable_hp_eft': {'units': 'C', 'value': design.ghe.sim_params.max_EFT_allowable},
            'minimum_allowable_hp_eft': {'units': 'C', 'value': design.ghe.sim_params.min_EFT_allowable},
            'maximum_allowable_height': {'units': 'm', 'value': design.ghe.sim_params.max_height},
            'minimum_allowable_height': {'units': 'm', 'value': design.ghe.sim_params.min_height},
            'simulation_time': {'units': 'years', 'value': int(design.ghe.sim_params.end_month / 12)},
            'simulation_load_method': "hybrid" if load_method == DesignMethod.Hybrid else "hourly"
        },
        'simulation_results': {

        }

    }

    # potentially add convection coefficient -- not sure why we wouldn't do it
    if hasattr(design.ghe.bhe, "h_f"):
        output_dict['ghe_system']['fluid_convection_coefficient'] = {
            'units': 'W/m-K', 'value': design.ghe.bhe.h_f
        }  # TODO: Should be W/m2-K?

    # add monthly load summary
    monthly_load_values = []
    m_cl = design.ghe.hybrid_load.monthly_cl
    m_hl = design.ghe.hybrid_load.monthly_hl
    p_cl = design.ghe.hybrid_load.monthly_peak_cl
    p_hl = design.ghe.hybrid_load.monthly_peak_hl
    d_cl = design.ghe.hybrid_load.monthly_peak_cl_duration
    d_hl = design.ghe.hybrid_load.monthly_peak_hl_duration
    n_months = len(design.ghe.hybrid_load.monthly_cl) - 1
    n_years = int(n_months / 12)
    months = n_years * [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    start_ind = 1
    stop_ind = n_months
    for i in range(start_ind, stop_ind + 1):
        monthly_load_values.append(
            [months[i - 1], m_hl[i], m_cl[i], p_hl[i], d_hl[i], p_cl[i], d_cl[i]]
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
        'units': ["", "KW-Hr", "KW-Hr", "KW", "hr", "KW", "hr"],
        'data': monthly_load_values
    }

    # add simulation results stuff
    n_years = 0
    out_array = []
    last_month = -1
    month_tb_vals = []
    month_eft_vals = []
    for tv, d_tb, eft in zip(design.ghe.times, design.ghe.dTb, design.ghe.hp_eft):
        current_month = floor(hours_to_month(tv))
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
    max_eft_time = hours_to_month(max_eft_time)
    min_eft_time = hours_to_month(min_eft_time)
    output_dict['simulation_results'] = {
        'max_hp_eft': {'units': 'C', 'value': max_eft},
        'max_hp_eft_time': {'units': 'months', 'value': max_eft_time},
        'min_hp_eft': {'units': 'C', 'value': min_eft},
        'min_hp_eft_time': {'units': 'months', 'value': min_eft_time},
        'monthly_temp_summary': {
            'titles': ["Time", "Tbw", "Max hp_eft", "Min hp_eft"],
            'units': ["(months)", "(C)", "(C)", "(C)"],
            'data': out_array
        }
    }

    return output_dict


def write_design_details(
        design: AnyBisectionType,
        time: float,
        project_name: str,
        notes: str,
        author: str,
        model_name: str,
        load_method: DesignMethod,
        output_directory: Path,
        file_suffix: str = ""
) -> None:
    # write design summary json file
    output_dict = design_summary(
        design,
        time,
        project_name,
        notes,
        author,
        model_name,
        load_method
    )
    output_directory.mkdir(exist_ok=True)
    with open(str(output_directory / f"SimulationSummary{file_suffix}.json"), "w", newline="") as txtF:
        txtF.write(dumps(output_dict, indent=2))

    # write time dependent values csv data
    csv1_array = []
    for i, (tv, d_tb, lv) in enumerate(zip(design.ghe.times, design.ghe.dTb, design.ghe.loading)):
        if i + 1 < len(design.ghe.times):
            current_time = tv
            loading = design.ghe.loading[i + 1]
            current_month = hours_to_month(tv)
            normalized_loading = loading / (design.ghe.bhe.b.H * design.ghe.nbh)
            wall_temperature = design.ghe.bhe.soil.ugt + d_tb
            hp_eft_val = design.ghe.hp_eft[i]
            csv1_row = list()
            csv1_row.append(tv)
            csv1_row.append(hours_to_month(tv))
            if i > 1:
                csv1_row.append(lv)
                csv1_row.append(lv / (design.ghe.bhe.b.H * design.ghe.nbh))
            else:
                csv1_row.append(0)
                csv1_row.append(0)
            csv1_row.append(design.ghe.bhe.soil.ugt + design.ghe.dTb[i - 1])
            csv1_row.append(design.ghe.hp_eft[i - 1])
            csv1_array.append(csv1_row)
        else:
            csv1_row = list()
            csv1_row.append(tv)
            csv1_row.append(hours_to_month(tv))
            if i > 1:
                csv1_row.append(lv)
                csv1_row.append(lv / (design.ghe.bhe.b.H * design.ghe.nbh))
            else:
                csv1_row.append(0)
                csv1_row.append(0)
            csv1_row.append(design.ghe.bhe.soil.ugt + design.ghe.dTb[i - 1])
            csv1_row.append(design.ghe.hp_eft[i - 1])
            csv1_array.append(csv1_row)
            current_time = tv
            loading = 0
            current_month = hours_to_month(tv)
            normalized_loading = loading / (design.ghe.bhe.b.H * design.ghe.nbh)
            wall_temperature = design.ghe.bhe.soil.ugt + d_tb
            hp_eft_val = design.ghe.hp_eft[i]
        csv1_row = list()
        csv1_row.append(current_time)
        csv1_row.append(current_month)
        csv1_row.append(loading)
        csv1_row.append(normalized_loading)
        csv1_row.append(wall_temperature)
        csv1_row.append(hp_eft_val)
        csv1_array.append(csv1_row)
    with open(os.path.join(output_directory, f"TimeDependentValues{file_suffix}.csv"), "w", newline="") as csv1OF:
        c_w = csv.writer(csv1OF)
        c_w.writerow(
            [
                "Time (hr)",
                "Time (month)",
                "Q (Rejection) (w) (before time)",
                "Q (Rejection) (W/m) (before time)",
                "Tb (C)",
                "GHE ExFT (C)",
            ]
        )
        c_w.writerows(csv1_array)

    # write borehole field data csv
    csv2_array = [["x", "y"], ]
    for bL in design.ghe.gFunction.bore_locations:
        csv2_array.append([bL[0], bL[1]])
    with open(os.path.join(output_directory, f"BoreFieldData{file_suffix}.csv"), "w", newline="") as csv2OF:
        c_w = csv.writer(csv2OF)
        c_w.writerows(csv2_array)

    # write hourly loading data csv
    hourly_loadings = design.ghe.hourly_extraction_ground_loads
    csv3_array = [["Month", "Day", "Hour", "Time (Hours)", "Loading (W) (Extraction)"], ]
    for hour, hour_load in enumerate(hourly_loadings):
        month, day_in_month, hour_in_day = ghe_time_convert(hour)
        csv3_array.append([month, day_in_month, hour_in_day, hour, hour_load])
    with open(os.path.join(output_directory, f"Loadings{file_suffix}.csv"), "w", newline="") as csv3OF:
        c_w = csv.writer(csv3OF)
        c_w.writerows(csv3_array)

    # write g-function csv
    csv4_array = [["ln(t/ts)", f"H:{design.ghe.bhe.b.H:0.2f}"], ]
    ghe_gf_adjusted = design.ghe.grab_g_function(design.ghe.B_spacing / float(design.ghe.bhe.b.H))
    gfunction_log_vals = ghe_gf_adjusted.x
    gfunction_g_vals = ghe_gf_adjusted.y
    for log_val, g_val in zip(gfunction_log_vals, gfunction_g_vals):
        gf_row = list()
        gf_row.append(log_val)
        gf_row.append(g_val)
        csv4_array.append(gf_row)
    with open(os.path.join(output_directory, f"Gfunction{file_suffix}.csv"), "w", newline="") as csv4OF:
        c_w = csv.writer(csv4OF)
        c_w.writerows(csv4_array)
