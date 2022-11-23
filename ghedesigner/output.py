import csv
import os
from datetime import datetime
from pathlib import Path

import math
import numpy as np

from ghedesigner.borehole_heat_exchangers import GHEDesignerBoreholeBase
from ghedesigner.utilities import DesignMethod


def create_title(allocated_width, title, filler_symbol=" "):
    return "{:{fS}^{L}s}\n".format(" " + title + " ", L=allocated_width, fS=filler_symbol)


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


def create_table(
        title, col_titles, rows, allocated_width, col_formats, filler_symbol=" ", centering=">"
):
    n_cols = len(col_titles[0])
    r_s = ""
    r_s += create_title(allocated_width, title, filler_symbol=filler_symbol)
    blank_line = create_line(allocated_width)
    r_s += blank_line
    header_format = ["s"] * n_cols
    for colT in col_titles:
        r_s += create_row(allocated_width, colT, header_format, centering="^")
    r_s += blank_line
    for row in rows:
        r_s += create_row(allocated_width, row, col_formats, centering=centering)
    r_s += blank_line
    return r_s


def create_d_row(row_allocation, entry_1, entry_2, d_type_1, d_type_2, b_tabs=0, a_tabs=0):
    tab_width = 8
    tab_offset = 0.5 * tab_width
    n_tabs = b_tabs + a_tabs
    initial_ratio = 0.5
    # reducedAllocation = rowAllocation-nTabs*tabWidth
    right_offset = initial_ratio * row_allocation
    left_offset = (1 - initial_ratio) * row_allocation

    right_offset = int(right_offset - n_tabs * tab_offset)
    left_offset = int(left_offset - tab_offset * n_tabs)
    if (right_offset + left_offset + n_tabs * tab_width) != row_allocation:
        right_offset += 1

    l_needed = len(str(entry_1))
    r_needed = len(str(entry_2))

    if (l_needed + r_needed) > (right_offset + left_offset):
        print("Allocation: ", row_allocation)
        print("Characters Needed: ", (l_needed + r_needed + n_tabs * tab_width))
        print("Allocated: ", (right_offset + left_offset + tab_width * n_tabs))
        print("Right Offset Allocated: ", right_offset)
        print("Left Offset Allocated: ", left_offset)
        print("Tab Space: ", tab_width * n_tabs)
        raise Exception("Not Enough Width Was Provided")
    if l_needed > left_offset:
        swing = l_needed - left_offset
        left_offset += swing
        right_offset -= swing
    if r_needed > right_offset:
        swing = r_needed - right_offset
        right_offset += swing
        left_offset -= swing
    if (right_offset + left_offset + tab_width * n_tabs) != row_allocation:
        print("Allocation: ", row_allocation)
        print("Allocated: ", (right_offset + left_offset + tab_width * n_tabs))
        print("Right Offset Allocated: ", right_offset)
        print("Left Offset Allocated: ", left_offset)
        print("Tab Space: ", tab_width * n_tabs)
        raise Exception("Width Allocation Error")
    r_s = ""
    for t in range(b_tabs):
        r_s += "\t"
    r_s += "{:<{lO}{f1}}{:>{rO}{f2}}".format(entry_1, entry_2, lO=left_offset, rO=right_offset, f1=d_type_1,
                                             f2=d_type_2)
    for t in range(a_tabs):
        r_s += "\t"
    r_s += "\n"
    return r_s


def create_line(row_allocation, character="*"):
    return character * row_allocation + "\n"


def hours_to_month(hours):
    days_in_year = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hours_in_year = 24 * days_in_year
    n_years = math.floor(hours / np.sum(hours_in_year))
    frac_month = n_years * len(days_in_year)
    month_in_year = 0
    for idx, _ in enumerate(days_in_year):
        hours_left = hours - n_years * np.sum(hours_in_year)
        if np.sum(hours_in_year[0: idx + 1]) >= hours_left:
            month_in_year = idx
            break
    # print("Year Months: ",fracMonth)
    # print("Month Months: ",monthInYear)
    frac_month += month_in_year
    h_l = hours - n_years * np.sum(hours_in_year) - np.sum(hours_in_year[0:month_in_year])
    frac_month += h_l / (hours_in_year[month_in_year])
    # print("Hour Months: ",hL/(hoursInYear[monthInYear]))
    # print(fracMonth)
    return frac_month


def ghe_time_convert(hours):
    days_in_year = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
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
    # print("Year Months: ",fracMonth)
    # print("Month Months: ",monthInYear)
    h_l = hours - np.sum(hours_in_year[0:month_in_year])
    day_in_month = int(math.floor(h_l / 24)) + 1
    hour_in_day = h_l % 24 + 1
    # print("Hour Months: ",hL/(hoursInYear[monthInYear]))
    # print(fracMonth)
    return month_in_year + 1, day_in_month, hour_in_day


def output_design_details(
        design,
        time,
        project_name,
        notes,
        author,
        model_name,
        load_method: DesignMethod,
        output_directory: Path,
        allocated_width=100,
        rounding_amount=10,
        summary_file="SimulationSummary.txt",
        csv_f_1="TimeDependentValues.csv",
        csv_f_2="BoreFieldData.csv",
        csv_f_3="Loadings.csv",
        csv_f_4="Gfunction.csv",

):
    try:
        ghe = design.ghe
    except:
        ghe = design
    bhe = ghe.bhe
    g_function = ghe.gFunction
    b_h = bhe.b
    b = g_function.bore_locations

    float_format = ".3f"
    string_format = "s"
    int_format = ".0f"
    # roundingAmount = 10
    sci_format = ".3e"

    blank_line = create_line(allocated_width)
    empty_line = create_line(allocated_width, character=" ")
    o_s = ""
    # oS += middleSpacingString.format("Project Name:",projectName,rO=rightOffset,lO=leftColLength) + "\n"
    o_s += create_d_row(allocated_width, "Project Name:", project_name, string_format, string_format)
    o_s += blank_line
    o_s += "Notes:\n\n" + notes + "\n"
    o_s += blank_line
    o_s += create_d_row(allocated_width, "File/Model Name:", model_name, string_format, string_format)
    now = datetime.now()
    time_string = now.strftime("%m/%d/%Y %H:%M:%S %p")
    o_s += create_d_row(allocated_width, "Simulated On:", time_string, string_format, string_format)
    o_s += create_d_row(allocated_width, "Simulated By:", author, string_format, string_format)
    o_s += create_d_row(allocated_width, "Calculation Time, s:", round(time, rounding_amount), string_format,
                        float_format, )
    o_s += empty_line
    o_s += create_title(allocated_width, "Design Selection", filler_symbol="-")

    design_header = [
        ["Field", "Excess Temperature", "Max Temperature", "Min Temperature"],
        [" ", "(C)", "(C)", "(C)"],
    ]
    try:
        design_values = design.searchTracker
    except:
        design_values = ""
    design_formats = ["s", ".3f", ".3f", ".3f"]

    o_s += create_table("Field Search Log", design_header, design_values, allocated_width, design_formats,
                        filler_symbol="-", centering="^", )

    o_s += empty_line
    o_s += create_title(allocated_width, "GHE System", filler_symbol="-")

    # gFunction LTS Table
    g_function_table_formats = [".3f"]
    gf_table_ff = [".3f"] * (len(g_function.g_lts) + 1)
    g_function_table_formats.extend(gf_table_ff)
    g_function_col_titles = ["ln(t/ts)"]

    for g_function_name in list(g_function.g_lts):
        g_function_col_titles.append("H:" + str(round(g_function_name, 0)) + "m")
    g_function_col_titles.append("H:" + str(round(b_h.H, 2)) + "m")

    g_function_data = []
    ghe_gf = g_function.g_function_interpolation(float(ghe.B_spacing) / b_h.H)[0]
    for i in range(len(g_function.log_time)):
        gf_row = list()
        gf_row.append(g_function.log_time[i])
        for g_function_name in list(g_function.g_lts):
            # print(gfunction.g_lts[gfunctionName][i])
            gf_row.append(g_function.g_lts[g_function_name][i])
        gf_row.append(ghe_gf[i])
        g_function_data.append(gf_row)

    o_s += create_table("gFunction LTS Values", [g_function_col_titles], g_function_data, allocated_width,
                        g_function_table_formats, filler_symbol="-", centering="^")
    o_s += empty_line

    """

    """

    o_s += "------ System parameters ------" + "\n"
    o_s += create_d_row(allocated_width, "Active Borehole Length, m:", b_h.H, string_format, int_format)
    o_s += create_d_row(allocated_width, "Borehole Radius, m:", round(b_h.r_b, rounding_amount), string_format,
                        float_format)
    o_s += create_d_row(allocated_width, "Borehole Spacing, m:", round(ghe.B_spacing, rounding_amount), string_format,
                        float_format)
    o_s += create_d_row(allocated_width, "Total Drilling, m:", round(b_h.H * len(b), rounding_amount), string_format,
                        float_format)

    indented_amount = 2

    o_s += "Field Geometry: " + "\n"
    # rightAd = rightOffset-indentedAmount*tabOffset+math.ceil(indentedAmount/2)
    # leftAd = leftColLength-tabOffset*indentedAmount+math.floor(indentedAmount/2)
    o_s += create_d_row(allocated_width, "Field Type:", ghe.fieldType, string_format, string_format,
                        b_tabs=indented_amount)
    # oS += middleSpacingIndentedString.format("\t\tField Type:",ghe.field_type,rO=rightAd,lO=leftAd)
    o_s += create_d_row(
        allocated_width,
        "Field Specifier:",
        ghe.fieldSpecifier,
        string_format,
        string_format,
        b_tabs=indented_amount,
    )
    # oS += middleSpacingIndentedString.format("\t\tField Specifier:",ghe.fieldSpecifier,rO=rightAd,lO=leftAd)
    o_s += create_d_row(allocated_width, "NBH:", len(b), string_format, int_format, b_tabs=indented_amount)
    # oS += middleSpacingIndentedString.format("\t\tNBH:",len(b),rO=rightAd,lO=leftAd)
    # Field NBH Borehole locations, field identification
    # System Details

    o_s += "Borehole Information: " + "\n"
    o_s += create_d_row(allocated_width, "Shank Spacing, m:", round(bhe.pipe.s, rounding_amount), string_format,
                        float_format, b_tabs=indented_amount)

    if isinstance(bhe.pipe.r_out, float):
        o_s += create_d_row(
            allocated_width,
            "Pipe Outer Radius, m:",
            round(bhe.pipe.r_out, rounding_amount),
            string_format,
            float_format,
            b_tabs=indented_amount,
        )
        o_s += create_d_row(
            allocated_width,
            "Pipe Inner Radius, m:",
            round(bhe.pipe.r_in, rounding_amount),
            string_format,
            float_format,
            b_tabs=indented_amount,
        )
    else:
        o_s += create_d_row(
            allocated_width,
            "Outer Pipe Outer Radius, m:",
            round(bhe.pipe.r_out[0], rounding_amount),
            string_format,
            float_format,
            b_tabs=indented_amount,
        )
        o_s += create_d_row(
            allocated_width,
            "Inner Pipe Outer Pipe Outer Radius, m:",
            round(bhe.pipe.r_out[1], rounding_amount),
            string_format,
            float_format,
            b_tabs=indented_amount,
        )
        o_s += create_d_row(
            allocated_width,
            "Outer Pipe Inner Radius, m:",
            round(bhe.pipe.r_in[0], rounding_amount),
            string_format,
            float_format,
            b_tabs=indented_amount,
        )
        o_s += create_d_row(
            allocated_width,
            "Inner Pipe Inner Radius, m:",
            round(bhe.pipe.r_in[1], rounding_amount),
            string_format,
            float_format,
            b_tabs=indented_amount,
        )

    o_s += create_d_row(
        allocated_width,
        "Pipe Roughness, m:",
        round(bhe.pipe.roughness, rounding_amount),
        string_format,
        sci_format,
        b_tabs=indented_amount,
    )
    o_s += create_d_row(
        allocated_width,
        "Shank Spacing, m:",
        round(bhe.pipe.s, rounding_amount),
        string_format,
        float_format,
        b_tabs=indented_amount,
    )
    o_s += create_d_row(
        allocated_width,
        "Grout Thermal Conductivity, W/(m*K):",
        round(bhe.grout.k, rounding_amount),
        string_format,
        float_format,
        b_tabs=indented_amount,
    )
    o_s += create_d_row(
        allocated_width,
        "Grout Volumetric Heat Capacity, kJ/(K*m^3):",
        round(bhe.pipe.s / 1000, rounding_amount),
        string_format,
        float_format,
        b_tabs=indented_amount,
    )
    if isinstance(bhe.pipe.r_out, float):
        o_s += create_d_row(
            allocated_width,
            "Reynold's Number:",
            round(GHEDesignerBoreholeBase.compute_reynolds(bhe.m_flow_borehole, bhe.pipe.r_in, bhe.fluid),
                  rounding_amount),
            string_format,
            float_format,
            b_tabs=indented_amount,
        )
    else:

        o_s += create_d_row(
            allocated_width,
            "Reynold's Number:",
            round(GHEDesignerBoreholeBase.compute_reynolds_concentric(bhe.m_flow_pipe, bhe.r_in_out, bhe.r_out_in,
                                                                      bhe.fluid), rounding_amount),
            string_format,
            float_format,
            b_tabs=indented_amount,
        )

    o_s += create_d_row(
        allocated_width,
        "Effective Borehole Resistance, W/(m*K):",
        round(bhe.calc_effective_borehole_resistance(), rounding_amount),
        string_format,
        float_format,
        b_tabs=indented_amount,
    )
    # Shank Spacing, Pipe Type, etc.

    o_s += "Soil Properties: " + "\n"
    o_s += create_d_row(
        allocated_width,
        "Thermal Conductivity, W/(m*K):",
        round(bhe.soil.k, rounding_amount),
        string_format,
        float_format,
        b_tabs=indented_amount,
    )
    o_s += create_d_row(
        allocated_width,
        "Volumetric Heat Capacity, kJ/(K*m^3):",
        round(bhe.soil.rhoCp / 1000, rounding_amount),
        string_format,
        float_format,
        b_tabs=indented_amount,
    )
    o_s += create_d_row(
        allocated_width,
        "Undisturbed Ground Temperature, C:",
        round(bhe.soil.ugt, rounding_amount),
        string_format,
        float_format,
        b_tabs=indented_amount,
    )

    o_s += "Fluid Properties" + "\n"
    o_s += create_d_row(
        allocated_width,
        "Volumetric Heat Capacity, kJ/(K*m^3):",
        round(bhe.fluid.rhoCp / 1000, rounding_amount),
        string_format,
        float_format,
        b_tabs=indented_amount,
    )
    o_s += create_d_row(
        allocated_width,
        "Thermal Conductivity, W/(m*K):",
        round(bhe.fluid.k, rounding_amount),
        string_format,
        float_format,
        b_tabs=indented_amount,
    )
    o_s += create_d_row(
        allocated_width,
        "Fluid Mix:",
        bhe.fluid.fluid.fluid_name,
        string_format,
        string_format,
        b_tabs=indented_amount,
    )
    o_s += create_d_row(
        allocated_width,
        "Density, kg/m^3:",
        round(bhe.fluid.rho, rounding_amount),
        string_format,
        float_format,
        b_tabs=indented_amount,
    )
    o_s += create_d_row(
        allocated_width,
        "Mass Flow Rate Per Borehole, kg/s:",
        round(bhe.m_flow_borehole, rounding_amount),
        string_format,
        float_format,
        b_tabs=indented_amount,
    )
    if hasattr(bhe, "h_f"):
        o_s += create_d_row(
            allocated_width,
            "Fluid Convection Coefficient, W/(m*K):",
            round(bhe.h_f, rounding_amount),
            string_format,
            float_format,
            b_tabs=indented_amount,
        )
    o_s += empty_line

    monthly_load_values = []
    m_cl = ghe.hybrid_load.monthly_cl
    m_hl = ghe.hybrid_load.monthly_hl
    p_cl = ghe.hybrid_load.monthly_peak_cl
    p_hl = ghe.hybrid_load.monthly_peak_hl
    d_cl = ghe.hybrid_load.monthly_peak_cl_duration
    d_hl = ghe.hybrid_load.monthly_peak_hl_duration
    n_months = len(ghe.hybrid_load.monthly_cl) - 1
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
    month_header = [
        [
            "Month",
            "Total Heating",
            "Total Cooling",
            "Peak Heating",
            "PH Duration",
            "Peak Cooling",
            "PC Duration",
        ],
        ["", "KW-Hr", "KW-Hr", "KW", "hr", "KW", "hr"],
    ]

    month_table_formats = ["s", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f"]

    o_s += create_table(
        "GLHE Monthly Loads",
        month_header,
        monthly_load_values,
        allocated_width,
        month_table_formats,
        filler_symbol="-",
        centering="^",
    )

    o_s += empty_line

    o_s += create_title(allocated_width, "Simulation Parameters")
    o_s += create_d_row(
        allocated_width,
        "Start Month: ",
        ghe.sim_params.start_month,
        string_format,
        int_format,
    )
    o_s += create_d_row(
        allocated_width, "End Month: ", ghe.sim_params.end_month, string_format, int_format
    )
    o_s += create_d_row(
        allocated_width,
        "Maximum Allowable hp_eft, C: ",
        ghe.sim_params.max_EFT_allowable,
        string_format,
        float_format,
    )
    o_s += create_d_row(
        allocated_width,
        "Minimum Allowable hp_eft, C: ",
        ghe.sim_params.min_EFT_allowable,
        string_format,
        float_format,
    )
    o_s += create_d_row(
        allocated_width,
        "Maximum Allowable Height, m: ",
        ghe.sim_params.max_Height,
        string_format,
        float_format,
    )
    o_s += create_d_row(
        allocated_width,
        "Minimum Allowable Height, m: ",
        ghe.sim_params.min_Height,
        string_format,
        float_format,
    )
    o_s += create_d_row(
        allocated_width,
        "Simulation Time, years: ",
        int(ghe.sim_params.end_month / 12),
        string_format,
        int_format,
    )
    load_method_string = "hybrid" if load_method == DesignMethod.Hybrid else "hourly"  # TODO: Use a method in the enum
    o_s += create_d_row(
        allocated_width,
        "Simulation Loading Type: ",
        load_method_string,
        string_format,
        string_format,
    )

    o_s += empty_line

    # Loading Stuff
    o_s += create_title(allocated_width, "Simulation Results")
    o_s += empty_line

    # Simulation Results
    eft_table_title = "Monthly Temperature Summary"
    # daysInYear = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    # hoursInYear = 24 * daysInYear
    time_vals = ghe.times
    eft_vals = []
    eft_vals.extend(ghe.hp_eft)
    d_tb_vals = []
    d_tb_vals.extend(ghe.dTb)
    n_years = 0
    # hTotalYear = np.sum(hoursInYear)
    out_array = []
    last_month = -1
    month_tb_vals = []
    month_eft_vals = []
    for tv, d_tb, eft in zip(time_vals, d_tb_vals, eft_vals):
        # currentHourMonth = timeVals[i] - hTotalYear * nYears
        current_month = int(math.floor(hours_to_month(tv)))
        # print(monthEFTVals)
        if current_month == last_month:
            month_tb_vals.append(d_tb)
            month_eft_vals.append(eft)
        elif current_month != last_month:
            if len(month_tb_vals) > 0:
                if len(out_array) == 0:
                    previous_temp = bhe.soil.ugt
                else:
                    # print(i)
                    # print(len(outArray))
                    # previousTemp = outArray[-1][1]
                    previous_temp = bhe.soil.ugt
                out_array.append(
                    [
                        current_month,
                        previous_temp + month_tb_vals[-1],
                        np.max(month_eft_vals),
                        np.min(month_eft_vals),
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
    eft_table_formats = [".0f", ".3f", ".3f", ".3f"]

    o_s += create_title(allocated_width, "Peak Temperature", filler_symbol="-")
    max_eft = np.max(eft_vals)
    min_eft = np.min(eft_vals)
    max_eft_time = time_vals[eft_vals.index(max(eft_vals))]
    min_eft_time = time_vals[eft_vals.index(min(eft_vals))]
    max_eft_time = hours_to_month(max_eft_time)
    min_eft_time = hours_to_month(min_eft_time)
    o_s += create_d_row(
        allocated_width,
        "Max hp_eft, C:",
        round(max_eft, rounding_amount),
        string_format,
        float_format,
    )
    o_s += create_d_row(
        allocated_width,
        "Max hp_eft Time, Months:",
        round(max_eft_time, rounding_amount),
        string_format,
        float_format,
    )
    o_s += create_d_row(
        allocated_width,
        "Min hp_eft, C:",
        round(min_eft, rounding_amount),
        string_format,
        float_format,
    )
    o_s += create_d_row(
        allocated_width,
        "Min hp_eft Time, Months:",
        round(min_eft_time, rounding_amount),
        string_format,
        float_format,
    )

    o_s += create_table(
        eft_table_title,
        header_array,
        out_array,
        allocated_width,
        eft_table_formats,
        filler_symbol="-",
        centering="^",
    )

    output_directory.mkdir(exist_ok=True)
    with open(str(output_directory / summary_file), "w", newline="") as txtF:
        txtF.write(o_s)

    csv1_array = []

    loading_values = ghe.loading
    # loadingValues_dt = np.hstack((loadingValues[1:] - loadingValues[:-1]))
    for i, (tv, d_tb, lv) in enumerate(zip(time_vals, d_tb_vals, loading_values)):
        if i + 1 < len(time_vals):
            current_time = tv
            loading = loading_values[i + 1]
            current_month = hours_to_month(tv)
            normalized_loading = loading / (ghe.bhe.b.H * ghe.nbh)
            wall_temperature = bhe.soil.ugt + d_tb
            hp_eft_val = eft_vals[i]
            csv1_row = list()
            csv1_row.append(tv)
            csv1_row.append(hours_to_month(tv))
            if i > 1:
                csv1_row.append(lv)
                csv1_row.append(lv / (ghe.bhe.b.H * ghe.nbh))
            else:
                csv1_row.append(0)
                csv1_row.append(0)
            csv1_row.append(bhe.soil.ugt + d_tb_vals[i - 1])
            csv1_row.append(eft_vals[i - 1])
            csv1_array.append(csv1_row)

        else:

            csv1_row = list()
            csv1_row.append(tv)
            csv1_row.append(hours_to_month(tv))
            if i > 1:
                csv1_row.append(lv)
                csv1_row.append(lv / (ghe.bhe.b.H * ghe.nbh))
            else:
                csv1_row.append(0)
                csv1_row.append(0)
            csv1_row.append(bhe.soil.ugt + d_tb_vals[i - 1])
            csv1_row.append(eft_vals[i - 1])
            csv1_array.append(csv1_row)

            current_time = tv
            loading = 0
            current_month = hours_to_month(tv)
            normalized_loading = loading / (ghe.bhe.b.H * ghe.nbh)
            wall_temperature = bhe.soil.ugt + d_tb
            hp_eft_val = eft_vals[i]
        csv1_row = list()
        csv1_row.append(current_time)
        csv1_row.append(current_month)
        csv1_row.append(loading)
        csv1_row.append(normalized_loading)
        csv1_row.append(wall_temperature)
        csv1_row.append(hp_eft_val)
        csv1_array.append(csv1_row)
    with open(os.path.join(output_directory, csv_f_1), "w", newline="") as csv1OF:
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

    csv2_array = list()

    csv2_array.append(["x", "y"])
    for bL in g_function.bore_locations:
        csv2_array.append([bL[0], bL[1]])

    with open(os.path.join(output_directory, csv_f_2), "w", newline="") as csv2OF:
        c_w = csv.writer(csv2OF)
        c_w.writerows(csv2_array)

    hourly_loadings = ghe.hourly_extraction_ground_loads
    csv3_array = list()
    csv3_array.append(
        ["Month", "Day", "Hour", "Time (Hours)", "Loading (W) (Extraction)"]
    )
    for hour, hour_load in enumerate(hourly_loadings):
        month, day_in_month, hour_in_day = ghe_time_convert(hour)
        csv3_array.append([month, day_in_month, hour_in_day, hour, hour_load])

    with open(os.path.join(output_directory, csv_f_3), "w", newline="") as csv3OF:
        c_w = csv.writer(csv3OF)
        c_w.writerows(csv3_array)

        # gFunction STS+LTS Table
    # gfunctionTableFormats = [".3f"]
    # gfTableFF = [".3f"] * (1)
    # gfunctionTableFormats.extend(gfTableFF)
    # gfunctionColTitles = ["ln(t/ts)"]

    # gfunctionColTitles.append("H:" + str(round(bH.H, 2)) + "m")

    csv4_array = [["ln(t/ts)", "H:{:.2f}".format(bhe.b.H)]]
    ghe_gf_adjusted = ghe.grab_g_function(ghe.B_spacing / float(ghe.bhe.b.H))
    gfunction_log_vals = ghe_gf_adjusted.x
    gfunction_g_vals = ghe_gf_adjusted.y
    for log_val, g_val in zip(gfunction_log_vals, gfunction_g_vals):
        gf_row = list()
        gf_row.append(log_val)
        gf_row.append(g_val)
        csv4_array.append(gf_row)

    with open(os.path.join(output_directory, csv_f_4), "w", newline="") as csv4OF:
        c_w = csv.writer(csv4OF)
        c_w.writerows(csv4_array)

    return o_s
