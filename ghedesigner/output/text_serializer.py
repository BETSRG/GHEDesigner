import re
from datetime import datetime
from math import floor
from typing import Any

from ghedesigner.ghe.boreholes.base import GHEDesignerBoreholeBase
from ghedesigner.ghe.boreholes.coaxial_borehole import CoaxialPipe
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.output.converters import hours_to_month
from ghedesigner.output.formatters import create_line, create_table, create_title, d_row


class TextSerializer:
    """Generate the plain-text SimulationSummary report."""

    @staticmethod
    def summary_text(
        width: int,
        project_name: str,
        model_name: str,
        notes: str,
        author: str,
        time_sec: float,
        ghe: GHE,
        search_tracker: Any,
    ) -> str:
        # format specifiers
        f_str = "s"
        f_int = ".0f"
        f_1f = ".1f"
        f_2f = ".2f"
        f_3f = ".3f"
        f_4f = ".4f"
        f_sci = ".3e"

        blank = create_line(width)
        empty = create_line(width, character=" ")

        out = blank
        out += d_row(width, "Project Name:", project_name, f_str)
        out += blank
        out += f"Notes:\n\n{notes}\n"
        out += blank
        out += d_row(width, "File/Model Name:", model_name, f_str)
        now = datetime.now().strftime("%m/%d/%Y %H:%M:%S %p")
        out += d_row(width, "Simulated On:", now, f_str)
        out += d_row(width, "Simulated By:", author, f_str)
        out += d_row(width, "Calculation Time, s:", time_sec, f_3f)
        out += empty

        # --- Design Selection Table ---
        out += create_title(width, "Design Selection", filler_symbol="-")
        header = [
            ["Field", "Excess Temperature", "Max Temperature", "Min Temperature"],
            [" ", "(C)", "(C)", "(C)"],
        ]
        col_fmts = [f_str, f_2f, f_2f, f_2f]
        out += create_table(
            "Field Search Log",
            header,
            search_tracker,
            width,
            col_fmts,
            filler_symbol="-",
            centering="^",
        )
        out += empty

        # --- GHE System (gFunction LTS table) ---
        out += create_title(width, "GHE System", filler_symbol="-")
        lts = ghe.gFunction.g_lts
        titles = ["ln(t/ts)"] + [f"H: {h:0.2f} m" for h in lts] + [f"H: {ghe.bhe.borehole.H:0.2f} m"]
        data = []
        interp = ghe.gFunction.g_function_interpolation(ghe.b_spacing / float(ghe.bhe.borehole.H))[0]
        for i, t in enumerate(ghe.gFunction.log_time):
            row = [t] + [lts[h][i] for h in lts] + [interp[i]]
            data.append(row)
        fmts = [f_3f] * len(titles)
        out += create_table("gFunction LTS Values", [titles], data, width, fmts, filler_symbol="-", centering="^")
        out += empty

        # --- System Parameters ---
        out += create_title(width, "System Parameters", filler_symbol="-")
        out += d_row(width, "Active Borehole Length, m:", ghe.bhe.borehole.H, f_int)
        out += d_row(width, "Borehole Diameter, mm:", ghe.bhe.borehole.r_b * 2e3, f_2f)
        out += d_row(width, "Borehole Spacing, m:", ghe.b_spacing, f_3f)
        out += d_row(width, "Borehole Depth, m:", ghe.bhe.borehole.D, f_2f)
        out += d_row(
            width,
            "Total Drilling, m:",
            ghe.bhe.borehole.H * len(ghe.gFunction.bore_locations),
            f_int,
        )
        out += "Field Geometry: \n"
        out += d_row(width, "Field Type:", ghe.field_type, f_str, n_tabs=1)
        out += d_row(width, "Field Specifier:", ghe.fieldSpecifier, f_str, n_tabs=1)
        out += d_row(width, "NBH:", len(ghe.gFunction.bore_locations), f_int, n_tabs=1)

        # --- Borehole Information ---
        out += "Borehole Information: \n"
        # pipe geometry
        if isinstance(ghe.bhe.pipe.r_out, float):
            out += d_row(width, "Pipe Outer Diameter, mm:", ghe.bhe.pipe.r_out * 2e3, f_2f, n_tabs=1)
            out += d_row(width, "Pipe Inner Diameter, mm:", ghe.bhe.pipe.r_in * 2e3, f_2f, n_tabs=1)
        else:
            out += d_row(width, "Outer Pipe Outer Diameter, mm:", ghe.bhe.pipe.r_out[1] * 2e3, f_2f, n_tabs=1)
            out += d_row(width, "Outer Pipe Inner Diameter, mm:", ghe.bhe.pipe.r_out[0] * 2e3, f_2f, n_tabs=1)
            out += d_row(width, "Inner Pipe Outer Diameter, mm:", ghe.bhe.pipe.r_in[1] * 2e3, f_2f, n_tabs=1)
            out += d_row(width, "Inner Pipe Inner Diameter, mm:", ghe.bhe.pipe.r_in[0] * 2e3, f_2f, n_tabs=1)

        out += d_row(width, "Pipe Roughness, m:", ghe.bhe.pipe.roughness, f_sci, n_tabs=1)
        if isinstance(ghe.bhe.pipe.k, float):
            out += d_row(width, "Pipe Thermal Conductivity, W/m-K:", ghe.bhe.pipe.k, f_3f, n_tabs=1)
        else:
            out += d_row(width, "Inner Pipe Thermal Conductivity, W/m-K:", ghe.bhe.pipe.k[0], f_3f, n_tabs=1)
            out += d_row(width, "Outer Pipe Thermal Conductivity, W/m-K:", ghe.bhe.pipe.k[1], f_3f, n_tabs=1)
        out += d_row(width, "Pipe Volumetric Heat Capacity, kJ/m3-K:", ghe.bhe.pipe.rho_cp / 1000, f_2f, n_tabs=1)
        out += d_row(width, "Shank Spacing, mm:", ghe.bhe.pipe.s * 1e3, f_2f, n_tabs=1)
        out += d_row(width, "Grout Thermal Conductivity, W/(m-K):", ghe.bhe.grout.k, f_3f, n_tabs=1)
        out += d_row(width, "Grout Volumetric Heat Capacity, kJ/m3-K:", ghe.bhe.grout.rho_cp / 1000, f_2f, n_tabs=1)

        # Reynolds & effective resistance
        if isinstance(ghe.bhe.pipe.r_out, float):
            reynolds = GHEDesignerBoreholeBase.compute_reynolds(
                ghe.bhe.m_flow_borehole, ghe.bhe.pipe.r_in, ghe.bhe.fluid
            )
        else:
            reynolds = CoaxialPipe.compute_reynolds_concentric(
                ghe.bhe.m_flow_borehole,
                ghe.bhe.r_in_out,
                ghe.bhe.r_out_in,
                ghe.bhe.fluid,
            )
        out += d_row(width, "Reynold's Number:", reynolds, f_int, n_tabs=1)
        out += d_row(
            width,
            "Effective Borehole Resistance, K/(W/m):",
            ghe.bhe.calc_effective_borehole_resistance(),
            f_4f,
            n_tabs=1,
        )

        # Soil & Fluid
        out += "Soil Properties: \n"
        out += d_row(width, "Thermal Conductivity, W/m-K:", ghe.bhe.soil.k, f_3f, n_tabs=1)
        out += d_row(width, "Volumetric Heat Capacity, kJ/m3-K:", ghe.bhe.soil.rho_cp / 1000, f_2f, n_tabs=1)
        out += d_row(width, "Undisturbed Ground Temperature, C:", ghe.bhe.soil.ugt, f_2f, n_tabs=1)

        out += "Fluid Properties\n"
        out += d_row(width, "Volumetric Heat Capacity, kJ/m3-K:", ghe.bhe.fluid.rho_cp / 1000, f_2f, n_tabs=1)
        out += d_row(width, "Thermal Conductivity, W/m-K:", ghe.bhe.fluid.k, f_2f, n_tabs=1)
        out += d_row(width, "Viscosity, Pa-s:", ghe.bhe.fluid.mu, f_sci, n_tabs=1)
        out += d_row(width, "Fluid Mix:", ghe.bhe.fluid.name, f_str, n_tabs=1)
        out += d_row(width, "Density, kg/m^3:", ghe.bhe.fluid.rho, f_2f, n_tabs=1)
        out += d_row(width, "Mass Flow Rate Per Borehole, kg/s:", ghe.bhe.m_flow_borehole, f_3f, n_tabs=1)
        if hasattr(ghe.bhe, "h_f"):
            out += d_row(width, "Fluid Convection Coefficient, W/m-K:", ghe.bhe.h_f, f_int, n_tabs=1)
        out += empty

        # --- Hybrid load monthly tables & simulation results (if present) ---
        if getattr(ghe, "hybrid_load", None):
            monthly_load_values: list[list[Any]] = []
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

            for i in range(1, n_months + 1):
                hl = ghe.hybrid_load
                monthly_load_values.append(
                    [
                        months[i - 1],
                        hl.monthly_hl[i],
                        hl.monthly_cl[i],
                        hl.monthly_peak_hl[i],
                        hl.monthly_peak_hl_duration[i],
                        hl.monthly_peak_cl[i],
                        hl.monthly_peak_cl_duration[i],
                    ]
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
                ["", "kWh", "kWh", "kW", "hr", "kW", "hr"],
            ]
            month_table_formats = [f_str, f_1f, f_1f, f_1f, f_1f, f_1f, f_1f]
            out += create_table(
                "GLHE Monthly Loads",
                month_header,
                monthly_load_values,
                width,
                month_table_formats,
                filler_symbol="-",
                centering="^",
            )
            out += empty

            # Simulation Parameters header (original code left details commented)
            out += create_title(width, "Simulation Parameters")
            out += empty

            # Simulation Results (peaks + monthly temp summary)
            out += create_title(width, "Simulation Results")
            out += empty

            # Peak Temperature section
            max_eft = max(ghe.hp_eft)
            min_eft = min(ghe.hp_eft)
            max_eft_time_hrs = ghe.times[ghe.hp_eft.index(max_eft)]
            min_eft_time_hrs = ghe.times[ghe.hp_eft.index(min_eft)]
            out += create_title(width, "Peak Temperature", filler_symbol="-")
            out += d_row(width, "Max HP EFT, C:", max_eft, f_3f)
            out += d_row(width, "Max HP EFT Time, Months:", hours_to_month(max_eft_time_hrs), f_3f)
            out += d_row(width, "Min HP EFT, C:", min_eft, f_3f)
            out += d_row(width, "Min HP EFT Time, Months:", hours_to_month(min_eft_time_hrs), f_3f)

            # Monthly temperature summary table
            out_array = []
            last_month = -1
            month_tb_vals: list[float] = []
            month_eft_vals: list[float] = []
            for tv, d_tb, eft in zip(ghe.times, ghe.dTb, ghe.hp_eft):
                current_month = floor(hours_to_month(tv))
                if current_month == last_month:
                    month_tb_vals.append(d_tb)
                    month_eft_vals.append(eft)
                else:
                    if month_tb_vals:
                        previous_temp = ghe.bhe.soil.ugt
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

            header_array = [
                ["Time", "BH Wall Temp", "Max HP EFT", "Min HP EFT"],
                ["(months)", "(C)", "(C)", "(C)"],
            ]
            eft_table_formats = [f_int, f_2f, f_2f, f_2f]
            out += create_table(
                "Monthly Temperature Summary",
                header_array,
                out_array,
                width,
                eft_table_formats,
                filler_symbol="-",
                centering="^",
            )

        # strip trailing spaces
        return re.sub(r"[ \t]+\n", "\n", out)
