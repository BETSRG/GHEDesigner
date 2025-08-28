from datetime import datetime
from math import floor
from typing import Any

from ghedesigner.enums import TimestepType
from ghedesigner.ghe.boreholes.base import GHEDesignerBoreholeBase
from ghedesigner.ghe.boreholes.coaxial_borehole import CoaxialPipe
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.output.converters import hours_to_month


class JsonSerializer:
    """Generate the JSON-serializable SimulationSummary object."""

    @staticmethod
    def summary_object(
        ghe: GHE,
        search_tracker: Any,
        time_sec: float,
        project_name: str,
        notes: str,
        author: str,
        model_name: str,
        load_method: TimestepType,
    ) -> dict[str, Any]:
        def unit(v: float, u: str) -> dict[str, float | str]:
            return {"value": v, "units": u}

        # gFunction LTS
        g_lts = ghe.gFunction.g_lts
        titles = ["ln(t/ts)"] + [f"H: {h:0.2f} m" for h in g_lts] + [f"H: {ghe.bhe.borehole.H:0.2f} m"]
        data = []
        ghe_gf = ghe.gFunction.g_function_interpolation(ghe.b_spacing / ghe.bhe.borehole.H)[0]
        for i, t in enumerate(ghe.gFunction.log_time):
            row = [t] + [g_lts[h][i] for h in g_lts] + [ghe_gf[i]]
            data.append(row)

        # pipe geometry + reynolds
        if isinstance(ghe.bhe.pipe.r_out, float):
            pipe_geometry = {
                "pipe_outer_diameter": unit(ghe.bhe.pipe.r_out * 2.0, "m"),
                "pipe_inner_diameter": unit(ghe.bhe.pipe.r_in * 2.0, "m"),
            }
            reynolds = GHEDesignerBoreholeBase.compute_reynolds(
                ghe.bhe.m_flow_borehole, ghe.bhe.pipe.r_in, ghe.bhe.fluid
            )
        else:
            pipe_geometry = {
                "inner_pipe_inner_diameter": unit(ghe.bhe.pipe.r_in[0] * 2.0, "m"),
                "inner_pipe_outer_diameter": unit(ghe.bhe.pipe.r_in[1] * 2.0, "m"),
                "outer_pipe_inner_diameter": unit(ghe.bhe.pipe.r_out[0] * 2.0, "m"),
                "outer_pipe_outer_diameter": unit(ghe.bhe.pipe.r_out[1] * 2.0, "m"),
            }
            reynolds = CoaxialPipe.compute_reynolds_concentric(
                ghe.bhe.m_flow_borehole,
                ghe.bhe.r_in_out,
                ghe.bhe.r_out_in,
                ghe.bhe.fluid,
            )

        obj: dict[str, Any] = {
            "project_name": project_name,
            "notes": notes,
            "model_name": model_name,
            "simulation_time_stamp": datetime.now().strftime("%m/%d/%Y %H:%M:%S %p"),
            "simulation_author": author,
            "simulation_runtime": unit(time_sec, "s"),
            "design_selection_search_log": {
                "titles": ["Field", "Excess Temperature", "Max Temperature", "Min Temperature"],
                "units": [" ", "(C)", "(C)", "(C)"],
                "data": search_tracker,
            },
            "ghe_system": {
                "search_log": {"titles": titles, "units": None, "data": data},
                "active_borehole_length": unit(ghe.bhe.borehole.H, "m"),
                "borehole_diameter": unit(ghe.bhe.borehole.r_b * 2.0, "m"),
                "borehole_buried_depth": unit(ghe.bhe.borehole.D, "m"),
                "borehole_spacing": unit(ghe.b_spacing, "m"),
                "total_drilling": unit(ghe.bhe.borehole.H * len(ghe.gFunction.bore_locations), "m"),
                "field_type": ghe.field_type,
                "field_specifier": ghe.fieldSpecifier,
                "number_of_boreholes": len(ghe.gFunction.bore_locations),
                "shank_spacing": unit(ghe.bhe.pipe.s, "m"),
                "pipe_geometry": pipe_geometry,
                "pipe_roughness": unit(ghe.bhe.pipe.roughness, "m"),
                "pipe_thermal_conductivity": unit(ghe.bhe.pipe.k, "W/m-K"),
                "pipe_volumetric_heat_capacity": unit(ghe.bhe.pipe.rho_cp / 1000, "kJ/m3-K"),
                "grout_thermal_conductivity": unit(ghe.bhe.grout.k, "W/m-K"),
                "grout_volumetric_heat_capacity": unit(ghe.bhe.grout.rho_cp / 1000, "kJ/m3-K"),
                "reynolds_number": reynolds,
                "effective_borehole_resistance": unit(ghe.bhe.calc_effective_borehole_resistance(), "W/m-K"),
                # TODO: are the units right here?
                "soil_thermal_conductivity": unit(ghe.bhe.soil.k, "W/m-K"),
                "soil_volumetric_heat_capacity": unit(ghe.bhe.soil.rho_cp / 1000, "kJ/m3-K"),
                "soil_undisturbed_ground_temp": unit(ghe.bhe.soil.ugt, "C"),
                "fluid_volumetric_heat_capacity": unit(ghe.bhe.fluid.rho_cp / 1000, "kJ/m3-K"),
                "fluid_thermal_conductivity": unit(ghe.bhe.fluid.k, "W/m-K"),
                "fluid_viscosity": unit(ghe.bhe.fluid.mu, "Pa-s"),
                "fluid_mixture": ghe.bhe.fluid.name,
                "fluid_density": unit(ghe.bhe.fluid.rho, "kg/m3"),
                "fluid_mass_flow_rate_per_borehole": unit(ghe.bhe.m_flow_borehole, "kg/s"),
            },
            "simulation_parameters": {
                "simulation_load_method": load_method.name,
            },
            "simulation_results": {},
        }

        # Optional convection coefficient
        if hasattr(ghe.bhe, "h_f"):
            obj["ghe_system"]["fluid_convection_coefficient"] = unit(ghe.bhe.h_f, "W/m-K")

        # Monthly loads (when hybrid loads are provided)
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
                monthly_load_values.append(
                    [
                        months[i - 1],
                        ghe.hybrid_load.monthly_hl[i],
                        ghe.hybrid_load.monthly_cl[i],
                        ghe.hybrid_load.monthly_peak_hl[i],
                        ghe.hybrid_load.monthly_peak_hl_duration[i],
                        ghe.hybrid_load.monthly_peak_cl[i],
                        ghe.hybrid_load.monthly_peak_cl_duration[i],
                    ]
                )
            obj["ghe_system"]["glhe_monthly_loads"] = {
                "titles": [
                    "Month",
                    "Total Heating",
                    "Total Cooling",
                    "Peak Heating",
                    "PH Duration",
                    "Peak Cooling",
                    "PC Duration",
                ],
                "units": ["", "kWh", "kWh", "kW", "hr", "kW", "hr"],
                "data": monthly_load_values,
            }

            # Peak EFTs + monthly temperature summary
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

            max_eft = max(ghe.hp_eft)
            min_eft = min(ghe.hp_eft)
            max_eft_time = hours_to_month(ghe.times[ghe.hp_eft.index(max_eft)])
            min_eft_time = hours_to_month(ghe.times[ghe.hp_eft.index(min_eft)])

            obj["simulation_results"] = {
                "max_hp_eft": unit(max_eft, "C"),
                "max_hp_eft_time": unit(max_eft_time, "months"),
                "min_hp_eft": unit(min_eft, "C"),
                "min_hp_eft_time": unit(min_eft_time, "months"),
                "monthly_temp_summary": {
                    "titles": ["Time", "BH Wall Temp", "Max HP EFT", "Min HP EFT"],
                    "units": ["(months)", "(C)", "(C)", "(C)"],
                    "data": out_array,
                },
            }

        return obj
