"""
LCOE bridge: merges GHEDesigner sizing results with a user-supplied cost JSON
and calls lcoe-ten's evaluate_project_ts to produce LCOE metrics.

Entry point
-----------
    run_lcoe(ghe_objects, hp_objects, lcoe_json_path, output_directory)

Output
------
    <output_directory>/LCOESummary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lcoe_ten.cost_items import (
    CapexScheduleTS,
    DebtScheduleTS,
    EnergyPriceTS,
    LoadTS,
    OpexFixedTS,
    OpexVariableTS,
)
from lcoe_ten.system_model import evaluate_project_ts

from ghedesigner.lcoe.quantities import GHEQuantities, extract_quantities
from ghedesigner.lcoe.schema import validate_lcoe_input

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ghedesigner.ghe.ground_heat_exchangers import GHE
    from ghedesigner.heat_pump_fixed_cop import HeatPumpFixedCOP


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_lcoe(
    ghe_objects: Sequence[GHE],
    hp_objects: Sequence[HeatPumpFixedCOP],
    lcoe_json_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """
    Run LCOE analysis and write LCOESummary.json to output_directory.

    Parameters
    ----------
    ghe_objects:
        Sized GHE instances returned by design_and_size_ghe.
    hp_objects:
        HeatPumpFixedCOP instances that drove the sizing (one per building).
        Pass an empty sequence when the GHE was driven by direct loads (no HP).
    lcoe_json_path:
        Path to the user's LCOE cost JSON file.
    output_directory:
        Directory where LCOESummary.json will be written (created if absent).

    Returns
    -------
    dict
        The full output object that was written to LCOESummary.json.
    """
    with open(lcoe_json_path) as fh:
        cost_data: dict[str, Any] = json.load(fh)

    validate_lcoe_input(cost_data)

    years: int = cost_data["years"]
    steps_per_year: int = cost_data["steps_per_year"]
    T: int = years * steps_per_year
    rate: float = cost_data["real_discount_rate"]

    q = extract_quantities(ghe_objects, hp_objects)

    ghe_inputs = _build_evaluate_inputs(cost_data, q, years, steps_per_year, T)
    ghe_result = evaluate_project_ts(**ghe_inputs)

    baseline_result: dict[str, Any] | None = None
    if "baseline" in cost_data:
        baseline_inputs = _build_baseline_inputs(
            cost_data["baseline"], q, years, steps_per_year, T,
            real_discount_rate=rate,
        )
        baseline_result = evaluate_project_ts(**baseline_inputs)

    output = _build_output(cost_data, q, ghe_result, baseline_result)

    output_directory.mkdir(parents=True, exist_ok=True)
    with open(output_directory / "LCOESummary.json", "w") as fh:
        json.dump(output, fh, indent=2)

    return output


# ---------------------------------------------------------------------------
# CAPEX helpers
# ---------------------------------------------------------------------------


def _build_capex(cost_data: dict, q: GHEQuantities) -> list[CapexScheduleTS]:
    """
    Combine unit-rate CAPEX (rate × GHEDesigner quantity) with fully-specified
    fixed_capex items from the cost JSON.
    """
    items: list[CapexScheduleTS] = []
    ur: dict = cost_data.get("unit_rate_capex", {})

    if "drilling_currency_per_meter" in ur:
        items.append(CapexScheduleTS(
            name="Drilling",
            cashflow_t0=ur["drilling_currency_per_meter"] * q.total_drilling_m,
        ))

    if "borehole_pipe_currency_per_meter" in ur:
        items.append(CapexScheduleTS(
            name="Borehole piping",
            cashflow_t0=ur["borehole_pipe_currency_per_meter"] * q.vertical_pipe_m,
        ))
    # TODO Currently, user must supply central loop and header pipe lengths - to be automatic with a future update
    if "central_loop_pipe_currency_per_meter" in ur and "central_loop_pipe_length_m" in ur:
        items.append(CapexScheduleTS(
            name="Central loop piping",
            cashflow_t0=ur["central_loop_pipe_currency_per_meter"] * ur["central_loop_pipe_length_m"],
        ))

    if "ghe_header_pipe_currency_per_meter" in ur and "ghe_header_pipe_length_m" in ur:
        items.append(CapexScheduleTS(
            name="GHE header piping",
            cashflow_t0=ur["ghe_header_pipe_currency_per_meter"] * ur["ghe_header_pipe_length_m"],
        ))

    if "grout_currency_per_m3" in ur:
        items.append(CapexScheduleTS(
            name="Grouting",
            cashflow_t0=ur["grout_currency_per_m3"] * q.grout_volume_m3,
        ))

    if "heat_pump_currency_per_unit" in ur:
        items.append(CapexScheduleTS(
            name="Heat pumps",
            cashflow_t0=ur["heat_pump_currency_per_unit"] * q.n_heat_pumps,
        ))

    for item in cost_data.get("fixed_capex", []):
        items.append(CapexScheduleTS(
            name=item["name"],
            cashflow_t0=item.get("cashflow_t0", 0.0),
            cashflow_ts=item.get("cashflow_ts", []),
            residual_at_end=item.get("residual_at_end", 0.0),
        ))

    return items


def _net_capex_t0(capex_ts: list[CapexScheduleTS]) -> float:
    """Sum of all t0 cash flows — used to fill null debt principals."""
    return sum(c.cashflow_t0 for c in capex_ts)


# ---------------------------------------------------------------------------
# Debt helpers
# ---------------------------------------------------------------------------


def _build_debt(
    section: dict,
    net_capex: float,
    project_years: int,
    project_steps_per_year: int,
) -> list[DebtScheduleTS]:
    items: list[DebtScheduleTS] = []
    for d in section.get("debt", []):
        principal = net_capex if d.get("principal") is None else float(d["principal"])
        items.append(DebtScheduleTS(
            name=d["name"],
            principal=principal,
            nominal_rate_ann=d["nominal_rate_ann"],
            years=d.get("years", project_years),
            steps_per_year=d.get("steps_per_year", project_steps_per_year),
            grace_steps=d.get("grace_steps", 0),
            fees_t0=d.get("fees_t0", 0.0),
            inflation_ann=d.get("inflation_ann", 0.0),
        ))
    return items


# ---------------------------------------------------------------------------
# Loads helper
# ---------------------------------------------------------------------------


def _build_loads(cost_data: dict, q: GHEQuantities, T: int) -> LoadTS:
    """
    Build LoadTS from GHE-derived year-1 annual energy totals.

    Values are divided evenly across steps within the year, then repeated for
    every year in the project horizon (V1: flat year-1 profile).
    """
    years: int = cost_data["years"]
    steps_per_year: int = cost_data["steps_per_year"]
    aux_kw: float = cost_data.get("aux_electric_kw", 0.0)
    hours_per_step = 8760.0 / steps_per_year

    return LoadTS(
        years=years,
        steps_per_year=steps_per_year,
        heat_MWh_ts=[q.heat_MWh_yr1 / steps_per_year] * T,
        cool_MWh_ts=[q.cool_MWh_yr1 / steps_per_year] * T,
        elec_heat_MWh_ts=[q.elec_heat_MWh_yr1 / steps_per_year] * T,
        elec_cool_MWh_ts=[q.elec_cool_MWh_yr1 / steps_per_year] * T,
        elec_aux_MWh_ts=[aux_kw * hours_per_step / 1000.0] * T,
    )


# ---------------------------------------------------------------------------
# evaluate_project_ts input builders
# ---------------------------------------------------------------------------


def _price_paths(section: dict) -> dict:
    ep = section.get("electricity_price")
    if not ep:
        return {}
    return {
        "electricity": EnergyPriceTS(
            name=ep.get("name", "electricity"),
            values_ts=ep["values_ts"],
        )
    }


def _build_evaluate_inputs(
    cost_data: dict,
    q: GHEQuantities,
    years: int,
    steps_per_year: int,
    T: int,
) -> dict:
    capex_ts = _build_capex(cost_data, q)
    net = _net_capex_t0(capex_ts)
    debt_ts = _build_debt(cost_data, net, years, steps_per_year)

    opex_fixed_ts = [
        OpexFixedTS(name=item["name"], series=item["series"])
        for item in cost_data.get("opex_fixed", [])
    ]
    opex_variable_ts = [
        OpexVariableTS(name=item["name"], unit=item["unit"], rates_ts=item["rates_ts"])
        for item in cost_data.get("opex_variable", [])
    ]

    return {
        "years": years,
        "steps_per_year": steps_per_year,
        "real_discount_rate": cost_data["real_discount_rate"],
        "capex_ts": capex_ts,
        "opex_fixed_ts": opex_fixed_ts,
        "opex_variable_ts": opex_variable_ts,
        "price_paths_ts": _price_paths(cost_data),
        "loads_ts": _build_loads(cost_data, q, T),
        "debt_ts": debt_ts or None,
    }


def _build_baseline_inputs(
    baseline: dict,
    q: GHEQuantities,
    years: int,
    steps_per_year: int,
    T: int,
    *,
    real_discount_rate: float,
) -> dict:
    """
    Baseline uses the same thermal loads as the GHE system (equal LCOx
    denominator for fair comparison) but its own cost structure.

    No unit_rate_capex section — the baseline does not depend on GHEDesigner
    sizing outputs.  Electricity for the baseline HP is zero; any baseline
    electricity should be modelled via opex_variable (unit=MWh_heat/cool) or
    aux_electric_kw.
    """
    aux_kw: float = baseline.get("aux_electric_kw", 0.0)
    hours_per_step = 8760.0 / steps_per_year

    loads_ts = LoadTS(
        years=years,
        steps_per_year=steps_per_year,
        heat_MWh_ts=[q.heat_MWh_yr1 / steps_per_year] * T,
        cool_MWh_ts=[q.cool_MWh_yr1 / steps_per_year] * T,
        elec_heat_MWh_ts=[0.0] * T,
        elec_cool_MWh_ts=[0.0] * T,
        elec_aux_MWh_ts=[aux_kw * hours_per_step / 1000.0] * T,
    )

    capex_ts = [
        CapexScheduleTS(
            name=item["name"],
            cashflow_t0=item.get("cashflow_t0", 0.0),
            cashflow_ts=item.get("cashflow_ts", []),
            residual_at_end=item.get("residual_at_end", 0.0),
        )
        for item in baseline.get("fixed_capex", [])
    ]
    net = _net_capex_t0(capex_ts)
    debt_ts = _build_debt(baseline, net, years, steps_per_year)

    opex_fixed_ts = [
        OpexFixedTS(name=item["name"], series=item["series"])
        for item in baseline.get("opex_fixed", [])
    ]
    opex_variable_ts = [
        OpexVariableTS(name=item["name"], unit=item["unit"], rates_ts=item["rates_ts"])
        for item in baseline.get("opex_variable", [])
    ]

    return {
        "years": years,
        "steps_per_year": steps_per_year,
        "real_discount_rate": real_discount_rate,
        "capex_ts": capex_ts,
        "opex_fixed_ts": opex_fixed_ts,
        "opex_variable_ts": opex_variable_ts,
        "price_paths_ts": _price_paths(baseline),
        "loads_ts": loads_ts,
        "debt_ts": debt_ts or None,
    }


# ---------------------------------------------------------------------------
# Output object builder
# ---------------------------------------------------------------------------


def _fmt(value: float, units: str) -> dict[str, Any]:
    return {"value": round(value, 4), "units": units}


def _build_output(
    cost_data: dict,
    q: GHEQuantities,
    ghe_result: dict,
    baseline_result: dict | None,
) -> dict[str, Any]:
    currency = cost_data.get("currency", "currency")

    output: dict[str, Any] = {
        "currency": currency,
        "ghe_quantities": {
            "total_drilling_m": _fmt(q.total_drilling_m, "m"),
            "n_boreholes": q.n_boreholes,
            "vertical_pipe_m": _fmt(q.vertical_pipe_m, "m"),
            "grout_volume_m3": _fmt(q.grout_volume_m3, "m3"),
            "n_heat_pumps": q.n_heat_pumps,
            "heat_MWh_yr1": _fmt(q.heat_MWh_yr1, "MWh"),
            "cool_MWh_yr1": _fmt(q.cool_MWh_yr1, "MWh"),
            "elec_heat_MWh_yr1": _fmt(q.elec_heat_MWh_yr1, "MWh"),
            "elec_cool_MWh_yr1": _fmt(q.elec_cool_MWh_yr1, "MWh"),
        },
        "ghe_system": {
            "LCOH": _fmt(ghe_result["LCOH_(currency_per_MWh_heat)"], f"{currency}/MWh_heat"),
            "LCOC": _fmt(ghe_result["LCOC_(currency_per_MWh_cool)"], f"{currency}/MWh_cool"),
            "LCOx_total": _fmt(
                ghe_result["LCOx_total_(currency_per_MWh_service)"], f"{currency}/MWh_service"
            ),
            "NPV_total_cost": _fmt(ghe_result["NPV_total_cost"], currency),
            "NPV_capex": _fmt(ghe_result["NPV_capex"], currency),
            "NPV_opex": _fmt(ghe_result["NPV_opex"], currency),
            "NPV_financing": _fmt(ghe_result["NPV_financing"], currency),
            "PV_heat_MWh": _fmt(ghe_result["PV_heat_MWh"], "MWh"),
            "PV_cool_MWh": _fmt(ghe_result["PV_cool_MWh"], "MWh"),
            "PV_service_MWh": _fmt(ghe_result["PV_service_MWh"], "MWh"),
        },
    }

    if baseline_result is not None:
        output["baseline_system"] = {
            "LCOH": _fmt(
                baseline_result["LCOH_(currency_per_MWh_heat)"], f"{currency}/MWh_heat"
            ),
            "LCOC": _fmt(
                baseline_result["LCOC_(currency_per_MWh_cool)"], f"{currency}/MWh_cool"
            ),
            "LCOx_total": _fmt(
                baseline_result["LCOx_total_(currency_per_MWh_service)"],
                f"{currency}/MWh_service",
            ),
            "NPV_total_cost": _fmt(baseline_result["NPV_total_cost"], currency),
        }
        delta_lcoh = (
            ghe_result["LCOH_(currency_per_MWh_heat)"]
            - baseline_result["LCOH_(currency_per_MWh_heat)"]
        )
        delta_lcox = (
            ghe_result["LCOx_total_(currency_per_MWh_service)"]
            - baseline_result["LCOx_total_(currency_per_MWh_service)"]
        )
        output["comparison"] = {
            "delta_LCOH": _fmt(delta_lcoh, f"{currency}/MWh_heat"),
            "delta_LCOx": _fmt(delta_lcox, f"{currency}/MWh_service"),
            "ghe_cheaper_than_baseline": delta_lcoh < 0.0,
        }

    return output
