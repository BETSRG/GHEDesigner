"""
LCOE bridge: merges GHEDesigner sizing results with a user-supplied cost JSON
and calls lcoe-ten's evaluate_project_ts to produce LCOE metrics.

Entry point
-----------
    run_lcoe(ghe_objects, hp_objects, lcoe_json_path, output_directory)

Outputs
-------
    <output_directory>/LCOESummary.csv        — full breakdown + amortization
"""

from __future__ import annotations

import csv
import json
import math
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
    Run LCOE analysis and write LCOESummary.csv in output_directory.

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
        Directory where outputs will be written (created if absent).

    Returns
    -------
    dict
        The result dict from evaluate_project_ts for the GHE system, plus an
        optional "baseline" key if a baseline section was present.
    """
    with open(lcoe_json_path) as fh:
        cost_data: dict[str, Any] = json.load(fh)

    validate_lcoe_input(cost_data)

    years: int = cost_data["years"]
    steps_per_year: int = cost_data["steps_per_year"]
    T: int = years * steps_per_year
    rate: float = cost_data["real_discount_rate"]
    currency: str = cost_data.get("currency", "currency")

    q = extract_quantities(ghe_objects, hp_objects)

    # Build GHE capex/debt separately so we hold references for amortization
    ghe_capex = _build_capex(cost_data, q)
    ghe_net = _net_capex_t0(ghe_capex)
    ghe_debt = _build_debt(cost_data, ghe_net, years, steps_per_year)

    ghe_inputs = _build_evaluate_inputs(
        cost_data, q, ghe_capex, ghe_debt, years, steps_per_year, T
    )
    ghe_result = evaluate_project_ts(**ghe_inputs)
    # After evaluate_project_ts, ghe_debt[i].annuity is populated

    baseline_result: dict[str, Any] | None = None
    baseline_debt: list[DebtScheduleTS] = []
    if "baseline" in cost_data:
        _, baseline_debt, baseline_inputs = _build_baseline_inputs(
            cost_data["baseline"], q, years, steps_per_year, T,
            real_discount_rate=rate,
        )
        baseline_result = evaluate_project_ts(**baseline_inputs)

    output_directory.mkdir(parents=True, exist_ok=True)

    # Write LCOESummary.csv
    csv_rows = _lcoe_csv_rows(
        cost_data, q, ghe_result, ghe_debt, baseline_result, baseline_debt, currency
    )
    with open(output_directory / "LCOESummary.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(csv_rows)

    result: dict[str, Any] = {"ghe_system": ghe_result}
    if baseline_result is not None:
        result["baseline_system"] = baseline_result
    return result


# ---------------------------------------------------------------------------
# CAPEX helpers
# ---------------------------------------------------------------------------


def _build_capex(cost_data: dict, q: GHEQuantities) -> list[CapexScheduleTS]:
    """
    Combine unit-rate CAPEX (rate * GHEDesigner quantity) with fully-specified
    fixed_capex items from the cost JSON.
    """
    items: list[CapexScheduleTS] = []
    ur: dict = cost_data.get("unit_rate_capex", {})

    if "drilling_currency_per_meter" in ur:
        items.append(CapexScheduleTS(
            name="Drilling",
            cashflow_t0=ur["drilling_currency_per_meter"] * q.total_drilling_m,
        ))

    # TODO Currently, user must supply central loop length - to be taken from GHED sizing output in a future update
    if "central_loop_pipe_currency_per_meter" in ur and "central_loop_pipe_length_m" in ur:
        items.append(CapexScheduleTS(
            name="Central loop piping",
            cashflow_t0=ur["central_loop_pipe_currency_per_meter"] * ur["central_loop_pipe_length_m"],
        ))
    # TODO Currently, user must supply header pipe length - to be taken from GHED sizing output in a future update
    if "ghe_header_pipe_currency_per_meter" in ur and "ghe_header_pipe_length_m" in ur:
        items.append(CapexScheduleTS(
            name="GHE header piping",
            cashflow_t0=ur["ghe_header_pipe_currency_per_meter"] * ur["ghe_header_pipe_length_m"],
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
    capex_ts: list[CapexScheduleTS],
    debt_ts: list[DebtScheduleTS],
    years: int,
    steps_per_year: int,
    T: int,
) -> dict:
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
) -> tuple[list[CapexScheduleTS], list[DebtScheduleTS], dict]:
    """
    Baseline uses the same thermal loads as the GHE system (equal LCOx
    denominator for fair comparison) but its own cost structure.

    Returns (capex_ts, debt_ts, evaluate_project_ts kwargs dict) so the caller
    can hold references to the debt objects for amortization reporting.
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

    inputs = {
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
    return capex_ts, debt_ts, inputs


# ---------------------------------------------------------------------------
# Amortization helper
# ---------------------------------------------------------------------------


def _amortization_rows(
    debt: DebtScheduleTS,
) -> list[tuple[int, float, float, float, float]]:
    """
    Reconstruct the amortization schedule for a single loan.

    Returns a list of (step, payment, interest, principal_paid, balance_end).
    Relies on debt.annuity already being set (populated by build_cashflows
    inside evaluate_project_ts).
    """
    r = debt.real_rate_step()
    N = debt.years * debt.steps_per_year
    repay_steps = max(N - debt.grace_steps, 0)

    annuity = debt.annuity
    if annuity is None and repay_steps > 0:
        annuity = (
            debt.principal * r / (1 - (1 + r) ** (-repay_steps))
            if r != 0
            else debt.principal / repay_steps
        )

    balance = debt.principal
    rows: list[tuple[int, float, float, float, float]] = []

    for t in range(N):
        if t < debt.grace_steps:
            interest = balance * r
            principal_paid = 0.0
            payment = interest
        else:
            loan_t = t - debt.grace_steps
            if loan_t >= repay_steps:
                break
            interest = balance * r
            if loan_t == repay_steps - 1:
                principal_paid = balance
                payment = interest + principal_paid
            else:
                payment = annuity  # type: ignore[assignment]
                principal_paid = payment - interest
            balance -= principal_paid

        rows.append((t + 1, payment, interest, principal_paid, balance))

    return rows


# ---------------------------------------------------------------------------
# Text output formatters
# ---------------------------------------------------------------------------

_SEP80 = "-" * 80


def _amortization_text(debt: DebtScheduleTS) -> str:
    rows = _amortization_rows(debt)
    col_sep = "-" * 66
    lines = [
        f"Loan: {debt.name}",
        f"  {'Year':>4} | {'Payment':>12} | {'Interest':>12} | {'Principal':>12} | {'Balance end':>13}",
        f"  {col_sep}",
    ]
    for step, payment, interest, principal, balance in rows:
        lines.append(
            f"  {step:>4} | {payment:>12,.2f} | {interest:>12,.2f}"
            f" | {principal:>12,.2f} | {balance:>13,.2f}"
        )
    return "\n".join(lines) + "\n"


def _breakdown_text(label: str, result: dict, currency: str) -> str:
    pv_service = result["PV_service_MWh"]
    total = result["NPV_total_cost"]

    def unit(npv: float) -> float:
        return npv / pv_service if pv_service else math.nan

    def share(npv: float) -> float:
        return 100.0 * npv / total if total else math.nan

    components = [
        ("CAPEX",               result["NPV_capex"]),
        ("OPEX fixed",          result["NPV_opex_fixed"]),
        ("Electricity (total)", result["NPV_elec_total"]),
        ("Other variable OPEX", result["NPV_opex_variable_other"]),
        ("Financing",           result["NPV_financing"]),
    ]

    hdr = (
        f"{'Component':<28} {'NPV [' + currency + ']':>15}"
        f" {'[' + currency + '/MWh service]':>20} {'Share':>7}"
    )
    lines = [
        f"=== {label}: LCOE breakdown (NPV basis) ===",
        hdr,
        _SEP80,
    ]
    for name, npv in components:
        lines.append(
            f"{name:<28} {npv:>15,.2f} {unit(npv):>20,.2f} {share(npv):>6.1f}%"
        )
    lines.append(_SEP80)
    lines.append(
        f"{'TOTAL (LCOE)':<28} {total:>15,.2f} {unit(total):>20,.2f} {'100.0%':>7}"
    )
    lines += [
        "",
        f"  LCOH:  {result['LCOH_(currency_per_MWh_heat)']:>12,.2f} {currency}/MWh_heat",
        f"  LCOC:  {result['LCOC_(currency_per_MWh_cool)']:>12,.2f} {currency}/MWh_cool",
        f"  LCOx:  {result['LCOx_total_(currency_per_MWh_service)']:>12,.2f} {currency}/MWh_service",
    ]
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def _breakdown_csv_rows(
    label: str,
    result: dict,
    debt: list[DebtScheduleTS],
    currency: str,
) -> list[list]:
    rows: list[list] = []

    pv_service = result["PV_service_MWh"]
    total = result["NPV_total_cost"]

    rows.append([f"{label} LCOE Breakdown (NPV basis)"])
    rows.append(["Component", f"NPV [{currency}]", f"Unit cost [{currency}/MWh service]", "Share [%]"])

    components = [
        ("CAPEX",               result["NPV_capex"]),
        ("OPEX fixed",          result["NPV_opex_fixed"]),
        ("Electricity (total)", result["NPV_elec_total"]),
        ("Other variable OPEX", result["NPV_opex_variable_other"]),
        ("Financing",           result["NPV_financing"]),
    ]
    for name, npv in components:
        uc = npv / pv_service if pv_service else math.nan
        sh = 100.0 * npv / total if total else math.nan
        rows.append([name, f"{npv:.2f}", f"{uc:.2f}", f"{sh:.1f}"])

    unit_total = total / pv_service if pv_service else math.nan
    rows.append(["TOTAL", f"{total:.2f}", f"{unit_total:.2f}", "100.0"])
    rows.append(["LCOH", f"{result['LCOH_(currency_per_MWh_heat)']:.2f}", f"{currency}/MWh_heat", ""])
    rows.append(["LCOC", f"{result['LCOC_(currency_per_MWh_cool)']:.2f}", f"{currency}/MWh_cool", ""])
    rows.append(["LCOx", f"{result['LCOx_total_(currency_per_MWh_service)']:.2f}", f"{currency}/MWh_service", ""])
    rows.append([])

    # Amortization tables
    for d in debt:
        rows.append([f"Amortization: {d.name} for {label}"])
        rows.append(["Year", "Payment", "Interest", "Principal", "Balance end"])
        for step, payment, interest, principal, balance in _amortization_rows(d):
            rows.append([step, f"{payment:.2f}", f"{interest:.2f}", f"{principal:.2f}", f"{balance:.2f}"])
        rows.append([])

    return rows


def _lcoe_csv_rows(
    cost_data: dict,
    q: GHEQuantities,
    ghe_result: dict,
    ghe_debt: list[DebtScheduleTS],
    baseline_result: dict | None,
    baseline_debt: list[DebtScheduleTS],
    currency: str,
) -> list[list]:
    rows: list[list] = []

    # Parameters
    rows.append(["Parameter", "Value"])
    rows.append(["Currency", currency])
    rows.append(["Years", cost_data["years"]])
    rows.append(["Real discount rate", f"{cost_data['real_discount_rate'] * 100:.2f}%"])
    rows.append(["Steps per year", cost_data["steps_per_year"]])
    rows.append([])

    # GHE quantities
    rows.append(["GHE Sizing Quantities", ""])
    rows.append(["Total drilling (m)", f"{q.total_drilling_m:.2f}"])
    rows.append(["Boreholes", q.n_boreholes])
    rows.append(["Heat pumps", q.n_heat_pumps])
    rows.append(["Heating year 1 (MWh)", f"{q.heat_MWh_yr1:.2f}"])
    rows.append(["Cooling year 1 (MWh)", f"{q.cool_MWh_yr1:.2f}"])
    rows.append(["HP elec heat yr1 (MWh)", f"{q.elec_heat_MWh_yr1:.2f}"])
    rows.append(["HP elec cool yr1 (MWh)", f"{q.elec_cool_MWh_yr1:.2f}"])
    rows.append([])

    rows += _breakdown_csv_rows("GHE System", ghe_result, ghe_debt, currency)

    if baseline_result is not None:
        rows += _breakdown_csv_rows("Baseline System", baseline_result, baseline_debt, currency)

        delta_lcoh = (
            ghe_result["LCOH_(currency_per_MWh_heat)"]
            - baseline_result["LCOH_(currency_per_MWh_heat)"]
        )
        delta_lcox = (
            ghe_result["LCOx_total_(currency_per_MWh_service)"]
            - baseline_result["LCOx_total_(currency_per_MWh_service)"]
        )
        rows.append(["Comparison"])
        rows.append(["delta_LCOH (GHE - baseline)", f"{delta_lcoh:+.2f}", f"{currency}/MWh_heat", ""])
        rows.append(["delta_LCOx (GHE - baseline)", f"{delta_lcox:+.2f}", f"{currency}/MWh_service", ""])

    return rows
