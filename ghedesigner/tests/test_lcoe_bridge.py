"""
Tests for ghedesigner.lcoe.bridge.

Organization
------------
TestBuildCapex       — unit tests for _build_capex
TestBuildDebt        — unit tests for _build_debt
TestRunLcoeNoBaseline    — run_lcoe integration, mocked extract_quantities, no baseline
TestRunLcoeWithBaseline  — run_lcoe integration, mocked extract_quantities, with baseline
TestRunLcoeRealSizing    — one end-to-end test that runs actual GHE sizing (slow)

Mock strategy
-------------
extract_quantities is patched to return a fixed GHEQuantities so that bridge
logic can be exercised without running a GHE sizing. The one real-sizing test
is the exception: it builds and sizes a GHE using the same parameters as
test_size_with_bldg_loads and then runs run_lcoe on the result.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ghedesigner.lcoe.bridge import _build_capex, _build_debt, run_lcoe
from ghedesigner.lcoe.quantities import GHEQuantities

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_YEARS = 5
_SPY = 1  # steps_per_year
_T = _YEARS * _SPY

# A GHEQuantities object with realistic-looking values used across mock tests.
_Q = GHEQuantities(
    total_drilling_m=600.0,
    n_boreholes=6,
    n_heat_pumps=1,
    heat_MWh_yr1=120.0,
    cool_MWh_yr1=60.0,
    elec_heat_MWh_yr1=40.0,
    elec_cool_MWh_yr1=20.0,
)

_MINIMAL_COST: dict = {
    "years": _YEARS,
    "steps_per_year": _SPY,
    "real_discount_rate": 0.05,
    "electricity_price": {"values_ts": [100.0] * _T},
    "fixed_capex": [{"name": "Equipment", "cashflow_t0": 50_000.0}],
    "opex_fixed": [{"name": "Maintenance", "series": [5_000.0] * _YEARS}],
    "debt": [
        {
            "name": "Loan",
            "principal": 50_000,
            "nominal_rate_ann": 0.04,
            "years": _YEARS,
            "steps_per_year": _SPY,
        }
    ],
}

_COST_WITH_BASELINE: dict = {
    **_MINIMAL_COST,
    "baseline": {
        "fixed_capex": [{"name": "Gas boiler", "cashflow_t0": 10_000.0}],
        "opex_fixed": [{"name": "Boiler maintenance", "series": [1_000.0] * _YEARS}],
        "opex_variable": [
            {
                "name": "Natural gas",
                "energy_unit": "MWh_heat",
                "rates_per_unit_ts": [50.0] * _T,
            }
        ],
    },
}


def _write_json(data: dict, directory: Path) -> Path:
    path = directory / "lcoe_inputs.json"
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# _build_capex
# ---------------------------------------------------------------------------


class TestBuildCapex:
    def test_no_unit_rate_no_fixed_returns_empty(self):
        items = _build_capex({}, _Q)
        assert items == []

    def test_drilling_rate_creates_item(self):
        cost = {"unit_rate_capex": {"drilling_currency_per_meter": 100.0}}
        items = _build_capex(cost, _Q)
        assert len(items) == 1
        assert items[0].name == "Drilling"
        assert items[0].cashflow_t0 == pytest.approx(100.0 * _Q.total_drilling_m)

    def test_central_loop_requires_both_keys(self):
        # only length provided — should not create an item
        cost = {"unit_rate_capex": {"central_loop_pipe_length_m": 150.0}}
        items = _build_capex(cost, _Q)
        assert not any(i.name == "Central loop piping" for i in items)

        # both keys provided — should create an item
        cost["unit_rate_capex"]["central_loop_pipe_currency_per_meter"] = 200.0
        items = _build_capex(cost, _Q)
        assert any(i.name == "Central loop piping" for i in items)

    def test_fixed_capex_passes_through(self):
        cost = {
            "fixed_capex": [
                {"name": "Heat pumps", "cashflow_t0": 80_000.0},
                {"name": "Controls", "cashflow_t0": 5_000.0},
            ]
        }
        items = _build_capex(cost, _Q)
        names = [i.name for i in items]
        assert "Heat pumps" in names
        assert "Controls" in names
        assert items[0].cashflow_t0 == pytest.approx(80_000.0)

    def test_mixed_unit_rate_and_fixed(self):
        cost = {
            "unit_rate_capex": {"drilling_currency_per_meter": 50.0},
            "fixed_capex": [{"name": "Earthworks", "cashflow_t0": 20_000.0}],
        }
        items = _build_capex(cost, _Q)
        assert len(items) == 2
        names = [i.name for i in items]
        assert "Drilling" in names
        assert "Earthworks" in names


# ---------------------------------------------------------------------------
# _build_debt
# ---------------------------------------------------------------------------


class TestBuildDebt:
    def test_empty_debt_section_returns_empty(self):
        assert _build_debt({}, net_capex=100_000.0, project_years=20, project_steps_per_year=1) == []

    def test_explicit_principal_used_as_is(self):
        section = {
            "debt": [{"name": "Loan", "principal": 200_000, "nominal_rate_ann": 0.03, "years": 10}]
        }
        items = _build_debt(section, net_capex=999_999.0, project_years=20, project_steps_per_year=1)
        assert len(items) == 1
        assert items[0].principal == pytest.approx(200_000.0)

    def test_null_principal_falls_back_to_net_capex(self):
        section = {
            "debt": [{"name": "Loan", "principal": None, "nominal_rate_ann": 0.03, "years": 10}]
        }
        items = _build_debt(section, net_capex=123_456.0, project_years=20, project_steps_per_year=1)
        assert items[0].principal == pytest.approx(123_456.0)

    def test_multiple_debt_items(self):
        section = {
            "debt": [
                {"name": "Loan A", "principal": 100_000, "nominal_rate_ann": 0.03, "years": 10},
                {"name": "Loan B", "principal": 50_000, "nominal_rate_ann": 0.05, "years": 5},
            ]
        }
        items = _build_debt(section, net_capex=0.0, project_years=20, project_steps_per_year=1)
        assert len(items) == 2
        names = [i.name for i in items]
        assert "Loan A" in names
        assert "Loan B" in names


# ---------------------------------------------------------------------------
# run_lcoe — integration with mocked extract_quantities, no baseline
# ---------------------------------------------------------------------------


class TestRunLcoeNoBaseline:
    @pytest.fixture(scope="class")
    def result_and_outdir(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("lcoe_no_baseline")
        json_path = _write_json(_MINIMAL_COST, out)
        with patch("ghedesigner.lcoe.bridge.extract_quantities", return_value=_Q):
            result = run_lcoe(
                ghe_objects=[],
                hp_objects=[],
                lcoe_json_path=json_path,
                output_directory=out,
            )
        return result, out

    def test_result_has_ghe_system_key(self, result_and_outdir):
        result, _ = result_and_outdir
        assert "ghe_system" in result

    def test_no_baseline_key_in_result(self, result_and_outdir):
        result, _ = result_and_outdir
        assert "baseline_system" not in result

    def test_npv_components_sum_to_total(self, result_and_outdir):
        result, _ = result_and_outdir
        r = result["ghe_system"]
        components = r["NPV_capex"] + r["NPV_opex"] + r["NPV_financing"]
        assert components == pytest.approx(r["NPV_total_cost"], rel=1e-6)

    def test_lcoh_positive(self, result_and_outdir):
        result, _ = result_and_outdir
        assert result["ghe_system"]["LCOH_(currency_per_MWh_heat)"] > 0.0

    def test_lcox_positive(self, result_and_outdir):
        result, _ = result_and_outdir
        assert result["ghe_system"]["LCOx_total_(currency_per_MWh_service)"] > 0.0

    def test_npv_capex_positive(self, result_and_outdir):
        """Net CAPEX is positive (equipment cost outweighs subsidies in this fixture)."""
        result, _ = result_and_outdir
        assert result["ghe_system"]["NPV_capex"] > 0.0

    def test_csv_exists_and_non_empty(self, result_and_outdir):
        _, out = result_and_outdir
        csv_path = out / "LCOESummary.csv"
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# run_lcoe — integration with mocked extract_quantities, with baseline
# ---------------------------------------------------------------------------


class TestRunLcoeWithBaseline:
    @pytest.fixture(scope="class")
    def result_and_outdir(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("lcoe_with_baseline")
        json_path = _write_json(_COST_WITH_BASELINE, out)
        with patch("ghedesigner.lcoe.bridge.extract_quantities", return_value=_Q):
            result = run_lcoe(
                ghe_objects=[],
                hp_objects=[],
                lcoe_json_path=json_path,
                output_directory=out,
            )
        return result, out

    def test_both_system_keys_present(self, result_and_outdir):
        result, _ = result_and_outdir
        assert "ghe_system" in result
        assert "baseline_system" in result

    def test_ghe_npv_components_sum_to_total(self, result_and_outdir):
        result, _ = result_and_outdir
        r = result["ghe_system"]
        components = r["NPV_capex"] + r["NPV_opex"] + r["NPV_financing"]
        assert components == pytest.approx(r["NPV_total_cost"], rel=1e-6)

    def test_baseline_npv_components_sum_to_total(self, result_and_outdir):
        result, _ = result_and_outdir
        r = result["baseline_system"]
        components = r["NPV_capex"] + r["NPV_opex"] + r["NPV_financing"]
        assert components == pytest.approx(r["NPV_total_cost"], rel=1e-6)

    def test_baseline_has_variable_opex_cost(self, result_and_outdir):
        """Baseline has natural-gas opex_variable — NPV_opex_variable_other must be > 0."""
        result, _ = result_and_outdir
        assert result["baseline_system"]["NPV_opex_variable_other"] > 0.0

    def test_ghe_has_no_variable_opex_cost(self, result_and_outdir):
        """GHE system has no opex_variable items — NPV_opex_variable_other should be 0."""
        result, _ = result_and_outdir
        assert result["ghe_system"]["NPV_opex_variable_other"] == pytest.approx(0.0)

    def test_both_lcoh_positive(self, result_and_outdir):
        result, _ = result_and_outdir
        assert result["ghe_system"]["LCOH_(currency_per_MWh_heat)"] > 0.0
        assert result["baseline_system"]["LCOH_(currency_per_MWh_heat)"] > 0.0

    def test_csv_exists_and_non_empty(self, result_and_outdir):
        _, out = result_and_outdir
        csv_path = out / "LCOESummary.csv"
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# run_lcoe — one real GHE sizing (slow)
# ---------------------------------------------------------------------------


class TestRunLcoeRealSizing:
    """
    Runs an actual GHE rectangular sizing using test_bldg_loads.csv, then
    passes the result directly into run_lcoe.  This exercises the full
    extract_quantities → bridge pipeline without mocking.
    """

    @pytest.fixture(scope="class")
    def result_and_outdir(self, tmp_path_factory):
        from ghedesigner.enums import TimestepType
        from ghedesigner.ghe.boreholes.core import Borehole
        from ghedesigner.ghe.design.rectangle import (
            DesignRectangle,
            GeometricConstraintsRectangle,
        )
        from ghedesigner.ghe.pipe import Pipe
        from ghedesigner.heat_pump_fixed_cop import HeatPumpFixedCOP
        from ghedesigner.media import Fluid, Grout, Soil

        test_data = Path(__file__).parent / "test_data"
        out = tmp_path_factory.mktemp("lcoe_real_sizing")

        # --- Build and size the GHE ---
        hp_data = {
            "total_load": {
                "column_number": 0,
                "file_path": test_data / "test_bldg_loads.csv",
                "heat_pump_cop": 3,
            }
        }
        heat_pump = HeatPumpFixedCOP("hp1", hp_data)
        ground_loads = heat_pump.get_ground_loads()

        pipe = Pipe.init_single_u_tube(
            inner_diameter=0.03404,
            outer_diameter=0.04216,
            shank_spacing=0.01856,
            roughness=1.0e-6,
            conductivity=0.4,
            rho_cp=1542000.0,
        )
        soil = Soil(k=2.0, rho_cp=2343493.0, ugt=18.3)
        fluid = Fluid("water")
        grout = Grout(1.0, 3901000.0)
        borehole = Borehole(burial_depth=2.0, borehole_radius=0.07)
        geometry = GeometricConstraintsRectangle(
            width=36.5, length=85.0, b_min=3.0, b_max=10
        )
        min_height, max_height = 60, 135

        design = DesignRectangle(
            v_flow=0.5,
            borehole=borehole,
            fluid=fluid,
            pipe=pipe,
            grout=grout,
            soil=soil,
            start_month=1,
            end_month=240,
            max_eft=35,
            min_eft=5,
            max_height=max_height,
            min_height=min_height,
            continue_if_design_unmet=True,
            max_boreholes=None,
            geometric_constraints=geometry,
            hourly_extraction_ground_loads=ground_loads,
            method=TimestepType.HYBRID,
        )
        search = design.find_design()
        search.ghe.compute_g_functions(min_height, max_height)
        search.ghe.size(
            method=TimestepType.HYBRID,
            min_height=min_height,
            max_height=max_height,
            design_min_eft=5,
            design_max_eft=35,
        )

        # --- LCOE cost data — minimal, 5 years ---
        cost_data = {
            "years": 5,
            "steps_per_year": 1,
            "real_discount_rate": 0.05,
            "electricity_price": {"values_ts": [100.0] * 5},
            "unit_rate_capex": {"drilling_currency_per_meter": 800.0},
            "opex_fixed": [{"name": "O&M", "series": [10_000.0] * 5}],
        }
        json_path = _write_json(cost_data, out)

        result = run_lcoe(
            ghe_objects=[search.ghe],
            hp_objects=[heat_pump],
            lcoe_json_path=json_path,
            output_directory=out,
        )
        return result, out

    def test_result_has_ghe_system_key(self, result_and_outdir):
        result, _ = result_and_outdir
        assert "ghe_system" in result

    def test_npv_components_sum_to_total(self, result_and_outdir):
        result, _ = result_and_outdir
        r = result["ghe_system"]
        components = r["NPV_capex"] + r["NPV_opex"] + r["NPV_financing"]
        assert components == pytest.approx(r["NPV_total_cost"], rel=1e-6)

    def test_npv_capex_reflects_actual_drilling(self, result_and_outdir):
        """Drilling CAPEX should be > 0 because the sizing produces boreholes."""
        result, _ = result_and_outdir
        assert result["ghe_system"]["NPV_capex"] > 0.0

    def test_lcoh_positive(self, result_and_outdir):
        result, _ = result_and_outdir
        assert result["ghe_system"]["LCOH_(currency_per_MWh_heat)"] > 0.0

    def test_csv_exists_and_non_empty(self, result_and_outdir):
        _, out = result_and_outdir
        csv_path = out / "LCOESummary.csv"
        assert csv_path.exists()
        assert csv_path.stat().st_size > 0