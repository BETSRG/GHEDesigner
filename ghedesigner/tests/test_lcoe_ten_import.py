"""
Tests that lcoe-ten integrates correctly with GHEDesigner by recreating the
Silkeborg LCOH-with-financing example in code. This test does not rely on any
inputs from GHEdesigner,it simply creates inputs and sends them to the
lcoe_ten package (installed as a dependancy) and checks the correct results
are returned.

The reference inputs come from Silkeborg_LCOH_with_financing.yaml
at https://github.com/mitchute/lcoe-ten/examples:
  - 15 buildings, 6 boreholes, 20-year project horizon
  - Currency: DKK, real discount rate: 3 %
  - A fixed-rate loan at 2.6 % nominal pays for the net CAPEX at t0
"""

from unittest import TestCase

from lcoe_ten.cost_items import (
    CapexScheduleTS,
    DebtScheduleTS,
    EnergyPriceTS,
    LoadTS,
    OpexFixedTS,
)
from lcoe_ten.system_model import evaluate_project_ts

# ---------------------------------------------------------------------------
# Reference CAPEX line items (DKK, at t=0).
# Positive values are costs; negative values are customer contributions /
# subsidies that offset the project CAPEX.
# ---------------------------------------------------------------------------
_CAPEX_ITEMS = [
    ("Earthworks - main pipeline network", 136693.0),
    ("Pipeline network - main pipes and connections", 147188.0),
    ("Electricity supply to HP", 123300.0),
    ("Energy meters (15 x 3,120 DKK/unit)", 46800.0),
    ("Excavation work - service lines (15 x 6,600 DKK)", 99000.0),
    ("Investment contribution (15 x -7,380 DKK)", -110700.0),
    ("Increased investment contribution (15 x -29,000 DKK)", -435000.0),
    ("Service line contribution (15 x -18,800 DKK)", -282000.0),
    ("Construction preparation contribution (15 x -18,357.33 DKK)", -275360.0),
    ("Heat pumps (15 x 58,867 DKK)", 883005.0),
    ("Drilling (6 x 79,746 DKK)", 478476.0),
]

# Net CAPEX = sum of all items = 811,402 DKK (also used as loan principal)
NET_CAPEX_T0 = sum(v for _, v in _CAPEX_ITEMS)  # 811_402.0

# Annual electricity prices (DKK/MWh) over 20 years
_ELEC_PRICES = [
    1004.46,
    1004.46,
    936.46,
    856.46,
    *([844.46] * 16),
]

# Electricity consumed for heating per year (MWh).
_ELEC_HEAT_MWH = [
    35.2941176470588,
    33.3333333333333,
    *([32.4324324324324] * 18),
]


def _build_silkeborg_inputs() -> dict:
    """Return kwargs ready to pass to evaluate_project_ts."""
    years = 20
    steps_per_year = 1

    capex_ts = [CapexScheduleTS(name=name, cashflow_t0=v) for name, v in _CAPEX_ITEMS]

    opex_fixed_ts = [
        OpexFixedTS(name="O&M contract", series=[37320.0] * years),
    ]

    price_paths_ts = {
        "electricity": EnergyPriceTS(name="electricity_heat", values_ts=_ELEC_PRICES),
    }

    loads_ts = LoadTS(
        years=years,
        steps_per_year=steps_per_year,
        heat_MWh_ts=[120.0] * years,
        elec_heat_MWh_ts=list(_ELEC_HEAT_MWH),
    )

    debt_ts = [
        DebtScheduleTS(
            name="Fixed-rate loan",
            principal=NET_CAPEX_T0,
            nominal_rate_ann=0.026,
            years=years,
            steps_per_year=steps_per_year,
            inflation_ann=0.0,
            grace_steps=0,
            fees_t0=0,
        )
    ]

    return {
        "years": years,
        "steps_per_year": steps_per_year,
        "real_discount_rate": 0.03,
        "capex_ts": capex_ts,
        "opex_fixed_ts": opex_fixed_ts,
        "opex_variable_ts": [],
        "price_paths_ts": price_paths_ts,
        "loads_ts": loads_ts,
        "debt_ts": debt_ts,
    }


class TestSilkeborgLCOH(TestCase):
    """Verify lcoe-ten produces correct results for the Silkeborg reference case."""

    @classmethod
    def setUpClass(cls):
        cls.result = evaluate_project_ts(**_build_silkeborg_inputs())

    # ------------------------------------------------------------------
    # Structural checks
    # ------------------------------------------------------------------

    def test_returns_expected_keys(self):
        expected = {
            "T",
            "NPV_total_cost",
            "NPV_capex",
            "NPV_opex",
            "NPV_financing",
            "NPV_opex_fixed",
            "NPV_elec_total",
            "PV_heat_MWh",
            "PV_cool_MWh",
            "PV_service_MWh",
            "LCOx_total_(currency_per_MWh_service)",
            "LCOH_(currency_per_MWh_heat)",
            "LCOC_(currency_per_MWh_cool)",
        }
        self.assertTrue(expected.issubset(self.result.keys()))

    def test_time_steps(self):
        self.assertEqual(self.result["T"], 20)

    # ------------------------------------------------------------------
    # CAPEX: all costs are at t=0, so NPV_capex == net CAPEX exactly
    # ------------------------------------------------------------------

    def test_npv_capex_exact(self):
        """With all CAPEX at t0 and no residual, NPV_capex equals the net t0 cost."""
        self.assertAlmostEqual(self.result["NPV_capex"], NET_CAPEX_T0, places=2)

    # ------------------------------------------------------------------
    # Fixed OPEX: constant 37,320 DKK/yr for 20 years discounted at 3 %
    # NPV = 37320 * (1 - 1.03^{-20}) / 0.03  (end-of-period annuity)
    # ------------------------------------------------------------------

    def test_npv_opex_fixed(self):
        annuity_factor = (1.0 - 1.03**-20) / 0.03
        expected = 37320.0 * annuity_factor
        self.assertAlmostEqual(self.result["NPV_opex_fixed"], expected, places=1)

    # ------------------------------------------------------------------
    # Discounted heat: 120 MWh/yr * annuity_factor(20yr, 3%)
    # ------------------------------------------------------------------

    def test_pv_heat_mwh(self):
        annuity_factor = (1.0 - 1.03**-20) / 0.03
        expected = 120.0 * annuity_factor
        self.assertAlmostEqual(self.result["PV_heat_MWh"], expected, places=4)

    def test_no_cooling_loads(self):
        """There is no cooling in this case; PV_cool_MWh should be zero."""
        self.assertAlmostEqual(self.result["PV_cool_MWh"], 0.0, places=6)

    def test_service_equals_heat(self):
        """Without cooling, total service energy equals heat energy."""
        self.assertAlmostEqual(
            self.result["PV_service_MWh"], self.result["PV_heat_MWh"], places=6
        )

    def test_lcoh_equals_lcox(self):
        """LCOH and LCOx_total should be identical when there is no cooling."""
        self.assertAlmostEqual(
            self.result["LCOH_(currency_per_MWh_heat)"],
            self.result["LCOx_total_(currency_per_MWh_service)"],
            places=6,
        )

    # ------------------------------------------------------------------
    # Financing: real discount rate (3 %) > loan rate (2.6 %), so the
    # PV of debt-service payments is less than the principal received.
    # NPV_financing should therefore be negative (net benefit to project).
    # ------------------------------------------------------------------

    def test_financing_npv_sign(self):
        """Financing NPV should be negative when discount rate > loan rate."""
        self.assertLess(self.result["NPV_financing"], 0.0)

    # ------------------------------------------------------------------
    # Sanity / plausibility checks
    # ------------------------------------------------------------------

    def test_lcoh_positive(self):
        self.assertGreater(self.result["LCOH_(currency_per_MWh_heat)"], 0.0)

    def test_npv_total_positive(self):
        self.assertGreater(self.result["NPV_total_cost"], 0.0)

    def test_npv_components_sum_to_total(self):
        """NPV_capex + NPV_opex + NPV_financing should equal NPV_total_cost."""
        components = (
            self.result["NPV_capex"]
            + self.result["NPV_opex"]
            + self.result["NPV_financing"]
        )
        self.assertAlmostEqual(components, self.result["NPV_total_cost"], places=4)
