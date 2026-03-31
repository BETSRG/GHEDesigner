"""
Unit tests for ghedesigner.lcoe.schema.validate_lcoe_input.

Covers:
  - Minimal valid input passes
  - Each required field missing raises ValidationError
  - Wrong types raise ValidationError
  - Unknown keys raise ValidationError (additionalProperties: False)
  - opex_variable unit enum enforcement
  - debt principal (explicit vs null)
  - baseline section structure
"""

import pytest
from jsonschema.exceptions import ValidationError

from ghedesigner.lcoe.schema import validate_lcoe_input

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_YEARS = 5
_T = _YEARS  # steps_per_year=1

_MINIMAL = {
    "years": _YEARS,
    "steps_per_year": 1,
    "real_discount_rate": 0.03,
    "electricity_price": {"values_ts": [100.0] * _T},
}


def _with(**overrides):
    """Return a copy of _MINIMAL with the given keys overridden or added."""
    return {**_MINIMAL, **overrides}


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------


class TestRequiredFields:
    def test_minimal_valid_passes(self):
        validate_lcoe_input(_MINIMAL)

    @pytest.mark.parametrize("field", ["years", "steps_per_year", "real_discount_rate", "electricity_price"])
    def test_missing_required_field_raises(self, field):
        data = {k: v for k, v in _MINIMAL.items() if k != field}
        with pytest.raises(ValidationError):
            validate_lcoe_input(data)

    def test_unknown_top_level_key_raises(self):
        with pytest.raises(ValidationError):
            validate_lcoe_input(_with(not_a_real_field=42))


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------


class TestTypes:
    def test_years_must_be_integer(self):
        with pytest.raises(ValidationError):
            validate_lcoe_input(_with(years=20.5))

    def test_years_as_string_raises(self):
        with pytest.raises(ValidationError):
            validate_lcoe_input(_with(years="twenty"))

    def test_steps_per_year_must_be_integer(self):
        with pytest.raises(ValidationError):
            validate_lcoe_input(_with(steps_per_year="monthly"))

    def test_real_discount_rate_must_be_number(self):
        with pytest.raises(ValidationError):
            validate_lcoe_input(_with(real_discount_rate="five_percent"))

    def test_electricity_price_values_ts_must_be_array(self):
        with pytest.raises(ValidationError):
            validate_lcoe_input(_with(electricity_price={"values_ts": 100.0}))


# ---------------------------------------------------------------------------
# opex_variable
# ---------------------------------------------------------------------------


class TestOpexVariable:
    @pytest.mark.parametrize("energy_unit", ["MWh_elec", "MWh_heat", "MWh_cool"])
    def test_valid_units_pass(self, energy_unit):
        data = _with(opex_variable=[
            {"name": "item", "energy_unit": energy_unit, "rates_per_unit_ts": [10.0] * _T}
        ])
        validate_lcoe_input(data)

    def test_invalid_unit_raises(self):
        data = _with(opex_variable=[
            {"name": "gas", "energy_unit": "MWh_gas", "rates_per_unit_ts": [50.0] * _T}
        ])
        with pytest.raises(ValidationError):
            validate_lcoe_input(data)

    def test_missing_rates_per_unit_ts_raises(self):
        data = _with(opex_variable=[{"name": "gas", "energy_unit": "MWh_heat"}])
        with pytest.raises(ValidationError):
            validate_lcoe_input(data)

    def test_unknown_opex_variable_key_raises(self):
        data = _with(opex_variable=[
            {"name": "gas", "energy_unit": "MWh_heat", "rates_per_unit_ts": [50.0] * _T, "extra": 1}
        ])
        with pytest.raises(ValidationError):
            validate_lcoe_input(data)


# ---------------------------------------------------------------------------
# debt
# ---------------------------------------------------------------------------


class TestDebt:
    def test_explicit_principal_passes(self):
        data = _with(debt=[
            {"name": "loan", "nominal_rate_ann": 0.03, "years": _YEARS, "principal": 500_000}
        ])
        validate_lcoe_input(data)

    def test_null_principal_passes(self):
        """null principal means use computed net CAPEX at runtime."""
        data = _with(debt=[
            {"name": "loan", "nominal_rate_ann": 0.03, "years": _YEARS, "principal": None}
        ])
        validate_lcoe_input(data)

    def test_missing_nominal_rate_raises(self):
        data = _with(debt=[{"name": "loan", "years": _YEARS}])
        with pytest.raises(ValidationError):
            validate_lcoe_input(data)

    def test_missing_years_in_debt_raises(self):
        data = _with(debt=[{"name": "loan", "nominal_rate_ann": 0.03}])
        with pytest.raises(ValidationError):
            validate_lcoe_input(data)

    def test_negative_nominal_rate_raises(self):
        data = _with(debt=[{"name": "loan", "nominal_rate_ann": -0.01, "years": _YEARS}])
        with pytest.raises(ValidationError):
            validate_lcoe_input(data)


# ---------------------------------------------------------------------------
# baseline
# ---------------------------------------------------------------------------


class TestBaseline:
    def test_empty_baseline_passes(self):
        validate_lcoe_input(_with(baseline={}))

    def test_baseline_with_fuel_opex_via_mwh_heat_passes(self):
        """Gas/propane costs should be modelled as opex_variable with unit MWh_heat."""
        data = _with(baseline={
            "opex_variable": [
                {"name": "Natural gas", "energy_unit": "MWh_heat", "rates_per_unit_ts": [50.0] * _T}
            ]
        })
        validate_lcoe_input(data)

    def test_baseline_with_fixed_capex_passes(self):
        data = _with(baseline={
            "fixed_capex": [{"name": "Gas boiler", "cashflow_t0": 10_000.0}]
        })
        validate_lcoe_input(data)

    def test_baseline_unit_rate_capex_raises(self):
        """Baseline has no unit_rate_capex — it does not depend on GHEDesigner sizing."""
        data = _with(baseline={"unit_rate_capex": {"drilling_currency_per_meter": 100.0}})
        with pytest.raises(ValidationError):
            validate_lcoe_input(data)

    def test_baseline_unknown_key_raises(self):
        with pytest.raises(ValidationError):
            validate_lcoe_input(_with(baseline={"not_a_field": 1}))