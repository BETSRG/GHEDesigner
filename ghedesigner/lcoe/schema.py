"""
JSON schema for the LCOE cost input file and validation helpers.

The schema is intentionally permissive about optional sections (unit_rate_capex,
fixed_capex, debt, etc.) so that minimal files work without errors.  Mandatory
fields are only those required by evaluate_project_ts: years, steps_per_year,
real_discount_rate, and electricity_price.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError

# ---------------------------------------------------------------------------
# Shared sub-schemas
# ---------------------------------------------------------------------------

_CAPEX_ITEM = {
    "type": "object",
    "required": ["name"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "cashflow_t0": {"type": "number", "default": 0.0},
        "cashflow_ts": {
            "type": "array",
            "items": {"type": "number"},
            "default": [],
        },
        "residual_at_end": {"type": "number", "default": 0.0},
    },
}

_OPEX_FIXED_ITEM = {
    "type": "object",
    "required": ["name", "series"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "series": {"type": "array", "items": {"type": "number"}, "minItems": 1},
    },
}

_OPEX_VARIABLE_ITEM = {
    "type": "object",
    "required": ["name", "energy_unit", "rates_per_unit_ts"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "energy_unit": {
            "type": "string",
            "enum": ["MWh_heat", "MWh_cool", "MWh_elec"],
            "description": (
                "Energy quantity that rates_per_unit_ts is multiplied against to compute cost. "
                "Use 'MWh_elec' for electricity tariffs. "
                "For fuel-based energy sources (natural gas, propane, district heat, etc.) "
                "that are proportional to delivered heat or cooling, use 'MWh_heat' or "
                "'MWh_cool' and set rates_per_unit_ts to the fuel price divided by the system's "
                "thermal efficiency (e.g. gas_price / boiler_efficiency)."
            ),
        },
        "rates_per_unit_ts": {"type": "array", "items": {"type": "number"}, "minItems": 1},
    },
}

_ELECTRICITY_PRICE = {
    "type": "object",
    "required": ["values_ts"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "values_ts": {"type": "array", "items": {"type": "number"}, "minItems": 1},
    },
}

_DEBT_ITEM = {
    "type": "object",
    "required": ["name", "nominal_rate_ann", "years"],
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        # null means "use computed net CAPEX at runtime"
        "principal": {"oneOf": [{"type": "number"}, {"type": "null"}]},
        "nominal_rate_ann": {"type": "number", "minimum": 0.0},
        "years": {"type": "integer", "minimum": 1},
        "steps_per_year": {"type": "integer", "minimum": 1, "default": 1},
        "grace_steps": {"type": "integer", "minimum": 0, "default": 0},
        "fees_t0": {"type": "number", "default": 0.0},
        "inflation_ann": {"type": "number", "default": 0.0},
    },
}

_UNIT_RATE_CAPEX = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "drilling_currency_per_meter": {"type": "number", "minimum": 0.0},
        # User must supply pipe lengths (GHEDesigner does not compute network lengths)
        "central_loop_pipe_length_m": {"type": "number", "minimum": 0.0},
        "central_loop_pipe_currency_per_meter": {"type": "number", "minimum": 0.0},
        "ghe_header_pipe_length_m": {"type": "number", "minimum": 0.0},
        "ghe_header_pipe_currency_per_meter": {"type": "number", "minimum": 0.0},
    },
}

# Baseline section reuses most of the top-level properties but has no
# unit_rate_capex (it does not depend on GHEDesigner sizing outputs).
_BASELINE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "fixed_capex": {"type": "array", "items": _CAPEX_ITEM},
        "opex_fixed": {"type": "array", "items": _OPEX_FIXED_ITEM},
        "opex_variable": {
            "type": "array",
            "items": _OPEX_VARIABLE_ITEM,
            "description": (
                "Variable operating costs scaled by an energy quantity. "
                "This is the correct place to model fuel costs for baseline systems "
                "(natural gas boiler, propane, district heating tariffs, etc.): "
                "set unit to 'MWh_heat' and rates_per_unit_ts to the fuel price per MWh of "
                "delivered heat, adjusted for the baseline system's thermal efficiency."
            ),
        },
        "electricity_price": _ELECTRICITY_PRICE,
        "aux_electric_kw": {"type": "number", "minimum": 0.0, "default": 0.0},
        "debt": {"type": "array", "items": _DEBT_ITEM},
    },
}

# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------

LCOE_INPUT_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "GHEDesigner LCOE cost input",
    "type": "object",
    "required": ["years", "steps_per_year", "real_discount_rate", "electricity_price"],
    "additionalProperties": False,
    "properties": {
        "currency": {"type": "string"},
        "years": {"type": "integer", "minimum": 1},
        "steps_per_year": {"type": "integer", "minimum": 1},
        "real_discount_rate": {"type": "number"},
        "unit_rate_capex": _UNIT_RATE_CAPEX,
        "fixed_capex": {"type": "array", "items": _CAPEX_ITEM},
        "opex_fixed": {"type": "array", "items": _OPEX_FIXED_ITEM},
        "opex_variable": {"type": "array", "items": _OPEX_VARIABLE_ITEM},
        "electricity_price": _ELECTRICITY_PRICE,
        "aux_electric_kw": {"type": "number", "minimum": 0.0, "default": 0.0},
        "debt": {"type": "array", "items": _DEBT_ITEM},
        "baseline": _BASELINE,
    },
}


def validate_lcoe_input(data: dict[str, Any]) -> None:
    """
    Validate a parsed LCOE cost dict against LCOE_INPUT_SCHEMA.

    Raises
    ------
    jsonschema.exceptions.ValidationError
        If the data does not conform to the schema.
    """
    validator = Draft7Validator(LCOE_INPUT_SCHEMA)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        # Re-raise the first error with a clear path prefix
        first = errors[0]
        path = " → ".join(str(p) for p in first.absolute_path) or "root"
        raise ValidationError(f"LCOE input error at [{path}]: {first.message}")


def validate_lcoe_input_file(path: Path) -> None:
    """Load a JSON file and validate it against the LCOE schema."""
    import json

    with open(path) as f:
        data = json.load(f)
    validate_lcoe_input(data)
