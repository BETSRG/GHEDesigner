#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError

# Note: JSON schema does not currently have a good way to handle case-insensitive enums.
#       I think we should enforce case sensitivity. The validation script should clearly alert the user.
#       More details here: https://github.com/json-schema-org/community/discussions/148


def _format_path(path: Sequence[Any]) -> str:
    """Human-readable path like Root → building → building1 → total_load."""
    if not path:
        return "Root"
    return "Root \u2192 " + " \u2192 ".join(map(str, path))


def _format_json_pointer(path: Sequence[Any]) -> str:
    """JSON Pointer-ish (RFC6901-style) path like /building/building1/total_load."""
    if not path:
        return "/"

    # escape ~ and /
    def esc(p: Any) -> str:
        s = str(p)
        return s.replace("~", "~0").replace("/", "~1")

    return "/" + "/".join(esc(p) for p in path)


def _schema_property_names(schema: dict) -> list[str]:
    props = schema.get("properties", {})
    if isinstance(props, dict):
        return sorted(props.keys())
    return []


def _extract_additional_properties_name(message: str) -> Optional[str]:
    # jsonschema messages often look like: "Additional properties are not allowed ('heat_pump_cop' was unexpected)"
    m = re.search(r"\('([^']+)' was unexpected\)", message)
    return m.group(1) if m else None


@dataclass(order=True)
class RankedError:
    # lower is better
    rank: tuple[int, int, int]
    # Do NOT include the ValidationError object in ordering comparisons.
    # When ranks tie, Python would otherwise attempt to compare ValidationError
    # instances and raise: TypeError: '<' not supported between instances of
    # 'ValidationError' and 'ValidationError'.
    error: ValidationError = field(compare=False)


def _rank_error(err: ValidationError) -> RankedError:
    """
    Heuristic ranking:
      - deeper paths first
      - prefer 'additionalProperties'/'required'/'enum'/'type' over composite oneOf noise
      - shorter messages slightly preferred
    """
    validator_weight = {
        "additionalProperties": 0,
        "required": 1,
        "enum": 2,
        "type": 3,
        "const": 4,
        "minimum": 5,
        "maximum": 5,
        "minItems": 6,
        "maxItems": 6,
        "oneOf": 20,
        "anyOf": 21,
        "allOf": 22,
    }
    vw = validator_weight.get(err.validator, 10)
    depth = len(list(err.path))
    msg_len = len(err.message or "")
    # rank tuple: (validator_weight, -depth, msg_len)
    return RankedError(rank=(vw, -depth, msg_len), error=err)


def _best_error(errors: Iterable[ValidationError]) -> ValidationError:
    ranked = sorted(_rank_error(e) for e in errors)
    return ranked[0].error


def _summarize_oneof(err: ValidationError) -> list[str]:
    """
    jsonschema stores per-branch errors in err.context.
    - If context is empty, message is all we have.
    - If multiple branches matched, jsonschema usually says "is valid under each of ..."
      and context may be empty; we can still hint likely ambiguity.
    """
    lines: list[str] = []
    if not err.context:
        # still provide a general hint
        lines.append("This value did not match the required schema shape for this field (oneOf).")
        lines.append(
            "Tip: ensure exactly one of the allowed variants applies; avoid mixing keys from multiple variants."
        )
        return lines

    # Group suberrors by which branch they came from (best-effort: keep in order)
    for i, sub in enumerate(err.context, start=1):
        p = _format_path([sub.path])
        lines.append(f"Variant {i} failed at {p}: {sub.message}")
    lines.append("Tip: adjust the object so it matches exactly one variant.")
    return lines


def _suggest_fix(err: ValidationError) -> list[str]:
    fix: list[str] = []

    if err.validator == "required":
        # err.validator_value is typically a list of required keys
        missing = err.message
        fix.append(f"Add the missing required field(s). Details: {missing}")
        return fix

    if err.validator == "additionalProperties":
        bad = _extract_additional_properties_name(err.message)
        allowed = _schema_property_names(err.schema)  # current schema node
        if bad:
            fix.append(f"Remove or rename the unexpected property: '{bad}'.")
        else:
            fix.append("Remove or rename unexpected properties at this location.")
        if allowed:
            fix.append("Allowed properties here are: " + ", ".join(allowed))
        # also detect a common pattern from your schema: branch uses a key not allowed at parent level
        fix.append("If this property is intended, the schema may need to include it in 'properties' at this level.")
        return fix

    if err.validator == "enum":
        allowed = err.validator_value if isinstance(err.validator_value, list) else err.schema.get("enum")
        if allowed:
            fix.append("Use one of the allowed values: " + ", ".join(map(str, allowed)))
            fix.append("Note: enum matching is case-sensitive.")
        return fix

    if err.validator == "type":
        expected = err.validator_value
        fix.append(f"Ensure the value is of type: {expected}")
        return fix

    if err.validator in ("oneOf", "anyOf"):
        fix.extend(_summarize_oneof(err))
        return fix

    # generic fallback
    if "type" in err.schema:
        fix.append(f"Ensure the value is of type: {err.schema['type']}")
    return fix


def validate_input_file(input_file_path: Path) -> None:
    """
    Validate input file against the schema with clearer, structured error messages.
    """
    instance = json.loads(input_file_path.read_text())
    schema_path = Path(__file__).parent / "schemas" / "ghedesigner.schema.json"
    schema = json.loads(schema_path.read_text())

    validator = Draft7Validator(schema)
    errors = list(validator.iter_errors(instance))
    if not errors:
        return

    err = _best_error(errors)

    # High-signal header
    print("\nValidation Error:", file=sys.stderr)
    path_list = list(err.path)
    print(f"  Location:        {_format_path(path_list)}", file=sys.stderr)
    print(f"  JSON Pointer:    {_format_json_pointer(path_list)}", file=sys.stderr)
    print(f"  Validator:       {err.validator}", file=sys.stderr)
    print(f"  Message:         {err.message}", file=sys.stderr)

    # Context (schema and instance hints)
    if err.validator == "additionalProperties":
        allowed = _schema_property_names(err.schema)
        if allowed:
            print(f"  Allowed keys:    {', '.join(allowed)}", file=sys.stderr)

    # Suggested fix section
    fix_lines = _suggest_fix(err)
    if fix_lines:
        print("\nSuggested Fix:", file=sys.stderr)
        for line in fix_lines:
            print(f"  - {line}", file=sys.stderr)

    # Optional: show a second-best error when the best is oneOf noise
    if err.validator in ("oneOf", "anyOf") and len(errors) > 1:
        alt = _best_error(e for e in errors if e is not err)
        print("\nAdditional Detail:", file=sys.stderr)
        print(f"  Another relevant issue at {_format_path(list(alt.path))}: {alt.message}", file=sys.stderr)

    print("\nFor example inputs, see the demo files.", file=sys.stderr)

    # preserve old behavior
    raise err


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate GHEDesigner input JSON against schema.")
    parser.add_argument("input_json", type=Path, help="Path to input JSON file")
    args = parser.parse_args(argv)

    instance_path = args.input_json.resolve()
    try:
        validate_input_file(instance_path)
    except ValidationError:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
