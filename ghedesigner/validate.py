#!/usr/bin/env python
import sys
from json import loads
from pathlib import Path

from jsonschema import ValidationError, validate

# Note: JSON schema does not currently have a good way to handle case-insensitive enums.
#       I think we should enforce case sensitivity.  The validation script should clearly alert the user.
#       More details here: https://github.com/json-schema-org/community/discussions/148


def validate_input_file(input_file_path: Path) -> None:
    """
    Validate input file against the schema
    """
    try:
        instance = loads(input_file_path.read_text())
        schema_path = Path(__file__).parent / "schemas" / "ghedesigner.schema.json"
        schema = loads(schema_path.read_text())
        validate(instance=instance, schema=schema)
    except ValidationError as error:
        print("\n Validation Error:")
        print(f"  Bad Input Location: {' → '.join(map(str, error.path)) if error.path else 'Root'}")
        print(f"  Problematic Key: {error.path[-1] if error.path else 'N/A'}")
        print(f"  Error: {error.message}")
        fix = "\n Suggested Fix:"
        if "required" in error.schema and isinstance(error.schema["required"], list):
            print(f"{fix}  Ensure that the following required keys are present: {', '.join(error.schema['required'])}")
        elif "enum" in error.schema:
            print(f"{fix}  Use one of the allowed values: {', '.join(map(str, error.schema['enum']))}")
        elif "type" in error.schema:
            print(f"{fix}  Ensure the value is of type: {error.schema['type']}")
        print("For example inputs, see the demo files.", file=sys.stderr)
        raise ValidationError("") from None


if __name__ == "__main__":
    instance_path = Path(sys.argv[1]).resolve()
    try:
        validate_input_file(instance_path)
    except ValidationError:
        sys.exit(1)
