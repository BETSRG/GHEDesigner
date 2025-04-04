#!/usr/bin/env python
import sys
from json import loads
from pathlib import Path

from jsonschema import ValidationError, validate

# Note: JSON schema does not currently have a good way to handle case-insensitive enums.
#       I think we should enforce case sensitivity.  The validation script should clearly alert the user.
#       More details here: https://github.com/json-schema-org/community/discussions/148


class ValidationResponse:
    def __init__(self, success: bool, message: str = ""):
        self.success = success
        self.message = message


def validate_input_file(input_file_path: Path) -> ValidationResponse:
    """
    Validate input file against the schema
    """
    try:
        instance = loads(input_file_path.read_text())
        schema_path = Path(__file__).parent / "ghedesigner.schema.json"
        schema = loads(schema_path.read_text())
        validate(instance=instance, schema=schema)
        return ValidationResponse(True)
    except ValidationError as error:
        summary = ""
        summary += "\n Validation Error:"
        summary += f"  Bad Input Location: {' → '.join(map(str, error.path)) if error.path else 'Root'}"
        summary += f"  Problematic Key: {error.path[-1] if error.path else 'N/A'}"
        summary += f"  Error: {error.message}"
        fix = "\n Suggested Fix:"
        if "required" in error.schema and isinstance(error.schema["required"], list):
            summary += (
                f"{fix}  Ensure that the following required keys are present: {', '.join(error.schema['required'])}"
            )
        elif "enum" in error.schema:
            summary += f"{fix}  Use one of the allowed values: {', '.join(map(str, error.schema['enum']))}"
        elif "type" in error.schema:
            summary += f"{fix}  Ensure the value is of type: {error.schema['type']}"
        summary += "For example inputs, see the demo files."
        return ValidationResponse(False, summary)


if __name__ == "__main__":
    instance_path = Path(sys.argv[1]).resolve()
    validation_response = validate_input_file(instance_path)
    sys.exit(0 if validation_response.success else 1)
