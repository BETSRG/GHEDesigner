import sys
from json import loads
from pathlib import Path

from jsonschema import validate, ValidationError


# Note: JSON schema does not currently have a good way to handle case-insentive enums.
#       Some fields are upper cased manually here for validation purposes.
#       More details here: https://github.com/json-schema-org/community/discussions/148


def validate_schema_instance(schema_file_name: str, instance: dict, error_msg: str) -> None:
    """
    Base-level worker function to validate schema instances
    """
    try:
        schema_dir = Path(__file__).parent / "schemas"
        schema_path = schema_dir / schema_file_name
        schema = loads(schema_path.read_text())
        validate(instance=instance, schema=schema)
    except ValidationError:
        raise ValidationError(error_msg)


def validate_file_structure(instance: dict) -> None:
    validate_schema_instance(
        schema_file_name="file_structure.schema.json",
        instance=instance,
        error_msg="Errors in input file structure. See demo files for examples."
    )


def validate_fluid(instance: dict) -> None:
    fluid_name = str(instance["fluid_name"]).upper()
    instance["fluid_name"] = fluid_name
    validate_schema_instance(
        schema_file_name="fluid.schema.json",
        instance=instance,
        error_msg="Errors in \"fluid\" input object."
    )


def validate_grout(instance: dict) -> None:
    validate_schema_instance(
        schema_file_name="grout.schema.json",
        instance=instance,
        error_msg="Errors in \"grout\" input object."
    )


def validate_soil(instance: dict) -> None:
    validate_schema_instance(
        schema_file_name="soil.schema.json",
        instance=instance,
        error_msg="Errors in \"soil\" input object."
    )


def validate_pipe(instance: dict) -> None:
    pipe_arrangement = str(instance["arrangement"]).upper()
    instance["arrangement"] = pipe_arrangement

    schema_map = {
        "SINGLEUTUBE": "pipe_single_double_u_tube.schema.json",
        "DOUBLEUTUBESERIES": "pipe_single_double_u_tube.schema.json",
        "DOUBLEUTUBEPARALLEL": "pipe_single_double_u_tube.schema.json",
        "COAXIAL": "pipe_coaxial.schema.json"
    }

    if pipe_arrangement not in schema_map.keys():
        raise ValidationError("Pipe arrangement not found.")

    validate_schema_instance(
        schema_file_name=schema_map[pipe_arrangement],
        instance=instance,
        error_msg="Errors in \"pipe\" input object."
    )


def validate_borehole(instance: dict) -> None:
    validate_schema_instance(
        schema_file_name="borehole.schema.json",
        instance=instance,
        error_msg="Errors in \"borehole\" input object."
    )


def validate_simulation(instance: dict) -> None:
    timestep = str(instance["timestep"]).upper()
    instance["timestep"] = timestep
    validate_schema_instance(
        schema_file_name="simulation.schema.json",
        instance=instance,
        error_msg="Errors in \"simulation\" input object."
    )


def validate_geometric(instance: dict) -> None:
    method = str(instance["method"]).upper()
    instance["method"] = method

    schema_map = {
        "NEARSQUARE": "geometric_near_square.schema.json",
        "BIRECTANGLE": "geometric_bi_rectangle.schema.json",
        "RECTANGLE": "geometric_rectangle.schema.json",
        "BIZONEDRECTANGLE": "geometric_bi_zoned_rectangle.schema.json",
        "BIRECTANGLECONSTRAINED": "geometric_bi_rectangle_constrained.schema.json",
        "ROWWISE": "geometric_rowwise.schema.json",
    }

    if method not in schema_map.keys():
        raise ValidationError("Geometric constraint method not recognized.")

    validate_schema_instance(
        schema_file_name=schema_map[method],
        instance=instance,
        error_msg="Errors in \"geometric_constraints\" input object."
    )


def validate_design(instance: dict) -> None:
    flow_type = str(instance["flow_type"]).upper()
    instance["flow_type"] = flow_type
    validate_schema_instance(
        schema_file_name="design.schema.json",
        instance=instance,
        error_msg="Errors in \"design\" input object."
    )


def validate_input_file(input_file_path: Path) -> None:
    """
    Validate input file against all schemas
    """

    # get instance data
    instance = loads(input_file_path.read_text())

    # validate
    validate_file_structure(instance)
    validate_fluid(instance["fluid"])
    validate_grout(instance["grout"])
    validate_soil(instance["soil"])
    validate_pipe(instance["pipe"])
    validate_borehole(instance["borehole"])
    validate_simulation(instance["simulation"])
    validate_geometric(instance["geometric_constraints"])
    validate_design(instance["design"])


if __name__ == "__main__":
    instance_path = Path(sys.argv[1]).resolve()
    validate_input_file(instance_path)
