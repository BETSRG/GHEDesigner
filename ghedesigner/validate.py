import sys
from json import loads
from pathlib import Path

from jsonschema import validate, ValidationError

from ghedesigner.enums import BHPipeType, DesignGeomType


# Note: JSON schema does not currently have a good way to handle case-insensitive enums.
#       Some fields are upper-cased manually here for validation purposes.
#       More details here: https://github.com/json-schema-org/community/discussions/148


def validate_schema_instance(schema_file_name: str, instance: dict, error_msg: str) -> int:
    """
    Base-level worker function to validate schema instances
    """
    try:
        schema_dir = Path(__file__).parent / "schemas"
        schema_path = schema_dir / schema_file_name
        schema = loads(schema_path.read_text())
        validate(instance=instance, schema=schema)
        return 0
    except ValidationError:
        print(error_msg, file=sys.stderr)
        return 1


def validate_file_structure(instance: dict) -> int:
    return validate_schema_instance(
        schema_file_name="file_structure.schema.json",
        instance=instance,
        error_msg="Errors in input file structure. See demo files for examples."
    )


def validate_fluid(instance: dict) -> int:
    fluid_name = str(instance["fluid_name"]).upper()
    instance["fluid_name"] = fluid_name
    return validate_schema_instance(
        schema_file_name="fluid.schema.json",
        instance=instance,
        error_msg="Errors in \"fluid\" input object."
    )


def validate_grout(instance: dict) -> int:
    return validate_schema_instance(
        schema_file_name="grout.schema.json",
        instance=instance,
        error_msg="Errors in \"grout\" input object."
    )


def validate_soil(instance: dict) -> int:
    return validate_schema_instance(
        schema_file_name="soil.schema.json",
        instance=instance,
        error_msg="Errors in \"soil\" input object."
    )


def validate_pipe(instance: dict) -> int:
    pipe_arrangement = str(instance["arrangement"]).upper()
    instance["arrangement"] = pipe_arrangement

    schema_map = {
        BHPipeType.SINGLEUTUBE.name: "pipe_single_double_u_tube.schema.json",
        BHPipeType.DOUBLEUTUBESERIES.name: "pipe_single_double_u_tube.schema.json",
        BHPipeType.DOUBLEUTUBEPARALLEL.name: "pipe_single_double_u_tube.schema.json",
        BHPipeType.COAXIAL.name: "pipe_coaxial.schema.json"
    }

    if pipe_arrangement not in schema_map.keys():
        print("Pipe arrangement not found.", file=sys.stderr)
        return 1

    return validate_schema_instance(
        schema_file_name=schema_map[pipe_arrangement],
        instance=instance,
        error_msg="Errors in \"pipe\" input object."
    )


def validate_borehole(instance: dict) -> int:
    return validate_schema_instance(
        schema_file_name="borehole.schema.json",
        instance=instance,
        error_msg="Errors in \"borehole\" input object."
    )


def validate_simulation(instance: dict) -> int:
    if "timestep" in instance:
        timestep = str(instance["timestep"]).upper()
        instance["timestep"] = timestep

    return validate_schema_instance(
        schema_file_name="simulation.schema.json",
        instance=instance,
        error_msg="Errors in \"simulation\" input object."
    )


def validate_geometric(instance: dict) -> int:
    method = str(instance["method"]).upper()
    instance["method"] = method

    schema_map = {
        DesignGeomType.BIRECTANGLE.name: "geometric_bi_rectangle.schema.json",
        DesignGeomType.BIRECTANGLECONSTRAINED.name: "geometric_bi_rectangle_constrained.schema.json",
        DesignGeomType.BIZONEDRECTANGLE.name: "geometric_bi_zoned_rectangle.schema.json",
        DesignGeomType.NEARSQUARE.name: "geometric_near_square.schema.json",
        DesignGeomType.RECTANGLE.name: "geometric_rectangle.schema.json",
        DesignGeomType.ROWWISE.name: "geometric_rowwise.schema.json",
    }

    if method not in schema_map.keys():
        print("Geometric constraint method not recognized.", file=sys.stderr)
        return 1

    return validate_schema_instance(
        schema_file_name=schema_map[method],
        instance=instance,
        error_msg="Errors in \"geometric_constraints\" input object."
    )


def validate_design(instance: dict) -> int:
    flow_type = str(instance["flow_type"]).upper()
    instance["flow_type"] = flow_type
    return validate_schema_instance(
        schema_file_name="design.schema.json",
        instance=instance,
        error_msg="Errors in \"design\" input object."
    )


def validate_input_file(input_file_path: Path) -> int:
    """
    Validate input file against all schemas
    """

    # get instance data
    instance = loads(input_file_path.read_text())

    # validate
    err_count = 0
    err_count += validate_file_structure(instance)
    err_count += validate_fluid(instance["fluid"])
    err_count += validate_grout(instance["grout"])
    err_count += validate_soil(instance["soil"])
    err_count += validate_pipe(instance["pipe"])
    err_count += validate_borehole(instance["borehole"])
    err_count += validate_simulation(instance["simulation"])
    err_count += validate_geometric(instance["geometric_constraints"])
    err_count += validate_design(instance["design"])
    return err_count


if __name__ == "__main__":
    instance_path = Path(sys.argv[1]).resolve()
    sys.exit(validate_input_file(instance_path))
