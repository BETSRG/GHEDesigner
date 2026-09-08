"""Add human-readable titles to every field in the GHEDesigner JSON schema."""

from __future__ import annotations

import argparse
import json
from json.decoder import scanstring
from pathlib import Path

TITLE_OVERRIDES = {
    "H": "Borehole active length",
    "a": "Coefficient a",
    "amplitude_1": "Annual ground temperature amplitude",
    "amplitude_2": "Semiannual ground temperature amplitude",
    "area": "Borefield plan area",
    "b": "Coefficient b",
    "b_max": "Maximum borehole spacing",
    "b_max_x": "Maximum borehole spacing in the x direction",
    "b_max_y": "Maximum borehole spacing in the y direction",
    "b_min": "Minimum borehole spacing",
    "borehole_removal_method": "Borehole removal method",
    "borehole_removal_options": "Borehole removal options",
    "boreholes_in_x_dimension": "Number of boreholes in the x direction",
    "boreholes_in_y_dimension": "Number of boreholes in the y direction",
    "building": "Buildings and heat pump loads",
    "buried_depth": "Borehole burial depth",
    "c": "Coefficient c",
    "c1": "Coefficient c1",
    "c2": "Coefficient c2",
    "c3": "Coefficient c3",
    "circulation_pump": "Circulation pump",
    "column_name": "CSV column name",
    "column_number": "CSV column number",
    "component": "Physical component",
    "component_pumps": "Component pump assignments",
    "concentration_percent": "Antifreeze concentration",
    "conductivity": "Thermal conductivity",
    "conductivity_inner": "Inner-pipe thermal conductivity",
    "conductivity_outer": "Outer-pipe thermal conductivity",
    "constant_cop": "Use constant COP",
    "continue_if_design_unmet": "Continue when design limits are unmet",
    "cooling_load": "Cooling load",
    "cooling_performance": "Cooling performance coefficients",
    "counter_flow": "Counterflow arrangement",
    "coupled_to": "Coupled horizontal pipe",
    "cut_in_temperature": "Cut-in temperature",
    "cut_out_temperature": "Cut-out temperature",
    "design": "GHE design limits",
    "design_cap": "Design capacity",
    "design_flow_rate": "Design mass flow rate",
    "design_pressure_loss": "Design pressure loss",
    "distribution_flow_multiplier": "Distribution flow multiplier",
    "distribution_pump": "Distribution pump",
    "effectiveness": "Heat-exchanger effectiveness",
    "exhaustive_search": "Use exhaustive search",
    "file_path": "Load file path",
    "flow_rate": "Design flow per borehole",
    "fluid": "Loop fluid",
    "fluid_name": "Fluid type",
    "from": "Upstream component",
    "geometric_constraints": "Borefield geometric constraints",
    "ground_heat_exchanger": "Ground heat exchangers",
    "ground_temperature_model": "Seasonal ground temperature model",
    "grout": "Borehole grout",
    "heat_pump": "Heat pump performance maps",
    "heat_pump_cop": "Heat pump COP",
    "heat_pump_name": "Heat pump performance map",
    "heating_load": "Heating load",
    "heating_performance": "Heating performance coefficients",
    "horizontal_piping": "Horizontal piping models",
    "horizontal_segments": "Horizontal pipe segments",
    "horizontal_simulation_considered": "Include horizontal pipe heat transfer",
    "hydraulics": "Component hydraulics",
    "id": "Component ID",
    "inner_diameter": "Pipe inner diameter",
    "inner_pipe_d_in": "Inner pipe inside diameter",
    "inner_pipe_d_out": "Inner pipe outside diameter",
    "line_segments": "Borehole removal line segments",
    "load_method": "Load time-step method",
    "load_values": "Hourly load values",
    "loads": "Ground loads",
    "max_boreholes": "Maximum number of boreholes",
    "max_eft": "Maximum entering fluid temperature",
    "max_height": "Maximum borehole active length",
    "max_rotation": "Maximum row rotation",
    "max_spacing": "Maximum row spacing",
    "maximum_curve_temperature": "Maximum performance-curve temperature",
    "min_eft": "Minimum entering fluid temperature",
    "min_height": "Minimum borehole active length",
    "min_rotation": "Minimum row rotation",
    "min_spacing": "Minimum row spacing",
    "minimum_curve_temperature": "Minimum performance-curve temperature",
    "minimum_distribution_mass_flow": "Minimum distribution mass flow rate",
    "minimum_flow_fraction": "Minimum GHE flow fraction",
    "minor_loss_coefficient": "Minor-loss coefficient",
    "method": "Borefield layout method",
    "network": "Distribution network",
    "no_go_boundaries": "No-go zone boundaries",
    "outer_diameter": "Pipe outer diameter",
    "outer_pipe_d_in": "Outer pipe inside diameter",
    "outer_pipe_d_out": "Outer pipe outside diameter",
    "perimeter_spacing_ratio": "Perimeter spacing ratio",
    "phase_lag_1": "Annual ground temperature phase lag",
    "phase_lag_2": "Semiannual ground temperature phase lag",
    "pipe": "Pipe construction",
    "pipe_defaults": "Network pipe defaults",
    "points": "Borehole removal points",
    "pre_designed": "Pre-designed borefield",
    "pressure_drop_multiplier": "Pressure-drop multiplier",
    "property_boundary": "Property boundary",
    "pump": "Pump model",
    "pump_efficiency": "Pump efficiency",
    "pumps": "Pump model library",
    "reference_mass_flow": "Reference mass flow rate",
    "reference_pressure_drop": "Reference pressure drop",
    "rho_cp": "Volumetric heat capacity",
    "rotate_step": "Row rotation increment",
    "roughness": "Pipe surface roughness",
    "search_method": "System GHE search method",
    "segments": "Distribution pipe segments",
    "shank_spacing": "U-tube shank spacing",
    "simulation_control": "Simulation controls",
    "simulation_years": "System simulation duration",
    "sizing_years": "GHE sizing duration",
    "soil": "Ground properties",
    "source_flow_rate": "Source-side flow rate",
    "source_sink_heat_exchanger": "Source and sink heat exchangers",
    "source_temperature": "Source-side temperature",
    "spacing": "Horizontal pipe spacing",
    "spacing_in_x_dimension": "Borehole spacing in the x direction",
    "spacing_in_y_dimension": "Borehole spacing in the y direction",
    "spacing_step": "Row spacing increment",
    "speed_fraction": "Pump speed fraction",
    "stations": "Ordered physical components",
    "temperature": "Fluid property temperature",
    "thermal_model": "Horizontal thermal model",
    "to": "Downstream component",
    "total_load": "Combined building load",
    "trench_depth": "Buried pipe depth",
    "undisturbed_temp": "Undisturbed ground temperature",
    "version": "Input schema version",
    "width": "Property width",
    "wire_to_water_efficiency": "Wire-to-water efficiency",
    "x": "Borehole x coordinates",
    "y": "Borehole y coordinates",
}

ACRONYMS = {"cop": "COP", "eft": "EFT", "ghe": "GHE", "hp": "HP", "id": "ID", "csv": "CSV"}

VARIANT_TITLE_OVERRIDES = {
    "COAXIAL": "Coaxial pipe",
    "DOUBLEUTUBEPARALLEL": "Double U-tube in parallel",
    "DOUBLEUTUBESERIES": "Double U-tube in series",
    "MANUAL": "Manual borefield coordinates",
    "RECTANGLE": "Rectangular borefield",
    "SINGLEUTUBE": "Single U-tube",
    "fixed": "Fixed mass flow",
    "load_proportional": "Load-proportional mass flow",
}


def natural_title(name: str) -> str:
    """Return the curated title or a readable sentence-case fallback."""
    if name in TITLE_OVERRIDES:
        return TITLE_OVERRIDES[name]
    words = [ACRONYMS.get(word.lower(), word.lower()) for word in name.replace("-", "_").split("_")]
    return " ".join(words).capitalize()


def variant_title(option: object, parent_title: str, index: int) -> str:
    """Name a schema alternative from its discriminator when one is available."""
    if isinstance(option, dict):
        required = option.get("required")
        if isinstance(required, list):
            required_fields = set(required)
            if {"heating_load", "cooling_load"} <= required_fields:
                return "Separate heating and cooling loads"
            if "total_load" in required_fields:
                return "Combined building load"
            if "cooling_load" in required_fields:
                return "Cooling load only"
            if "heating_load" in required_fields:
                return "Heating load only"
        properties = option.get("properties")
        if isinstance(properties, dict):
            for discriminator in ("type", "arrangement", "method"):
                discriminator_schema = properties.get(discriminator)
                if not isinstance(discriminator_schema, dict) or "const" not in discriminator_schema:
                    continue
                value = discriminator_schema["const"]
                if isinstance(value, str):
                    return VARIANT_TITLE_OVERRIDES.get(value, natural_title(value))
    return f"{parent_title} option {index + 1}"


class SchemaScanner:
    """Locate schema-object opening braces without reformatting the source JSON."""

    def __init__(self, source: str):
        self.source = source
        self.insertions: list[tuple[int, str]] = []

    def skip_space(self, index: int) -> int:
        while index < len(self.source) and self.source[index].isspace():
            index += 1
        return index

    def parse_string(self, index: int) -> tuple[str, int]:
        return scanstring(self.source, index + 1, True)

    def parse_value(
        self,
        index: int,
        *,
        title: str | None = None,
        container: str | None = None,
        context_title: str | None = None,
        variant_parent_title: str | None = None,
    ) -> int:
        index = self.skip_space(index)
        character = self.source[index]
        if character == "{":
            return self.parse_object(index, title=title, container=container, context_title=context_title)
        if character == "[":
            return self.parse_array(
                index,
                context_title=context_title,
                variant_parent_title=variant_parent_title,
            )
        if character == '"':
            _, end = self.parse_string(index)
            return end
        while index < len(self.source) and self.source[index] not in ",]} \t\r\n":
            index += 1
        return index

    def parse_array(
        self,
        index: int,
        *,
        context_title: str | None = None,
        variant_parent_title: str | None = None,
    ) -> int:
        index = self.skip_space(index + 1)
        if self.source[index] == "]":
            return index + 1
        option_index = 0
        while True:
            opening_brace = index if self.source[index] == "{" else None
            index = self.parse_value(index, context_title=context_title)
            if opening_brace is not None and variant_parent_title is not None:
                option = json.loads(self.source[opening_brace:index])
                if isinstance(option, dict) and "title" not in option:
                    self.insertions.append(
                        (opening_brace + 1, variant_title(option, variant_parent_title, option_index))
                    )
            option_index += 1
            index = self.skip_space(index)
            if self.source[index] == "]":
                return index + 1
            if self.source[index] != ",":
                raise ValueError(f"Expected ',' in array at character {index}.")
            index = self.skip_space(index + 1)

    def parse_object(
        self,
        index: int,
        *,
        title: str | None,
        container: str | None,
        context_title: str | None,
    ) -> int:
        opening_brace = index
        effective_title = title or context_title
        keys: set[str] = set()
        index = self.skip_space(index + 1)
        if self.source[index] == "}":
            if title is not None:
                self.insertions.append((opening_brace + 1, title))
            return index + 1
        while True:
            if self.source[index] != '"':
                raise ValueError(f"Expected object key at character {index}.")
            key, index = self.parse_string(index)
            keys.add(key)
            index = self.skip_space(index)
            if self.source[index] != ":":
                raise ValueError(f"Expected ':' after object key at character {index}.")
            index = self.skip_space(index + 1)
            child_title = natural_title(key) if container in {"definitions", "properties"} else None
            child_container = "properties" if key == "properties" else "definitions" if key == "$defs" else None
            index = self.parse_value(
                index,
                title=child_title,
                container=child_container,
                context_title=child_title or effective_title,
                variant_parent_title=effective_title if key in {"oneOf", "anyOf"} else None,
            )
            index = self.skip_space(index)
            if self.source[index] == "}":
                if title is not None and "title" not in keys:
                    self.insertions.append((opening_brace + 1, title))
                return index + 1
            if self.source[index] != ",":
                raise ValueError(f"Expected ',' in object at character {index}.")
            index = self.skip_space(index + 1)

    def add_titles(self) -> str:
        end = self.parse_value(0, title="GHEDesigner input file")
        if self.skip_space(end) != len(self.source):
            raise ValueError("Unexpected content after the root schema object.")
        result = self.source
        for index, title in sorted(self.insertions, reverse=True):
            line_start = result.rfind("\n", 0, index) + 1
            line_prefix = result[line_start:index]
            indent = line_prefix[: len(line_prefix) - len(line_prefix.lstrip())]
            encoded = json.dumps(title)
            if index < len(result) and result[index] == "\n":
                insertion = f'\n{indent}  "title": {encoded},'
            elif index < len(result) and result[index] == "}":
                insertion = f' "title": {encoded} '
            else:
                insertion = f' "title": {encoded},'
            result = result[:index] + insertion + result[index:]
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "schema",
        nargs="?",
        type=Path,
        default=Path("ghedesigner/schemas/ghedesigner.schema.json"),
    )
    args = parser.parse_args()
    source = args.schema.read_text()
    json.loads(source)
    updated = SchemaScanner(source).add_titles()
    json.loads(updated)
    args.schema.write_text(updated)


if __name__ == "__main__":
    main()
