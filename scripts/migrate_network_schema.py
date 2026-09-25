from __future__ import annotations

import json
import math
from csv import DictReader
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "demos"
LAMINAR_REYNOLDS_NUMBER = 2300.0
REFERENCE_WATER_HEAT_CAPACITY = 4180.0
FIXED_COP_DESIGN_DELTA_T = 10.0


def canonicalize_components(data: dict[str, Any]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for collection_name in (
        "building",
        "ground_heat_exchanger",
        "source_sink_heat_exchanger",
    ):
        collection = data.get(collection_name, {})
        canonical: dict[str, Any] = {}
        for old_id, component in collection.items():
            component_id = component.pop("id", old_id)
            aliases[old_id] = component_id
            aliases[component_id] = component_id
            component.pop("type", None)
            component.pop("inlet_nodeID", None)
            component.pop("outlet_nodeID", None)
            canonical[component_id] = component
        if collection:
            data[collection_name] = canonical
    return aliases


def building_pump(
    building_id: str,
    building: dict[str, Any],
    heat_pumps: dict[str, Any],
    default_efficiency: float,
) -> tuple[str, dict[str, Any]]:
    hp_name = None
    for load_name in ("heating_load", "cooling_load"):
        hp_name = building.get(load_name, {}).get("heat_pump_name")
        if hp_name is not None:
            break
    hp_data = heat_pumps.get(hp_name, {})
    efficiency = float(hp_data.get("pump_efficiency", default_efficiency))
    pump_id = f"network_pump_{building_id}"
    return pump_id, {
        "wire_to_water_efficiency": efficiency,
    }


def building_design_flow(building: dict[str, Any], heat_pumps: dict[str, Any]) -> float:
    design_flows: list[float] = []
    for load_name, performance_name in (
        ("heating_load", "heating_performance"),
        ("cooling_load", "cooling_performance"),
    ):
        load_data = building.get(load_name, {})
        hp_name = load_data.get("heat_pump_name")
        if not load_data:
            continue
        load_values = load_data.get("load_values")
        if load_values is not None:
            peak_load = max((abs(float(value)) for value in load_values), default=0.0)
        else:
            load_path = (DEMOS / load_data["file_path"]).resolve()
            with load_path.open(newline="") as input_file:
                peak_load = max(
                    (abs(float(row[load_data["column_name"]])) for row in DictReader(input_file)),
                    default=0.0,
                )
        if hp_name is None:
            design_flows.append(peak_load / (REFERENCE_WATER_HEAT_CAPACITY * FIXED_COP_DESIGN_DELTA_T))
        else:
            hp_data = heat_pumps[hp_name]
            unit_flow = float(hp_data["design_flow_rate"])
            design_capacity = float(hp_data[performance_name]["design_cap"])
            design_flows.append(unit_flow * max(1.0, peak_load / design_capacity))
    return max(design_flows, default=1.0)


def design_pipe_diameter(length: float, mass_flow: float, target_pressure_drop: float) -> float:
    density = 1000.0
    dynamic_viscosity = 0.003
    roughness = 1.0e-6

    def pressure_drop(diameter: float) -> float:
        area = math.pi * diameter**2 / 4.0
        velocity = mass_flow / (density * area)
        reynolds_number = density * velocity * diameter / dynamic_viscosity
        relative_roughness = roughness / diameter
        if reynolds_number < LAMINAR_REYNOLDS_NUMBER:
            friction_factor = 64.0 / max(reynolds_number, 1.0)
        else:
            friction_factor = 1.0 / (-1.8 * math.log10((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds_number)) ** 2
        return friction_factor * length / diameter * density * velocity**2 / 2.0

    low, high = 0.01, 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if pressure_drop(midpoint) > target_pressure_drop:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2.0


def component_order(topology: list[dict[str, str]], aliases: dict[str, str]) -> list[str]:
    component_types = {
        "building",
        "ground_heat_exchanger",
        "source_sink_heat_exchanger",
    }
    return [aliases[item["name"]] for item in topology if item["type"].lower() in component_types]


def compact_segments(
    topology: list[dict[str, str]],
    aliases: dict[str, str],
    horizontal_data: dict[str, Any],
    network_type: str,
    loop_length: float,
) -> list[dict[str, Any]]:
    order = component_order(topology, aliases)
    pair_count = len(order) if network_type == "one_pipe" else max(0, len(order) - 1)
    horizontal_between: dict[tuple[str, str], str] = {}
    latest_component: str | None = None
    pending_horizontal: str | None = None
    for item in topology:
        item_type = item["type"].lower()
        if item_type in ("building", "ground_heat_exchanger", "source_sink_heat_exchanger"):
            current_component = aliases[item["name"]]
            if latest_component is not None and pending_horizontal is not None:
                horizontal_between[(latest_component, current_component)] = pending_horizontal
            latest_component = current_component
            pending_horizontal = None
        elif item_type in ("isolated_horizontal_pipe", "coupled_horizontal_pipe"):
            pending_horizontal = item["name"]
    if network_type == "one_pipe" and latest_component is not None and pending_horizontal is not None:
        horizontal_between[(latest_component, order[0])] = pending_horizontal

    default_length = loop_length / max(pair_count, 1)
    segments: list[dict[str, Any]] = []
    for index in range(pair_count):
        from_component = order[index]
        to_component = order[(index + 1) % len(order)]
        thermal_model = horizontal_between.get((from_component, to_component))
        length = float(horizontal_data[thermal_model]["length"]) if thermal_model is not None else default_length
        segment: dict[str, Any] = {
            "id": f"network_segment_{index + 1}",
            "from": from_component,
            "to": to_component,
            "length": length,
        }
        if thermal_model is not None:
            segment["thermal_model"] = thermal_model
        segments.append(segment)
    return segments


def compact_network(
    data: dict[str, Any], topology: list[dict[str, str]], aliases: dict[str, str], central_loop: dict[str, Any]
) -> dict[str, Any]:
    network_type = "one_pipe" if central_loop["pipe_configuration"] == "ONEPIPE" else "two_pipe"
    order = component_order(topology, aliases)
    default_efficiency = float(central_loop["pump_efficiency"])
    pumps: dict[str, Any] = {}
    component_pumps: dict[str, str] = {}
    for building_id, building in data.get("building", {}).items():
        pump_id, pump_data = building_pump(building_id, building, data.get("heat_pump", {}), default_efficiency)
        pumps[pump_id] = pump_data
        component_pumps[building_id] = pump_id

    design_loop_flow = max(
        sum(
            building_design_flow(building, data.get("heat_pump", {}))
            for building_id, building in data.get("building", {}).items()
            if building_id in order
        )
        * float(central_loop["flow_factor"]),
        0.1,
    )
    target_pressure_drop = max(
        float(central_loop["loop_length"]) * float(central_loop["design_pressure_loss_per_meter"]),
        1000.0,
    )
    pipe_diameter = design_pipe_diameter(float(central_loop["loop_length"]), design_loop_flow, target_pressure_drop)
    network: dict[str, Any] = {
        "type": network_type,
        "stations": [{"component": component_id} for component_id in order],
        "pipe_defaults": {"diameter": pipe_diameter, "roughness": 1.0e-6},
        "segments": compact_segments(
            topology,
            aliases,
            data.get("horizontal_piping", {}),
            network_type,
            float(central_loop["loop_length"]),
        ),
        "pumps": pumps,
        "component_pumps": component_pumps,
    }
    if network_type == "one_pipe":
        network["mass_flow_control"] = {
            "distribution_flow_multiplier": float(central_loop["flow_factor"]),
            "minimum_distribution_mass_flow": 0.1,
        }
        distribution_pump_id = "network_distribution_pump"
        network["pumps"][distribution_pump_id] = {
            "wire_to_water_efficiency": float(central_loop["pump_efficiency"]),
        }
        network["distribution_pump"] = {
            "type": "pump",
            "pump": distribution_pump_id,
        }
    return network


def migrate_file(path: Path) -> None:
    data = json.loads(path.read_text())
    data["version"] = 4
    if "topology" not in data:
        network = data.get("network")
        if network is not None and network["type"] == "one_pipe":
            network.setdefault(
                "mass_flow_control",
                {
                    "distribution_flow_multiplier": 1.5,
                    "minimum_distribution_mass_flow": 0.1,
                },
            )
        path.write_text(json.dumps(data, indent=2) + "\n")
        return
    aliases = canonicalize_components(data)
    topology = data.pop("topology")
    central_loop = data.pop("central_loop", None)
    if central_loop is not None:
        if central_loop["pipe_configuration"] not in ("ONEPIPE", "TWOPIPE"):
            raise ValueError("Only ONEPIPE and TWOPIPE configurations can be migrated.")
        data["network"] = compact_network(data, topology, aliases, central_loop)
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    for path in sorted(DEMOS.glob("*.json")):
        migrate_file(path)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
