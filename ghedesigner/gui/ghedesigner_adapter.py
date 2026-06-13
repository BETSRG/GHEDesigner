from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ghedesigner.gui.models import ComponentNode, NetworkDocument


class NetworkValidationError(ValueError):
    """Raised when the editor graph cannot be exported safely."""


def validate_network(document: NetworkDocument) -> list[str]:
    messages: list[str] = []
    if not document.nodes:
        return ["Add at least one component."]

    names = [node.name for node in document.nodes]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        messages.append("Component names must be unique: " + ", ".join(duplicate_names))

    node_ids = {node.id for node in document.nodes}
    for edge in document.edges:
        if edge.source not in node_ids or edge.target not in node_ids:
            messages.append("Remove connections with missing endpoints.")
            break

    if not any(node.component_type == "ground_heat_exchanger" for node in document.nodes):
        messages.append("Add at least one ground heat exchanger before exporting.")

    outgoing: dict[str, int] = {node.id: 0 for node in document.nodes}
    incoming: dict[str, int] = {node.id: 0 for node in document.nodes}
    for edge in document.edges:
        outgoing[edge.source] = outgoing.get(edge.source, 0) + 1
        incoming[edge.target] = incoming.get(edge.target, 0) + 1

    branching = [node.name for node in document.nodes if outgoing[node.id] > 1]
    merging = [node.name for node in document.nodes if incoming[node.id] > 1]
    if branching:
        messages.append(
            "One-pipe draft export expects one downstream connection per component: " + ", ".join(branching)
        )
    if merging:
        messages.append("One-pipe draft export expects one upstream connection per component: " + ", ".join(merging))

    disconnected = [node.name for node in document.nodes if incoming[node.id] == 0 and outgoing[node.id] == 0]
    if disconnected and len(document.nodes) > 1:
        messages.append("Connect or remove disconnected components: " + ", ".join(disconnected))

    connected_ids = _connected_component_ids(document)
    if len(connected_ids) != len(document.nodes):
        missing = [node.name for node in document.nodes if node.id not in connected_ids]
        messages.append("Connect all components into one topology before export: " + ", ".join(missing))

    return messages


def export_to_ghedesigner(document: NetworkDocument) -> dict[str, Any]:
    errors = validate_network(document)
    if errors:
        raise NetworkValidationError("\n".join(errors))

    ordered_nodes = _topology_order(document)
    output: dict[str, Any] = {
        "version": 2,
        "topology": [{"type": node.component_type, "name": node.name} for node in ordered_nodes],
        "central_loop": deepcopy(document.settings["central_loop"]),
        "fluid": deepcopy(document.settings["fluid"]),
        "simulation_control": deepcopy(document.settings["simulation_control"]),
        "ground_heat_exchanger": {},
    }

    building: dict[str, Any] = {}
    ground_heat_exchanger: dict[str, Any] = {}
    source_sink_heat_exchanger: dict[str, Any] = {}
    horizontal_piping: dict[str, Any] = {}
    uses_horizontal = False

    for node in ordered_nodes:
        config = deepcopy(node.config)
        if node.component_type == "building":
            building[node.name] = config
        elif node.component_type == "ground_heat_exchanger":
            ground_heat_exchanger[node.name] = config
        elif node.component_type == "source_sink_heat_exchanger":
            source_sink_heat_exchanger[node.name] = config
        elif node.component_type in {"isolated_horizontal_pipe", "coupled_horizontal_pipe"}:
            horizontal_piping[node.name] = config
            uses_horizontal = True

    output["ground_heat_exchanger"] = ground_heat_exchanger
    if building:
        output["building"] = building
    if source_sink_heat_exchanger:
        output["source_sink_heat_exchanger"] = source_sink_heat_exchanger
    if horizontal_piping:
        output["horizontal_piping"] = horizontal_piping
        output["ground_temperature_model"] = deepcopy(document.settings["ground_temperature_model"])
        output["simulation_control"]["horizontal_simulation_considered"] = uses_horizontal

    normalize_file_paths(output)
    return output


def normalize_file_paths(data: dict[str, Any], base_dir: Path | None = None) -> None:
    root = base_dir or Path.cwd()

    def _walk(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "file_path" and isinstance(value, str):
                    file_path = Path(value).expanduser()
                    if not file_path.is_absolute():
                        file_path = root / file_path
                    obj[key] = str(file_path.resolve())
                else:
                    _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(data)


def validate_file_paths(data: dict[str, Any]) -> list[str]:
    missing: list[str] = []

    def _walk(obj: Any, location: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                child_location = f"{location}/{key}"
                if key == "file_path" and isinstance(value, str):
                    if not Path(value).expanduser().exists():
                        missing.append(f"{child_location}: {value}")
                else:
                    _walk(value, child_location)
        elif isinstance(obj, list):
            for index, item in enumerate(obj):
                _walk(item, f"{location}/{index}")

    _walk(data, "")
    return missing


def _connected_component_ids(document: NetworkDocument) -> set[str]:
    if not document.nodes:
        return set()
    neighbors: dict[str, set[str]] = {node.id: set() for node in document.nodes}
    for edge in document.edges:
        if edge.source in neighbors and edge.target in neighbors:
            neighbors[edge.source].add(edge.target)
            neighbors[edge.target].add(edge.source)
    seen: set[str] = set()
    stack = [document.nodes[0].id]
    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(sorted(neighbors[node_id] - seen))
    return seen


def _topology_order(document: NetworkDocument) -> list[ComponentNode]:
    node_by_id = {node.id: node for node in document.nodes}
    outgoing = {edge.source: edge.target for edge in document.edges}
    incoming_targets = {edge.target for edge in document.edges}

    start = next((node for node in document.nodes if node.id not in incoming_targets), document.nodes[0])
    ordered: list[ComponentNode] = []
    seen: set[str] = set()
    current = start.id

    while current in node_by_id and current not in seen:
        seen.add(current)
        ordered.append(node_by_id[current])
        current = outgoing.get(current, "")

    ordered.extend(sorted((node for node in document.nodes if node.id not in seen), key=lambda node: (node.y, node.x)))
    return ordered
