from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

ComponentType = str

COMPONENT_TYPES: dict[ComponentType, str] = {
    "building": "Building",
    "ground_heat_exchanger": "Ground heat exchanger",
    "source_sink_heat_exchanger": "Source/sink heat exchanger",
    "isolated_horizontal_pipe": "Isolated horizontal pipe",
    "coupled_horizontal_pipe": "Coupled horizontal pipe",
}

DEFAULT_BUILDING_LOADS_PATH = (
    Path(__file__).parents[2] / "ghedesigner" / "tests" / "test_data" / "3GHE_6HP_Example_Loads.csv"
)


NODE_COLORS: dict[ComponentType, str] = {
    "building": "#2563eb",
    "ground_heat_exchanger": "#15803d",
    "source_sink_heat_exchanger": "#b45309",
    "isolated_horizontal_pipe": "#6d28d9",
    "coupled_horizontal_pipe": "#9333ea",
}


def _default_component_config(component_type: ComponentType) -> dict[str, Any]:
    if component_type == "building":
        return {
            "heating_load": {
                "file_path": str(DEFAULT_BUILDING_LOADS_PATH),
                "column_name": "B1Z1_HPHtgLd_W",
                "heat_pump_cop": 4.4,
            },
            "cooling_load": {
                "file_path": str(DEFAULT_BUILDING_LOADS_PATH),
                "column_name": "B1Z1_HPClgLd_W",
                "heat_pump_cop": 4.5,
            },
        }
    if component_type == "ground_heat_exchanger":
        return {
            "flow_rate": 0.3375,
            "flow_type": "BOREHOLE",
            "grout": {"conductivity": 1.0, "rho_cp": 3901000},
            "soil": {"conductivity": 2.0, "rho_cp": 2343493, "undisturbed_temp": 6.1},
            "pipe": {
                "inner_diameter": 0.03404,
                "outer_diameter": 0.04216,
                "shank_spacing": 0.01856,
                "roughness": 0.000001,
                "conductivity": 0.4,
                "rho_cp": 1542000,
                "arrangement": "SINGLEUTUBE",
            },
            "borehole": {"buried_depth": 2.0, "diameter": 0.14},
            "pre_designed": {
                "arrangement": "RECTANGLE",
                "H": 100,
                "spacing_in_x_dimension": 5,
                "spacing_in_y_dimension": 5,
                "boreholes_in_x_dimension": 10,
                "boreholes_in_y_dimension": 8,
            },
        }
    if component_type == "source_sink_heat_exchanger":
        return {
            "effectiveness": 0.7,
            "source_temperature": 10.0,
            "source_flow_rate": 1.0,
            "cut_in_temperature": 35.0,
            "cut_out_temperature": 30.0,
        }
    if component_type in {"isolated_horizontal_pipe", "coupled_horizontal_pipe"}:
        config: dict[str, Any] = {
            "length": 25.0,
            "trench_depth": 1.5,
            "soil": {"conductivity": 2.0, "rho_cp": 2343493},
            "pipe": {
                "inner_diameter": 0.03404,
                "outer_diameter": 0.04216,
                "roughness": 0.000001,
                "conductivity": 0.4,
                "rho_cp": 1542000,
            },
        }
        if component_type == "coupled_horizontal_pipe":
            config.update({"coupled_to": "", "counter_flow": False, "spacing": 0.5})
        return config
    raise ValueError(f"Unsupported component type: {component_type}")


@dataclass
class ComponentNode:
    id: str
    component_type: ComponentType
    name: str
    x: float
    y: float
    config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, component_type: ComponentType, x: float, y: float, index: int) -> ComponentNode:
        label = COMPONENT_TYPES[component_type].lower().replace("/", " ").replace(" ", "_")
        return cls(
            id=str(uuid4()),
            component_type=component_type,
            name=f"{label}_{index}",
            x=x,
            y=y,
            config=_default_component_config(component_type),
        )


@dataclass
class ConnectionEdge:
    source: str
    target: str


@dataclass
class NetworkDocument:
    version: int = 1
    title: str = "Untitled GHEDesigner network"
    nodes: list[ComponentNode] = field(default_factory=list)
    edges: list[ConnectionEdge] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def starter(cls) -> NetworkDocument:
        doc = cls(settings=default_settings())
        building = doc.add_node("building", 150, 160)
        ghe = doc.add_node("ground_heat_exchanger", 430, 160)
        doc.add_edge(building.id, ghe.id)
        doc.add_edge(ghe.id, building.id)
        return doc

    def add_node(self, component_type: ComponentType, x: float, y: float) -> ComponentNode:
        index = 1 + sum(node.component_type == component_type for node in self.nodes)
        node = ComponentNode.create(component_type, x, y, index)
        self.nodes.append(node)
        return node

    def add_edge(self, source: str, target: str) -> None:
        if source == target:
            raise ValueError("A component cannot be connected to itself.")
        if not self.find_node(source) or not self.find_node(target):
            raise ValueError("Both connection endpoints must exist.")
        if any(edge.source == source and edge.target == target for edge in self.edges):
            return
        self.edges.append(ConnectionEdge(source=source, target=target))

    def set_downstream(self, source: str, target: str) -> None:
        if source == target:
            raise ValueError("A component cannot be connected to itself.")
        if not self.find_node(source) or not self.find_node(target):
            raise ValueError("Both connection endpoints must exist.")
        self.edges = [edge for edge in self.edges if edge.source != source and edge.target != target]
        self.edges.append(ConnectionEdge(source=source, target=target))

    def delete_node(self, node_id: str) -> None:
        self.nodes = [node for node in self.nodes if node.id != node_id]
        self.edges = [edge for edge in self.edges if node_id not in (edge.source, edge.target)]

    def find_node(self, node_id: str | None) -> ComponentNode | None:
        if node_id is None:
            return None
        return next((node for node in self.nodes if node.id == node_id), None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NetworkDocument:
        return cls(
            version=data.get("version", 1),
            title=data.get("title", "Untitled GHEDesigner network"),
            nodes=[ComponentNode(**node) for node in data.get("nodes", [])],
            edges=[ConnectionEdge(**edge) for edge in data.get("edges", [])],
            settings={**default_settings(), **data.get("settings", {})},
        )


def default_settings() -> dict[str, Any]:
    return {
        "central_loop": {
            "pipe_configuration": "ONEPIPE",
            "flow_factor": 1.5,
            "pump_efficiency": 0.5,
            "loop_length": 200,
            "design_pressure_loss_per_meter": 200,
        },
        "fluid": {
            "fluid_name": "PROPYLENEGLYCOL",
            "concentration_percent": 30,
            "temperature": 20,
        },
        "simulation_control": {
            "simulation_years": 1,
            "load_method": "HOURLY",
            "search_method": "SIMULATION_ONLY",
        },
        "ground_temperature_model": {
            "annual_average": 10.0,
            "amplitude_1": 8.0,
            "amplitude_2": 1.5,
            "phase_lag_1": 30.0,
            "phase_lag_2": 45.0,
        },
    }
