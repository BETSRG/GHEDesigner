from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import log10, pi
from typing import Any

import numpy as np

FLOW_TOLERANCE = 1.0e-9
MASS_BALANCE_TOLERANCE = 1.0e-4
LAMINAR_REYNOLDS_NUMBER = 2300.0
MINIMUM_TWO_PIPE_STATIONS = 2


class NetworkType(StrEnum):
    ONE_PIPE = "one_pipe"
    TWO_PIPE = "two_pipe"


class BranchType(StrEnum):
    PIPE = "pipe"
    BUILDING = "building"
    GROUND_HEAT_EXCHANGER = "ground_heat_exchanger"
    SOURCE_SINK_HEAT_EXCHANGER = "source_sink_heat_exchanger"
    PUMP = "pump"
    BYPASS = "bypass"


class HydraulicType(StrEnum):
    PASSIVE = "passive"
    CONTROLLED_FLOW = "controlled_flow"
    PUMP = "pump"


@dataclass(frozen=True)
class PumpModel:
    id: str
    wire_to_water_efficiency: float


@dataclass
class NetworkNode:
    id: str
    branch_ids: list[str] = field(default_factory=list)
    temperature: float = 0.0


@dataclass
class NetworkBranch:
    id: str
    branch_type: BranchType
    node_a: str
    node_b: str
    component_id: str | None = None
    hydraulic_type: HydraulicType = HydraulicType.PASSIVE
    length: float = 0.0
    diameter: float = 0.0
    roughness: float = 1.0e-6
    minor_loss_coefficient: float = 0.0
    reference_mass_flow: float | None = None
    reference_pressure_drop: float | None = None
    pump_id: str | None = None
    thermal_model_id: str | None = None
    mass_flow: float = 0.0
    pressure_difference: float = 0.0
    pump_power: float = 0.0

    def passive_pressure_drop(self, mass_flow: float, density: float, dynamic_viscosity: float) -> float:
        """Return signed pressure drop from node A to node B."""
        if abs(mass_flow) <= FLOW_TOLERANCE:
            return 0.0

        if self.reference_mass_flow is not None and self.reference_pressure_drop is not None:
            ratio = abs(mass_flow) / self.reference_mass_flow
            return np.sign(mass_flow) * self.reference_pressure_drop * ratio**2

        if self.branch_type == BranchType.BYPASS or self.component_id is not None:
            return 0.0

        if self.length <= 0.0 or self.diameter <= 0.0:
            raise ValueError(f"Passive branch '{self.id}' requires positive length and diameter.")

        area = pi * self.diameter**2 / 4.0
        velocity = abs(mass_flow) / (density * area)
        reynolds_number = density * velocity * self.diameter / dynamic_viscosity
        relative_roughness = self.roughness / self.diameter
        if reynolds_number < LAMINAR_REYNOLDS_NUMBER:
            friction_factor = 64.0 / max(reynolds_number, 1.0)
        else:
            friction_factor = 1.0 / (-1.8 * log10((relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds_number)) ** 2
        loss_coefficient = friction_factor * self.length / self.diameter + self.minor_loss_coefficient
        pressure_drop = loss_coefficient * density * velocity**2 / 2.0
        return float(np.sign(mass_flow) * pressure_drop)


@dataclass
class HydraulicSolution:
    node_pressures: dict[str, float]
    branch_flows: dict[str, float]


@dataclass
class NetworkGraph:
    nodes: dict[str, NetworkNode]
    branches: dict[str, NetworkBranch]
    pumps: dict[str, PumpModel]

    def __post_init__(self) -> None:
        for branch in self.branches.values():
            if branch.node_a in self.nodes:
                self.nodes[branch.node_a].branch_ids.append(branch.id)
            if branch.node_b in self.nodes:
                self.nodes[branch.node_b].branch_ids.append(branch.id)

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("Network must contain at least one node.")
        if not self.branches:
            raise ValueError("Network must contain at least one branch.")

        for branch in self.branches.values():
            if branch.node_a not in self.nodes:
                raise ValueError(f"Branch '{branch.id}' references missing node '{branch.node_a}'.")
            if branch.node_b not in self.nodes:
                raise ValueError(f"Branch '{branch.id}' references missing node '{branch.node_b}'.")
            if branch.node_a == branch.node_b:
                raise ValueError(f"Branch '{branch.id}' cannot connect a node to itself.")
            if (
                branch.hydraulic_type in (HydraulicType.PUMP, HydraulicType.CONTROLLED_FLOW)
                and branch.pump_id not in self.pumps
            ):
                raise ValueError(f"Branch '{branch.id}' requires a valid pump reference.")
            if branch.hydraulic_type == HydraulicType.PASSIVE:
                has_reference_curve = (
                    branch.reference_mass_flow is not None and branch.reference_pressure_drop is not None
                )
                has_pipe_geometry = branch.length > 0.0 and branch.diameter > 0.0
                is_zero_loss_component = branch.branch_type == BranchType.BYPASS or branch.component_id is not None
                if not is_zero_loss_component and not (has_reference_curve or has_pipe_geometry):
                    raise ValueError(
                        f"Passive branch '{branch.id}' requires pipe geometry or a reference design point."
                    )
        for pump in self.pumps.values():
            if not 0.0 < pump.wire_to_water_efficiency <= 1.0:
                raise ValueError(f"Pump '{pump.id}' wire-to-water efficiency must be in (0, 1].")

        referenced_pumps = {branch.pump_id for branch in self.branches.values() if branch.pump_id is not None}
        unused_pumps = sorted(set(self.pumps) - referenced_pumps)
        if unused_pumps:
            raise ValueError(f"Network contains unused pumps: {', '.join(unused_pumps)}.")

        pending = [next(iter(self.nodes))]
        visited: set[str] = set()
        while pending:
            node_id = pending.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            for branch_id in self.nodes[node_id].branch_ids:
                branch = self.branches[branch_id]
                pending.append(branch.node_b if branch.node_a == node_id else branch.node_a)
        if visited != set(self.nodes):
            missing = ", ".join(sorted(set(self.nodes) - visited))
            raise ValueError(f"Network must be connected; unreachable nodes: {missing}.")

    def solve_hydraulics(
        self,
        density: float,
        dynamic_viscosity: float,
        controlled_flows: dict[str, float] | None = None,
    ) -> HydraulicSolution:
        """Solve continuity from prescribed flow and post-process pressure."""
        return self._solve_mass_flow_driven(
            density=density,
            dynamic_viscosity=dynamic_viscosity,
            prescribed_flows=controlled_flows,
        )

    def _solve_mass_flow_driven(
        self,
        density: float,
        dynamic_viscosity: float,
        prescribed_flows: dict[str, float] | None = None,
    ) -> HydraulicSolution:
        """Solve continuity from prescribed flows, then post-process pressure.

        Pressure never participates in the flow solution. Cyclic networks must
        prescribe enough independent branch flows to make continuity unique.
        """
        self.validate()
        prescribed_flows = prescribed_flows or {}
        unknown_prescribed = sorted(set(prescribed_flows) - set(self.branches))
        if unknown_prescribed:
            raise ValueError(f"Prescribed flows reference unknown branches: {', '.join(unknown_prescribed)}.")

        reference_node = next(iter(self.nodes))
        balance_nodes = [node_id for node_id in self.nodes if node_id != reference_node]
        unknown_branches = [branch for branch in self.branches.values() if branch.id not in prescribed_flows]
        incidence = np.zeros((len(balance_nodes), len(unknown_branches)), dtype=float)
        rhs = np.zeros(len(balance_nodes), dtype=float)
        node_index = {node_id: index for index, node_id in enumerate(balance_nodes)}

        for branch in self.branches.values():
            if branch.id in prescribed_flows:
                branch_flow = float(prescribed_flows[branch.id])
                if branch.node_a != reference_node:
                    rhs[node_index[branch.node_a]] += branch_flow
                if branch.node_b != reference_node:
                    rhs[node_index[branch.node_b]] -= branch_flow

        for column, branch in enumerate(unknown_branches):
            if branch.node_a != reference_node:
                incidence[node_index[branch.node_a], column] = -1.0
            if branch.node_b != reference_node:
                incidence[node_index[branch.node_b], column] = 1.0

        if unknown_branches:
            solved_flows, _, rank, _ = np.linalg.lstsq(incidence, rhs, rcond=None)
            if rank != len(unknown_branches):
                cycle_count = len(unknown_branches) - rank
                raise ValueError(
                    "Prescribed-flow network is underconstrained; add "
                    f"{cycle_count} independent prescribed-flow controller(s)."
                )
            continuity_residual = float(np.linalg.norm(incidence @ solved_flows - rhs, ord=np.inf))
            if continuity_residual > MASS_BALANCE_TOLERANCE:
                raise ValueError(
                    "Prescribed branch flows are inconsistent with network continuity "
                    f"(mass residual {continuity_residual:.3g} kg/s)."
                )
        else:
            solved_flows = np.empty(0, dtype=float)

        branch_flows = {branch_id: float(value) for branch_id, value in prescribed_flows.items()}
        branch_flows.update({branch.id: float(solved_flows[index]) for index, branch in enumerate(unknown_branches)})

        # Recover a diagnostic pressure field only after the flows are fixed.
        # This least-squares post-process cannot alter branch flow.
        pressure_matrix_rows: list[np.ndarray] = []
        pressure_rhs: list[float] = []
        pressure_nodes = balance_nodes
        pressure_index = {node_id: index for index, node_id in enumerate(pressure_nodes)}
        for branch in self.branches.values():
            if branch.hydraulic_type == HydraulicType.PUMP:
                continue
            row = np.zeros(len(pressure_nodes), dtype=float)
            if branch.node_a != reference_node:
                row[pressure_index[branch.node_a]] = 1.0
            if branch.node_b != reference_node:
                row[pressure_index[branch.node_b]] = -1.0
            try:
                pressure_drop = branch.passive_pressure_drop(branch_flows[branch.id], density, dynamic_viscosity)
            except ValueError:
                # A prescribed component without its own loss model still
                # participates in flow routing but not pressure recovery.
                continue
            pressure_matrix_rows.append(row)
            pressure_rhs.append(pressure_drop)

        node_pressures = {reference_node: 0.0}
        if pressure_matrix_rows:
            pressure_matrix = np.asarray(pressure_matrix_rows, dtype=float)
            pressure_values, _, _, _ = np.linalg.lstsq(
                pressure_matrix,
                np.asarray(pressure_rhs, dtype=float),
                rcond=None,
            )
            node_pressures.update({node_id: float(pressure_values[index]) for node_id, index in pressure_index.items()})
        else:
            node_pressures.update(dict.fromkeys(pressure_nodes, 0.0))

        for branch in self.branches.values():
            branch.mass_flow = branch_flows[branch.id]
            branch.pressure_difference = node_pressures[branch.node_a] - node_pressures[branch.node_b]
            branch.pump_power = 0.0
            if branch.pump_id is None:
                continue
            pump = self.pumps[branch.pump_id]
            flow_direction = float(np.sign(branch.mass_flow))
            required_pressure_rise = max(
                0.0,
                flow_direction * (node_pressures[branch.node_b] - node_pressures[branch.node_a]),
            )
            hydraulic_power = abs(branch.mass_flow) * required_pressure_rise / density
            branch.pump_power = hydraulic_power / pump.wire_to_water_efficiency

        return HydraulicSolution(node_pressures, branch_flows)

    def incoming_branches(self, node_id: str, tolerance: float = FLOW_TOLERANCE) -> list[NetworkBranch]:
        incoming: list[NetworkBranch] = []
        for branch_id in self.nodes[node_id].branch_ids:
            branch = self.branches[branch_id]
            enters_at_b = branch.mass_flow > tolerance and branch.node_b == node_id
            enters_at_a = branch.mass_flow < -tolerance and branch.node_a == node_id
            if enters_at_a or enters_at_b:
                incoming.append(branch)
        return incoming

    def outgoing_branches(self, node_id: str, tolerance: float = FLOW_TOLERANCE) -> list[NetworkBranch]:
        outgoing: list[NetworkBranch] = []
        for branch_id in self.nodes[node_id].branch_ids:
            branch = self.branches[branch_id]
            leaves_at_a = branch.mass_flow > tolerance and branch.node_a == node_id
            leaves_at_b = branch.mass_flow < -tolerance and branch.node_b == node_id
            if leaves_at_a or leaves_at_b:
                outgoing.append(branch)
        return outgoing


def _component_type(component_id: str, component_types: dict[str, BranchType]) -> BranchType:
    try:
        return component_types[component_id]
    except KeyError as error:
        raise ValueError(f"Network references unknown component '{component_id}'.") from error


def _pump_models(network_data: dict[str, Any]) -> dict[str, PumpModel]:
    pumps: dict[str, PumpModel] = {}
    for pump_id, pump_data in network_data.get("pumps", {}).items():
        pumps[pump_id] = PumpModel(
            id=pump_id,
            wire_to_water_efficiency=float(pump_data["wire_to_water_efficiency"]),
        )
    return pumps


def _hydraulic_fields(branch_data: dict[str, Any]) -> dict[str, Any]:
    hydraulic_data = branch_data.get("hydraulics", {"type": "passive"})
    return {
        "hydraulic_type": HydraulicType(hydraulic_data["type"]),
        "reference_mass_flow": hydraulic_data.get("reference_mass_flow"),
        "reference_pressure_drop": hydraulic_data.get("reference_pressure_drop"),
        "pump_id": hydraulic_data.get("pump"),
    }


def compile_network(
    network_data: dict[str, Any],
    component_types: dict[str, BranchType],
    component_data: dict[str, Any] | None = None,
) -> NetworkGraph:
    """Compile a one- or two-pipe public input into the internal network graph."""
    network_type = NetworkType(network_data["type"])
    return _compile_compact_network(network_data, component_types, network_type, component_data or {})


def _compile_compact_network(
    network_data: dict[str, Any],
    component_types: dict[str, BranchType],
    network_type: NetworkType,
    component_data: dict[str, Any],
) -> NetworkGraph:
    station_ids = [station["component"] for station in network_data["stations"]]
    if network_type == NetworkType.TWO_PIPE and len(station_ids) < MINIMUM_TWO_PIPE_STATIONS:
        raise ValueError("A compact two_pipe network requires at least two stations.")
    if len(set(station_ids)) != len(station_ids):
        raise ValueError("Compact network stations must reference each component exactly once.")
    for component_id in station_ids:
        _component_type(component_id, component_types)

    building_ids = {
        component_id for component_id in station_ids if component_types[component_id] == BranchType.BUILDING
    }
    actual_pump_ids = set(network_data.get("component_pumps", {}))
    if actual_pump_ids != building_ids:
        raise ValueError(
            f"Compact network 'component_pumps' keys must match {sorted(building_ids)}; "
            f"received {sorted(actual_pump_ids)}."
        )

    collection_by_type = {
        BranchType.BUILDING: "building",
        BranchType.GROUND_HEAT_EXCHANGER: "ground_heat_exchanger",
        BranchType.SOURCE_SINK_HEAT_EXCHANGER: "source_sink_heat_exchanger",
    }

    def component_hydraulic_fields(component_id: str) -> dict[str, Any]:
        branch_type = component_types[component_id]
        if branch_type == BranchType.BUILDING:
            return {
                "hydraulic_type": HydraulicType.CONTROLLED_FLOW,
                "pump_id": network_data["component_pumps"][component_id],
            }
        component = component_data.get(collection_by_type[branch_type], {}).get(component_id, {})
        hydraulics = component.get("hydraulics", {"type": "passive"})
        if hydraulics.get("type", "passive") != HydraulicType.PASSIVE:
            raise ValueError(f"Compact network component '{component_id}' must use passive component-owned hydraulics.")
        return {
            "hydraulic_type": HydraulicType.PASSIVE,
            "reference_mass_flow": hydraulics.get("reference_mass_flow"),
            "reference_pressure_drop": hydraulics.get("reference_pressure_drop"),
        }

    defaults = network_data.get("pipe_defaults", {})
    segment_lookup = {segment["id"]: segment for segment in network_data["segments"]}
    branches: dict[str, NetworkBranch] = {}
    nodes: dict[str, NetworkNode] = {}

    if network_type == NetworkType.ONE_PIPE:
        distribution_pump_node = "__distribution_pump_in"
        nodes[distribution_pump_node] = NetworkNode(distribution_pump_node)
        for component_id in station_ids:
            node_a = f"__{component_id}_a"
            node_b = f"__{component_id}_b"
            nodes[node_a] = NetworkNode(node_a)
            nodes[node_b] = NetworkNode(node_b)
            branches[f"__{component_id}_device"] = NetworkBranch(
                id=f"__{component_id}_device",
                branch_type=component_types[component_id],
                node_a=node_a,
                node_b=node_b,
                component_id=component_id,
                **component_hydraulic_fields(component_id),
            )
            branches[f"__{component_id}_bypass"] = NetworkBranch(
                id=f"__{component_id}_bypass",
                branch_type=BranchType.BYPASS,
                node_a=node_a,
                node_b=node_b,
            )
        expected_segments = len(station_ids)
    else:
        for component_id in station_ids:
            supply_node = f"__{component_id}_supply"
            return_node = f"__{component_id}_return"
            nodes[supply_node] = NetworkNode(supply_node)
            nodes[return_node] = NetworkNode(return_node)
            component_type = component_types[component_id]
            is_ghe = component_type == BranchType.GROUND_HEAT_EXCHANGER
            node_a = return_node if is_ghe else supply_node
            node_b = supply_node if is_ghe else return_node
            branches[f"__{component_id}_device"] = NetworkBranch(
                id=f"__{component_id}_device",
                branch_type=component_types[component_id],
                node_a=node_a,
                node_b=node_b,
                component_id=component_id,
                **component_hydraulic_fields(component_id),
            )
        expected_segments = max(0, len(station_ids) - 1)

    if len(segment_lookup) != expected_segments:
        raise ValueError(
            f"{network_type.value} network requires {expected_segments} distribution segments; "
            f"received {len(segment_lookup)}."
        )

    expected_pairs = [
        (station_ids[index], station_ids[(index + 1) % len(station_ids)]) for index in range(expected_segments)
    ]
    actual_pairs = [(segment["from"], segment["to"]) for segment in segment_lookup.values()]
    if actual_pairs != expected_pairs:
        raise ValueError(
            f"{network_type.value} segments must connect consecutive stations in station order; "
            f"expected {expected_pairs}, received {actual_pairs}."
        )

    for segment in segment_lookup.values():
        from_component = segment["from"]
        to_component = segment["to"]
        if from_component not in station_ids or to_component not in station_ids:
            raise ValueError(f"Segment '{segment['id']}' references a component outside the station list.")
        properties = {**defaults, **segment}
        if network_type == NetworkType.ONE_PIPE:
            target_node = distribution_pump_node if to_component == station_ids[0] else f"__{to_component}_a"
            pairs = [("", f"__{from_component}_b", target_node)]
        else:
            pairs = [
                ("_supply", f"__{from_component}_supply", f"__{to_component}_supply"),
                ("_return", f"__{to_component}_return", f"__{from_component}_return"),
            ]
        for suffix, node_a, node_b in pairs:
            branch_id = f"{segment['id']}{suffix}"
            branches[branch_id] = NetworkBranch(
                id=branch_id,
                branch_type=BranchType.PIPE,
                node_a=node_a,
                node_b=node_b,
                length=float(properties["length"]),
                diameter=float(properties["diameter"]),
                roughness=float(properties.get("roughness", 1.0e-6)),
                minor_loss_coefficient=float(properties.get("minor_loss_coefficient", 0.0)),
                thermal_model_id=properties.get("thermal_model"),
            )

    if network_type == NetworkType.ONE_PIPE:
        pump_data = network_data["distribution_pump"]
        branches["__distribution_pump"] = NetworkBranch(
            id="__distribution_pump",
            branch_type=BranchType.PUMP,
            node_a=distribution_pump_node,
            node_b=f"__{station_ids[0]}_a",
            **_hydraulic_fields({"hydraulics": pump_data}),
        )

    graph = NetworkGraph(nodes, branches, _pump_models(network_data))
    graph.validate()
    return graph


def component_type_map(data: dict[str, Any]) -> dict[str, BranchType]:
    result: dict[str, BranchType] = {}
    for component_id in data.get("building", {}):
        result[component_id] = BranchType.BUILDING
    for component_id in data.get("ground_heat_exchanger", {}):
        result[component_id] = BranchType.GROUND_HEAT_EXCHANGER
    for component_id in data.get("source_sink_heat_exchanger", {}):
        result[component_id] = BranchType.SOURCE_SINK_HEAT_EXCHANGER
    return result


def _validate_mass_flow_strategy(network_type: NetworkType, network_data: dict[str, Any], graph: NetworkGraph) -> None:
    if network_type == NetworkType.ONE_PIPE:
        if not network_data.get("mass_flow_control"):
            raise ValueError("A one_pipe network requires 'mass_flow_control'.")
        return

    prescribed_branch_ids = {
        branch.id
        for branch in graph.branches.values()
        if branch.branch_type in (BranchType.BUILDING, BranchType.GROUND_HEAT_EXCHANGER)
    }
    unknown_branches = [branch for branch in graph.branches.values() if branch.id not in prescribed_branch_ids]
    if not unknown_branches:
        return

    reference_node = next(iter(graph.nodes))
    balance_nodes = [node_id for node_id in graph.nodes if node_id != reference_node]
    node_index = {node_id: index for index, node_id in enumerate(balance_nodes)}
    incidence = np.zeros((len(balance_nodes), len(unknown_branches)), dtype=float)
    for column, branch in enumerate(unknown_branches):
        if branch.node_a != reference_node:
            incidence[node_index[branch.node_a], column] = -1.0
        if branch.node_b != reference_node:
            incidence[node_index[branch.node_b], column] = 1.0

    missing_controls = len(unknown_branches) - int(np.linalg.matrix_rank(incidence))
    if missing_controls:
        raise ValueError(
            "Prescribed-flow network is underconstrained; add "
            f"{missing_controls} independent branch mass-flow control(s)."
        )


def validate_network_data(data: dict[str, Any]) -> None:
    """Perform cross-reference checks that JSON Schema cannot express."""
    network_data = data.get("network")
    if network_data is None:
        return

    id_locations: dict[str, str] = {}
    collections = {
        "building": data.get("building", {}),
        "ground_heat_exchanger": data.get("ground_heat_exchanger", {}),
        "source_sink_heat_exchanger": data.get("source_sink_heat_exchanger", {}),
        "horizontal_piping": data.get("horizontal_piping", {}),
        "pump": network_data.get("pumps", {}),
    }

    def register_id(item_id: str, collection_name: str) -> None:
        if item_id.startswith("__"):
            raise ValueError(f"ID '{item_id}' uses the reserved canonical-network prefix '__'.")
        if item_id in id_locations:
            raise ValueError(
                f"ID '{item_id}' is used by both {id_locations[item_id]} and {collection_name}; IDs must be global."
            )
        id_locations[item_id] = collection_name

    for collection_name, collection in collections.items():
        for item_id in collection:
            register_id(item_id, collection_name)
    for segment in network_data.get("segments", []):
        register_id(segment["id"], "segment")

    for ghe_id, ghe_data in data.get("ground_heat_exchanger", {}).items():
        hydraulics = ghe_data.get("hydraulics", {})
        has_component_loss = "reference_mass_flow" in hydraulics or "reference_pressure_drop" in hydraulics
        if ghe_data.get("circulation_pump") is not None and has_component_loss:
            raise ValueError(
                f"Ground heat exchanger '{ghe_id}' cannot define both component hydraulic loss and "
                "circulation_pump pressure loss."
            )

    graph = compile_network(network_data, component_type_map(data), data)
    _validate_mass_flow_strategy(NetworkType(network_data["type"]), network_data, graph)
    placement_counts: dict[str, int] = {}
    for branch in graph.branches.values():
        if branch.component_id is not None:
            placement_counts[branch.component_id] = placement_counts.get(branch.component_id, 0) + 1
    placed_components = set(placement_counts)
    available_components = set(component_type_map(data))
    if placed_components != available_components:
        omitted = sorted(available_components - placed_components)
        duplicated_or_unknown = sorted(placed_components - available_components)
        details = []
        if omitted:
            details.append(f"unplaced components: {', '.join(omitted)}")
        if duplicated_or_unknown:
            details.append(f"unknown components: {', '.join(duplicated_or_unknown)}")
        raise ValueError("Invalid network component placement (" + "; ".join(details) + ").")
    duplicated = sorted(component_id for component_id, count in placement_counts.items() if count != 1)
    if duplicated:
        raise ValueError(
            "Each component must be placed on exactly one branch; duplicate placements: " + ", ".join(duplicated) + "."
        )

    horizontal_ids = set(data.get("horizontal_piping", {}))
    for branch in graph.branches.values():
        if branch.thermal_model_id is not None and branch.thermal_model_id not in horizontal_ids:
            raise ValueError(f"Branch '{branch.id}' references unknown thermal model '{branch.thermal_model_id}'.")
