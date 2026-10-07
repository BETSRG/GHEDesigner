from __future__ import annotations

from typing import Any

import pytest

from ghedesigner.network import (
    BranchType,
    NetworkBranch,
    NetworkGraph,
    NetworkNode,
    compile_network,
    validate_network_data,
)


def test_mass_flow_driven_solver_uses_continuity_then_postprocesses_pressure() -> None:
    nodes = {node_id: NetworkNode(node_id) for node_id in ("a", "b", "c")}
    branches = {
        "load": NetworkBranch(
            "load",
            BranchType.BUILDING,
            "a",
            "b",
            reference_mass_flow=1.0,
            reference_pressure_drop=1000.0,
        ),
        "pipe": NetworkBranch(
            "pipe",
            BranchType.PIPE,
            "b",
            "c",
            reference_mass_flow=1.0,
            reference_pressure_drop=2000.0,
        ),
        "ghe": NetworkBranch(
            "ghe",
            BranchType.GROUND_HEAT_EXCHANGER,
            "c",
            "a",
            reference_mass_flow=1.0,
            reference_pressure_drop=3000.0,
        ),
    }
    graph = NetworkGraph(nodes, branches, {})

    solution = graph.solve_hydraulics(
        1000.0,
        0.001,
        controlled_flows={"load": 2.0, "ghe": 2.0},
    )

    assert solution.branch_flows == pytest.approx({"load": 2.0, "pipe": 2.0, "ghe": 2.0})
    assert solution.node_pressures["b"] != solution.node_pressures["c"]


def test_mass_flow_driven_solver_requires_a_controller_for_each_cycle() -> None:
    nodes = {node_id: NetworkNode(node_id) for node_id in ("a", "b", "c")}
    branches = {
        branch_id: NetworkBranch(
            branch_id,
            BranchType.PIPE,
            node_a,
            node_b,
            reference_mass_flow=1.0,
            reference_pressure_drop=1000.0,
        )
        for branch_id, node_a, node_b in (
            ("ab", "a", "b"),
            ("bc", "b", "c"),
            ("ca", "c", "a"),
        )
    }
    graph = NetworkGraph(nodes, branches, {})

    with pytest.raises(ValueError, match=r"add 1 independent prescribed-flow controller"):
        graph.solve_hydraulics(1000.0, 0.001)

    solution = graph.solve_hydraulics(1000.0, 0.001, controlled_flows={"ab": 0.5})
    assert solution.branch_flows == pytest.approx({"ab": 0.5, "bc": 0.5, "ca": 0.5})


def test_compact_one_pipe_generates_zero_loss_station_bypasses() -> None:
    pump = {
        "wire_to_water_efficiency_fraction": 0.7,
    }
    network_data = {
        "type": "one_pipe",
        "stations": [{"component_id": "building_1"}, {"component_id": "ghe_1"}],
        "segments": [
            {
                "id": "segment_1",
                "from_component_id": "building_1",
                "to_component_id": "ghe_1",
                "length_m": 10.0,
                "diameter_m": 0.1,
                "surface_roughness_m": 1.0e-6,
                "minor_loss_coefficient": 0.0,
            },
            {
                "id": "segment_2",
                "from_component_id": "ghe_1",
                "to_component_id": "building_1",
                "length_m": 10.0,
                "diameter_m": 0.1,
                "surface_roughness_m": 1.0e-6,
                "minor_loss_coefficient": 0.0,
            },
        ],
        "pumps": {"building_pump": pump, "distribution_pump": pump},
        "component_pumps": {"building_1": "building_pump"},
        "distribution_pump": {"type": "pump", "pump_id": "distribution_pump"},
        "mass_flow_control": {
            "distribution_flow_multiplier": 1.5,
            "minimum_distribution_mass_flow_rate_kg_per_s": 0.1,
        },
    }
    component_data: dict[str, Any] = {
        "buildings": {"building_1": {}},
        "ground_heat_exchangers": {"ghe_1": {"circulation_pump": {"reference_pressure_drop_pa": 50_000.0}}},
    }

    graph = compile_network(
        network_data,
        {"building_1": BranchType.BUILDING, "ghe_1": BranchType.GROUND_HEAT_EXCHANGER},
        component_data,
    )

    bypasses = [branch for branch in graph.branches.values() if branch.branch_type == BranchType.BYPASS]
    assert len(bypasses) == 2
    assert all(branch.reference_mass_flow is None for branch in bypasses)
    assert all(branch.reference_pressure_drop is None for branch in bypasses)
    assert all(branch.passive_pressure_drop(1.0, 1000.0, 0.001) == 0.0 for branch in bypasses)
    assert graph.branches["__ghe_1_device"].passive_pressure_drop(1.0, 1000.0, 0.001) == 0.0


def test_linked_horizontal_model_supplies_segment_hydraulic_geometry() -> None:
    pump = {"wire_to_water_efficiency_fraction": 0.7}
    network_data = {
        "type": "one_pipe",
        "stations": [{"component_id": "building_1"}, {"component_id": "ghe_1"}],
        "segments": [
            {
                "id": "horizontal_segment",
                "from_component_id": "building_1",
                "to_component_id": "ghe_1",
                "length_m": 100.0,
                "minor_loss_coefficient": 0.0,
                "thermal_model_id": "horizontal_1",
            },
            {
                "id": "return_segment",
                "from_component_id": "ghe_1",
                "to_component_id": "building_1",
                "length_m": 100.0,
                "diameter_m": 0.1,
                "surface_roughness_m": 1.0e-6,
                "minor_loss_coefficient": 0.0,
            },
        ],
        "pumps": {"building_pump": pump, "distribution_pump": pump},
        "component_pumps": {"building_1": "building_pump"},
        "distribution_pump": {"type": "pump", "pump_id": "distribution_pump"},
        "mass_flow_control": {
            "distribution_flow_multiplier": 1.5,
            "minimum_distribution_mass_flow_rate_kg_per_s": 0.1,
        },
    }
    component_data: dict[str, Any] = {
        "buildings": {"building_1": {}},
        "ground_heat_exchangers": {"ghe_1": {}},
        "horizontal_piping": {
            "horizontal_1": {
                "pipe": {
                    "inner_diameter_m": 0.08,
                    "surface_roughness_m": 2.0e-6,
                }
            }
        },
    }

    graph = compile_network(
        network_data,
        {"building_1": BranchType.BUILDING, "ghe_1": BranchType.GROUND_HEAT_EXCHANGER},
        component_data,
    )
    horizontal_branch = graph.branches["horizontal_segment"]
    original_loss = horizontal_branch.passive_pressure_drop(1.0, 1000.0, 0.001)

    component_data["horizontal_piping"]["horizontal_1"]["pipe"]["inner_diameter_m"] = 0.04
    smaller_graph = compile_network(
        network_data,
        {"building_1": BranchType.BUILDING, "ghe_1": BranchType.GROUND_HEAT_EXCHANGER},
        component_data,
    )
    smaller_branch = smaller_graph.branches["horizontal_segment"]

    assert horizontal_branch.diameter == 0.08
    assert horizontal_branch.roughness == 2.0e-6
    assert smaller_branch.diameter == 0.04
    assert smaller_branch.passive_pressure_drop(1.0, 1000.0, 0.001) > original_loss


def test_semantic_validation_rejects_duplicate_ghe_pressure_loss() -> None:
    data = {
        "network": {"type": "one_pipe"},
        "ground_heat_exchangers": {
            "ghe_1": {
                "circulation_pump": {"reference_pressure_drop_pa": 50_000.0},
                "hydraulics": {
                    "type": "passive",
                    "reference_mass_flow_rate_kg_per_s": 1.0,
                    "reference_pressure_drop_pa": 10_000.0,
                },
            }
        },
    }

    with pytest.raises(ValueError, match="cannot define both component hydraulic loss"):
        validate_network_data(data)


def test_semantic_validation_requires_ghe_circulation_pumps_for_one_pipe() -> None:
    data = {
        "network": {"type": "one_pipe"},
        "ground_heat_exchangers": {
            "ghe_1": {"circulation_pump": {"reference_pressure_drop_pa": 50_000.0}},
            "ghe_2": {},
        },
    }

    with pytest.raises(
        ValueError,
        match=r"one_pipe network requires 'circulation_pump'.*missing for: ghe_2",
    ):
        validate_network_data(data)
