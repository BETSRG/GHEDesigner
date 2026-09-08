from __future__ import annotations

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
        "wire_to_water_efficiency": 0.7,
    }
    network_data = {
        "type": "one_pipe",
        "stations": [{"component": "building_1"}, {"component": "ghe_1"}],
        "pipe_defaults": {"diameter": 0.1},
        "segments": [
            {"id": "segment_1", "from": "building_1", "to": "ghe_1", "length": 10.0},
            {"id": "segment_2", "from": "ghe_1", "to": "building_1", "length": 10.0},
        ],
        "pumps": {"building_pump": pump, "distribution_pump": pump},
        "component_pumps": {"building_1": "building_pump"},
        "distribution_pump": {"type": "pump", "pump": "distribution_pump"},
        "mass_flow_control": {
            "distribution_flow_multiplier": 1.5,
            "minimum_distribution_mass_flow": 0.1,
        },
    }
    component_data = {
        "building": {"building_1": {}},
        "ground_heat_exchanger": {"ghe_1": {"circulation_pump": {"reference_pressure_drop": 50_000.0}}},
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


def test_semantic_validation_rejects_duplicate_ghe_pressure_loss() -> None:
    data = {
        "network": {"type": "one_pipe"},
        "ground_heat_exchanger": {
            "ghe_1": {
                "circulation_pump": {"reference_pressure_drop": 50_000.0},
                "hydraulics": {
                    "type": "passive",
                    "reference_mass_flow": 1.0,
                    "reference_pressure_drop": 10_000.0,
                },
            }
        },
    }

    with pytest.raises(ValueError, match="cannot define both component hydraulic loss"):
        validate_network_data(data)
