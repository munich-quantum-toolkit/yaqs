# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small structural and numerical tests for WP20 resource fairness."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.wp20_resources import (
    DEFAULT_NORMALIZED_COMPUTE_POLICY,
    CircuitResourceMetrics,
    InfeasibleResourceBudget,
    NormalizedComputePolicy,
    ParetoPoint,
    ReachableResourceStratum,
    ResourceBudget,
    SelectedResourceStratum,
    WP20WorkLedger,
    deterministic_pareto_frontier,
    measure_circuit_resources,
    resource_selection_outcome_from_dict,
    select_reachable_resource_stratum,
    wp20_work_from_noisy_krotov,
)
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate
from tests.benchmarks.test_state_preparation_wp17_noisy_krotov import _stage, _successful


def _reseal(document: dict[str, object]) -> None:
    """Refresh a top-level test document checksum after deliberate tampering."""
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })


def _mixed_circuit() -> ParameterizedCircuit:
    """Return a q2 circuit with one compiled RXX basis-change round trip."""
    return ParameterizedCircuit(
        num_qubits=2,
        gates=[
            ParameterizedGate("rx", (0,), param_index=0, logical_gate_id="left-rx"),
            ParameterizedGate("rxx", (0, 1), param_index=1, logical_gate_id="entangler"),
            ParameterizedGate("ry", (1,), param_index=2, logical_gate_id="right-ry"),
        ],
        num_params=3,
    )


def _resource_circuit(native_two_qubit_gates: int, *, num_params: int | None = None) -> ParameterizedCircuit:
    """Return a q2 circuit with a requested compiled two-qubit count."""
    gates = [
        ParameterizedGate("rzz", (0, 1), param_index=index, logical_gate_id=f"rzz-{index}")
        for index in range(native_two_qubit_gates)
    ]
    return ParameterizedCircuit(
        num_qubits=2,
        gates=gates,
        num_params=native_two_qubit_gates if num_params is None else num_params,
    )


def _chain_resource_circuit(edge_counts: tuple[int, ...]) -> ParameterizedCircuit:
    """Return a chain circuit with the requested compiled count on each edge."""
    gates: list[ParameterizedGate] = []
    for edge, count in enumerate(edge_counts):
        for occurrence in range(count):
            parameter = len(gates)
            gates.append(
                ParameterizedGate(
                    "rzz",
                    (edge, edge + 1),
                    param_index=parameter,
                    logical_gate_id=f"edge-{edge}-{occurrence}",
                )
            )
    return ParameterizedCircuit(
        num_qubits=len(edge_counts) + 1,
        gates=gates,
        num_params=len(gates),
    )


def _stratum(
    stratum_id: str,
    native_two_qubit_gates: int,
    normalized_compute: int,
    *,
    num_params: int | None = None,
) -> ReachableResourceStratum:
    """Build compact resource evidence for selection and frontier tests.

    Returns:
        The requested reachable stratum.
    """
    return ReachableResourceStratum(
        stratum_id=stratum_id,
        circuit_resources=measure_circuit_resources(_resource_circuit(native_two_qubit_gates, num_params=num_params)),
        work=WP20WorkLedger(forward_circuit_evaluations=normalized_compute),
    )


def test_work_ledger_is_additive_sealed_and_projects_without_changing_wp16() -> None:
    """Detailed accounting has exact additive and compatibility semantics."""
    ledger = WP20WorkLedger().plus(
        forward_circuit_evaluations=2,
        backward_circuit_evaluations=1,
        trajectory_gate_applications=7,
        training_trajectories=3,
        checkpoint_validation_trajectories=5,
        test_trajectories=11,
        objective_calls=4,
        gradient_calls=2,
        cross_trajectory_pairings=18,
        wall_time_seconds=0.25,
        peak_memory_bytes=100,
    )
    merged = ledger.plus(wall_time_seconds=0.75, peak_memory_bytes=80, objective_calls=1)

    assert merged.total_sampled_trajectories == 19
    assert merged.wall_time_seconds == pytest.approx(1.0)
    assert merged.peak_memory_bytes == 100
    assert merged.objective_calls == 5
    assert merged.normalized_compute() == pytest.approx(28.0)
    assert merged.phase2_projection() == {
        "objective_evaluations": 5,
        "gradient_evaluations": 2,
        "training_trajectories": 3,
        "checkpoint_validation_trajectories": 5,
        "test_trajectories": 11,
        "trajectory_gate_applications": 7,
    }
    assert WP20WorkLedger.from_json(merged.to_json()) == merged
    assert WP20WorkLedger.from_dict(merged.to_dict()).content_checksum == merged.content_checksum
    with pytest.raises(FrozenInstanceError):
        merged.objective_calls = 0  # ty: ignore[invalid-assignment]


def test_work_ledger_rejects_forged_totals_types_and_unknown_counters() -> None:
    """No caller can substitute a summary for mechanically tracked work."""
    with pytest.raises(TypeError, match="an int"):
        WP20WorkLedger(objective_calls=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 0"):
        WP20WorkLedger(training_trajectories=-1)
    with pytest.raises(ValueError, match="Unknown WP20 work counters"):
        WP20WorkLedger().plus(native_two_qubit_gates=1)

    document = WP20WorkLedger(training_trajectories=2, test_trajectories=3).to_dict()
    document["total_sampled_trajectories"] = 99
    _reseal(document)
    with pytest.raises(ValueError, match="sum of the three trajectory roles"):
        WP20WorkLedger.from_dict(document)


def test_cross_pairing_and_optimizer_work_have_unambiguous_units() -> None:
    """Cross work scales as updates times parameters times R squared."""
    cross = WP20WorkLedger(
        forward_circuit_evaluations=6,
        backward_circuit_evaluations=6,
        training_trajectories=6,
        objective_calls=2,
        gradient_calls=0,
        cross_trajectory_pairings=2 * 3 * 3**2,
    )
    adam = WP20WorkLedger(forward_circuit_evaluations=2 * 3 * 2, gradient_calls=2)
    spsa = WP20WorkLedger(forward_circuit_evaluations=2 * 2, gradient_calls=2)

    assert cross.cross_trajectory_pairings == 54
    assert cross.phase2_projection()["gradient_evaluations"] == 0
    assert adam.forward_circuit_evaluations == 12
    assert spsa.forward_circuit_evaluations == 4


def test_krotov_projection_counts_forward_backward_and_quadratic_cross_work() -> None:
    """Verified WP17 evidence maps mechanically onto the shared WP20 units."""
    cross_stage = _stage(update="cross", iterations=2, trajectories=3)
    cross_execution = _successful(cross_stage)
    cross = wp20_work_from_noisy_krotov(
        cross_stage,
        cross_execution,
        wall_time_seconds=0.5,
        peak_memory_bytes=128,
    )
    assert cross.forward_circuit_evaluations == 18
    assert cross.backward_circuit_evaluations == 6
    assert cross.trajectory_gate_applications == 54
    assert cross.cross_trajectory_pairings == 54
    assert cross.gradient_calls == 0
    assert cross.wall_time_seconds == pytest.approx(0.5)
    assert cross.peak_memory_bytes == 128

    noiseless_stage = _stage(noise_id="noiseless", iterations=2)
    noiseless = wp20_work_from_noisy_krotov(noiseless_stage, _successful(noiseless_stage))
    assert noiseless.forward_circuit_evaluations == 5
    assert noiseless.backward_circuit_evaluations == 2
    assert noiseless.objective_calls == 5
    assert noiseless.gradient_calls == 2


def test_normalized_compute_policy_is_explicit_versioned_and_frozen() -> None:
    """Only the checksum-addressed unit policy can define the compute cap."""
    policy = NormalizedComputePolicy.from_dict(DEFAULT_NORMALIZED_COMPUTE_POLICY.to_dict())
    ledger = WP20WorkLedger(
        forward_circuit_evaluations=2,
        backward_circuit_evaluations=3,
        trajectory_gate_applications=5,
        cross_trajectory_pairings=7,
        objective_calls=100,
        training_trajectories=100,
    )

    assert policy.compute(ledger) == pytest.approx(17.0)
    assert policy.content_checksum == DEFAULT_NORMALIZED_COMPUTE_POLICY.content_checksum
    with pytest.raises(ValueError, match="frozen unit weight"):
        NormalizedComputePolicy(forward_circuit_evaluation_weight=2.0)


def test_circuit_resources_use_the_frozen_compiler_and_exact_dependency_depth() -> None:
    """Logical/native counts come from gate semantics and Quantinuum compilation."""
    resources = measure_circuit_resources(_mixed_circuit())

    assert resources.trainable_parameter_count == 3
    assert resources.logical_one_qubit_gates == 2
    assert resources.logical_two_qubit_gates == 1
    assert resources.logical_depth == 3
    assert resources.native_one_qubit_gates == 6
    assert resources.native_two_qubit_gates == 1
    assert resources.native_two_qubit_gates_per_chain_edge == (1,)
    assert resources.native_depth == 5
    assert tuple(event.name for event in resources.native_events) == (
        "rx",
        "h",
        "h",
        "rzz",
        "h",
        "h",
        "ry",
    )
    assert tuple(event.native_gate_id for event in resources.native_events) == tuple(range(7))
    assert resources.native_events[3].source_gate_name == "rxx"
    assert resources.native_events[3].basis_change_relationship == "rxx_h"
    assert CircuitResourceMetrics.from_json(resources.to_json()) == resources


def test_circuit_resource_aliases_are_recomputed_and_data_maps_reject() -> None:
    """Re-sealing aggregate aliases cannot override event-level evidence."""
    document = measure_circuit_resources(_mixed_circuit()).to_dict()
    document["native_two_qubit_gates"] = 99
    _reseal(document)
    with pytest.raises(ValueError, match="not derived from the event evidence"):
        CircuitResourceMetrics.from_dict(document)

    document = measure_circuit_resources(_mixed_circuit()).to_dict()
    document["native_two_qubit_gates_per_chain_edge"] = [0]
    _reseal(document)
    with pytest.raises(ValueError, match="not derived from the event evidence"):
        CircuitResourceMetrics.from_dict(document)

    data_dependent = ParameterizedCircuit(
        num_qubits=2,
        gates=[ParameterizedGate("rx", (0,), param_index=0, data_map=lambda _x: 0.0)],
        num_params=1,
    )
    with pytest.raises(ValueError, match="data-dependent"):
        measure_circuit_resources(data_dependent)


def test_per_edge_counts_are_mechanical_and_reject_edge_concentration() -> None:
    """A total below the aggregate chain capacity cannot hide one overfull edge."""
    concentrated = ReachableResourceStratum(
        "edge-concentrated",
        measure_circuit_resources(_chain_resource_circuit((3, 0))),
        WP20WorkLedger(),
    )
    balanced = ReachableResourceStratum(
        "balanced",
        measure_circuit_resources(_chain_resource_circuit((2, 2))),
        WP20WorkLedger(),
    )
    budget = ResourceBudget(
        native_two_qubit_gate_cap_per_chain_edge=2,
        normalized_compute_cap=0.0,
    )

    assert concentrated.circuit_resources.native_two_qubit_gates == 3
    assert concentrated.circuit_resources.native_two_qubit_gates_per_chain_edge == (3, 0)
    assert balanced.circuit_resources.native_two_qubit_gates == 4
    assert balanced.circuit_resources.native_two_qubit_gates_per_chain_edge == (2, 2)
    assert CircuitResourceMetrics.from_json(balanced.circuit_resources.to_json()) == balanced.circuit_resources
    assert isinstance(select_reachable_resource_stratum((concentrated,), budget), InfeasibleResourceBudget)

    outcome = select_reachable_resource_stratum((concentrated, balanced), budget)
    assert isinstance(outcome, SelectedResourceStratum)
    assert outcome.selected.stratum_id == "balanced"
    assert outcome.native_two_qubit_residuals_per_chain_edge == (0, 0)
    assert outcome.exact_native_match


def test_per_edge_residuals_report_every_unmatched_chain_edge() -> None:
    """Reaching the cap on one edge does not claim an exact vector match."""
    asymmetric = ReachableResourceStratum(
        "asymmetric",
        measure_circuit_resources(_chain_resource_circuit((2, 1))),
        WP20WorkLedger(),
    )
    outcome = select_reachable_resource_stratum(
        (asymmetric,),
        ResourceBudget(
            native_two_qubit_gate_cap_per_chain_edge=2,
            normalized_compute_cap=0.0,
        ),
    )

    assert isinstance(outcome, SelectedResourceStratum)
    assert outcome.native_two_qubit_residuals_per_chain_edge == (0, 1)
    assert not outcome.exact_native_match
    assert resource_selection_outcome_from_dict(outcome.to_dict()) == outcome


def test_joint_budget_selection_is_order_independent_and_reports_residuals() -> None:
    """The richest jointly feasible attempted stratum is selected mechanically."""
    strata = (
        _stratum("stage-three", 3, 5),
        _stratum("stage-one", 1, 30),
        _stratum("stage-two", 2, 20),
    )
    budget = ResourceBudget(native_two_qubit_gate_cap_per_chain_edge=2, normalized_compute_cap=25.0)
    forward = select_reachable_resource_stratum(strata, budget)
    reverse = select_reachable_resource_stratum(tuple(reversed(strata)), budget)

    assert isinstance(forward, SelectedResourceStratum)
    assert isinstance(reverse, SelectedResourceStratum)
    assert forward.selected.stratum_id == reverse.selected.stratum_id == "stage-two"
    assert forward.native_two_qubit_residuals_per_chain_edge == (0,)
    assert forward.normalized_compute_residual == pytest.approx(5.0)
    assert forward.exact_native_match
    assert not forward.exact_compute_match
    assert resource_selection_outcome_from_dict(forward.to_dict()) == forward


def test_infeasible_budget_is_typed_and_proves_all_attempts_violate_a_cap() -> None:
    """No reachable candidate produces a fabricated matched result."""
    attempts = (_stratum("one", 1, 3), _stratum("two", 2, 2))
    outcome = select_reachable_resource_stratum(
        attempts,
        ResourceBudget(native_two_qubit_gate_cap_per_chain_edge=0, normalized_compute_cap=1.0),
    )

    assert isinstance(outcome, InfeasibleResourceBudget)
    assert outcome.status == "infeasible"
    assert outcome.reason == "no_reachable_stratum_within_joint_caps"
    assert tuple(item.stratum_id for item in outcome.attempted_strata) == ("one", "two")
    assert resource_selection_outcome_from_dict(outcome.to_dict()) == outcome
    with pytest.raises(ValueError, match="cannot contain a stratum"):
        InfeasibleResourceBudget(
            budget=ResourceBudget(native_two_qubit_gate_cap_per_chain_edge=2, normalized_compute_cap=10.0),
            attempted_strata=attempts,
        )


def test_parameter_count_cannot_substitute_for_native_two_qubit_count() -> None:
    """Feasibility reads compiled entanglers, never trainable-vector length."""
    one_parameter_two_entanglers = ParameterizedCircuit(
        num_qubits=2,
        gates=[
            ParameterizedGate("rzz", (0, 1), param_index=0, logical_gate_id="first"),
            ParameterizedGate("rzz", (0, 1), param_index=0, logical_gate_id="second"),
        ],
        num_params=1,
    )
    ten_parameters_no_entangler = ParameterizedCircuit(
        num_qubits=2,
        gates=[ParameterizedGate("rx", (0,), param_index=0)],
        num_params=10,
    )
    entangled = ReachableResourceStratum(
        "entangled",
        measure_circuit_resources(one_parameter_two_entanglers),
        WP20WorkLedger(),
    )
    local = ReachableResourceStratum(
        "local",
        measure_circuit_resources(ten_parameters_no_entangler),
        WP20WorkLedger(),
    )

    assert entangled.circuit_resources.trainable_parameter_count == 1
    assert entangled.circuit_resources.native_two_qubit_gates == 2
    assert isinstance(
        select_reachable_resource_stratum(
            (entangled,),
            ResourceBudget(native_two_qubit_gate_cap_per_chain_edge=1, normalized_compute_cap=0.0),
        ),
        InfeasibleResourceBudget,
    )
    assert local.circuit_resources.trainable_parameter_count == 10
    assert local.circuit_resources.native_two_qubit_gates == 0
    assert isinstance(
        select_reachable_resource_stratum(
            (local,),
            ResourceBudget(native_two_qubit_gate_cap_per_chain_edge=0, normalized_compute_cap=0.0),
        ),
        SelectedResourceStratum,
    )


def test_pareto_frontier_is_deterministic_and_removes_only_dominated_points() -> None:
    """The frozen sweep keeps genuine resource/fidelity trade-offs."""
    points = (
        ParetoPoint(_stratum("high", 3, 20), 0.90),
        ParetoPoint(_stratum("dominated", 2, 8), 0.75),
        ParetoPoint(_stratum("low", 1, 10), 0.70),
        ParetoPoint(_stratum("trade", 2, 5), 0.80),
    )
    frontier = deterministic_pareto_frontier(points)
    reversed_frontier = deterministic_pareto_frontier(tuple(reversed(points)))

    assert tuple(point.stratum.stratum_id for point in frontier) == ("low", "trade", "high")
    assert tuple(point.content_checksum for point in reversed_frontier) == tuple(
        point.content_checksum for point in frontier
    )
    with pytest.raises(ValueError, match="unique stratum identifiers"):
        deterministic_pareto_frontier((points[0], points[0]))
