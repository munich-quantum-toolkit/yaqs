# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Focused mathematical and integrity tests for the WP21 pruning core."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from benchmarks.state_preparation.circuits import compile_quantinuum_native
from benchmarks.state_preparation.phase2.canonical import canonical_checksum
from benchmarks.state_preparation.phase2.noisy_krotov import NoisyKrotovCircuitBinding
from benchmarks.state_preparation.phase2.pruning import (
    TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
    TOPDOWN_MAGNITUDE_METHOD_ID,
    TOPDOWN_RANDOM_METHOD_ID,
    ParameterShiftRequest,
    PruningRoundResult,
    PruningStagePolicy,
    PruningStageSpec,
    PruningUnitKind,
    RemovalSchedule,
    ScoringObjectiveKind,
    build_pruning_units,
    generalized_parameter_shift_derivative,
    prune_circuit,
    rank_pruning_units,
    run_pruning_round,
)
from benchmarks.state_preparation.phase2.wp20_resources import WP20WorkLedger
from mqt.yaqs.optimization import ParameterizedCircuit, ParameterizedGate

if TYPE_CHECKING:
    from collections.abc import Collection

    from numpy.typing import NDArray


def _circuit(
    gates: list[ParameterizedGate],
    *,
    num_qubits: int = 1,
    num_params: int | None = None,
) -> ParameterizedCircuit:
    """Build one small parameterized circuit.

    Returns:
        The validated circuit used by a focused test.
    """
    return ParameterizedCircuit(num_qubits, gates, num_params=num_params)


def _binding(circuit: ParameterizedCircuit, topology_id: str = "wp21_test_input") -> NoisyKrotovCircuitBinding:
    """Bind a small logical circuit to the frozen Phase II policy.

    Returns:
        The immutable circuit binding.
    """
    return NoisyKrotovCircuitBinding(circuit, topology_id)


def _state(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    *,
    skipped_gate_indices: Collection[int] = (),
) -> NDArray[np.complex128]:
    """Return a dense one- or two-qubit state for an analytic test.

    Returns:
        The state obtained from the computational-zero input.

    Raises:
        ValueError: If the circuit contains more than two qubits.
    """
    if circuit.num_qubits not in {1, 2}:
        msg = "The focused dense helper supports only one or two qubits."
        raise ValueError(msg)
    skipped = set(skipped_gate_indices)
    state = np.zeros(2**circuit.num_qubits, dtype=np.complex128)
    state[0] = 1.0
    identity = np.eye(2, dtype=np.complex128)
    for gate_index, gate in enumerate(circuit.gates):
        if gate_index in skipped:
            continue
        matrix, sites = circuit.gate_matrix(gate, theta)
        if circuit.num_qubits == 1 or len(sites) == 2:
            embedded = matrix
        elif sites == (0,):
            embedded = np.kron(matrix, identity)
        else:
            embedded = np.kron(identity, matrix)
        state = embedded @ state
    return state


def _zero_fidelity(
    circuit: ParameterizedCircuit,
    parameters: NDArray[np.float64],
    request: ParameterShiftRequest,
) -> float:
    """Return computational-zero fidelity after verifying request binding.

    Returns:
        The exact dense-state fidelity.
    """
    shifted_binding = NoisyKrotovCircuitBinding(circuit, "shared_rotation")
    assert request.shifted_circuit_checksum == shifted_binding.content_checksum
    return float(abs(_state(circuit, parameters)[0]) ** 2)


def _policy(
    *,
    unit: PruningUnitKind = "gate",
    objective: ScoringObjectiveKind = "none",
    schedule: RemovalSchedule = "fixed_count",
    count: int | None = 1,
    fraction: float | None = None,
    relax: bool = False,
) -> PruningStagePolicy:
    """Construct one exact pruning policy.

    Returns:
        The checksum-sealed policy value.
    """
    return PruningStagePolicy(
        pruning_unit=unit,
        scoring_objective_kind=objective,
        removal_schedule=schedule,
        removal_count=count,
        removal_fraction=fraction,
        relax_after_round=relax,
    )


def _spec(
    method_id: str,
    policy: PruningStagePolicy,
    *,
    seed: int | None = None,
) -> PruningStageSpec:
    """Resolve a policy against the corresponding method identity.

    Returns:
        The complete pruning-stage specification.
    """
    rules = {
        TOPDOWN_RANDOM_METHOD_ID: "random",
        TOPDOWN_MAGNITUDE_METHOD_ID: "magnitude",
        TOPDOWN_IMPACT_ITERATIVE_METHOD_ID: "impact_iterative",
    }
    return PruningStageSpec(method_id=method_id, score_rule=rules[method_id], policy=policy, random_seed=seed)


def _separate_rotation_circuit(count: int) -> ParameterizedCircuit:
    """Return a one-qubit circuit with one parameter per rotation.

    Returns:
        A deterministic circuit with ``count`` trainable gates.
    """
    return _circuit(
        [
            ParameterizedGate("ry", (0,), param_index=index, logical_gate_id=f"rotation_{index}")
            for index in range(count)
        ],
        num_params=count,
    )


def test_occurrence_parameter_shift_matches_analytic_and_finite_difference_shared_gradient() -> None:
    """Occurrence shifts must sum the exact derivative of a shared parameter."""
    circuit = _circuit(
        [
            ParameterizedGate("ry", (0,), param_index=0, angle_scale=0.7, angle_offset=0.2),
            ParameterizedGate("ry", (0,), param_index=0, angle_scale=-0.4, angle_offset=-0.1),
        ],
        num_params=1,
    )
    binding = _binding(circuit, "shared_rotation")
    theta = np.array([0.37], dtype=np.float64)

    gradient, occurrences, evaluations, work = generalized_parameter_shift_derivative(
        binding,
        theta,
        _zero_fidelity,
    )

    total_angle = 0.3 * theta[0] + 0.1
    analytic = -0.5 * 0.3 * np.sin(total_angle)
    epsilon = 1e-6
    finite_difference = (
        abs(_state(circuit, theta + epsilon)[0]) ** 2 - abs(_state(circuit, theta - epsilon)[0]) ** 2
    ) / (2 * epsilon)
    assert gradient == pytest.approx([analytic], abs=1e-14)
    assert gradient[0] == pytest.approx(finite_difference, abs=1e-9)
    assert gradient[0] == pytest.approx(sum(occurrences.values()), abs=1e-15)
    assert tuple(occurrences) == (0, 1)
    assert [(item.request.gate_occurrence_index, item.request.sign) for item in evaluations] == [
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
    ]
    assert all(item.request.parameter_index == 0 for item in evaluations)
    assert work.forward_circuit_evaluations == 4
    assert work.objective_calls == 4
    assert work.gradient_calls == 1


def test_unit_construction_groups_shared_parameters_and_complete_native_entanglers() -> None:
    """Shared and compiled units must contain their complete atomic membership."""
    shared = _circuit(
        [
            ParameterizedGate("ry", (0,), param_index=0),
            ParameterizedGate("rz", (0,), param_index=0),
            ParameterizedGate("rx", (0,), param_index=1),
        ],
        num_params=2,
    )
    groups = build_pruning_units(shared, "shared_parameter_group")
    assert [unit.gate_indices for unit in groups] == [(0, 1), (2,)]
    assert [unit.parameter_indices for unit in groups] == [(0,), (1,)]
    with pytest.raises(ValueError, match="exactly one gate"):
        build_pruning_units(shared, "parameter")

    entanglers = _circuit(
        [
            ParameterizedGate("rxx", (0, 1), param_index=0, logical_gate_id="entangler_xx"),
            ParameterizedGate("ryy", (0, 1), param_index=1, logical_gate_id="entangler_yy"),
            ParameterizedGate("ry", (0,), param_index=2, logical_gate_id="local_y"),
        ],
        num_qubits=2,
        num_params=3,
    )
    compilation = compile_quantinuum_native(entanglers)
    units = build_pruning_units(entanglers, "compiled_entangler_group")
    assert len(units) == 2
    for unit, source in zip(units, compilation.mapping[:2], strict=True):
        expected_ids = tuple(compilation.circuit.gates[index].native_gate_id for index in source.native_gate_indices)
        assert unit.gate_indices == (source.source_logical_gate_index,)
        assert unit.native_gate_ids == expected_ids
        assert len(unit.native_gate_ids) == 5  # full basis-change round trip plus RZZ

    output, _theta, remap = prune_circuit(
        entanglers,
        np.array([0.2, 0.3, 0.4]),
        units,
        (units[0].unit_id,),
        output_topology_id="without_first_entangler",
    )
    assert remap.removed_input_gate_indices == (0,)
    assert [gate.logical_gate_id for gate in output.circuit.gates] == ["entangler_yy", "local_y"]
    assert all(
        event.logical_gate_id != "entangler_xx" for event in compile_quantinuum_native(output.circuit).circuit.gates
    )


def test_shared_parameter_impact_removes_every_occurrence_and_remaps_state() -> None:
    """A selected shared-parameter group disappears atomically with exact semantics."""
    circuit = _circuit(
        [
            ParameterizedGate("ry", (0,), param_index=0, logical_gate_id="shared-first"),
            ParameterizedGate("ry", (0,), param_index=0, logical_gate_id="shared-second"),
            ParameterizedGate("ry", (0,), param_index=1, logical_gate_id="retained"),
        ],
        num_params=2,
    )
    theta = np.array([0.0, 0.4], dtype=np.float64)
    result = run_pruning_round(
        _binding(circuit, "shared_rotation"),
        theta,
        _spec(
            TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
            _policy(unit="shared_parameter_group", objective="noiseless_fidelity"),
        ),
        round_index=0,
        output_topology_id="shared_group_output",
        objective=_zero_fidelity,
    )

    assert result.removed_unit_ids == ("shared_parameter_000000",)
    assert result.parameter_remap.removed_input_gate_indices == (0, 1)
    assert result.parameter_remap.old_to_new_parameter_indices == ((1, 0),)
    assert [gate.logical_gate_id for gate in result.output_circuit_binding.circuit.gates] == ["retained"]
    assert result.output_theta == pytest.approx([theta[1]])
    assert _state(result.output_circuit_binding.circuit, result.output_theta) == pytest.approx(
        _state(circuit, theta, skipped_gate_indices=(0, 1)),
        abs=1e-14,
    )


def test_magnitude_ties_and_random_rankings_are_deterministic() -> None:
    """Stable IDs break exact ties and random repetitions are seed-controlled."""
    circuit = _separate_rotation_circuit(4)
    units = build_pruning_units(circuit, "gate")
    magnitude = _spec(TOPDOWN_MAGNITUDE_METHOD_ID, _policy())
    tied = rank_pruning_units(circuit, np.array([0.25, -0.25, 1.0, -1.0]), units, magnitude)
    assert [item.unit_id for item in tied] == [
        "gate_000000",
        "gate_000001",
        "gate_000002",
        "gate_000003",
    ]

    random_a = _spec(TOPDOWN_RANDOM_METHOD_ID, _policy(), seed=91)
    random_b = _spec(TOPDOWN_RANDOM_METHOD_ID, _policy(), seed=92)
    theta = np.zeros(4)
    first = rank_pruning_units(circuit, theta, units, random_a)
    repetition = rank_pruning_units(circuit, theta, tuple(reversed(units)), random_a)
    independent = rank_pruning_units(circuit, theta, units, random_b)
    assert [(item.unit_id, item.score) for item in first] == [(item.unit_id, item.score) for item in repetition]
    assert [(item.unit_id, item.score) for item in first] != [(item.unit_id, item.score) for item in independent]


@pytest.mark.parametrize(
    ("policy", "expected_removed"),
    [
        (_policy(count=2), 2),
        (_policy(schedule="fraction_floor", count=None, fraction=0.4), 2),
    ],
)
def test_fixed_count_and_fraction_floor_schedules(
    policy: PruningStagePolicy,
    expected_removed: int,
) -> None:
    """Both schedules remove the exact deterministic number of units."""
    circuit = _separate_rotation_circuit(5)
    result = run_pruning_round(
        _binding(circuit),
        np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
        _spec(TOPDOWN_MAGNITUDE_METHOD_ID, policy),
        round_index=0,
        output_topology_id="wp21_test_output",
    )
    assert len(result.removed_unit_ids) == expected_removed
    assert result.output_circuit_binding.circuit.num_params == 5 - expected_removed
    assert PruningRoundResult.from_dict(result.to_dict()).to_dict() == result.to_dict()


def test_fraction_floor_rejects_a_zero_or_all_unit_removal() -> None:
    """A pruning round cannot silently become a no-op or remove the whole model."""
    circuit = _separate_rotation_circuit(4)
    spec = _spec(
        TOPDOWN_MAGNITUDE_METHOD_ID,
        _policy(schedule="fraction_floor", count=None, fraction=0.2),
    )
    with pytest.raises(ValueError, match="at least one"):
        run_pruning_round(
            _binding(circuit),
            np.arange(1.0, 5.0),
            spec,
            round_index=0,
            output_topology_id="fraction_noop",
        )


def test_parameter_remap_preserves_retained_state_and_all_gate_metadata() -> None:
    """Compaction must change only parameter indices and selected gate presence."""
    circuit = _circuit(
        [
            ParameterizedGate("ry", (0,), param_index=0, angle_scale=-0.5, angle_offset=0.1),
            ParameterizedGate("rxx", (0, 1), param_index=1, angle_scale=0.7, angle_offset=-0.2),
            ParameterizedGate("rz", (1,), param_index=2, angle_scale=1.3, angle_offset=0.4),
        ],
        num_qubits=2,
        num_params=3,
    )
    theta = np.array([0.31, -0.47, 0.19])
    units = build_pruning_units(circuit, "gate")
    output, output_theta, remap = prune_circuit(
        circuit,
        theta,
        units,
        (units[1].unit_id,),
        output_topology_id="remapped_circuit",
    )
    assert remap.old_to_new_parameter_indices == ((0, 0), (2, 1))
    assert remap.removed_parameter_indices == (1,)
    assert np.array_equal(output_theta, theta[[0, 2]])
    assert _state(output.circuit, output_theta) == pytest.approx(
        _state(circuit, theta, skipped_gate_indices=(1,)),
        abs=1e-14,
    )

    retained_input = [circuit.gates[index] for index in remap.retained_input_gate_indices]
    for old_gate, new_gate in zip(retained_input, output.circuit.gates, strict=True):
        assert new_gate.name == old_gate.name
        assert new_gate.sites == old_gate.sites
        assert new_gate.angle_scale == old_gate.angle_scale
        assert new_gate.angle_offset == old_gate.angle_offset
        assert new_gate.fixed_params == old_gate.fixed_params
        assert new_gate.logical_gate_id == old_gate.logical_gate_id
        assert new_gate.native_gate_id == old_gate.native_gate_id
        assert new_gate.noise_enabled == old_gate.noise_enabled


def test_policy_and_round_decoders_reject_resealed_semantic_tampering() -> None:
    """Checksums and reconstruction reject structural and scientifically false data."""
    policy = _policy(count=1)
    damaged_policy = policy.to_mapping()
    damaged_policy["removal_count"] = 2
    with pytest.raises(ValueError, match="checksum"):
        PruningStagePolicy.from_mapping(damaged_policy)

    circuit = _separate_rotation_circuit(4)
    result = run_pruning_round(
        _binding(circuit),
        np.array([0.1, 0.2, 0.3, 0.4]),
        _spec(TOPDOWN_MAGNITUDE_METHOD_ID, policy),
        round_index=0,
        output_topology_id="tamper_output",
    )
    document = copy.deepcopy(result.to_dict())
    scores = cast("list[dict[str, object]]", document["scores"])
    score = scores[-1]
    member_scores = cast("list[float]", score["member_scores"])
    member_scores[0] = 0.45
    score["score"] = 0.45
    score["content_checksum"] = canonical_checksum({
        key: value for key, value in score.items() if key != "content_checksum"
    })
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })
    with pytest.raises(ValueError, match="Pruning scores"):
        PruningRoundResult.from_dict(document)


@pytest.mark.parametrize(
    "field",
    [
        "forward_circuit_evaluations",
        "backward_circuit_evaluations",
        "trajectory_gate_applications",
        "training_trajectories",
        "checkpoint_validation_trajectories",
        "test_trajectories",
        "objective_calls",
        "gradient_calls",
        "cross_trajectory_pairings",
        "wall_time_seconds",
        "peak_memory_bytes",
    ],
)
def test_impact_round_rejects_every_resealed_false_work_field(field: str) -> None:
    """Every impact-work field is derived from shifts and scoring trajectories."""
    circuit = _separate_rotation_circuit(2)
    trajectories = 3
    result = run_pruning_round(
        _binding(circuit, "shared_rotation"),
        np.array([0.1, 0.2], dtype=np.float64),
        _spec(
            TOPDOWN_IMPACT_ITERATIVE_METHOD_ID,
            _policy(objective="fixed_map_sample_average_fidelity"),
        ),
        round_index=0,
        output_topology_id="impact_work_output",
        objective=_zero_fidelity,
        scoring_trajectory_count=trajectories,
        sampling_work=WP20WorkLedger(
            forward_circuit_evaluations=trajectories,
            trajectory_gate_applications=trajectories * len(circuit.gates),
            training_trajectories=trajectories,
        ),
    )
    document = copy.deepcopy(result.to_dict())
    work = cast("dict[str, object]", document["work"])
    work[field] = cast("float", work[field]) + (0.25 if field == "wall_time_seconds" else 1)
    work["total_sampled_trajectories"] = sum(
        cast("int", work[name])
        for name in ("training_trajectories", "checkpoint_validation_trajectories", "test_trajectories")
    )
    work["content_checksum"] = canonical_checksum({
        key: value for key, value in work.items() if key != "content_checksum"
    })
    document["content_checksum"] = canonical_checksum({
        key: value for key, value in document.items() if key != "content_checksum"
    })

    with pytest.raises(ValueError, match="work is not exactly implied"):
        PruningRoundResult.from_dict(document)


def test_policy_records_are_immutable_and_exactly_decodable() -> None:
    """The embedded stage mapping round-trips and rejects a changed frozen rule."""
    policy = _policy(unit="shared_parameter_group", objective="none", count=1)
    assert PruningStagePolicy.from_mapping(policy.to_mapping()) == policy
    with pytest.raises(ValueError, match="score_aggregation"):
        replace(policy, score_aggregation="mean_member_scores_v1")
