# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Numerical validation for the WP17 fixed-rate noisy Krotov stage.

The tests in this module intentionally use an exhaustive branch oracle for the
small systems.  This separates probability-law validation from the production
sampler and makes the boundary of the pathwise approximation explicit.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from benchmarks.state_preparation.noise import (
    STANDARD_ONE_QUBIT_GATE_STRENGTH,
    STANDARD_TWO_QUBIT_GATE_STRENGTH,
    create_scaled_standard_noise_provider,
)
from benchmarks.state_preparation.phase2.fixed_rate_validation import (
    FixedRateBranchTree,
    enumerate_fixed_rate_pauli_branches,
)
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.optimization import (
    KrotovNoiseMap,
    KrotovTJMOptions,
    KrotovTruncation,
    ParameterizedCircuit,
    ParameterizedGate,
    noisy_state_preparation_contribution,
    noisy_state_preparation_cross_contribution,
    noisy_state_preparation_metrics,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.optimization import GateNoiseProvider

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _random_target(num_qubits: int, seed: int) -> NDArray[np.complex128]:
    """Return a deterministic normalized dense target state."""
    rng = np.random.default_rng(seed)
    target = rng.normal(size=2**num_qubits) + 1j * rng.normal(size=2**num_qubits)
    return np.asarray(target / np.linalg.norm(target), dtype=np.complex128)


def _replay_options(options: KrotovTJMOptions) -> KrotovTJMOptions:
    """Return single-trajectory options suitable for one exact branch."""
    return replace(options, num_trajectories=1, trajectory_update="independent")


def _weighted_branch_gradient(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    tree: FixedRateBranchTree,
    options: KrotovTJMOptions,
    truncation: KrotovTruncation,
) -> NDArray[np.float64]:
    """Return the probability-weighted pathwise gradient over an exact tree."""
    replay_options = _replay_options(options)
    gradient = np.zeros(circuit.num_params, dtype=np.float64)
    for branch in tree.circuit_branches:
        contribution, _loss, _fidelity, _trajectories = noisy_state_preparation_contribution(
            circuit,
            theta,
            target,
            None,
            replay_options,
            MPS(circuit.num_qubits),
            truncation,
            fixed_noise_maps=[list(branch.noise_maps)],
        )
        gradient += branch.probability * contribution
    return gradient


def _weighted_branch_loss(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    tree: FixedRateBranchTree,
    options: KrotovTJMOptions,
    truncation: KrotovTruncation,
) -> float:
    """Return the probability-weighted infidelity over an exact tree."""
    replay_options = _replay_options(options)
    loss = 0.0
    for branch in tree.circuit_branches:
        branch_loss, _fidelity, _fidelities = noisy_state_preparation_metrics(
            circuit,
            theta,
            target,
            None,
            replay_options,
            truncation=truncation,
            fixed_noise_maps=[list(branch.noise_maps)],
        )
        loss += branch.probability * branch_loss
    return loss


def _central_difference_with_tree(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    target: NDArray[np.complex128],
    tree: FixedRateBranchTree,
    options: KrotovTJMOptions,
    truncation: KrotovTruncation,
    *,
    step: float = 1e-6,
) -> NDArray[np.float64]:
    """Differentiate a branch-weighted loss while holding branch weights fixed.

    Returns:
        Central finite differences for every circuit parameter.
    """
    derivative = np.zeros(circuit.num_params, dtype=np.float64)
    for parameter_index in range(circuit.num_params):
        plus = theta.copy()
        minus = theta.copy()
        plus[parameter_index] += step
        minus[parameter_index] -= step
        derivative[parameter_index] = (
            _weighted_branch_loss(circuit, plus, target, tree, options, truncation)
            - _weighted_branch_loss(circuit, minus, target, tree, options, truncation)
        ) / (2.0 * step)
    return derivative


def _fixed_rate_case(
    num_qubits: int,
) -> tuple[
    ParameterizedCircuit,
    NDArray[np.float64],
    NDArray[np.complex128],
    GateNoiseProvider,
    KrotovTJMOptions,
    int,
]:
    """Build one deterministic two-, three-, or four-qubit exact case.

    Returns:
        Circuit, parameters, target, provider, options, and exact branch count.

    Raises:
        ValueError: If ``num_qubits`` is outside the covered two-to-four range.
    """
    if num_qubits == 2:
        circuit = ParameterizedCircuit(
            2,
            [
                ParameterizedGate("ry", (0,), param_index=0),
                ParameterizedGate("rzz", (0, 1), param_index=1),
            ],
            num_params=2,
        )
        theta = np.array([0.31, -0.47], dtype=np.float64)
        provider = create_scaled_standard_noise_provider("dephasing_1s_all", 100.0)
        options = KrotovTJMOptions(num_trajectories=1, dt=0.73)
        expected_branch_count = 6
    elif num_qubits == 3:
        circuit = ParameterizedCircuit(
            3,
            [
                ParameterizedGate("rx", (0,), param_index=0),
                ParameterizedGate("rzz", (0, 2), param_index=1),
                ParameterizedGate("ry", (1,), param_index=2),
            ],
            num_params=3,
        )
        theta = np.array([0.20, -0.40, 0.60], dtype=np.float64)
        provider = create_scaled_standard_noise_provider("dephasing_2s_2q", 100.0)
        options = KrotovTJMOptions(num_trajectories=1, dt=1.3, noisy_gate_indices=(1,))
        expected_branch_count = 2
    elif num_qubits == 4:
        circuit = ParameterizedCircuit(
            4,
            [
                ParameterizedGate("ry", (0,), param_index=0),
                ParameterizedGate("ry", (1,), param_index=1),
                ParameterizedGate("rzz", (1, 2), param_index=2),
                ParameterizedGate("ry", (3,), param_index=3),
            ],
            num_params=4,
        )
        theta = np.array([0.11, 0.29, -0.27, 0.43], dtype=np.float64)
        provider = create_scaled_standard_noise_provider("depolarizing_1s_1q", 100.0)
        options = KrotovTJMOptions(num_trajectories=1, dt=0.9, noisy_gate_indices=(0, 3))
        expected_branch_count = 16
    else:
        msg = f"Unsupported exact validation size {num_qubits}."
        raise ValueError(msg)
    return circuit, theta, _random_target(num_qubits, 100 + num_qubits), provider, options, expected_branch_count


def test_exhaustive_branch_tree_has_exact_no_jump_jump_and_placement_law() -> None:
    """The oracle enumerates p0, one jump, placement, and normalization exactly."""
    circuit = ParameterizedCircuit(
        2,
        [
            ParameterizedGate("ry", (0,), param_index=0),
            ParameterizedGate("rzz", (0, 1), param_index=1),
        ],
        num_params=2,
    )
    theta = np.array([0.31, -0.47], dtype=np.float64)
    dt = 0.73
    provider = create_scaled_standard_noise_provider("dephasing_1s_all", 1.0)
    options = KrotovTJMOptions(num_trajectories=1, dt=dt)

    tree = enumerate_fixed_rate_pauli_branches(circuit, theta, provider, options)

    assert tuple(len(branches) for branches in tree.gate_branches) == (2, 3)
    assert len(tree.circuit_branches) == 6
    one_qubit, two_qubit = tree.gate_branches
    assert tuple(branch.outcome_label for branch in one_qubit) == ("no_jump", "pauli_z")
    assert tuple(branch.outcome_label for branch in two_qubit) == ("no_jump", "pauli_z", "pauli_z")
    assert one_qubit[0].probability == pytest.approx(math.exp(-dt * STANDARD_ONE_QUBIT_GATE_STRENGTH))
    assert two_qubit[0].probability == pytest.approx(math.exp(-dt * 2.0 * STANDARD_TWO_QUBIT_GATE_STRENGTH))
    assert tuple(branch.noise_map.operators[0][1] for branch in two_qubit[1:]) == ((0,), (1,))

    for gate_branches in tree.gate_branches:
        assert math.fsum(branch.probability for branch in gate_branches) == pytest.approx(1.0, abs=1e-14)
        for branch in gate_branches:
            assert branch.noise_map.normalized
            assert len(branch.noise_map.operators) == (0 if branch.process_index is None else 1)
            assert branch.noise_map.source_gate_index == branch.gate_index
    for branch in tree.circuit_branches:
        assert branch.probability == pytest.approx(
            math.prod(gate_branch.probability for gate_branch in branch.gate_branches),
            abs=1e-15,
        )
    assert math.fsum(branch.probability for branch in tree.circuit_branches) == pytest.approx(1.0, abs=2e-14)


@pytest.mark.parametrize("num_qubits", [2, 3, 4])
def test_fixed_rate_pauli_expected_gradient_matches_branchwise_paths_and_finite_difference(
    num_qubits: int,
) -> None:
    """Weighted branch paths equal the independent expected-loss derivative."""
    circuit, theta, target, provider, options, expected_branch_count = _fixed_rate_case(num_qubits)
    truncation = KrotovTruncation()
    base_tree = enumerate_fixed_rate_pauli_branches(circuit, theta, provider, options)
    gradient = _weighted_branch_gradient(circuit, theta, target, base_tree, options, truncation)
    base_labels = tuple(branch.outcome_labels for branch in base_tree.circuit_branches)
    base_probabilities = np.asarray(
        [branch.probability for branch in base_tree.circuit_branches],
        dtype=np.float64,
    )
    finite_difference = np.zeros(circuit.num_params, dtype=np.float64)
    step = 1e-6

    assert len(base_tree.circuit_branches) == expected_branch_count
    for parameter_index in range(circuit.num_params):
        plus = theta.copy()
        minus = theta.copy()
        plus[parameter_index] += step
        minus[parameter_index] -= step
        plus_tree = enumerate_fixed_rate_pauli_branches(circuit, plus, provider, options)
        minus_tree = enumerate_fixed_rate_pauli_branches(circuit, minus, provider, options)
        plus_probabilities = np.asarray(
            [branch.probability for branch in plus_tree.circuit_branches],
            dtype=np.float64,
        )
        minus_probabilities = np.asarray(
            [branch.probability for branch in minus_tree.circuit_branches],
            dtype=np.float64,
        )

        assert tuple(branch.outcome_labels for branch in plus_tree.circuit_branches) == base_labels
        assert tuple(branch.outcome_labels for branch in minus_tree.circuit_branches) == base_labels
        np.testing.assert_array_equal(plus_probabilities, base_probabilities)
        np.testing.assert_array_equal(minus_probabilities, base_probabilities)
        np.testing.assert_array_equal((plus_probabilities - minus_probabilities) / (2.0 * step), 0.0)
        finite_difference[parameter_index] = (
            _weighted_branch_loss(circuit, plus, target, plus_tree, options, truncation)
            - _weighted_branch_loss(circuit, minus, target, minus_tree, options, truncation)
        ) / (2.0 * step)

    np.testing.assert_allclose(gradient, finite_difference, atol=5e-8, rtol=2e-7)


def test_nonunitary_state_dependent_counterexample_quantifies_pathwise_bias() -> None:
    """A synthetic state-dependent jump law has nonzero score and pathwise bias."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.37], dtype=np.float64)
    target = np.array([0.2, 0.3, 0.4, np.sqrt(0.71)], dtype=np.complex128)
    gamma = 0.8
    dt = 0.7
    drift = np.diag([np.exp(-0.5 * gamma * dt), 1.0]).astype(np.complex128)
    fixed_maps = [[KrotovNoiseMap(operators=((drift, (0,)),), normalized=True)]]
    options = KrotovTJMOptions(num_trajectories=1)
    truncation = KrotovTruncation()

    pathwise, _loss, _fidelity, _trajectories = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        None,
        options,
        MPS(2),
        truncation,
        fixed_noise_maps=fixed_maps,
    )
    step = 1e-6
    plus = theta + step
    minus = theta - step
    finite_difference = (
        noisy_state_preparation_metrics(
            circuit,
            plus,
            target,
            None,
            options,
            truncation=truncation,
            fixed_noise_maps=fixed_maps,
        )[0]
        - noisy_state_preparation_metrics(
            circuit,
            minus,
            target,
            None,
            options,
            truncation=truncation,
            fixed_noise_maps=fixed_maps,
        )[0]
    ) / (2.0 * step)

    attenuation_probability = math.exp(-gamma * dt)

    def jump_probability(angle: float) -> float:
        """Return the synthetic state-dependent jump probability."""
        return 1.0 - (attenuation_probability * math.cos(angle / 2.0) ** 2 + math.sin(angle / 2.0) ** 2)

    probability_derivative = (jump_probability(theta[0] + step) - jump_probability(theta[0] - step)) / (2.0 * step)
    pathwise_bias = float(pathwise[0] - finite_difference)

    assert probability_derivative == pytest.approx(-0.0775287098, abs=2e-9)
    assert pathwise_bias == pytest.approx(0.0124215011, abs=2e-9)
    assert abs(probability_derivative) > 0.05
    assert abs(pathwise_bias) > 0.01


def test_state_dependent_two_branch_derivative_includes_probability_score_term() -> None:
    """The complete expected derivative includes the nonzero branch-probability score."""
    theta = 0.73
    target_angle = 1.2
    gamma = 0.8
    dt = 0.7
    attenuation = math.exp(-gamma * dt)
    attenuation_amplitude = math.sqrt(attenuation)
    target_zero = math.cos(target_angle / 2.0)
    target_one = math.sin(target_angle / 2.0)

    def expected_loss(angle: float) -> float:
        """Return the exact two-branch expected loss at one circuit angle."""
        cosine = math.cos(angle / 2.0)
        sine = math.sin(angle / 2.0)
        no_jump_probability = cosine**2 + attenuation * sine**2
        jump_probability = (1.0 - attenuation) * sine**2
        no_jump_overlap = target_zero * cosine + target_one * attenuation_amplitude * sine
        no_jump_loss = 1.0 - no_jump_overlap**2 / no_jump_probability
        jump_loss = target_one**2
        return no_jump_probability * no_jump_loss + jump_probability * jump_loss

    cosine = math.cos(theta / 2.0)
    sine = math.sin(theta / 2.0)
    no_jump_probability = cosine**2 + attenuation * sine**2
    jump_probability = (1.0 - attenuation) * sine**2
    no_jump_probability_derivative = (attenuation - 1.0) * cosine * sine
    jump_probability_derivative = -no_jump_probability_derivative

    no_jump_overlap = target_zero * cosine + target_one * attenuation_amplitude * sine
    no_jump_overlap_derivative = 0.5 * (-target_zero * sine + target_one * attenuation_amplitude * cosine)
    no_jump_fidelity = no_jump_overlap**2 / no_jump_probability
    no_jump_fidelity_derivative = (
        2.0 * no_jump_overlap * no_jump_overlap_derivative / no_jump_probability
        - no_jump_overlap**2 * no_jump_probability_derivative / no_jump_probability**2
    )
    no_jump_loss = 1.0 - no_jump_fidelity
    no_jump_loss_derivative = -no_jump_fidelity_derivative
    jump_loss = target_one**2

    frozen_branch_pathwise_term = no_jump_probability * no_jump_loss_derivative
    no_jump_score = no_jump_probability_derivative / no_jump_probability
    jump_score = jump_probability_derivative / jump_probability
    probability_score_term = (
        no_jump_probability * no_jump_loss * no_jump_score + jump_probability * jump_loss * jump_score
    )
    complete_expected_derivative = frozen_branch_pathwise_term + probability_score_term

    step = 1e-5
    finite_difference = (expected_loss(theta + step) - expected_loss(theta - step)) / (2.0 * step)

    assert no_jump_probability + jump_probability == pytest.approx(1.0, abs=1e-15)
    assert probability_score_term == pytest.approx(
        no_jump_probability_derivative * no_jump_loss + jump_probability_derivative * jump_loss,
        abs=1e-15,
    )
    assert abs(probability_score_term) > 0.03
    assert complete_expected_derivative == pytest.approx(finite_difference, abs=5e-10)
    assert abs(complete_expected_derivative - frozen_branch_pathwise_term) > 0.03


def test_cross_trajectory_signal_matches_explicit_dense_r_squared_sum() -> None:
    """Cross mode equals the explicit dense double sum with the required R-squared scale."""
    circuit = ParameterizedCircuit(2, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.63], dtype=np.float64)
    target = _random_target(2, 77)
    fixed_maps = [
        [KrotovNoiseMap(is_identity=True)],
        [KrotovNoiseMap(operators=((_X, (0,)),), is_identity=False)],
        [KrotovNoiseMap(operators=((_Z, (0,)),), is_identity=False)],
    ]
    options = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross")
    truncation = KrotovTruncation()

    contribution, _loss, _fidelity, trajectories = noisy_state_preparation_cross_contribution(
        circuit,
        theta,
        target,
        None,
        options,
        MPS(2),
        truncation,
        fixed_noise_maps=fixed_maps,
    )

    derivative, sites = circuit.derivative_operator(circuit.gates[0])
    assert sites == (0,)
    dense_derivative = np.kron(_I2, derivative)
    dense_noise_operators = (np.eye(4, dtype=np.complex128), np.kron(_I2, _X), np.kron(_I2, _Z))
    forward_gate_outputs = [trajectory.gate_outputs[0].to_vec() for trajectory in trajectories]
    backward_states = [operator.conj().T @ target for operator in dense_noise_operators]
    raw_signal = 0.0
    for forward_state in forward_gate_outputs:
        for backward_state in backward_states:
            raw_signal -= (
                circuit.gates[0].angle_scale
                * 2.0
                * float(
                    np.real(
                        np.vdot(backward_state, dense_derivative @ forward_state)
                        * np.vdot(forward_state, backward_state)
                    )
                )
            )
    expected = raw_signal / options.num_trajectories**2

    assert abs(raw_signal) > 1e-3
    assert contribution[0] == pytest.approx(expected, abs=1e-12)


def test_deterministic_resampling_converges_to_exact_branch_gradient() -> None:
    """Deterministic resamples reproduce and their RMSE decreases at Monte Carlo scale."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    theta = np.array([0.37], dtype=np.float64)
    target = np.array([np.cos(0.9 / 2.0), np.sin(0.9 / 2.0)], dtype=np.complex128)
    provider = create_scaled_standard_noise_provider("dephasing_1s_1q", 1000.0)
    oracle_options = KrotovTJMOptions(num_trajectories=1, random_seed=314, dt=1.0)
    truncation = KrotovTruncation()
    tree = enumerate_fixed_rate_pauli_branches(circuit, theta, provider, oracle_options)
    exact_gradient = _weighted_branch_gradient(circuit, theta, target, tree, oracle_options, truncation)[0]

    repeated_options = replace(oracle_options, num_trajectories=16)
    first = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        None,
        repeated_options,
        MPS(1),
        truncation,
        iteration=4,
        noise_provider=provider,
    )
    repeated = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        None,
        repeated_options,
        MPS(1),
        truncation,
        iteration=4,
        noise_provider=provider,
    )
    np.testing.assert_array_equal(first[0], repeated[0])
    assert [trajectory.noise_maps[0].jump_process_index for trajectory in first[3]] == [
        trajectory.noise_maps[0].jump_process_index for trajectory in repeated[3]
    ]

    sample_sizes = (16, 256)
    root_mean_square_errors = []
    for sample_size in sample_sizes:
        sample_options = replace(oracle_options, num_trajectories=sample_size)
        sampled_gradients = []
        for iteration in range(12):
            contribution, _loss, _fidelity, _trajectories = noisy_state_preparation_contribution(
                circuit,
                theta,
                target,
                None,
                sample_options,
                MPS(1),
                truncation,
                iteration=iteration,
                noise_provider=provider,
            )
            sampled_gradients.append(contribution[0])
        errors = np.asarray(sampled_gradients, dtype=np.float64) - exact_gradient
        root_mean_square_errors.append(float(np.sqrt(np.mean(errors**2))))

    small_error, large_error = root_mean_square_errors
    assert large_error < 0.5 * small_error
    scaled_error_ratio = large_error * math.sqrt(sample_sizes[1]) / (small_error * math.sqrt(sample_sizes[0]))
    assert 0.5 < scaled_error_ratio < 2.0


def test_fixed_crn_training_has_a_positive_fresh_validation_gap() -> None:
    """A tiny fixed CRN ensemble can overfit and must be checked on fresh trajectories."""
    circuit = ParameterizedCircuit(1, [ParameterizedGate("ry", (0,), param_index=0)])
    target = np.array([np.cos(0.9 / 2.0), np.sin(0.9 / 2.0)], dtype=np.complex128)
    provider = create_scaled_standard_noise_provider("dephasing_1s_1q", 1000.0)
    training_options = KrotovTJMOptions(num_trajectories=4, random_seed=32, dt=1.0)
    truncation = KrotovTruncation()
    theta = np.array([0.15], dtype=np.float64)
    sampled = noisy_state_preparation_contribution(
        circuit,
        theta,
        target,
        None,
        training_options,
        MPS(1),
        truncation,
        noise_provider=provider,
    )
    fixed_maps = [list(trajectory.noise_maps) for trajectory in sampled[3]]
    assert all(noise_map[0].jump_process_index == 0 for noise_map in fixed_maps)

    for _iteration in range(50):
        contribution, _loss, _fidelity, _trajectories = noisy_state_preparation_contribution(
            circuit,
            theta,
            target,
            None,
            training_options,
            MPS(1),
            truncation,
            fixed_noise_maps=fixed_maps,
        )
        theta -= 0.4 * contribution

    _fixed_loss, fixed_fidelity, _fixed_fidelities = noisy_state_preparation_metrics(
        circuit,
        theta,
        target,
        None,
        training_options,
        truncation=truncation,
        fixed_noise_maps=fixed_maps,
    )
    validation_options = KrotovTJMOptions(num_trajectories=512, random_seed=929, dt=1.0)
    validation = noisy_state_preparation_metrics(
        circuit,
        theta,
        target,
        None,
        validation_options,
        truncation=truncation,
        iteration=17,
        noise_provider=provider,
    )
    repeated_validation = noisy_state_preparation_metrics(
        circuit,
        theta,
        target,
        None,
        validation_options,
        truncation=truncation,
        iteration=17,
        noise_provider=provider,
    )
    fresh_fidelity = validation[1]
    signed_gap = fixed_fidelity - fresh_fidelity

    assert validation == repeated_validation
    assert fixed_fidelity > 0.9999
    assert 0.6 < fresh_fidelity < 0.75
    assert signed_gap > 0.25


def test_selected_bond_truncation_has_measurable_adjoint_bias() -> None:
    """A selected bond cap records the expected backward-truncation gradient bias."""
    circuit = ParameterizedCircuit(
        4,
        [
            ParameterizedGate("ry", (0,), param_index=0),
            ParameterizedGate("ry", (1,), param_index=1),
            ParameterizedGate("rxx", (0, 1), param_index=2),
            ParameterizedGate("ry", (2,), param_index=3),
            ParameterizedGate("rxx", (1, 2), param_index=4),
            ParameterizedGate("ry", (3,), param_index=5),
            ParameterizedGate("rxx", (2, 3), param_index=6),
        ],
        num_params=7,
    )
    theta = np.array([0.31, -0.27, 0.82, 0.43, -0.71, 0.19, 0.91], dtype=np.float64)
    target = _random_target(4, 444)
    provider = create_scaled_standard_noise_provider("dephasing_1s_all", 500.0)
    options = KrotovTJMOptions(num_trajectories=1, dt=0.8, noisy_gate_indices=(2,))
    tree = enumerate_fixed_rate_pauli_branches(circuit, theta, provider, options)
    exact_truncation = KrotovTruncation()
    selected_truncation = KrotovTruncation(max_bond_dim=2, min_bond_dim=1)

    exact_gradient = _weighted_branch_gradient(circuit, theta, target, tree, options, exact_truncation)
    exact_finite_difference = _central_difference_with_tree(
        circuit,
        theta,
        target,
        tree,
        options,
        exact_truncation,
    )
    truncated_gradient = _weighted_branch_gradient(circuit, theta, target, tree, options, selected_truncation)
    truncated_finite_difference = _central_difference_with_tree(
        circuit,
        theta,
        target,
        tree,
        options,
        selected_truncation,
    )

    np.testing.assert_allclose(exact_gradient, exact_finite_difference, atol=5e-8, rtol=2e-7)
    np.testing.assert_allclose(truncated_finite_difference, exact_finite_difference, atol=5e-8, rtol=2e-7)
    bias_norm = float(np.linalg.norm(truncated_gradient - truncated_finite_difference))
    assert 0.015 < bias_norm < 0.025
