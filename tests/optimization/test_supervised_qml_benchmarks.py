# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: PLC2701

"""Tests for supervised QML Krotov benchmark experiments."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.krotov_supervised_qml_benchmarks import (
    HybridSettings,
    PerezSalinasCrownModel,
    build_parity_circuit,
    generate_crown_dataset,
    generate_parity_split,
    parity_basis_state,
    parity_initial_theta,
    run_parity_seed,
)
from experiments.krotov_supervised_qml_noise_sweep import (
    _crown_gate_sites,
    _crown_noisy_online_sample_update,
    _crown_noisy_sample_contribution,
    _parity_noisy_online_sample_update,
    build_pauli_noise_model,
)
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.optimization import KrotovReadout, KrotovTJMOptions, KrotovTruncation
from mqt.yaqs.optimization.krotov import _online_sample_update


def test_parity_split_and_ansatz_match_requested_contract() -> None:
    """The parity benchmark uses all unique bitstrings and 24 gate parameters plus bias."""
    x_train, x_test, y_train, y_test = generate_parity_split(train_size=10, test_size=6, seed=0)

    assert x_train.shape == (10, 4)
    assert x_test.shape == (6, 4)
    assert len({tuple(row) for row in np.vstack([x_train, x_test])}) == 16
    assert np.array_equal(y_train, np.where(np.sum(x_train, axis=1) % 2 == 1, 1.0, -1.0))
    assert np.array_equal(y_test, np.where(np.sum(x_test, axis=1) % 2 == 1, 1.0, -1.0))

    circuit = build_parity_circuit(n_layers=2)
    assert circuit.num_qubits == 4
    assert circuit.num_params == 2 * 4 * 3
    assert circuit.num_params + 1 == 25
    assert [(gate.name, gate.sites) for gate in circuit.gates[:3]] == [("rz", (0,)), ("ry", (0,)), ("rz", (0,))]
    assert [(gate.name, gate.sites) for gate in circuit.gates[12:16]] == [
        ("cx", (0, 1)),
        ("cx", (1, 2)),
        ("cx", (2, 3)),
        ("cx", (3, 0)),
    ]


def test_crown_dataset_and_model_match_requested_contract() -> None:
    """The crown benchmark uses stratified 70/30 split and 168 trainable parameters."""
    x_train, x_test, y_train, y_test = generate_crown_dataset(n_samples=60, test_fraction=0.3, seed=2)

    assert x_train.shape == (42, 2)
    assert x_test.shape == (18, 2)
    assert set(np.unique(y_train)) == {0, 1}
    assert set(np.unique(y_test)) == {0, 1}
    assert {tuple(np.round(row, 12)) for row in x_train}.isdisjoint({tuple(np.round(row, 12)) for row in x_test})

    model = PerezSalinasCrownModel(n_qubits=4, n_layers=8)
    params = model.init_params(seed=0)
    gates, states = model.gate_sequence_and_states(params, x_train[0])

    assert model.n_quantum_params == 4 * 8 * 5
    assert model.n_weight_params == 2 * 4
    assert model.n_params == 168
    assert len(gates) == 4 * 8 * 5 + 2 * (8 - 1)
    assert len(states) == len(gates) + 1


@pytest.mark.parametrize("param_index", [0, 7, 40, 47])
def test_crown_gate_local_gradient_matches_finite_difference(param_index: int) -> None:
    """The crown gate-local gradient matches central finite differences."""
    x_train, _x_test, y_train, _y_test = generate_crown_dataset(n_samples=24, test_fraction=0.25, seed=3)
    model = PerezSalinasCrownModel(n_qubits=4, n_layers=2)
    params = model.init_params(seed=5)
    gradient = model.loss_gradient(params, x_train[:6], y_train[:6])

    eps = 1e-6
    shifted_plus = params.copy()
    shifted_minus = params.copy()
    shifted_plus[param_index] += eps
    shifted_minus[param_index] -= eps
    numeric = (
        model.loss(shifted_plus, x_train[:6], y_train[:6])
        - model.loss(shifted_minus, x_train[:6], y_train[:6])
    ) / (2.0 * eps)

    assert gradient[param_index] == pytest.approx(numeric, abs=2e-6)


def test_parity_runner_records_train_and_test_accuracy() -> None:
    """A short parity run records train loss, train accuracy, and test accuracy."""
    result = run_parity_seed(
        0,
        HybridSettings(max_iterations=2, switch_iteration=1, online_step_size=0.1, batch_step_size=0.1),
    )

    assert result.trace["phase"] == ["init", "online", "batch"]
    assert len(result.trace["loss"]) == 3
    assert all(0.0 <= float(value) <= 1.0 for value in result.trace["train_acc"])
    assert all(0.0 <= float(value) <= 1.0 for value in result.trace["test_acc"])


def test_noisy_crown_fixed_trajectory_gradient_matches_finite_difference() -> None:
    """The noisy crown pathwise contribution matches fixed-map finite differences."""
    model = PerezSalinasCrownModel(n_qubits=4, n_layers=2)
    params = model.init_params(seed=7)
    x_value = np.array([0.2, -0.4], dtype=np.float64)
    label = 1
    gamma = 0.03
    gate_sites = _crown_gate_sites(model)
    contribution, _loss, _scores, trajectories = _crown_noisy_sample_contribution(
        model,
        params,
        x_value,
        label,
        gamma,
        3,
        1234,
        gate_sites,
    )
    fixed_maps = [trajectory.noise_maps for trajectory in trajectories]

    param_index = 9
    eps = 1e-6
    shifted_plus = params.copy()
    shifted_minus = params.copy()
    shifted_plus[param_index] += eps
    shifted_minus[param_index] -= eps
    _grad_plus, loss_plus, _scores_plus, _trajectories_plus = _crown_noisy_sample_contribution(
        model,
        shifted_plus,
        x_value,
        label,
        gamma,
        3,
        1234,
        gate_sites,
        fixed_maps,
    )
    _grad_minus, loss_minus, _scores_minus, _trajectories_minus = _crown_noisy_sample_contribution(
        model,
        shifted_minus,
        x_value,
        label,
        gamma,
        3,
        1234,
        gate_sites,
        fixed_maps,
    )
    numeric = (loss_plus - loss_minus) / (2.0 * eps)

    assert contribution[param_index] == pytest.approx(numeric, abs=2e-6)


def test_noisy_crown_online_update_reduces_to_noiseless_at_zero_noise() -> None:
    """The noisy benchmark driver must match the noiseless online update at gamma zero."""
    model = PerezSalinasCrownModel(n_qubits=4, n_layers=2)
    params = model.init_params(seed=11)
    x_value = np.array([0.35, -0.2], dtype=np.float64)
    label = 1
    step_size = 0.07

    noiseless_params, noiseless_contribution = model.online_sample_update(params, x_value, label, step_size)
    noisy_params, noisy_contribution = _crown_noisy_online_sample_update(
        model,
        params,
        x_value,
        label,
        0.0,
        3,
        _crown_gate_sites(model),
        222,
        step_size,
    )

    assert noisy_params == pytest.approx(noiseless_params, abs=1e-12)
    assert noisy_contribution == pytest.approx(noiseless_contribution, abs=1e-12)


def test_noisy_parity_online_update_reduces_to_noiseless_at_zero_noise() -> None:
    """The noisy parity driver must match the native online update at gamma zero."""
    circuit = build_parity_circuit(n_layers=1)
    theta = parity_initial_theta(seed=3, num_params=circuit.num_params)
    x_value = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    label = -1.0
    bias = 0.2
    step_size = 0.05
    readout = KrotovReadout(Observable("z", 0), loss="mse", use_bias=True)
    truncation = KrotovTruncation()

    noiseless_theta, noiseless_contribution = _online_sample_update(
        circuit,
        theta,
        x_value,
        label,
        readout,
        bias,
        step_size,
        parity_basis_state(x_value),
        truncation,
    )
    noisy_theta, noisy_contribution = _parity_noisy_online_sample_update(
        circuit,
        theta,
        bias,
        x_value,
        label,
        readout,
        build_pauli_noise_model(circuit.num_qubits, 0.0),
        KrotovTJMOptions(num_trajectories=3, random_seed=123, apply_noise_to="two-qubit"),
        truncation,
        0,
        step_size,
    )

    assert noisy_theta == pytest.approx(noiseless_theta, abs=1e-12)
    assert noisy_contribution == pytest.approx(noiseless_contribution, abs=1e-12)
