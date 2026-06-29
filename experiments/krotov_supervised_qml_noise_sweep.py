# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: DOC201, PLC0415, PLC2701, T201

"""Noise-strength sweeps for supervised Krotov parity and crown diagnostics."""

from __future__ import annotations

import copy
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.krotov_supervised_qml_benchmarks import (  # noqa: E402
    HybridSettings,
    PerezSalinasCrownModel,
    _sign_accuracy,
    build_parity_circuit,
    generate_crown_dataset,
    generate_parity_split,
    parity_basis_state,
    parity_initial_theta,
    run_crown_seed,
    run_parity_seed,
)
from mqt.yaqs.core.data_structures.noise_model import NoiseModel  # noqa: E402
from mqt.yaqs.core.data_structures.simulation_parameters import Observable  # noqa: E402
from mqt.yaqs.optimization import KrotovReadout, KrotovTJMOptions, KrotovTruncation, ParameterizedCircuit  # noqa: E402
from mqt.yaqs.optimization.krotov import (  # noqa: E402
    KrotovNoiseMap,
    _apply_noise_map,
    _apply_operator,
    _expectation,
    _forward_tjm_trajectories,
    _gate_contribution,
    _loss_and_costate_factor,
    _trajectory_chi_tildes,
    noisy_sample_contribution,
    noisy_sample_loss,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

NUM_STEPS = int(os.environ.get("YAQS_SUPERVISED_SWEEP_STEPS", "20"))
NUM_GAMMAS = int(os.environ.get("YAQS_SUPERVISED_SWEEP_POINTS", "10"))
GAMMA_MIN = float(os.environ.get("YAQS_SUPERVISED_SWEEP_GAMMA_MIN", "0.001"))
GAMMA_MAX = float(os.environ.get("YAQS_SUPERVISED_SWEEP_GAMMA_MAX", "0.1"))
SEEDS = tuple(int(seed) for seed in os.environ.get("YAQS_SUPERVISED_SWEEP_SEEDS", "0,1,2").split(","))

PARITY_TRAIN_TRAJECTORIES = int(os.environ.get("YAQS_PARITY_TRAIN_TRAJECTORIES", "16"))
PARITY_VALIDATION_TRAJECTORIES = int(os.environ.get("YAQS_PARITY_VALIDATION_TRAJECTORIES", "32"))
CROWN_TRAIN_TRAJECTORIES = int(os.environ.get("YAQS_CROWN_TRAIN_TRAJECTORIES", "4"))
CROWN_VALIDATION_TRAJECTORIES = int(os.environ.get("YAQS_CROWN_VALIDATION_TRAJECTORIES", "8"))
NUM_WORKERS = int(os.environ.get("YAQS_SUPERVISED_SWEEP_WORKERS", str(os.cpu_count() or 1)))

PARITY_SWITCH_ITERATION = int(os.environ.get("YAQS_PARITY_NOISY_SWITCH_ITERATION", "10"))
PARITY_ONLINE_STEP_SIZE = float(os.environ.get("YAQS_PARITY_NOISY_ONLINE_STEP_SIZE", "0.3"))
PARITY_BATCH_STEP_SIZE = float(os.environ.get("YAQS_PARITY_NOISY_BATCH_STEP_SIZE", "1.0"))
CROWN_SWITCH_ITERATION = int(os.environ.get("YAQS_CROWN_NOISY_SWITCH_ITERATION", "10"))
CROWN_ONLINE_STEP_SIZE = float(os.environ.get("YAQS_CROWN_NOISY_ONLINE_STEP_SIZE", "0.3"))
CROWN_BATCH_STEP_SIZE = float(os.environ.get("YAQS_CROWN_NOISY_BATCH_STEP_SIZE", "0.5"))
CROWN_SAMPLES = int(os.environ.get("YAQS_CROWN_SWEEP_SAMPLES", "600"))


@dataclass(frozen=True)
class SweepResult:
    """Container for one benchmark, seed, and gamma run."""

    detail_rows: list[dict[str, float | int | str]]
    summary_row: dict[str, float | int | str]


@dataclass(frozen=True)
class CrownTrajectory:
    """Dense fixed-noise trajectory storage for the crown readout."""

    states: list[NDArray[np.complex128]]
    gate_outputs: list[NDArray[np.complex128]]
    noise_maps: list[NDArray[np.complex128] | None]


def build_pauli_noise_model(num_qubits: int, gamma: float) -> NoiseModel:
    """Build local Pauli X/Y/Z noise on every qubit."""
    return NoiseModel([
        {"name": name, "sites": [site], "strength": gamma}
        for site in range(num_qubits)
        for name in ("pauli_x", "pauli_y", "pauli_z")
    ])


def tjm_options(num_trajectories: int, seed: int) -> KrotovTJMOptions:
    """Build YAQS-style circuit-TJM options matching the state-preparation sweep."""
    return KrotovTJMOptions(
        num_trajectories=num_trajectories,
        random_seed=seed,
        apply_noise_to="two-qubit",
    )


def _parity_fixed_maps(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    bias: float,
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    readout: KrotovReadout,
    noise_model: NoiseModel,
    options: KrotovTJMOptions,
    truncation: KrotovTruncation,
) -> list[list[list[KrotovNoiseMap]]]:
    fixed_maps = []
    for sample_index, (x_value, label) in enumerate(zip(inputs, labels, strict=True)):
        _contribution, _loss, _zbar, _readouts, trajectories = noisy_sample_contribution(
            circuit,
            theta,
            x_value,
            float(label),
            readout,
            bias,
            noise_model,
            options,
            parity_basis_state(x_value),
            truncation,
            iteration=50_000 + sample_index,
        )
        fixed_maps.append([trajectory.noise_maps for trajectory in trajectories])
    return fixed_maps


def _parity_noisy_metrics(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    bias: float,
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    readout: KrotovReadout,
    noise_model: NoiseModel,
    options: KrotovTJMOptions,
    truncation: KrotovTruncation,
    fixed_maps: list[list[list[KrotovNoiseMap]]] | None,
) -> tuple[float, float]:
    losses = []
    scores = []
    for sample_index, (x_value, label) in enumerate(zip(inputs, labels, strict=True)):
        maps = None if fixed_maps is None else fixed_maps[sample_index]
        loss, zbar, _readouts = noisy_sample_loss(
            circuit,
            theta,
            x_value,
            float(label),
            readout,
            bias,
            noise_model,
            options,
            initial_state=parity_basis_state(x_value),
            truncation=truncation,
            iteration=70_000 + sample_index,
            fixed_noise_maps=maps,
        )
        losses.append(loss)
        scores.append(zbar + bias)
    return float(np.mean(losses)), _sign_accuracy(np.asarray(scores), labels)


def _parity_noisy_gradient(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    bias: float,
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    readout: KrotovReadout,
    noise_model: NoiseModel,
    options: KrotovTJMOptions,
    truncation: KrotovTruncation,
    iteration: int,
) -> tuple[NDArray[np.float64], float]:
    contribution = np.zeros(circuit.num_params, dtype=np.float64)
    bias_gradient = 0.0
    for sample_index, (x_value, label) in enumerate(zip(inputs, labels, strict=True)):
        sample_contribution, _loss, zbar, _readouts, _trajectories = noisy_sample_contribution(
            circuit,
            theta,
            x_value,
            float(label),
            readout,
            bias,
            noise_model,
            options,
            parity_basis_state(x_value),
            truncation,
            iteration=iteration * 10_000 + sample_index,
        )
        contribution += sample_contribution
        bias_gradient += 2.0 * (zbar + bias - float(label))
    return contribution / len(inputs), bias_gradient / len(inputs)


def _parity_noisy_online_sample_update(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    bias: float,
    x_value: NDArray[np.float64],
    label: float,
    readout: KrotovReadout,
    noise_model: NoiseModel,
    options: KrotovTJMOptions,
    truncation: KrotovTruncation,
    iteration: int,
    step_size: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one supervised stale-adjoint online update through fixed TJM maps."""
    trajectories = _forward_tjm_trajectories(
        circuit,
        theta,
        x_value,
        parity_basis_state(x_value),
        truncation,
        noise_model,
        options,
        iteration,
    )
    trajectory_readouts = [_expectation(trajectory.states[-1], readout.observable) for trajectory in trajectories]
    zbar = float(np.mean(trajectory_readouts))
    _loss_value, factor = _loss_and_costate_factor(zbar, bias, float(label), readout.loss)

    scale = 1.0 / options.num_trajectories
    stale_costates = []
    for trajectory in trajectories:
        chi = copy.deepcopy(trajectory.states[-1])
        chi.apply_local(readout.observable)
        chi.tensors[0] *= factor * scale
        stale_costates.append(_trajectory_chi_tildes(circuit, theta, x_value, trajectory, chi, truncation))

    new_theta = theta.copy()
    current_states = [parity_basis_state(x_value) for _trajectory in trajectories]
    contribution = np.zeros(circuit.num_params, dtype=np.float64)
    for gate_index, gate in enumerate(circuit.gates):
        if gate.param_index is not None:
            signal = 0.0
            for trajectory_index, current in enumerate(current_states):
                gate_output = copy.deepcopy(current)
                matrix, sites = circuit.gate_matrix(gate, new_theta, x_value)
                _apply_operator(gate_output, matrix, sites, truncation)
                signal += _gate_contribution(
                    circuit,
                    gate,
                    stale_costates[trajectory_index][gate_index],
                    gate_output,
                    truncation,
                )
            contribution[gate.param_index] += signal
            new_theta[gate.param_index] -= step_size * signal

        for trajectory_index, current in enumerate(current_states):
            matrix, sites = circuit.gate_matrix(gate, new_theta, x_value)
            _apply_operator(current, matrix, sites, truncation)
            _apply_noise_map(current, trajectories[trajectory_index].noise_maps[gate_index], truncation)

    return new_theta, contribution


def _parity_noisy_bias_gradient(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    bias: float,
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    readout: KrotovReadout,
    noise_model: NoiseModel,
    options: KrotovTJMOptions,
    truncation: KrotovTruncation,
    iteration: int,
) -> float:
    bias_gradient = 0.0
    for sample_index, (x_value, label) in enumerate(zip(inputs, labels, strict=True)):
        _loss, zbar, _readouts = noisy_sample_loss(
            circuit,
            theta,
            x_value,
            float(label),
            readout,
            bias,
            noise_model,
            options,
            initial_state=parity_basis_state(x_value),
            truncation=truncation,
            iteration=iteration * 10_000 + sample_index,
        )
        bias_gradient += 2.0 * (zbar + bias - float(label))
    return bias_gradient / len(inputs)


def _parity_noisy_online_epoch(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    bias: float,
    inputs: NDArray[np.float64],
    labels: NDArray[np.float64],
    readout: KrotovReadout,
    noise_model: NoiseModel,
    options: KrotovTJMOptions,
    truncation: KrotovTruncation,
    epoch_seed: int,
    step_size: float,
) -> tuple[NDArray[np.float64], float, float]:
    rng = np.random.default_rng(epoch_seed)
    contributions = []
    for order_index, sample_index in enumerate(rng.permutation(len(inputs))):
        theta, contribution = _parity_noisy_online_sample_update(
            circuit,
            theta,
            bias,
            inputs[sample_index],
            float(labels[sample_index]),
            readout,
            noise_model,
            options,
            truncation,
            epoch_seed * 10_000 + order_index,
            step_size,
        )
        contributions.append(contribution)

    bias_gradient = 0.0
    if readout.use_bias:
        bias_gradient = _parity_noisy_bias_gradient(
            circuit,
            theta,
            bias,
            inputs,
            labels,
            readout,
            noise_model,
            options,
            truncation,
            epoch_seed,
        )
        bias -= step_size * bias_gradient

    mean_contribution = np.mean(np.asarray(contributions), axis=0)
    gradient_norm = float(np.sqrt(np.linalg.norm(mean_contribution) ** 2 + bias_gradient**2))
    return theta, bias, gradient_norm


def run_parity_noisy_strength(seed: int, gamma: float, noiseless_test_acc: float) -> SweepResult:
    """Run the noisy parity optimizer for one seed and gamma."""
    x_train, x_test, y_train, y_test = generate_parity_split(seed=seed)
    circuit = build_parity_circuit(n_layers=2)
    readout = KrotovReadout(Observable("z", 0), loss="mse", use_bias=True)
    truncation = KrotovTruncation()
    theta = parity_initial_theta(seed, circuit.num_params)
    bias = 0.0
    noise_model = build_pauli_noise_model(circuit.num_qubits, gamma)
    train_options = tjm_options(PARITY_TRAIN_TRAJECTORIES, 111 + seed)
    validation_options = tjm_options(PARITY_VALIDATION_TRAJECTORIES, 777 + seed)
    fixed_test_maps = _parity_fixed_maps(
        circuit,
        theta,
        bias,
        x_test,
        y_test,
        readout,
        noise_model,
        validation_options,
        truncation,
    )

    detail_rows: list[dict[str, float | int | str]] = []
    best_test_acc = 0.0
    best_iteration = 0
    final_test_acc = 0.0
    final_loss = 0.0
    for iteration in range(NUM_STEPS + 1):
        loss, test_acc = _parity_noisy_metrics(
            circuit,
            theta,
            bias,
            x_test,
            y_test,
            readout,
            noise_model,
            validation_options,
            truncation,
            fixed_test_maps,
        )
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_iteration = iteration
        gradient_norm = 0.0
        detail_rows.append({
            "benchmark": "4-bit parity",
            "seed": seed,
            "gamma": gamma,
            "iteration": iteration,
            "test_loss": loss,
            "test_accuracy": test_acc,
            "gradient_norm": gradient_norm,
            "train_trajectories": PARITY_TRAIN_TRAJECTORIES,
            "validation_trajectories": PARITY_VALIDATION_TRAJECTORIES,
        })
        final_loss = loss
        final_test_acc = test_acc
        if iteration < NUM_STEPS:
            phase_iteration = iteration + 1
            if phase_iteration <= PARITY_SWITCH_ITERATION:
                theta, bias, gradient_norm = _parity_noisy_online_epoch(
                    circuit,
                    theta,
                    bias,
                    x_train,
                    y_train,
                    readout,
                    noise_model,
                    train_options,
                    truncation,
                    seed + phase_iteration,
                    PARITY_ONLINE_STEP_SIZE,
                )
            else:
                contribution, bias_gradient = _parity_noisy_gradient(
                    circuit,
                    theta,
                    bias,
                    x_train,
                    y_train,
                    readout,
                    noise_model,
                    train_options,
                    truncation,
                    phase_iteration - PARITY_SWITCH_ITERATION,
                )
                gradient_norm = float(np.sqrt(np.linalg.norm(contribution) ** 2 + bias_gradient**2))
                theta -= PARITY_BATCH_STEP_SIZE * contribution
                bias -= PARITY_BATCH_STEP_SIZE * bias_gradient
            detail_rows[-1]["gradient_norm"] = gradient_norm

    return SweepResult(
        detail_rows=detail_rows,
        summary_row={
            "benchmark": "4-bit parity",
            "seed": seed,
            "gamma": gamma,
            "initial_test_accuracy": detail_rows[0]["test_accuracy"],
            "final_test_accuracy": final_test_acc,
            "best_test_accuracy": best_test_acc,
            "best_iteration": best_iteration,
            "final_test_loss": final_loss,
            "noiseless_test_accuracy": noiseless_test_acc,
            "train_trajectories": PARITY_TRAIN_TRAJECTORIES,
            "validation_trajectories": PARITY_VALIDATION_TRAJECTORIES,
        },
    )


def _crown_gate_sites(model: PerezSalinasCrownModel) -> list[tuple[int, int] | None]:
    sites: list[tuple[int, int] | None] = []
    for layer in range(model.n_layers):
        sites.extend([None] * (model.n_qubits * model.params_per_block))
        if layer < model.n_layers - 1:
            if layer % 2 == 0:
                sites.extend([(0, 1), (2, 3)])
            else:
                sites.extend([(1, 2), (0, 3)])
    return sites


def _crown_noise_processes(
    model: PerezSalinasCrownModel,
    sites: tuple[int, int],
) -> list[NDArray[np.complex128]]:
    operators = []
    for site in sites:
        operators.extend((
            model.pauli_y_ops[site] @ model.pauli_z_ops[site] * 1j,
            model.pauli_y_ops[site],
            model.pauli_z_ops[site],
        ))
    return operators


def _sample_crown_noise_map(
    model: PerezSalinasCrownModel,
    sites: tuple[int, int] | None,
    gamma: float,
    rng: np.random.Generator,
) -> NDArray[np.complex128] | None:
    if sites is None or gamma <= 0.0:
        return None
    operators = _crown_noise_processes(model, sites)
    jump_probability = 1.0 - float(np.exp(-len(operators) * gamma))
    if rng.random() >= jump_probability:
        return None
    return operators[int(rng.integers(0, len(operators)))]


def _crown_forward_trajectory(
    model: PerezSalinasCrownModel,
    gates: list[tuple[NDArray[np.complex128], int | None]],
    gamma: float,
    rng: np.random.Generator,
    gate_sites: list[tuple[int, int] | None],
    fixed_maps: list[NDArray[np.complex128] | None] | None = None,
) -> CrownTrajectory:
    state = np.zeros(model.dim, dtype=np.complex128)
    state[0] = 1.0
    states = [state.copy()]
    gate_outputs = []
    noise_maps = []
    for gate_index, ((gate, _param_index), sites) in enumerate(zip(gates, gate_sites, strict=True)):
        gate_output = gate @ state
        gate_outputs.append(gate_output.copy())
        noise_map = (
            fixed_maps[gate_index]
            if fixed_maps is not None
            else _sample_crown_noise_map(model, sites, gamma, rng)
        )
        state = gate_output if noise_map is None else noise_map @ gate_output
        noise_maps.append(noise_map)
        states.append(state.copy())
    return CrownTrajectory(states=states, gate_outputs=gate_outputs, noise_maps=noise_maps)


def _crown_noisy_sample_contribution(
    model: PerezSalinasCrownModel,
    params: NDArray[np.float64],
    x_value: NDArray[np.float64],
    label: int,
    gamma: float,
    num_trajectories: int,
    seed: int,
    gate_sites: list[tuple[int, int] | None],
    fixed_maps: list[list[NDArray[np.complex128] | None]] | None = None,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64], list[CrownTrajectory]]:
    gates = model.gate_sequence(params, x_value)
    trajectories = []
    for trajectory_index in range(num_trajectories):
        rng = np.random.default_rng(seed + 1_000_003 * trajectory_index)
        maps = None if fixed_maps is None else fixed_maps[trajectory_index]
        trajectories.append(_crown_forward_trajectory(model, gates, gamma, rng, gate_sites, maps))

    fidelities = np.asarray([model.sample_fidelities(trajectory.states[-1]) for trajectory in trajectories])
    mean_fidelities = np.mean(fidelities, axis=0)
    scores = np.sum(model.weights(params) * mean_fidelities, axis=1)
    residual = scores - model.target_vector(label)
    loss = 0.5 * float(residual @ residual)

    contribution = np.zeros_like(params)
    contribution[model.weight_slice()] = (residual[:, None] * mean_fidelities).reshape(-1)
    terminal_operator = np.zeros((model.dim, model.dim), dtype=np.complex128)
    weights = model.weights(params)
    for class_index in range(model.n_classes):
        for qubit in range(model.n_qubits):
            terminal_operator += (
                residual[class_index] * weights[class_index, qubit] * model.projector_ops[class_index][qubit]
            )

    for trajectory in trajectories:
        costate = terminal_operator @ trajectory.states[-1] / num_trajectories
        for gate_index in range(len(gates) - 1, -1, -1):
            gate, param_index = gates[gate_index]
            noise_map = trajectory.noise_maps[gate_index]
            chi_tilde = costate if noise_map is None else noise_map.conj().T @ costate
            if param_index is not None:
                derivative_state = (
                    model.gate_derivative_generator(param_index, x_value) @ trajectory.gate_outputs[gate_index]
                )
                contribution[param_index] += 2.0 * float(np.real(chi_tilde.conj() @ derivative_state))
            costate = gate.conj().T @ chi_tilde
    return contribution, loss, scores, trajectories


def _crown_fixed_maps(
    model: PerezSalinasCrownModel,
    params: NDArray[np.float64],
    inputs: NDArray[np.float64],
    labels: NDArray[np.int_],
    gamma: float,
    num_trajectories: int,
    gate_sites: list[tuple[int, int] | None],
    seed: int,
) -> list[list[list[NDArray[np.complex128] | None]]]:
    fixed_maps = []
    for sample_index, (x_value, label) in enumerate(zip(inputs, labels, strict=True)):
        _contribution, _loss, _scores, trajectories = _crown_noisy_sample_contribution(
            model,
            params,
            x_value,
            int(label),
            gamma,
            num_trajectories,
            seed + sample_index,
            gate_sites,
        )
        fixed_maps.append([trajectory.noise_maps for trajectory in trajectories])
    return fixed_maps


def _crown_noisy_metrics(
    model: PerezSalinasCrownModel,
    params: NDArray[np.float64],
    inputs: NDArray[np.float64],
    labels: NDArray[np.int_],
    gamma: float,
    num_trajectories: int,
    gate_sites: list[tuple[int, int] | None],
    fixed_maps: list[list[list[NDArray[np.complex128] | None]]] | None,
    seed: int,
) -> tuple[float, float]:
    losses = []
    predictions = []
    for sample_index, (x_value, label) in enumerate(zip(inputs, labels, strict=True)):
        maps = None if fixed_maps is None else fixed_maps[sample_index]
        _contribution, loss, scores, _trajectories = _crown_noisy_sample_contribution(
            model,
            params,
            x_value,
            int(label),
            gamma,
            num_trajectories,
            seed + sample_index,
            gate_sites,
            maps,
        )
        losses.append(loss)
        predictions.append(int(np.argmax(scores)))
    return float(np.mean(losses)), float(np.mean(np.asarray(predictions, dtype=int) == labels))


def _crown_noisy_gradient(
    model: PerezSalinasCrownModel,
    params: NDArray[np.float64],
    inputs: NDArray[np.float64],
    labels: NDArray[np.int_],
    gamma: float,
    num_trajectories: int,
    gate_sites: list[tuple[int, int] | None],
    iteration: int,
    seed: int,
) -> NDArray[np.float64]:
    gradient = np.zeros_like(params)
    for sample_index, (x_value, label) in enumerate(zip(inputs, labels, strict=True)):
        contribution, _loss, _scores, _trajectories = _crown_noisy_sample_contribution(
            model,
            params,
            x_value,
            int(label),
            gamma,
            num_trajectories,
            seed + iteration * 10_000 + sample_index,
            gate_sites,
        )
        gradient += contribution
    return gradient / len(inputs)


def _crown_noisy_online_sample_update(
    model: PerezSalinasCrownModel,
    params: NDArray[np.float64],
    x_value: NDArray[np.float64],
    label: int,
    gamma: float,
    num_trajectories: int,
    gate_sites: list[tuple[int, int] | None],
    seed: int,
    step_size: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply one stale-adjoint online update through fixed noisy trajectories."""
    gates = model.gate_sequence(params, x_value)
    trajectories = []
    for trajectory_index in range(num_trajectories):
        rng = np.random.default_rng(seed + 1_000_003 * trajectory_index)
        trajectories.append(_crown_forward_trajectory(model, gates, gamma, rng, gate_sites))

    fidelities = np.asarray([model.sample_fidelities(trajectory.states[-1]) for trajectory in trajectories])
    mean_fidelities = np.mean(fidelities, axis=0)
    scores = np.sum(model.weights(params) * mean_fidelities, axis=1)
    residual = scores - model.target_vector(label)

    terminal_operator = np.zeros((model.dim, model.dim), dtype=np.complex128)
    weights = model.weights(params)
    for class_index in range(model.n_classes):
        for qubit in range(model.n_qubits):
            terminal_operator += (
                residual[class_index] * weights[class_index, qubit] * model.projector_ops[class_index][qubit]
            )

    stale_chi_tildes: list[list[NDArray[np.complex128]]] = []
    for trajectory in trajectories:
        costate = terminal_operator @ trajectory.states[-1] / num_trajectories
        trajectory_chi_tildes = [np.empty(0, dtype=np.complex128)] * len(gates)
        for gate_index in range(len(gates) - 1, -1, -1):
            noise_map = trajectory.noise_maps[gate_index]
            chi_tilde = costate if noise_map is None else noise_map.conj().T @ costate
            trajectory_chi_tildes[gate_index] = chi_tilde
            costate = gates[gate_index][0].conj().T @ chi_tilde
        stale_chi_tildes.append(trajectory_chi_tildes)

    new_params = params.copy()
    current_states = [np.eye(1, model.dim, 0, dtype=np.complex128).reshape(-1) for _ in trajectories]
    contribution = np.zeros_like(params)
    for gate_index, (gate, param_index) in enumerate(gates):
        active_gate = gate
        gate_outputs = [active_gate @ current_state for current_state in current_states]
        if param_index is not None:
            generator = model.gate_derivative_generator(param_index, x_value)
            gradient = 0.0
            for trajectory_index, gate_output in enumerate(gate_outputs):
                derivative_state = generator @ gate_output
                gradient += 2.0 * float(
                    np.real(stale_chi_tildes[trajectory_index][gate_index].conj() @ derivative_state)
                )
            contribution[param_index] = gradient
            new_params[param_index] -= step_size * gradient
            active_gate = model.rebuild_param_gate(param_index, new_params, x_value)
            gate_outputs = [active_gate @ current_state for current_state in current_states]

        next_states = []
        for trajectory_index, gate_output in enumerate(gate_outputs):
            noise_map = trajectories[trajectory_index].noise_maps[gate_index]
            next_states.append(gate_output if noise_map is None else noise_map @ gate_output)
        current_states = next_states
    return new_params, contribution


def _crown_noisy_nongate_gradient(
    model: PerezSalinasCrownModel,
    params: NDArray[np.float64],
    inputs: NDArray[np.float64],
    labels: NDArray[np.int_],
    gamma: float,
    num_trajectories: int,
    gate_sites: list[tuple[int, int] | None],
    seed: int,
) -> NDArray[np.float64]:
    gradient = np.zeros_like(params)
    for sample_index, (x_value, label) in enumerate(zip(inputs, labels, strict=True)):
        contribution, _loss, _scores, _trajectories = _crown_noisy_sample_contribution(
            model,
            params,
            x_value,
            int(label),
            gamma,
            num_trajectories,
            seed + sample_index,
            gate_sites,
        )
        gradient[model.weight_slice()] += contribution[model.weight_slice()]
    return gradient / len(inputs)


def _crown_noisy_online_epoch(
    model: PerezSalinasCrownModel,
    params: NDArray[np.float64],
    inputs: NDArray[np.float64],
    labels: NDArray[np.int_],
    gamma: float,
    num_trajectories: int,
    gate_sites: list[tuple[int, int] | None],
    seed: int,
    step_size: float,
) -> tuple[NDArray[np.float64], float]:
    rng = np.random.RandomState(seed)
    contributions = []
    new_params = params.copy()
    for order_index, sample_index in enumerate(rng.permutation(len(inputs))):
        new_params, contribution = _crown_noisy_online_sample_update(
            model,
            new_params,
            inputs[sample_index],
            int(labels[sample_index]),
            gamma,
            num_trajectories,
            gate_sites,
            seed + 10_000 * order_index + int(sample_index),
            step_size,
        )
        contributions.append(contribution)
    classical_gradient = _crown_noisy_nongate_gradient(
        model,
        new_params,
        inputs,
        labels,
        gamma,
        num_trajectories,
        gate_sites,
        seed + 900_000,
    )
    new_params -= step_size * classical_gradient
    mean_contribution = np.mean(np.asarray(contributions), axis=0) if contributions else np.zeros_like(params)
    return new_params, float(np.linalg.norm(mean_contribution + classical_gradient))


def _crown_noisy_batch_update(
    model: PerezSalinasCrownModel,
    params: NDArray[np.float64],
    inputs: NDArray[np.float64],
    labels: NDArray[np.int_],
    gamma: float,
    num_trajectories: int,
    gate_sites: list[tuple[int, int] | None],
    iteration: int,
    seed: int,
    step_size: float,
) -> tuple[NDArray[np.float64], float]:
    gradient = _crown_noisy_gradient(
        model,
        params,
        inputs,
        labels,
        gamma,
        num_trajectories,
        gate_sites,
        iteration,
        seed,
    )
    return params - step_size * gradient, float(np.linalg.norm(gradient))


def run_crown_noisy_strength(seed: int, gamma: float, noiseless_test_acc: float) -> SweepResult:
    """Run the noisy crown optimizer for one seed and gamma."""
    x_train, x_test, y_train, y_test = generate_crown_dataset(n_samples=CROWN_SAMPLES, test_fraction=0.3, seed=seed)
    model = PerezSalinasCrownModel(n_qubits=4, n_layers=8)
    params = model.init_params(seed)
    gate_sites = _crown_gate_sites(model)
    fixed_test_maps = _crown_fixed_maps(
        model,
        params,
        x_test,
        y_test,
        gamma,
        CROWN_VALIDATION_TRAJECTORIES,
        gate_sites,
        3_000 + seed,
    )

    detail_rows: list[dict[str, float | int | str]] = []
    best_test_acc = 0.0
    best_iteration = 0
    final_loss = 0.0
    final_test_acc = 0.0
    for iteration in range(NUM_STEPS + 1):
        loss, test_acc = _crown_noisy_metrics(
            model,
            params,
            x_test,
            y_test,
            gamma,
            CROWN_VALIDATION_TRAJECTORIES,
            gate_sites,
            fixed_test_maps,
            4_000 + seed,
        )
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_iteration = iteration
        gradient_norm = 0.0
        detail_rows.append({
            "benchmark": "Perez-Salinas crown",
            "seed": seed,
            "gamma": gamma,
            "iteration": iteration,
            "test_loss": loss,
            "test_accuracy": test_acc,
            "gradient_norm": gradient_norm,
            "train_trajectories": CROWN_TRAIN_TRAJECTORIES,
            "validation_trajectories": CROWN_VALIDATION_TRAJECTORIES,
        })
        final_loss = loss
        final_test_acc = test_acc
        if iteration < NUM_STEPS:
            phase_iteration = iteration + 1
            if phase_iteration <= CROWN_SWITCH_ITERATION:
                params, gradient_norm = _crown_noisy_online_epoch(
                    model,
                    params,
                    x_train,
                    y_train,
                    gamma,
                    CROWN_TRAIN_TRAJECTORIES,
                    gate_sites,
                    seed + phase_iteration,
                    CROWN_ONLINE_STEP_SIZE,
                )
            else:
                params, gradient_norm = _crown_noisy_batch_update(
                    model,
                    params,
                    x_train,
                    y_train,
                    gamma,
                    CROWN_TRAIN_TRAJECTORIES,
                    gate_sites,
                    phase_iteration - CROWN_SWITCH_ITERATION,
                    5_000 + seed,
                    CROWN_BATCH_STEP_SIZE,
                )
            detail_rows[-1]["gradient_norm"] = gradient_norm

    return SweepResult(
        detail_rows=detail_rows,
        summary_row={
            "benchmark": "Perez-Salinas crown",
            "seed": seed,
            "gamma": gamma,
            "initial_test_accuracy": detail_rows[0]["test_accuracy"],
            "final_test_accuracy": final_test_acc,
            "best_test_accuracy": best_test_acc,
            "best_iteration": best_iteration,
            "final_test_loss": final_loss,
            "noiseless_test_accuracy": noiseless_test_acc,
            "train_trajectories": CROWN_TRAIN_TRAJECTORIES,
            "validation_trajectories": CROWN_VALIDATION_TRAJECTORIES,
        },
    )


def _noiseless_baseline_job(benchmark: str, seed: int) -> tuple[str, int, float]:
    if benchmark == "4-bit parity":
        settings = HybridSettings(max_iterations=12, switch_iteration=10, online_step_size=0.3, batch_step_size=1.0)
        return benchmark, seed, run_parity_seed(seed, settings).final_test_acc
    settings = HybridSettings(max_iterations=20, switch_iteration=10, online_step_size=0.3, batch_step_size=0.5)
    return benchmark, seed, run_crown_seed(seed, settings, n_samples=CROWN_SAMPLES).final_test_acc


def _sweep_job(benchmark: str, seed: int, gamma: float, noiseless_test_acc: float) -> SweepResult:
    if benchmark == "4-bit parity":
        return run_parity_noisy_strength(seed, gamma, noiseless_test_acc)
    return run_crown_noisy_strength(seed, gamma, noiseless_test_acc)


def _save_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_benchmark(summary_rows: list[dict[str, float | int | str]], benchmark: str, path: Path) -> None:
    import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]
    from matplotlib.ticker import FormatStrFormatter  # ty: ignore[unresolved-import]

    rows = [row for row in summary_rows if row["benchmark"] == benchmark]
    gammas = sorted({float(row["gamma"]) for row in rows})
    final_mean = []
    best_mean = []
    noiseless = []
    for gamma in gammas:
        gamma_rows = [row for row in rows if np.isclose(float(row["gamma"]), gamma)]
        final_mean.append(float(np.mean([float(row["final_test_accuracy"]) for row in gamma_rows])))
        best_mean.append(float(np.mean([float(row["best_test_accuracy"]) for row in gamma_rows])))
        noiseless.append(float(np.mean([float(row["noiseless_test_accuracy"]) for row in gamma_rows])))

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(gammas, final_mean, marker="o", linewidth=2.0, color="#9b2226", label="noisy final")
    ax.plot(gammas, best_mean, marker="x", linestyle="--", linewidth=1.8, color="#ca6702", label="noisy best")
    ax.axhline(float(np.mean(noiseless)), color="#005f73", linestyle=":", linewidth=1.8, label="noiseless final")
    ax.set_xlabel("Pauli noise strength gamma")
    ax.set_ylabel("fixed-test accuracy")
    ax.set_xlim(min(gammas), max(gammas))
    ax.set_xticks(gammas)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.tick_params(axis="x", labelrotation=30)
    ax.set_title(f"{benchmark} noisy supervised Krotov sweep")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    """Run the supervised noisy sweep."""
    start = time.perf_counter()
    output_dir = Path("experiments/supervised_qml_noise_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)
    gammas = np.linspace(GAMMA_MIN, GAMMA_MAX, NUM_GAMMAS)

    detail_rows: list[dict[str, float | int | str]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    total_jobs = len(SEEDS) * len(gammas) * 2
    print(
        f"steps={NUM_STEPS}, gammas={','.join(f'{gamma:.6g}' for gamma in gammas)}, seeds={SEEDS}, "
        f"crown_samples={CROWN_SAMPLES}, workers={NUM_WORKERS}",
        flush=True,
    )

    noiseless_parity: dict[int, float] = {}
    noiseless_crown: dict[int, float] = {}
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [
            executor.submit(_noiseless_baseline_job, benchmark, seed)
            for benchmark in ("4-bit parity", "Perez-Salinas crown")
            for seed in SEEDS
        ]
        for future in as_completed(futures):
            benchmark, seed, test_acc = future.result()
            if benchmark == "4-bit parity":
                noiseless_parity[seed] = test_acc
            else:
                noiseless_crown[seed] = test_acc
            print(
                f"baseline benchmark={benchmark} seed={seed} test_acc={test_acc:.6f} "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )

    print(f"noiseless_parity_mean={np.mean(list(noiseless_parity.values())):.6f}")
    print(f"noiseless_crown_mean={np.mean(list(noiseless_crown.values())):.6f}")

    completed_jobs = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for seed in SEEDS:
            for gamma in gammas:
                futures.append(executor.submit(_sweep_job, "4-bit parity", seed, float(gamma), noiseless_parity[seed]))
                futures.append(
                    executor.submit(_sweep_job, "Perez-Salinas crown", seed, float(gamma), noiseless_crown[seed])
                )
        for future in as_completed(futures):
            result = future.result()
            detail_rows.extend(result.detail_rows)
            summary_rows.append(result.summary_row)
            completed_jobs += 1
            print(
                f"progress={100 * completed_jobs / total_jobs:6.2f}% "
                f"benchmark={result.summary_row['benchmark']} seed={result.summary_row['seed']} "
                f"gamma={float(result.summary_row['gamma']):.6g} "
                f"final={float(result.summary_row['final_test_accuracy']):.6f} "
                f"best={float(result.summary_row['best_test_accuracy']):.6f} "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )

    detail_path = output_dir / f"supervised_qml_noise_sweep_steps{NUM_STEPS}_detail.csv"
    summary_path = output_dir / f"supervised_qml_noise_sweep_steps{NUM_STEPS}_summary.csv"
    parity_plot = output_dir / f"supervised_qml_noise_sweep_parity_steps{NUM_STEPS}.png"
    crown_plot = output_dir / f"supervised_qml_noise_sweep_crown_steps{NUM_STEPS}.png"
    _save_csv(detail_rows, detail_path)
    _save_csv(summary_rows, summary_path)
    _plot_benchmark(summary_rows, "4-bit parity", parity_plot)
    _plot_benchmark(summary_rows, "Perez-Salinas crown", crown_plot)
    print(f"detail_csv={detail_path}")
    print(f"summary_csv={summary_path}")
    print(f"parity_plot={parity_plot}")
    print(f"crown_plot={crown_plot}")
    print(f"elapsed_seconds={time.perf_counter() - start:.3f}")


if __name__ == "__main__":
    main()
