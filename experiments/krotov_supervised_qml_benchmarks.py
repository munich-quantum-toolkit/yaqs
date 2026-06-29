# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: DOC201, DOC501, PLC0415, PLC2701, PLR6301, T201

"""Supervised Krotov diagnostics for 4-bit parity and Perez-Salinas crown.

The parity benchmark uses the YAQS MPS Krotov implementation directly with a
single ``<Z_0> + bias`` MSE readout. The Perez-Salinas crown benchmark has a
multi-class weighted-fidelity readout that is not expressible by the current
scalar ``KrotovReadout`` API, so the experiment keeps the same gate-local
forward/costate/backward update in a compact state-vector model.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.optimization import KrotovReadout, KrotovTruncation, ParameterizedCircuit, ParameterizedGate
from mqt.yaqs.optimization.krotov import (
    _apply_operator,
    _batch_epoch,
    _expectation,
    _online_epoch,
    _step_size,
    empirical_loss,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray


Schedule = Literal["constant", "inverse", "exp"]

_I2 = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_CZ_LOCAL = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)


@dataclass(frozen=True)
class HybridSettings:
    """Optimizer settings for an online-to-batch Krotov run."""

    max_iterations: int
    switch_iteration: int
    online_step_size: float
    batch_step_size: float
    online_schedule: Schedule = "constant"
    batch_schedule: Schedule = "constant"
    online_decay: float = 0.05
    batch_decay: float = 0.05


@dataclass
class BenchmarkRun:
    """Result of one benchmark and seed."""

    benchmark: str
    seed: int
    final_loss: float
    final_train_acc: float
    final_test_acc: float
    wall_time_total: float
    initial_params: list[float]
    final_params: list[float]
    trace: dict[str, list[float | int | str]]


def parity_truth_table() -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Return all unique 4-bit inputs and odd/even parity labels."""
    bitstrings = np.array(
        [[(value >> shift) & 1 for shift in range(3, -1, -1)] for value in range(16)],
        dtype=int,
    )
    labels = np.where(np.sum(bitstrings, axis=1) % 2 == 1, 1, -1).astype(int)
    return bitstrings, labels


def generate_parity_split(
    train_size: int = 10,
    test_size: int = 6,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate the requested leakage-free unique 4-bit parity split."""
    if train_size <= 0 or test_size <= 0 or train_size + test_size != 16:
        msg = "4-bit parity requires positive train/test sizes summing to 16."
        raise ValueError(msg)
    bitstrings, labels = parity_truth_table()
    permutation = np.random.RandomState(seed).permutation(16)
    train_idx = permutation[:train_size]
    test_idx = permutation[train_size : train_size + test_size]
    return (
        bitstrings[train_idx].astype(np.float64),
        bitstrings[test_idx].astype(np.float64),
        labels[train_idx].astype(np.float64),
        labels[test_idx].astype(np.float64),
    )


def build_parity_circuit(n_layers: int = 2) -> ParameterizedCircuit:
    """Build the requested 4-qubit Rot/CNOT-ring parity ansatz."""
    if n_layers <= 0:
        msg = "n_layers must be positive."
        raise ValueError(msg)

    gates: list[ParameterizedGate] = []
    param_index = 0
    for _layer in range(n_layers):
        for qubit in range(4):
            gates.extend((
                ParameterizedGate("rz", (qubit,), param_index=param_index),
                ParameterizedGate("ry", (qubit,), param_index=param_index + 1),
                ParameterizedGate("rz", (qubit,), param_index=param_index + 2),
            ))
            param_index += 3
        gates.extend((
            ParameterizedGate("cx", (0, 1)),
            ParameterizedGate("cx", (1, 2)),
            ParameterizedGate("cx", (2, 3)),
            ParameterizedGate("cx", (3, 0)),
        ))
    return ParameterizedCircuit(num_qubits=4, gates=gates, num_params=param_index)


def parity_initial_theta(seed: int, num_params: int) -> NDArray[np.float64]:
    """Initialize parity rotations from ``Normal(0, 0.01)``."""
    return np.random.RandomState(seed).normal(loc=0.0, scale=0.01, size=num_params).astype(np.float64)


def parity_basis_state(x: NDArray[np.float64]) -> MPS:
    """Prepare one 4-bit input as a YAQS basis-state MPS."""
    basis = "".join(str(int(bit)) for bit in np.asarray(x, dtype=int))
    return MPS(4, state="basis", basis_string=basis)


def _parity_scores(
    circuit: ParameterizedCircuit,
    theta: NDArray[np.float64],
    bias: float,
    inputs: NDArray[np.float64],
    readout: KrotovReadout,
    truncation: KrotovTruncation,
) -> NDArray[np.float64]:
    scores = np.empty(len(inputs), dtype=np.float64)
    for idx, x in enumerate(inputs):
        state = parity_basis_state(x)
        for gate in circuit.gates:
            matrix, sites = circuit.gate_matrix(gate, theta, x)
            _apply_operator(state, matrix, sites, truncation)
        scores[idx] = _expectation(state, readout.observable) + bias
    return scores


def _sign_accuracy(scores: NDArray[np.float64], labels: NDArray[np.float64]) -> float:
    predictions = np.where(np.asarray(scores, dtype=np.float64) >= 0.0, 1.0, -1.0)
    return float(np.mean(predictions == np.asarray(labels, dtype=np.float64)))


def _init_trace() -> dict[str, list[float | int | str]]:
    return {
        "step": [],
        "phase": [],
        "loss": [],
        "train_acc": [],
        "test_acc": [],
        "wall_time": [],
        "step_size": [],
        "update_norm": [],
        "gradient_norm": [],
    }


def _append_trace(
    trace: dict[str, list[float | int | str]],
    *,
    step: int,
    phase: str,
    loss: float,
    train_acc: float,
    test_acc: float,
    wall_time: float,
    step_size: float,
    update_norm: float,
    gradient_norm: float,
) -> None:
    trace["step"].append(int(step))
    trace["phase"].append(str(phase))
    trace["loss"].append(float(loss))
    trace["train_acc"].append(float(train_acc))
    trace["test_acc"].append(float(test_acc))
    trace["wall_time"].append(float(wall_time))
    trace["step_size"].append(float(step_size))
    trace["update_norm"].append(float(update_norm))
    trace["gradient_norm"].append(float(gradient_norm))


def run_parity_seed(
    seed: int,
    settings: HybridSettings,
    *,
    train_size: int = 10,
    test_size: int = 6,
    n_layers: int = 2,
) -> BenchmarkRun:
    """Run one 4-bit parity seed with the native YAQS MPS Krotov path."""
    x_train, x_test, y_train, y_test = generate_parity_split(train_size, test_size, seed)
    circuit = build_parity_circuit(n_layers)
    readout = KrotovReadout(Observable("z", 0), loss="mse", use_bias=True)
    truncation = KrotovTruncation()
    theta = parity_initial_theta(seed, circuit.num_params)
    initial_params = [*theta.tolist(), 0.0]
    bias = 0.0
    trace = _init_trace()
    start = time.time()

    def record(step: int, phase: str, step_size: float, update_norm: float, gradient_norm: float) -> None:
        loss = empirical_loss(circuit, theta, x_train, y_train, readout, bias, parity_basis_state, truncation)
        train_acc = _sign_accuracy(_parity_scores(circuit, theta, bias, x_train, readout, truncation), y_train)
        test_acc = _sign_accuracy(_parity_scores(circuit, theta, bias, x_test, readout, truncation), y_test)
        _append_trace(
            trace,
            step=step,
            phase=phase,
            loss=loss,
            train_acc=train_acc,
            test_acc=test_acc,
            wall_time=time.time() - start,
            step_size=step_size,
            update_norm=update_norm,
            gradient_norm=gradient_norm,
        )

    record(0, "init", 0.0, 0.0, 0.0)
    for iteration in range(1, settings.max_iterations + 1):
        theta_before = theta.copy()
        bias_before = bias
        if iteration <= settings.switch_iteration:
            step = _step_size(
                settings.online_step_size,
                iteration,
                settings.online_schedule,
                settings.online_decay,
            )
            theta, bias, gradient_norm = _online_epoch(
                circuit,
                theta,
                bias,
                x_train,
                y_train,
                readout,
                step,
                seed + iteration,
                parity_basis_state,
                truncation,
            )
            phase = "online"
        else:
            phase_iteration = iteration - settings.switch_iteration
            step = _step_size(
                settings.batch_step_size,
                phase_iteration,
                settings.batch_schedule,
                settings.batch_decay,
            )
            theta, bias, gradient_norm, _pre_update_loss = _batch_epoch(
                circuit,
                theta,
                bias,
                x_train,
                y_train,
                readout,
                step,
                parity_basis_state,
                truncation,
            )
            phase = "batch"
        update_norm = float(np.sqrt(np.linalg.norm(theta - theta_before) ** 2 + (bias - bias_before) ** 2))
        record(iteration, phase, step, update_norm, gradient_norm)

    return BenchmarkRun(
        benchmark="4-bit parity",
        seed=int(seed),
        final_loss=float(trace["loss"][-1]),
        final_train_acc=float(trace["train_acc"][-1]),
        final_test_acc=float(trace["test_acc"][-1]),
        wall_time_total=time.time() - start,
        initial_params=initial_params,
        final_params=[*theta.tolist(), float(bias)],
        trace=trace,
    )


def _kron_n(matrices: Sequence[NDArray[np.complex128]]) -> NDArray[np.complex128]:
    result = np.asarray(matrices[0], dtype=np.complex128)
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return np.asarray(result, dtype=np.complex128)


def _single_qubit_gate(gate: NDArray[np.complex128], qubit: int, n_qubits: int) -> NDArray[np.complex128]:
    matrices = [_I2.copy() for _ in range(n_qubits)]
    matrices[qubit] = gate
    return _kron_n(matrices)


def _embed_two_qubit_operator(
    local_operator: NDArray[np.complex128],
    qubit_a: int,
    qubit_b: int,
    n_qubits: int,
) -> NDArray[np.complex128]:
    if qubit_a == qubit_b:
        msg = "Two-qubit operators require distinct qubits."
        raise ValueError(msg)

    qa, qb = sorted((qubit_a, qubit_b))
    dim = 2**n_qubits
    full = np.zeros((dim, dim), dtype=np.complex128)
    for column in range(dim):
        bits = [(column >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        local_column = (bits[qa] << 1) | bits[qb]
        for local_row in range(4):
            amplitude = local_operator[local_row, local_column]
            if np.isclose(amplitude, 0.0):
                continue
            out_bits = bits.copy()
            out_bits[qa] = (local_row >> 1) & 1
            out_bits[qb] = local_row & 1
            row = sum(bit << (n_qubits - 1 - idx) for idx, bit in enumerate(out_bits))
            full[row, column] += amplitude
    return full


def _ry(angle: float) -> NDArray[np.complex128]:
    cos = np.cos(angle / 2.0)
    sin = np.sin(angle / 2.0)
    return np.array([[cos, -sin], [sin, cos]], dtype=np.complex128)


def _rz(angle: float) -> NDArray[np.complex128]:
    return np.array(
        [[np.exp(-0.5j * angle), 0.0], [0.0, np.exp(0.5j * angle)]],
        dtype=np.complex128,
    )


def _zero_state(n_qubits: int) -> NDArray[np.complex128]:
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0
    return state


def _stratified_train_test_split(
    x_values: NDArray[np.float64],
    labels: NDArray[np.int_],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_], NDArray[np.int_]]:
    if not 0.0 < test_fraction < 1.0:
        msg = "test_fraction must lie strictly between zero and one."
        raise ValueError(msg)
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels, dtype=int)
    target_test = round(len(labels) * test_fraction)
    test_indices: list[int] = []
    class_indices = [np.flatnonzero(labels == class_label) for class_label in np.unique(labels)]
    desired_by_class = [round(len(indices) * test_fraction) for indices in class_indices]
    while sum(desired_by_class) < target_test:
        fractions = [
            len(indices) * test_fraction - count
            for indices, count in zip(class_indices, desired_by_class, strict=False)
        ]
        desired_by_class[int(np.argmax(fractions))] += 1
    while sum(desired_by_class) > target_test:
        fractions = [
            count - len(indices) * test_fraction
            for indices, count in zip(class_indices, desired_by_class, strict=False)
        ]
        desired_by_class[int(np.argmax(fractions))] -= 1

    for indices, num_test in zip(class_indices, desired_by_class, strict=True):
        permuted = rng.permutation(indices)
        test_indices.extend(int(index) for index in permuted[:num_test])

    test_index_array = np.asarray(sorted(test_indices), dtype=int)
    train_mask = np.ones(len(labels), dtype=bool)
    train_mask[test_index_array] = False
    train_index_array = np.flatnonzero(train_mask)
    train_index_array = rng.permutation(train_index_array)
    test_index_array = rng.permutation(test_index_array)
    return (
        x_values[train_index_array].copy(),
        x_values[test_index_array].copy(),
        labels[train_index_array].copy(),
        labels[test_index_array].copy(),
    )


def generate_crown_dataset(
    *,
    n_samples: int = 600,
    test_fraction: float = 0.3,
    seed: int = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int_], NDArray[np.int_]]:
    """Generate the Perez-Salinas crown dataset with a stratified split."""
    if n_samples <= 1:
        msg = "n_samples must be greater than one."
        raise ValueError(msg)
    rng = np.random.RandomState(seed)
    x_values = 2.0 * rng.rand(n_samples, 2) - 1.0
    radius = np.linalg.norm(x_values, axis=1)
    inner = np.sqrt(0.8 - 2.0 / np.pi)
    outer = np.sqrt(0.8)
    labels = ((radius > inner) & (radius < outer)).astype(int)
    x_train, x_test, y_train, y_test = _stratified_train_test_split(
        x_values,
        labels,
        test_fraction=test_fraction,
        seed=seed,
    )
    _assert_no_overlap(x_train, x_test)
    return x_train, x_test, y_train, y_test


def _assert_no_overlap(x_train: NDArray[np.float64], x_test: NDArray[np.float64]) -> None:
    train_points = {tuple(np.round(row, 12)) for row in x_train}
    test_points = {tuple(np.round(row, 12)) for row in x_test}
    overlap = train_points & test_points
    if overlap:
        msg = f"Train/test split contains {len(overlap)} duplicate points."
        raise RuntimeError(msg)


class PerezSalinasCrownModel:
    """Four-qubit Perez-Salinas weighted-fidelity crown classifier."""

    _slot_order = (1, 4, 0, 3, 2)

    def __init__(self, n_qubits: int = 4, n_layers: int = 8) -> None:
        """Initialize the requested 4-qubit, 2-class crown model."""
        if n_qubits != 4:
            msg = "The requested crown diagnostic uses exactly four qubits."
            raise ValueError(msg)
        if n_layers <= 0:
            msg = "n_layers must be positive."
            raise ValueError(msg)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.n_classes = 2
        self.params_per_block = 5
        self.n_quantum_params = self.n_qubits * self.n_layers * self.params_per_block
        self.n_weight_params = self.n_classes * self.n_qubits
        self.n_params = self.n_quantum_params + self.n_weight_params
        self.dim = 2**self.n_qubits
        self.identity = np.eye(self.dim, dtype=np.complex128)
        self.pauli_y_ops = tuple(_single_qubit_gate(_Y, qubit, self.n_qubits) for qubit in range(self.n_qubits))
        self.pauli_z_ops = tuple(_single_qubit_gate(_Z, qubit, self.n_qubits) for qubit in range(self.n_qubits))
        self.projector_ops = (
            tuple(0.5 * (self.identity + self.pauli_z_ops[qubit]) for qubit in range(self.n_qubits)),
            tuple(0.5 * (self.identity - self.pauli_z_ops[qubit]) for qubit in range(self.n_qubits)),
        )
        self.cz_layers = (
            (
                _embed_two_qubit_operator(_CZ_LOCAL, 0, 1, self.n_qubits),
                _embed_two_qubit_operator(_CZ_LOCAL, 2, 3, self.n_qubits),
            ),
            (
                _embed_two_qubit_operator(_CZ_LOCAL, 1, 2, self.n_qubits),
                _embed_two_qubit_operator(_CZ_LOCAL, 0, 3, self.n_qubits),
            ),
        )

    def init_params(self, seed: int = 0) -> NDArray[np.float64]:
        """Initialize all quantum and readout parameters from ``Uniform[0, 1)``."""
        return np.random.RandomState(seed).rand(self.n_params).astype(np.float64)

    def weight_slice(self) -> slice:
        """Return the flat slice containing classical readout weights."""
        return slice(self.n_quantum_params, self.n_params)

    def weights(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return class-by-qubit readout weights."""
        return np.asarray(params[self.weight_slice()], dtype=np.float64).reshape(self.n_classes, self.n_qubits)

    def target_vector(self, label: int) -> NDArray[np.float64]:
        """Return the one-hot binary class target."""
        target = np.zeros(self.n_classes, dtype=np.float64)
        target[int(label)] = 1.0
        return target

    def _block_base(self, layer: int, qubit: int) -> int:
        return (layer * self.n_qubits + qubit) * self.params_per_block

    def _slot_spec(self, slot: int) -> tuple[str, int | None, float]:
        if slot == 0:
            return "ry", None, 1.0
        if slot == 1:
            return "rz", None, -1.0
        if slot == 2:
            return "rz", None, -1.0
        if slot == 3:
            return "ry", 0, 1.0
        if slot == 4:
            return "rz", 1, -1.0
        msg = f"Unknown Perez-Salinas slot {slot}."
        raise ValueError(msg)

    def gate_factor(self, param_index: int, x_value: NDArray[np.float64]) -> float:
        """Return the scalar feature factor multiplying one quantum parameter."""
        slot = param_index % self.params_per_block
        _axis, feature_index, sign = self._slot_spec(slot)
        factor = sign
        if feature_index is not None:
            factor *= float(x_value[feature_index])
        return factor

    def rebuild_param_gate(
        self,
        param_index: int,
        params: NDArray[np.float64],
        x_value: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        """Rebuild the embedded unitary associated with one quantum parameter."""
        block = param_index // self.params_per_block
        qubit = block % self.n_qubits
        slot = param_index % self.params_per_block
        axis, _feature_index, _sign = self._slot_spec(slot)
        angle = self.gate_factor(param_index, x_value) * float(params[param_index])
        pauli = self.pauli_y_ops[qubit] if axis == "ry" else self.pauli_z_ops[qubit]
        return np.cos(angle / 2.0) * self.identity - 1j * np.sin(angle / 2.0) * pauli

    def gate_derivative_generator(self, param_index: int, x_value: NDArray[np.float64]) -> NDArray[np.complex128]:
        """Return ``dU/dtheta * U^dagger`` for one quantum parameter."""
        block = param_index // self.params_per_block
        qubit = block % self.n_qubits
        slot = param_index % self.params_per_block
        axis, _feature_index, _sign = self._slot_spec(slot)
        pauli = self.pauli_y_ops[qubit] if axis == "ry" else self.pauli_z_ops[qubit]
        return self.gate_factor(param_index, x_value) * (-0.5j) * pauli

    def gate_sequence_and_states(
        self,
        params: NDArray[np.float64],
        x_value: NDArray[np.float64],
    ) -> tuple[list[tuple[NDArray[np.complex128], int | None]], list[NDArray[np.complex128]]]:
        """Return dense gate factors and all forward states for one sample."""
        gates = self.gate_sequence(params, x_value)
        state = _zero_state(self.n_qubits)
        states = [state.copy()]
        for gate, _param_index in gates:
            state = gate @ state
            states.append(state.copy())
        return gates, states

    def gate_sequence(
        self,
        params: NDArray[np.float64],
        x_value: NDArray[np.float64],
    ) -> list[tuple[NDArray[np.complex128], int | None]]:
        """Return dense gate factors for one sample."""
        gates: list[tuple[NDArray[np.complex128], int | None]] = []
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                block_base = self._block_base(layer, qubit)
                for slot in self._slot_order:
                    param_index = block_base + slot
                    gates.append((self.rebuild_param_gate(param_index, params, x_value), param_index))
            if layer < self.n_layers - 1:
                gates.extend((entangler, None) for entangler in self.cz_layers[layer % 2])
        return gates

    def sample_fidelities(self, final_state: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Compute local class-label fidelities for one final state."""
        fidelities = np.empty((self.n_classes, self.n_qubits), dtype=np.float64)
        for class_index in range(self.n_classes):
            for qubit in range(self.n_qubits):
                value = final_state.conj() @ self.projector_ops[class_index][qubit] @ final_state
                fidelities[class_index, qubit] = float(np.clip(np.real(value), 0.0, 1.0))
        return fidelities

    def class_scores(self, params: NDArray[np.float64], x_value: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the two class scores for one sample."""
        _gates, states = self.gate_sequence_and_states(params, x_value)
        return np.sum(self.weights(params) * self.sample_fidelities(states[-1]), axis=1)

    def predict(self, params: NDArray[np.float64], inputs: NDArray[np.float64]) -> NDArray[np.int_]:
        """Predict class labels by ``argmax`` over class scores."""
        return np.array([int(np.argmax(self.class_scores(params, x_value))) for x_value in inputs], dtype=int)

    def accuracy(self, params: NDArray[np.float64], inputs: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
        """Compute classification accuracy."""
        return float(np.mean(self.predict(params, inputs) == np.asarray(labels, dtype=int)))

    def loss(self, params: NDArray[np.float64], inputs: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
        """Compute mean weighted-fidelity squared loss."""
        losses = []
        for x_value, label in zip(inputs, labels, strict=True):
            scores = self.class_scores(params, x_value)
            residual = scores - self.target_vector(int(label))
            losses.append(0.5 * float(residual @ residual))
        return float(np.mean(losses))

    def terminal_costate(
        self,
        params: NDArray[np.float64],
        final_state: NDArray[np.complex128],
        label: int,
    ) -> NDArray[np.complex128]:
        """Build the terminal costate from weighted-fidelity residuals."""
        fidelities = self.sample_fidelities(final_state)
        residual = np.sum(self.weights(params) * fidelities, axis=1) - self.target_vector(label)
        operator = np.zeros((self.dim, self.dim), dtype=np.complex128)
        weights = self.weights(params)
        for class_index in range(self.n_classes):
            for qubit in range(self.n_qubits):
                operator += residual[class_index] * weights[class_index, qubit] * self.projector_ops[class_index][qubit]
        return operator @ final_state

    def loss_gradient(
        self,
        params: NDArray[np.float64],
        inputs: NDArray[np.float64],
        labels: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Compute the exact empirical gate-local gradient."""
        gradient = np.zeros_like(params)
        for x_value, label in zip(inputs, labels, strict=True):
            gates, states = self.gate_sequence_and_states(params, x_value)
            final_state = states[-1]
            fidelities = self.sample_fidelities(final_state)
            scores = np.sum(self.weights(params) * fidelities, axis=1)
            residual = scores - self.target_vector(int(label))
            gradient[self.weight_slice()] += (residual[:, None] * fidelities).reshape(-1)

            costates: list[NDArray[np.complex128]] = [np.empty(0, dtype=np.complex128)] * len(states)
            costates[-1] = self.terminal_costate(params, final_state, int(label))
            for gate_index in range(len(gates) - 1, -1, -1):
                costates[gate_index] = gates[gate_index][0].conj().T @ costates[gate_index + 1]
            for gate_index, (_gate, param_index) in enumerate(gates):
                if param_index is None:
                    continue
                derivative_state = self.gate_derivative_generator(param_index, x_value) @ states[gate_index + 1]
                gradient[param_index] += 2.0 * float(np.real(costates[gate_index + 1].conj() @ derivative_state))
        return gradient / len(inputs)

    def nongate_loss_gradient(
        self,
        params: NDArray[np.float64],
        inputs: NDArray[np.float64],
        labels: NDArray[np.int_],
    ) -> NDArray[np.float64]:
        """Compute only the classical readout-weight gradient."""
        gradient = np.zeros_like(params)
        for x_value, label in zip(inputs, labels, strict=True):
            _gates, states = self.gate_sequence_and_states(params, x_value)
            fidelities = self.sample_fidelities(states[-1])
            residual = np.sum(self.weights(params) * fidelities, axis=1) - self.target_vector(int(label))
            gradient[self.weight_slice()] += (residual[:, None] * fidelities).reshape(-1)
        return gradient / len(inputs)

    def sample_krotov_contribution(
        self,
        params: NDArray[np.float64],
        x_value: NDArray[np.float64],
        label: int,
    ) -> NDArray[np.float64]:
        """Compute one sample contribution for gate-supported parameters."""
        gates, states = self.gate_sequence_and_states(params, x_value)
        costates: list[NDArray[np.complex128]] = [np.empty(0, dtype=np.complex128)] * len(states)
        costates[-1] = self.terminal_costate(params, states[-1], int(label))
        for gate_index in range(len(gates) - 1, -1, -1):
            costates[gate_index] = gates[gate_index][0].conj().T @ costates[gate_index + 1]

        contribution = np.zeros_like(params)
        for gate_index, (_gate, param_index) in enumerate(gates):
            if param_index is None:
                continue
            derivative_state = self.gate_derivative_generator(param_index, x_value) @ states[gate_index + 1]
            contribution[param_index] = 2.0 * float(np.real(costates[gate_index + 1].conj() @ derivative_state))
        return contribution

    def online_sample_update(
        self,
        params: NDArray[np.float64],
        x_value: NDArray[np.float64],
        label: int,
        step_size: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Apply one stale-adjoint online update to quantum parameters."""
        gates, states = self.gate_sequence_and_states(params, x_value)
        costates: list[NDArray[np.complex128]] = [np.empty(0, dtype=np.complex128)] * len(states)
        costates[-1] = self.terminal_costate(params, states[-1], int(label))
        for gate_index in range(len(gates) - 1, -1, -1):
            costates[gate_index] = gates[gate_index][0].conj().T @ costates[gate_index + 1]

        new_params = params.copy()
        current_state = states[0].copy()
        contribution = np.zeros_like(params)
        for gate_index, (gate, param_index) in enumerate(gates):
            active_gate = gate
            if param_index is not None:
                gate_output = active_gate @ current_state
                derivative_state = self.gate_derivative_generator(param_index, x_value) @ gate_output
                gradient = 2.0 * float(np.real(costates[gate_index + 1].conj() @ derivative_state))
                contribution[param_index] = gradient
                new_params[param_index] -= step_size * gradient
                active_gate = self.rebuild_param_gate(param_index, new_params, x_value)
            current_state = active_gate @ current_state
        return new_params, contribution


def _crown_batch_update(
    model: PerezSalinasCrownModel,
    params: NDArray[np.float64],
    inputs: NDArray[np.float64],
    labels: NDArray[np.int_],
    step_size: float,
) -> tuple[NDArray[np.float64], float]:
    gradient = model.loss_gradient(params, inputs, labels)
    return params - step_size * gradient, float(np.linalg.norm(gradient))


def _crown_online_epoch(
    model: PerezSalinasCrownModel,
    params: NDArray[np.float64],
    inputs: NDArray[np.float64],
    labels: NDArray[np.int_],
    step_size: float,
    seed: int,
) -> tuple[NDArray[np.float64], float]:
    rng = np.random.RandomState(seed)
    contributions = []
    new_params = params.copy()
    for sample_index in rng.permutation(len(inputs)):
        new_params, contribution = model.online_sample_update(
            new_params,
            inputs[sample_index],
            int(labels[sample_index]),
            step_size,
        )
        contributions.append(contribution)
    classical_gradient = model.nongate_loss_gradient(new_params, inputs, labels)
    new_params -= step_size * classical_gradient
    mean_contribution = np.mean(np.asarray(contributions), axis=0) if contributions else np.zeros_like(params)
    return new_params, float(np.linalg.norm(mean_contribution + classical_gradient))


def run_crown_seed(
    seed: int,
    settings: HybridSettings,
    *,
    n_samples: int = 600,
    test_fraction: float = 0.3,
    n_layers: int = 8,
) -> BenchmarkRun:
    """Run one Perez-Salinas crown seed with gate-local Krotov updates."""
    x_train, x_test, y_train, y_test = generate_crown_dataset(
        n_samples=n_samples,
        test_fraction=test_fraction,
        seed=seed,
    )
    model = PerezSalinasCrownModel(n_qubits=4, n_layers=n_layers)
    params = model.init_params(seed)
    initial_params = params.tolist()
    trace = _init_trace()
    start = time.time()

    def record(step: int, phase: str, step_size: float, update_norm: float, gradient_norm: float) -> None:
        _append_trace(
            trace,
            step=step,
            phase=phase,
            loss=model.loss(params, x_train, y_train),
            train_acc=model.accuracy(params, x_train, y_train),
            test_acc=model.accuracy(params, x_test, y_test),
            wall_time=time.time() - start,
            step_size=step_size,
            update_norm=update_norm,
            gradient_norm=gradient_norm,
        )

    record(0, "init", 0.0, 0.0, 0.0)
    for iteration in range(1, settings.max_iterations + 1):
        params_before = params.copy()
        if iteration <= settings.switch_iteration:
            step = _step_size(
                settings.online_step_size,
                iteration,
                settings.online_schedule,
                settings.online_decay,
            )
            params, gradient_norm = _crown_online_epoch(model, params, x_train, y_train, step, seed + iteration)
            phase = "online"
        else:
            phase_iteration = iteration - settings.switch_iteration
            step = _step_size(
                settings.batch_step_size,
                phase_iteration,
                settings.batch_schedule,
                settings.batch_decay,
            )
            params, gradient_norm = _crown_batch_update(model, params, x_train, y_train, step)
            phase = "batch"
        record(iteration, phase, step, float(np.linalg.norm(params - params_before)), gradient_norm)

    return BenchmarkRun(
        benchmark="Perez-Salinas crown",
        seed=int(seed),
        final_loss=float(trace["loss"][-1]),
        final_train_acc=float(trace["train_acc"][-1]),
        final_test_acc=float(trace["test_acc"][-1]),
        wall_time_total=time.time() - start,
        initial_params=initial_params,
        final_params=params.tolist(),
        trace=trace,
    )


def summarize_runs(runs: Sequence[BenchmarkRun]) -> list[dict[str, float | str | int]]:
    """Summarize runs by benchmark, ranked by test accuracy, loss, and time."""
    summaries: list[dict[str, float | str | int]] = []
    for benchmark in sorted({run.benchmark for run in runs}):
        group = [run for run in runs if run.benchmark == benchmark]
        summaries.append(
            {
                "benchmark": benchmark,
                "num_seeds": len(group),
                "mean_final_test_acc": float(np.mean([run.final_test_acc for run in group])),
                "mean_final_train_acc": float(np.mean([run.final_train_acc for run in group])),
                "mean_final_loss": float(np.mean([run.final_loss for run in group])),
                "mean_wall_time_total": float(np.mean([run.wall_time_total for run in group])),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            -float(row["mean_final_test_acc"]),
            float(row["mean_final_loss"]),
            float(row["mean_wall_time_total"]),
        ),
    )


def _jsonable_run(run: BenchmarkRun) -> dict[str, Any]:
    return {
        "benchmark": run.benchmark,
        "seed": run.seed,
        "wall_time_total": run.wall_time_total,
        "final_loss": run.final_loss,
        "final_train_acc": run.final_train_acc,
        "final_test_acc": run.final_test_acc,
        "initial_params": run.initial_params,
        "final_params": run.final_params,
        "trace": run.trace,
    }


def write_outputs(runs: Sequence[BenchmarkRun], output_dir: Path, *, make_plot: bool = True) -> dict[str, Path]:
    """Write trace CSV, summary CSV, JSON, and optional plot artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary_csv": output_dir / "supervised_qml_benchmark_summary.csv",
        "trace_csv": output_dir / "supervised_qml_benchmark_traces.csv",
        "json": output_dir / "supervised_qml_benchmark_results.json",
    }

    with paths["summary_csv"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "benchmark",
            "num_seeds",
            "mean_final_test_acc",
            "mean_final_train_acc",
            "mean_final_loss",
            "mean_wall_time_total",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summarize_runs(runs))

    with paths["trace_csv"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "benchmark",
            "seed",
            "step",
            "phase",
            "loss",
            "train_acc",
            "test_acc",
            "wall_time",
            "step_size",
            "update_norm",
            "gradient_norm",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for idx, step in enumerate(run.trace["step"]):
                writer.writerow(
                    {
                        "benchmark": run.benchmark,
                        "seed": run.seed,
                        "step": step,
                        "phase": run.trace["phase"][idx],
                        "loss": run.trace["loss"][idx],
                        "train_acc": run.trace["train_acc"][idx],
                        "test_acc": run.trace["test_acc"][idx],
                        "wall_time": run.trace["wall_time"][idx],
                        "step_size": run.trace["step_size"][idx],
                        "update_norm": run.trace["update_norm"][idx],
                        "gradient_norm": run.trace["gradient_norm"][idx],
                    }
                )

    with paths["json"].open("w", encoding="utf-8") as handle:
        json.dump({"summary": summarize_runs(runs), "runs": [_jsonable_run(run) for run in runs]}, handle, indent=2)

    if make_plot:
        paths["plot"] = output_dir / "supervised_qml_benchmark_traces.png"
        _plot_runs(runs, paths["plot"])
    return paths


def _mean_trace(runs: Sequence[BenchmarkRun], field: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    max_len = max(len(run.trace[field]) for run in runs)
    values = np.full((len(runs), max_len), np.nan, dtype=np.float64)
    for idx, run in enumerate(runs):
        trace_values = np.asarray(run.trace[field], dtype=np.float64)
        values[idx, : len(trace_values)] = trace_values
    steps = np.arange(max_len, dtype=np.float64)
    return steps, np.nanmean(values, axis=0)


def _plot_runs(runs: Sequence[BenchmarkRun], path: Path) -> None:
    import matplotlib as mpl  # ty: ignore[unresolved-import]

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # ty: ignore[unresolved-import]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=False)
    for benchmark in sorted({run.benchmark for run in runs}):
        group = [run for run in runs if run.benchmark == benchmark]
        steps, losses = _mean_trace(group, "loss")
        _steps, test_acc = _mean_trace(group, "test_acc")
        axes[0].plot(steps, losses, marker="o", linewidth=1.6, label=benchmark)
        axes[1].plot(steps, test_acc, marker="o", linewidth=1.6, label=benchmark)
    axes[0].set_xlabel("Krotov iteration")
    axes[0].set_ylabel("Mean train loss")
    axes[0].set_yscale("log")
    axes[1].set_xlabel("Krotov iteration")
    axes[1].set_ylabel("Mean test accuracy")
    axes[1].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Supervised Krotov diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _parse_seeds(values: Iterable[str]) -> list[int]:
    return [int(value) for value in values]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", nargs="+", choices=["parity", "crown"], default=["parity", "crown"])
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "2"])
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/supervised_qml_benchmarks"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--parity-train-size", type=int, default=10)
    parser.add_argument("--parity-test-size", type=int, default=6)
    parser.add_argument("--parity-layers", type=int, default=2)
    parser.add_argument("--parity-iterations", type=int, default=12)
    parser.add_argument("--parity-switch", type=int, default=10)
    parser.add_argument("--parity-online-step", type=float, default=0.3)
    parser.add_argument("--parity-batch-step", type=float, default=1.0)
    parser.add_argument("--crown-samples", type=int, default=600)
    parser.add_argument("--crown-test-fraction", type=float, default=0.3)
    parser.add_argument("--crown-layers", type=int, default=8)
    parser.add_argument("--crown-iterations", type=int, default=20)
    parser.add_argument("--crown-switch", type=int, default=10)
    parser.add_argument("--crown-online-step", type=float, default=0.3)
    parser.add_argument("--crown-batch-step", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    """Run the requested supervised diagnostics."""
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    parity_settings = HybridSettings(
        max_iterations=args.parity_iterations,
        switch_iteration=args.parity_switch,
        online_step_size=args.parity_online_step,
        batch_step_size=args.parity_batch_step,
    )
    crown_settings = HybridSettings(
        max_iterations=args.crown_iterations,
        switch_iteration=args.crown_switch,
        online_step_size=args.crown_online_step,
        batch_step_size=args.crown_batch_step,
    )

    runs: list[BenchmarkRun] = []
    for seed in seeds:
        if "parity" in args.benchmarks:
            run = run_parity_seed(
                seed,
                parity_settings,
                train_size=args.parity_train_size,
                test_size=args.parity_test_size,
                n_layers=args.parity_layers,
            )
            runs.append(run)
            print(
                f"parity seed={seed}: loss={run.final_loss:.6f}, "
                f"train_acc={run.final_train_acc:.3f}, test_acc={run.final_test_acc:.3f}"
            )
        if "crown" in args.benchmarks:
            run = run_crown_seed(
                seed,
                crown_settings,
                n_samples=args.crown_samples,
                test_fraction=args.crown_test_fraction,
                n_layers=args.crown_layers,
            )
            runs.append(run)
            print(
                f"crown seed={seed}: loss={run.final_loss:.6f}, "
                f"train_acc={run.final_train_acc:.3f}, test_acc={run.final_test_acc:.3f}"
            )

    paths = write_outputs(runs, args.output_dir, make_plot=not args.no_plot)
    print("wrote outputs:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
