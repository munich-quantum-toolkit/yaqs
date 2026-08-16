# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Construct stochastic circuit realizations from standard YAQS noise models."""

from __future__ import annotations

import math
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import XGate, YGate, ZGate

from ..core.data_structures.noise_model import NoiseModel, is_pauli
from .digital_tjm import create_local_noise_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


_PAULI_MATRICES = {
    "x": NoiseModel.get_operator("pauli_x"),
    "y": NoiseModel.get_operator("pauli_y"),
    "z": NoiseModel.get_operator("pauli_z"),
}
_PAULI_GATES = {"x": XGate(), "y": YGate(), "z": ZGate()}


def _match_pauli_matrix(matrix: NDArray[np.complex128]) -> tuple[str, float]:
    """Identify a one-qubit Pauli matrix and its unit-modulus phase."""
    for name, reference in _PAULI_MATRICES.items():
        index = np.unravel_index(int(np.argmax(np.abs(reference))), reference.shape)
        phase = complex(matrix[index] / reference[index])
        if np.isclose(abs(phase), 1.0, atol=1e-10, rtol=0.0) and np.allclose(
            matrix, phase * reference, atol=1e-10, rtol=0.0
        ):
            return name, float(np.angle(phase))
    msg = "Operator does not match a one-qubit Pauli up to a unit-modulus phase."
    raise ValueError(msg)


def _match_pauli_product(matrix: NDArray[np.complex128]) -> tuple[tuple[str, str], float]:
    """Identify a two-qubit Pauli product and its unit-modulus phase."""
    for first, second in product(_PAULI_MATRICES, repeat=2):
        reference = np.kron(_PAULI_MATRICES[first], _PAULI_MATRICES[second])
        index = np.unravel_index(int(np.argmax(np.abs(reference))), reference.shape)
        phase = complex(matrix[index] / reference[index])
        if np.isclose(abs(phase), 1.0, atol=1e-10, rtol=0.0) and np.allclose(
            matrix, phase * reference, atol=1e-10, rtol=0.0
        ):
            return (first, second), float(np.angle(phase))
    msg = "Operator does not match a two-qubit Pauli product up to a unit-modulus phase."
    raise ValueError(msg)


def _unsupported_process_message(process: dict[str, Any]) -> str:
    return (
        "Stochastic circuit sampling supports only Pauli jump processes that can be represented as explicit "
        f"X, Y, or Z gates; process {process['name']!r} on sites {process['sites']} is unsupported."
    )


def _append_pauli_process(circuit: QuantumCircuit, process: dict[str, Any]) -> None:
    """Append one sampled Pauli process as explicit single-qubit gates."""
    if not is_pauli(process):
        raise ValueError(_unsupported_process_message(process))

    sites = [int(site) for site in process["sites"]]
    try:
        if len(sites) == 1:
            name, phase = _match_pauli_matrix(np.asarray(process["matrix"], dtype=np.complex128))
            names = (name,)
        elif "factors" in process:
            matches = [_match_pauli_matrix(np.asarray(factor, dtype=np.complex128)) for factor in process["factors"]]
            names = tuple(name for name, _phase in matches)
            phase = math.fsum(match_phase for _name, match_phase in matches)
        else:
            names, phase = _match_pauli_product(np.asarray(process["matrix"], dtype=np.complex128))
    except (KeyError, ValueError) as error:
        raise ValueError(_unsupported_process_message(process)) from error

    for name, site in zip(names, sites, strict=True):
        circuit.append(_PAULI_GATES[name], [site])
    circuit.global_phase += phase


def _sample_process(processes: Sequence[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any] | None:
    """Sample at most one process using the support-level jump convention."""
    if not processes:
        return None

    rates = np.asarray([float(process["strength"]) for process in processes], dtype=np.float64)
    for process, rate in zip(processes, rates, strict=True):
        if rate > 0.0 and not is_pauli(process):
            raise ValueError(_unsupported_process_message(process))

    max_rate = float(np.max(rates))
    if max_rate == 0.0:
        return None

    scaled_rates = rates / max_rate
    scaled_total = math.fsum(float(rate) for rate in scaled_rates)
    total_rate = max_rate * scaled_total
    event_probability = 1.0 if math.isinf(total_rate) else -math.expm1(-total_rate)
    if rng.random() >= event_probability:
        return None

    threshold = float(rng.random()) * scaled_total
    cumulative = 0.0
    last_positive_index = 0
    for index, rate in enumerate(scaled_rates):
        if rate > 0.0:
            last_positive_index = index
        cumulative += float(rate)
        if threshold < cumulative:
            return processes[index]
    return processes[last_positive_index]


def sample_stochastic_circuit(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """Sample one stochastic circuit realization from an ideal circuit.

    Distribution-valued process strengths are sampled once when construction
    starts, yielding one concrete noise-model realization for the returned
    trajectory. After each gate acting on one or two qubits, processes whose
    complete support is contained in that gate's qubit support are considered.
    If their rates are ``gamma_i``, an event occurs with probability
    ``1 - exp(-sum(gamma_i))`` and exactly one process is then selected with
    conditional probability ``gamma_i / sum(gamma_i)``.

    Only Pauli processes representable as explicit X, Y, or Z circuit gates are
    supported. The input circuit is not mutated, and native two-qubit gates are
    copied without decomposition. Scheduled jumps are not part of this
    gate-local preprocessing convention.

    Args:
        circuit: Ideal Qiskit circuit.
        noise_model: Standard YAQS noise model. Process strengths retain their
            meaning as nonnegative Lindblad rates.
        rng: Generator used for both one-time strength sampling and gate-level
            event/process sampling.

    Returns:
        A new circuit representing one stochastic trajectory.

    Raises:
        ValueError: If a relevant positive-rate process is not a Pauli jump that
            can be represented by explicit X, Y, or Z gates.
    """
    has_distributed_strength = any(isinstance(process["strength"], dict) for process in noise_model.processes)
    concrete_noise_model = noise_model.sample(rng=rng) if has_distributed_strength else noise_model
    stochastic_circuit = circuit.copy_empty_like()

    for instruction in circuit.data:
        qubits = [stochastic_circuit.qubits[circuit.find_bit(qubit).index] for qubit in instruction.qubits]
        clbits = [stochastic_circuit.clbits[circuit.find_bit(clbit).index] for clbit in instruction.clbits]
        stochastic_circuit.append(instruction.operation.copy(), qubits, clbits)

        if not isinstance(instruction.operation, Gate) or instruction.operation.num_qubits not in {1, 2}:
            continue

        gate_sites = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        local_processes = create_local_noise_model(concrete_noise_model, gate_sites).processes
        sampled_process = _sample_process(local_processes, rng)
        if sampled_process is not None:
            _append_pauli_process(stochastic_circuit, sampled_process)

    return stochastic_circuit
