# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Construct stochastic circuit realizations from standard YAQS noise models."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import XGate, YGate, ZGate

from .digital_tjm import create_local_noise_model

if TYPE_CHECKING:
    import numpy as np

    from ..core.data_structures.noise_model import NoiseModel


_ONE_QUBIT_PAULIS = {
    "pauli_x": "x",
    "pauli_y": "y",
    "pauli_z": "z",
    "x": "x",
    "y": "y",
    "z": "z",
}
_PAULI_GATES = {"x": XGate(), "y": YGate(), "z": ZGate()}
_PAULI_ERROR = "Explicit stochastic circuit sampling supports recognized YAQS Pauli processes only."


def _pauli_labels(process: dict[str, Any]) -> tuple[str, ...] | None:
    """Decode a recognized YAQS Pauli process name.

    Returns:
        One Pauli label per process site, or ``None`` if unsupported.
    """
    name = str(process["name"])
    if len(process["sites"]) == 1:
        label = _ONE_QUBIT_PAULIS.get(name)
        return (label,) if label is not None else None

    if name.startswith("crosstalk_"):
        labels = name.removeprefix("crosstalk_")
        if len(labels) == 2 and set(labels) <= {"x", "y", "z"}:
            return labels[0], labels[1]
    return None


def _validate_pauli_noise_model(noise_model: NoiseModel) -> None:
    """Validate that a noise model can be materialized as Pauli gates.

    Raises:
        ValueError: If scheduled jumps or positive-rate non-Pauli processes are present.
    """
    if noise_model.scheduled_jumps:
        msg = "Explicit stochastic circuit sampling does not support scheduled jumps."
        raise ValueError(msg)
    if any(float(process["strength"]) > 0.0 and _pauli_labels(process) is None for process in noise_model.processes):
        raise ValueError(_PAULI_ERROR)


def sample_stochastic_circuit(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """Sample one explicit Pauli-noise circuit realization.

    Distribution-valued strengths are resolved once per helper call. Every
    original one- or two-qubit gate is a noise opportunity, and processes are
    eligible when their complete support is contained in the gate support. The
    input circuit is copied without decomposition.

    Args:
        circuit: Circuit to sample.
        noise_model: Existing YAQS noise model containing recognized Pauli processes.
        rng: Random-number generator for disorder, event, and process draws.

    Returns:
        A concrete circuit containing the original gates and sampled Pauli gates.
    """
    if any(isinstance(process["strength"], dict) for process in noise_model.processes):
        noise_model = noise_model.sample(rng=rng)
    _validate_pauli_noise_model(noise_model)

    sampled_circuit = circuit.copy_empty_like()
    for instruction in circuit.data:
        sites = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        qubits = [sampled_circuit.qubits[site] for site in sites]
        clbits = [sampled_circuit.clbits[circuit.find_bit(clbit).index] for clbit in instruction.clbits]
        sampled_circuit.append(instruction.operation.copy(), qubits, clbits)

        if not isinstance(instruction.operation, Gate) or instruction.operation.num_qubits not in {1, 2}:
            continue

        processes = create_local_noise_model(noise_model, sites).processes
        rates = [float(process["strength"]) for process in processes]
        total_rate = sum(rates)
        if not total_rate or rng.random() >= -math.expm1(-total_rate):
            continue

        threshold = float(rng.random()) * total_rate
        cumulative = 0.0
        selected = next(process for process, rate in zip(reversed(processes), reversed(rates), strict=True) if rate > 0)
        for process, rate in zip(processes, rates, strict=True):
            cumulative += rate
            if threshold < cumulative:
                selected = process
                break

        labels = _pauli_labels(selected)
        assert labels is not None
        for label, site in zip(labels, selected["sites"], strict=True):
            sampled_circuit.append(_PAULI_GATES[label], [sampled_circuit.qubits[int(site)]])

    return sampled_circuit
