# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Construct stochastic circuit realizations from standard YAQS noise models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.circuit import Gate, QuantumCircuit
from qiskit.circuit.library import XGate, YGate, ZGate

from ..core.data_structures.noise_model import NoiseModel, _identify_pauli_process
from ..core.random_utils import make_trajectory_rng
from .digital_tjm import _digital_tjm_impl, create_local_noise_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ..core.data_structures.mps import MPS
    from ..core.data_structures.simulation_parameters import DigitalSimParams


_PAULI_GATES = {"x": XGate(), "y": YGate(), "z": ZGate()}


@dataclass(frozen=True)
class _StochasticCircuitSchedule:
    """Pair a circuit realization with any execution-time noise.

    Pauli noise is materialized in ``circuit``. Models containing dissipative
    processes remain in ``post_gate_noise_model`` for state-dependent execution.
    """

    circuit: QuantumCircuit
    post_gate_noise_model: NoiseModel | None


def _unsupported_process_message(process: dict[str, Any]) -> str:
    return (
        "Stochastic circuit sampling supports only Pauli jump processes that can be represented as explicit "
        f"X, Y, or Z gates; process {process['name']!r} on sites {process['sites']} is unsupported."
    )


def _append_pauli_process(circuit: QuantumCircuit, process: dict[str, Any]) -> None:
    """Append one sampled Pauli process as explicit single-qubit gates.

    Raises:
        ValueError: If the process cannot be represented by explicit Pauli gates.
    """
    sites = [int(site) for site in process["sites"]]
    match = _identify_pauli_process(process)
    if match is None:
        raise ValueError(_unsupported_process_message(process))
    names, phase = match

    for name, site in zip(names, sites, strict=True):
        circuit.append(_PAULI_GATES[name], [site])
    circuit.global_phase += phase


def _sample_process(processes: Sequence[dict[str, Any]], rng: np.random.Generator) -> dict[str, Any] | None:
    """Sample at most one process using the support-level jump convention.

    Returns:
        The selected process, or ``None`` if no event occurs.

    Raises:
        ValueError: If a positive-rate process cannot be represented by explicit Pauli gates.
    """
    if not processes:
        return None

    rates = np.asarray([float(process["strength"]) for process in processes], dtype=np.float64)
    for process, rate in zip(processes, rates, strict=True):
        if rate > 0.0 and _identify_pauli_process(process) is None:
            raise ValueError(_unsupported_process_message(process))

    max_rate = float(np.max(rates))
    if not max_rate:
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


def _sample_concrete_noise_model(noise_model: NoiseModel, rng: np.random.Generator) -> NoiseModel:
    """Resolve distribution-valued strengths for one circuit realization.

    Returns:
        A model with concrete strengths, or the original concrete model.
    """
    has_distributed_strength = any(isinstance(process["strength"], dict) for process in noise_model.processes)
    return noise_model.sample(rng=rng) if has_distributed_strength else noise_model


def sample_stochastic_circuit(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    rng: np.random.Generator,
) -> QuantumCircuit:
    """Sample one Pauli-noise circuit realization.

    After every one- or two-qubit gate, processes supported on the gate qubits
    are sampled using the same rate convention as digital TJM with ``dt=1``.
    The input circuit is copied without decomposition.

    Args:
        circuit: Circuit to sample.
        noise_model: Noise model containing Pauli processes.
        rng: Random-number generator.

    Returns:
        Sampled circuit realization.
    """
    concrete_noise_model = _sample_concrete_noise_model(noise_model, rng)
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


def _sample_stochastic_schedule(
    circuit: QuantumCircuit,
    noise_model: NoiseModel,
    rng: np.random.Generator,
) -> _StochasticCircuitSchedule:
    """Construct one post-gate stochastic realization.

    Pauli-only models are sampled into explicit gates. Models containing a
    positive-rate dissipative process remain intact for digital TJM execution.

    Args:
        circuit: Circuit to sample.
        noise_model: Concrete noise model for the run.
        rng: Trajectory random-number generator.

    Returns:
        Circuit realization and optional execution-time noise model.
    """
    requires_state_dependent_execution = any(
        float(process["strength"]) > 0.0 and _identify_pauli_process(process) is None
        for process in noise_model.processes
    )
    if requires_state_dependent_execution:
        return _StochasticCircuitSchedule(circuit.copy(), noise_model)

    return _StochasticCircuitSchedule(sample_stochastic_circuit(circuit, noise_model, rng), None)


def _run_stochastic_trajectory(
    initial_state: MPS,
    schedule: _StochasticCircuitSchedule,
    sim_params: DigitalSimParams,
    trajectory_index: int = 0,
    *,
    copy_initial_state: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, dict[int, int] | None, MPS | None]:
    """Execute one stochastic circuit trajectory with digital TJM.

    Args:
        initial_state: Initial MPS for this trajectory.
        schedule: Sampled stochastic realization.
        sim_params: Digital simulation parameters.
        trajectory_index: Index used for trajectory seeding and shot allocation.
        copy_initial_state: Whether to copy the initial MPS.
        rng: Optional trajectory random-number generator.

    Returns:
        Observable values, diagnostics, counts, and an optional final MPS.
    """
    execution_noise_model = schedule.post_gate_noise_model
    return _digital_tjm_impl(
        (trajectory_index, initial_state, execution_noise_model, sim_params, schedule.circuit),
        copy_initial_state=copy_initial_state,
        rng=rng,
        compiled_circuit=None,
        post_gate_noise=execution_noise_model is not None,
    )


def _sample_and_run_stochastic_trajectory(
    args: tuple[int, MPS, NoiseModel | None, DigitalSimParams, QuantumCircuit],
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None, dict[int, int] | None, MPS | None]:
    """Construct and execute one schedule for the simulator's ensemble loop.

    Returns:
        The unaggregated output of one digital trajectory.
    """
    trajectory_index, initial_state, noise_model, sim_params, circuit = args
    rng = make_trajectory_rng(trajectory_index, base_seed=sim_params.random_seed)
    schedule = _sample_stochastic_schedule(circuit, noise_model or NoiseModel(), rng)
    return _run_stochastic_trajectory(
        initial_state,
        schedule,
        sim_params,
        trajectory_index,
        rng=rng,
    )
