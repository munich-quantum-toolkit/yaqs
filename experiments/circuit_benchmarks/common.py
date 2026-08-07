# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Shared state, evolution, and metric helpers for the circuit campaign."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import DigitalSimParams
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate, apply_two_qubit_gate

from .circuits import GateOp, TrotterStep, build_schedule, step_qiskit_circuit
from .config import (
    KRYLOV_TOL,
    METHOD_TO_GATE_MODE,
    SVD_THRESHOLD,
    TDVP_MODE,
    TRUNC_MODE,
    BenchmarkCase,
    Method,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from qiskit.dagcircuit import DAGOpNode


@dataclass(frozen=True)
class CompiledGate:
    """A protocol gate paired with its YAQS-compatible DAG node."""

    gate: GateOp
    node: DAGOpNode


@dataclass(frozen=True)
class CompiledStep:
    """One precompiled Trotter step, excluded from publication timings."""

    index: int
    gates: tuple[CompiledGate, ...]


def initial_basis_string(case: BenchmarkCase) -> str:
    """Return the MPS-chain-order product state for one benchmark case."""
    if case.initial_state == "zeros":
        return "0" * case.n_qubits
    if case.geometry == "1d":
        return "".join("0" if site % 2 == 0 else "1" for site in range(case.n_qubits))

    from .circuits import snake_index

    chain = ["0"] * case.n_qubits
    for row in range(case.rows):
        for col in range(case.cols):
            chain[snake_index(row, col, case.cols)] = "0" if (row + col) % 2 == 0 else "1"
    return "".join(chain)


def initial_mps(case: BenchmarkCase) -> MPS:
    """Construct the frozen product-state MPS for a benchmark case."""
    return MPS(case.n_qubits, state="basis", basis_string=initial_basis_string(case))


def initial_vector(case: BenchmarkCase) -> np.ndarray:
    """Return the dense vector from the same MPS used by approximate methods."""
    return np.asarray(initial_mps(case).to_vec(), dtype=np.complex128)


def compile_step(step: TrotterStep, n_qubits: int) -> CompiledStep:
    """Compile all gates before entering an accuracy or timing loop."""
    return CompiledStep(
        index=step.index,
        gates=tuple(CompiledGate(gate, gate.to_dag_node(n_qubits)) for gate in step.gates),
    )


def compile_schedule(schedule: Sequence[TrotterStep], n_qubits: int) -> tuple[CompiledStep, ...]:
    """Compile a complete schedule to YAQS-compatible DAG nodes."""
    return tuple(compile_step(step, n_qubits) for step in schedule)


def digital_params(method: Method, chi: int, *, n_sub: int) -> DigitalSimParams:
    """Build the common numerical settings for one MPS comparison method."""
    if method not in METHOD_TO_GATE_MODE:
        msg = f"Unknown comparison method {method!r}."
        raise ValueError(msg)
    if chi < 1:
        msg = "chi must be at least one."
        raise ValueError(msg)
    return DigitalSimParams(
        observables=[],
        get_state=True,
        preset="exact",
        max_bond_dim=int(chi),
        trunc_mode=TRUNC_MODE,
        svd_threshold=SVD_THRESHOLD,
        krylov_tol=KRYLOV_TOL,
        gate_mode=METHOD_TO_GATE_MODE[method],  # type: ignore[arg-type]
        tdvp_sweeps=int(n_sub),
        tdvp_mode=TDVP_MODE,
    )


def apply_mps_step(state: MPS, step: CompiledStep, sim_params: DigitalSimParams) -> None:
    """Apply one precompiled step in its specified gate order."""
    for compiled in step.gates:
        if len(compiled.gate.qubits) == 1:
            apply_single_qubit_gate(state, compiled.node)
        else:
            apply_two_qubit_gate(state, compiled.node, sim_params)
            # Match the noiseless production ``digital_tjm`` circuit path.
            state.normalize(form="B", decomposition="QR")


def apply_dense_step(vector: np.ndarray, step: TrotterStep, n_qubits: int) -> np.ndarray:
    """Apply the identical Trotter step with Qiskit dense statevector evolution."""
    circuit = step_qiskit_circuit(step, n_qubits)
    state = Statevector(np.asarray(vector, dtype=np.complex128).reshape(-1))
    return np.asarray(state.evolve(circuit).data, dtype=np.complex128)


def dense_reference_trajectory(
    case: BenchmarkCase,
    schedule: Sequence[TrotterStep] | None = None,
) -> np.ndarray:
    """Return the dense state at time zero and after every identical circuit step."""
    selected = build_schedule(case) if schedule is None else tuple(schedule)
    trajectory = [initial_vector(case)]
    step_circuits = [step_qiskit_circuit(step, case.n_qubits) for step in selected]
    state = Statevector(trajectory[0])
    for circuit in step_circuits:
        state = state.evolve(circuit)
        trajectory.append(np.asarray(state.data, dtype=np.complex128).copy())
    return np.stack(trajectory)


def normalized_state_fidelity(exact: np.ndarray, approx: np.ndarray) -> dict[str, float]:
    """Return phase-insensitive normalized fidelity and norm diagnostics."""
    exact_vec = np.asarray(exact, dtype=np.complex128).reshape(-1)
    approx_vec = np.asarray(approx, dtype=np.complex128).reshape(-1)
    exact_norm_sq = float(np.real(np.vdot(exact_vec, exact_vec)))
    approx_norm_sq = float(np.real(np.vdot(approx_vec, approx_vec)))
    if exact_norm_sq <= 0.0 or approx_norm_sq <= 0.0:
        msg = "Fidelity requires two nonzero statevectors."
        raise ValueError(msg)
    fidelity = float(abs(np.vdot(exact_vec, approx_vec)) ** 2 / (exact_norm_sq * approx_norm_sq))
    if fidelity < -1e-12 or fidelity > 1.0 + 1e-12:
        msg = f"Normalized fidelity {fidelity} lies outside [0, 1] beyond roundoff."
        raise ValueError(msg)
    fidelity = min(1.0, max(0.0, fidelity))
    exact_norm = float(np.sqrt(exact_norm_sq))
    approx_norm = float(np.sqrt(approx_norm_sq))
    return {
        "fidelity_normalized": fidelity,
        "infidelity_normalized": 1.0 - fidelity,
        "norm_exact": exact_norm,
        "norm_approx": approx_norm,
        "norm_drift": approx_norm - exact_norm,
    }


def phase_aligned_distance(reference: np.ndarray, state: np.ndarray) -> float:
    """Return the Euclidean distance after removing a global phase."""
    reference_vec = np.asarray(reference, dtype=np.complex128).reshape(-1)
    state_vec = np.asarray(state, dtype=np.complex128).reshape(-1)
    phase = np.vdot(state_vec, reference_vec)
    if abs(phase) > 0.0:
        state_vec = state_vec * phase / abs(phase)
    return float(np.linalg.norm(reference_vec - state_vec))


def bond_profile(state: MPS) -> list[int]:
    """Return the full MPS bond profile, including unit boundary bonds."""
    if not state.tensors:
        return []
    return [int(state.tensors[0].shape[1]), *(int(tensor.shape[2]) for tensor in state.tensors)]


def parameter_count(state: MPS) -> int:
    """Return the number of stored complex MPS coefficients."""
    return int(sum(np.asarray(tensor).size for tensor in state.tensors))


def parameter_count_from_profile(profile: Sequence[int], *, local_dimension: int = 2) -> int:
    """Return the MPS parameter count implied by a bond profile."""
    if len(profile) < 2:
        return 0
    return int(
        sum(
            local_dimension * left * right
            for left, right in itertools.pairwise(profile)
        )
    )


def protocol_metadata(case: BenchmarkCase, schedule: Sequence[TrotterStep]) -> dict[str, Any]:
    """Return compact physical metadata suitable for manifests and output rows."""
    from .circuits import circuit_fingerprint, physical_edges

    return {
        "case": case.key,
        "model": case.model,
        "geometry": case.geometry,
        "rows": case.rows,
        "cols": case.cols,
        "n_qubits": case.n_qubits,
        "initial_basis_string": initial_basis_string(case),
        "n_edges": len(physical_edges(case)),
        "n_steps": len(schedule),
        "circuit_fingerprint": circuit_fingerprint(case, schedule),
    }
