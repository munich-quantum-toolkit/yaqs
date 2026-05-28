# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Hybrid TDVP benchmark + diagnostic helpers.

This module intentionally lives under `benchmarks/` (not `tests/`) so the
benchmark can be used as a *diagnostic suite* without being coupled to pytest.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate, apply_two_qubit_gate

GateMode = Literal["hybrid", "tebd", "tdvp"]
SVD_THRESHOLD = 1e-14

PASS_TOL = 1e-10
WARN_TOL = 1e-6


def build_probes() -> dict[str, QuantumCircuit]:
    """Named probe circuits for hybrid long-range TDVP evaluation (CNOT-style)."""
    probes: dict[str, QuantumCircuit] = {}

    qc = QuantumCircuit(4)
    qc.h(1)
    qc.cx(1, 3)
    probes["interior_lr_4q"] = qc

    qc = QuantumCircuit(8)
    qc.h(3)
    qc.cx(3, 7)
    probes["interior_lr_8q"] = qc

    qc = QuantumCircuit(8)
    qc.h(0)
    qc.cx(0, 7)
    probes["endpoint_lr_8q"] = qc

    qc = QuantumCircuit(8)
    qc.h(4)
    qc.cx(4, 0)
    probes["mid_ctrl_8q"] = qc

    qc = QuantumCircuit(6)
    qc.h(1)
    qc.cx(1, 5)
    qc.h(2)
    qc.cx(2, 4)
    probes["double_lr_6q"] = qc

    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 3)
    probes["mixed_nn_lr_4q"] = qc

    qc = QuantumCircuit(6)
    qc.h(0)
    for i in range(5):
        qc.cx(i, i + 1)
    probes["ghz_chain_6q"] = qc

    return probes


# Single-qubit Ry(π/4) plus two-qubit RZZ(θ) — analogues of H + long-range CX in the CNOT suite.
_ROTATION_PREP_ANGLE = float(np.pi / 4)
_ROTATION_ENTANGLE_ANGLE = 0.42
_ROTATION_NN_ANGLE = 0.22


def build_rotation_probes() -> dict[str, QuantumCircuit]:
    """Rotation-only probes (legacy broad suite)."""
    probes: dict[str, QuantumCircuit] = {}
    prep = _ROTATION_PREP_ANGLE
    ent = _ROTATION_ENTANGLE_ANGLE
    nn = _ROTATION_NN_ANGLE

    qc = QuantumCircuit(4)
    qc.ry(prep, 1)
    qc.rzz(ent, 1, 3)
    probes["interior_lr_4q"] = qc

    qc = QuantumCircuit(8)
    qc.ry(prep, 3)
    qc.rzz(ent, 3, 7)
    probes["interior_lr_8q"] = qc

    qc = QuantumCircuit(8)
    qc.ry(prep, 0)
    qc.rzz(ent, 0, 7)
    probes["endpoint_lr_8q"] = qc

    qc = QuantumCircuit(8)
    qc.ry(prep, 4)
    qc.rzz(ent, 4, 0)
    probes["mid_ctrl_8q"] = qc

    qc = QuantumCircuit(6)
    qc.ry(prep, 1)
    qc.rzz(ent, 1, 5)
    qc.ry(prep, 2)
    qc.rzz(ent, 2, 4)
    probes["double_lr_6q"] = qc

    qc = QuantumCircuit(4)
    qc.ry(prep, 0)
    qc.rzz(nn, 0, 1)
    qc.rzz(ent, 0, 3)
    probes["mixed_nn_lr_4q"] = qc

    qc = QuantumCircuit(6)
    qc.ry(prep, 0)
    for i in range(5):
        qc.rzz(prep, i, i + 1)
    probes["nn_rotation_chain_6q"] = qc

    qc = QuantumCircuit(4)
    qc.rx(0.37, 0)
    qc.ry(0.51, 1)
    qc.rz(0.23, 2)
    qc.rxx(ent, 0, 2)
    probes["single_qubit_plus_nn_rxx_4q"] = qc

    qc = QuantumCircuit(4)
    qc.rx(0.37, 0)
    qc.ry(0.51, 1)
    qc.rz(0.23, 2)
    qc.ryy(ent, 1, 3)
    probes["single_qubit_plus_lr_ryy_4q"] = qc

    # The known failing regression in the broad suite.
    qc = QuantumCircuit(12)
    qc.ry(prep, 1)
    qc.ry(prep, 4)
    qc.ry(prep, 7)
    qc.rzz(0.19, 1, 10)
    qc.rzz(0.27, 4, 11)
    qc.rzz(0.33, 0, 7)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    probes["lr_stack_mixed_12q"] = qc

    return probes


def build_rotation_diagnostic_probes() -> dict[str, QuantumCircuit]:
    """Minimal diagnostic probes for long-range Pauli rotations.

    Each circuit is intended to test a single hypothesis.
    """
    probes: dict[str, QuantumCircuit] = {}
    pi = float(np.pi)

    # Group A: RZZ phase-like cases
    qc = QuantumCircuit(6)
    qc.ry(pi / 4, 0)
    qc.rzz(0.42, 0, 5)
    probes["rzz_one_active_lr_6q"] = qc

    qc = QuantumCircuit(6)
    qc.ry(pi / 4, 0)
    qc.ry(pi / 5, 5)
    qc.rzz(0.42, 0, 5)
    probes["rzz_both_active_lr_6q"] = qc

    # Group B: RXX/RYY pair creation from product states
    qc = QuantumCircuit(6)
    qc.rxx(0.42, 0, 5)
    probes["rxx_vacuum_lr_6q"] = qc

    qc = QuantumCircuit(6)
    qc.ryy(0.42, 0, 5)
    probes["ryy_vacuum_lr_6q"] = qc

    # Group C: RXX/RYY with one or both endpoints pre-rotated
    qc = QuantumCircuit(6)
    qc.ry(pi / 4, 0)
    qc.rxx(0.42, 0, 5)
    probes["rxx_one_active_lr_6q"] = qc

    qc = QuantumCircuit(6)
    qc.ry(pi / 4, 0)
    qc.ryy(0.42, 0, 5)
    probes["ryy_one_active_lr_6q"] = qc

    qc = QuantumCircuit(6)
    qc.ry(pi / 4, 0)
    qc.ry(pi / 5, 5)
    qc.rxx(0.42, 0, 5)
    probes["rxx_both_active_lr_6q"] = qc

    qc = QuantumCircuit(6)
    qc.ry(pi / 4, 0)
    qc.ry(pi / 5, 5)
    qc.ryy(0.42, 0, 5)
    probes["ryy_both_active_lr_6q"] = qc

    # Group D: repeated same-axis long-range rotations
    qc = QuantumCircuit(6)
    qc.ry(pi / 4, 0)
    qc.ry(pi / 5, 5)
    for theta in (0.05, 0.10, 0.20, 0.40):
        qc.rzz(theta, 0, 5)
    probes["rzz_repeated_same_pair_6q"] = qc

    qc = QuantumCircuit(6)
    qc.ry(pi / 4, 0)
    qc.ry(pi / 5, 5)
    for theta in (0.05, 0.10, 0.20, 0.40):
        qc.rxx(theta, 0, 5)
    probes["rxx_repeated_same_pair_6q"] = qc

    # Group E: mixed-axis same-pair rotations
    qc = QuantumCircuit(6)
    qc.ry(pi / 4, 0)
    qc.ry(pi / 5, 5)
    qc.rzz(0.19, 0, 5)
    qc.rxx(0.21, 0, 5)
    qc.ryy(0.25, 0, 5)
    probes["mixed_axes_same_pair_6q"] = qc

    # Group F: disjoint long-range rotations
    qc = QuantumCircuit(8)
    qc.ry(pi / 4, 0)
    qc.ry(pi / 5, 7)
    qc.rzz(0.19, 0, 7)
    qc.rxx(0.21, 1, 6)
    qc.ryy(0.25, 2, 5)
    probes["mixed_axes_disjoint_pairs_8q"] = qc

    # Group G: minimal versions of current failing mixed stack
    qc = QuantumCircuit(10)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    probes["stack_core_rxx_ryy_vacuum_10q"] = qc

    qc = QuantumCircuit(10)
    for q in (2, 3, 8, 9):
        qc.ry(pi / 4, q)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    probes["stack_core_rxx_ryy_active_10q"] = qc

    qc = QuantumCircuit(10)
    qc.ry(pi / 4, 1)
    qc.ry(pi / 5, 4)
    qc.rzz(0.19, 1, 8)
    qc.rzz(0.27, 4, 9)
    qc.rxx(0.21, 2, 7)
    qc.ryy(0.25, 3, 6)
    probes["minimal_mixed_stack_10q"] = qc

    # Keep the original broad-suite regression as final probe.
    probes["lr_stack_mixed_12q"] = build_rotation_probes()["lr_stack_mixed_12q"]

    return probes


@dataclass(frozen=True)
class BenchmarkSuiteConfig:
    """Configuration for a full hybrid benchmark report."""

    title: str
    description: str
    basename: str
    probes: Callable[[], dict[str, QuantumCircuit]]
    gate_family_note: str
    sweep_counts: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    pad_values: tuple[int | None, ...] = (None, 2, 4, 8, 16)
    sweep_probes: tuple[str, ...] = ()
    padding_probes: tuple[str, ...] = ()
    takeaways: tuple[str, ...] = ()


DEFAULT_BENCHMARK_CONFIG = BenchmarkSuiteConfig(
    title="Hybrid TDVP Benchmark Report",
    description="Hybrid MPS circuit simulation benchmark vs Qiskit Statevector (CNOT-style probes)",
    basename="hybrid_tdvp_benchmark",
    probes=build_probes,
    gate_family_note="Discrete Clifford gates (H, CX) plus NN chains where noted",
    sweep_probes=("interior_lr_4q", "interior_lr_8q", "mid_ctrl_8q", "double_lr_6q"),
    padding_probes=("interior_lr_4q", "interior_lr_8q", "mid_ctrl_8q", "double_lr_6q"),
    takeaways=(
        "Fidelity vs observables: Z-only checks can miss errors; include Y (and X) in the panel.",
        "Padding: chi=1 (pad=None, direct path) fails interior LR probes; chi>=2 fixes single-LR cases.",
        "Sweeps: help mid-control partial LR converge in fidelity; do not fix double-LR or symmetric mode.",
        "TEBD reference: use gate_mode=tebd rows in baseline for exact comparison on hard probes.",
    ),
)


ROTATION_DIAGNOSTIC_CONFIG = BenchmarkSuiteConfig(
    title="Hybrid TDVP Rotation Diagnostic Benchmark",
    description="Minimal diagnostic probes for long-range Pauli rotations",
    basename="hybrid_rotation_tdvp_diagnostic",
    probes=build_rotation_diagnostic_probes,
    gate_family_note="Minimal RX/RY/RZ/RXX/RYY/RZZ probes for TDVP padding/sweep/failure diagnosis",
    sweep_counts=(1, 2, 4, 8, 16, 32),
    pad_values=(None, 2, 4, 8, 16),
)


ROTATION_BENCHMARK_CONFIG = BenchmarkSuiteConfig(
    title="Hybrid TDVP Rotation-Gate Benchmark Report",
    description="Hybrid MPS circuit simulation with rotation-only probes vs Qiskit Statevector",
    basename="hybrid_rotation_tdvp_benchmark",
    probes=build_rotation_probes,
    gate_family_note="Rotation-only RX/RY/RZ/RXX/RYY/RZZ probes (broad suite)",
    sweep_counts=(1, 2, 4, 8, 16, 32),
    pad_values=(None, 2, 4, 8, 16),
    sweep_probes=("lr_stack_mixed_12q",),
    padding_probes=("lr_stack_mixed_12q",),
    takeaways=(
        "Use the diagnostic suite for minimal debuggable probes.",
        "This broad suite keeps `lr_stack_mixed_12q` as a regression case.",
    ),
)


def pauli_label(num_qubits: int, site: int, letter: str) -> str:
    """Qiskit Pauli string for a single-site Pauli (site ``i`` = Qiskit qubit ``i``)."""
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - site] = letter
    return "".join(chars)


def xyz_observable_panel(num_qubits: int) -> list[Observable]:
    """Pauli X, Y, Z on every qubit."""
    observables: list[Observable] = []
    for site in range(num_qubits):
        observables.extend([Observable(X(), site), Observable(Y(), site), Observable(Z(), site)])
    return observables


def qiskit_expectation_from_vector(state_vector: np.ndarray, qc: QuantumCircuit, obs: Observable) -> float:
    """Pauli expectation for ``state_vector`` using Qiskit conventions."""
    site = obs.sites[0] if isinstance(obs.sites, list) else obs.sites
    letter = obs.gate.name.upper()
    pauli = Pauli(pauli_label(qc.num_qubits, site, letter))
    return float(np.real(Statevector(state_vector).expectation_value(pauli)))


def qiskit_expectation(qc: QuantumCircuit, obs: Observable) -> float:
    """Reference Pauli expectation from Qiskit ``Statevector``."""
    return qiskit_expectation_from_vector(np.asarray(Statevector(qc).data, dtype=np.complex128), qc, obs)


def fidelity_vs_qiskit(qc: QuantumCircuit, simulated: np.ndarray) -> float:
    """Squared overlap fidelity against Qiskit ``Statevector``."""
    reference = np.asarray(Statevector(qc).data, dtype=np.complex128)
    return float(abs(np.vdot(reference, simulated)) ** 2)


def max_virtual_bond_dim(mps: MPS) -> int:
    """Largest virtual bond dimension in an MPS."""
    bond_max = 1
    for tensor in mps.tensors:
        _, chi_l, chi_r = tensor.shape
        bond_max = max(bond_max, chi_l, chi_r)
    return bond_max


@dataclass
class WorstObservable:
    """Largest single Pauli expectation deviation."""

    name: str
    site: int
    reference: float
    simulated: float
    abs_error: float


@dataclass
class BenchmarkRow:
    """One benchmark evaluation point (one circuit, one path/config)."""

    probe: str
    num_qubits: int
    gate_mode: GateMode
    tdvp_sweeps: int
    tdvp_circuit_full_sweep: bool
    initial_pad: int | None
    path: Literal["simulator", "direct_mps"]
    initial_bond_dim: int
    fidelity: float
    fidelity_error: float
    obs_max_error: float
    obs_l2_error: float
    obs_mean_error: float
    worst_observable: WorstObservable | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _observable_errors_from_vector(qc: QuantumCircuit, vec: np.ndarray) -> tuple[float, float, float, WorstObservable]:
    panel = xyz_observable_panel(qc.num_qubits)
    errors: list[float] = []
    worst: WorstObservable | None = None
    for obs in panel:
        ref = qiskit_expectation(qc, obs)
        sim = qiskit_expectation_from_vector(vec, qc, obs)
        err = abs(sim - ref)
        errors.append(err)
        if worst is None or err > worst.abs_error:
            site = obs.sites[0] if isinstance(obs.sites, list) else obs.sites
            worst = WorstObservable(obs.gate.name, int(site), float(ref), float(sim), float(err))
    arr = np.asarray(errors, dtype=float)
    return float(np.max(arr)), float(np.linalg.norm(arr)), float(np.mean(arr)), worst or WorstObservable("?", -1, 0.0, 0.0, 0.0)


def evaluate_row(
    probe_id: str,
    qc: QuantumCircuit,
    vec: np.ndarray,
    *,
    gate_mode: GateMode,
    tdvp_sweeps: int,
    tdvp_circuit_full_sweep: bool,
    initial_pad: int | None,
    path: Literal["simulator", "direct_mps"],
    initial_bond_dim: int,
    meta: dict[str, Any] | None = None,
) -> BenchmarkRow:
    fidelity = fidelity_vs_qiskit(qc, vec)
    obs_max, obs_l2, obs_mean, worst = _observable_errors_from_vector(qc, vec)
    return BenchmarkRow(
        probe=probe_id,
        num_qubits=qc.num_qubits,
        gate_mode=gate_mode,
        tdvp_sweeps=tdvp_sweeps,
        tdvp_circuit_full_sweep=tdvp_circuit_full_sweep,
        initial_pad=initial_pad,
        path=path,
        initial_bond_dim=initial_bond_dim,
        fidelity=float(fidelity),
        fidelity_error=float(1.0 - fidelity),
        obs_max_error=float(obs_max),
        obs_l2_error=float(obs_l2),
        obs_mean_error=float(obs_mean),
        worst_observable=worst,
        meta=meta or {},
    )


def row_to_dict(row: BenchmarkRow) -> dict[str, Any]:
    data = asdict(row)
    for key in ("fidelity", "fidelity_error", "obs_max_error", "obs_l2_error", "obs_mean_error"):
        data[key] = float(data[key])
    if row.worst_observable is not None:
        for key in ("reference", "simulated", "abs_error"):
            data["worst_observable"][key] = float(data["worst_observable"][key])
    return data


def _as_gate_list_text(qc: QuantumCircuit) -> str:
    lines: list[str] = []
    for inst, qargs, _ in qc.data:
        q_text = ",".join(f"q{qc.find_bit(q).index}" for q in qargs)
        if inst.params:
            params = ",".join(str(p) for p in inst.params)
            lines.append(f"{inst.name}({params}) {q_text}")
        else:
            lines.append(f"{inst.name} {q_text}")
    return "\n".join(lines)


def debug_gate_by_gate(
    qc: QuantumCircuit,
    *,
    mode: GateMode = "hybrid",
    pad: int | None = None,
    normalize_after_gate: bool = True,
) -> None:
    """Apply a circuit gate-by-gate on the direct MPS path and compare to Qiskit after each gate."""
    qiskit_prefix = QuantumCircuit(qc.num_qubits)
    state = State(qc.num_qubits, initial="zeros", representation="mps", pad=pad)
    mps = state.mps

    params = StrongSimParams(
        svd_threshold=SVD_THRESHOLD,
        preset="exact",
        gate_mode=mode,
        tdvp_sweeps=1,
        tdvp_circuit_full_sweep=False,
        get_state=True,
    )

    dag = circuit_to_dag(qc)
    for k, node in enumerate(dag.topological_op_nodes(), start=1):
        q_indices = [qc.find_bit(q).index for q in node.qargs]
        qiskit_prefix.append(
            node.op,
            [qiskit_prefix.qubits[qc.find_bit(q).index] for q in node.qargs],
        )

        if node.op.num_qubits == 1:
            apply_single_qubit_gate(mps, node)
        elif node.op.num_qubits == 2:
            vec_before = np.asarray(mps.to_vec(), dtype=np.complex128).copy()
            apply_two_qubit_gate(mps, node, params)
            if normalize_after_gate:
                # Mirror the strong-sim circuit path which re-normalizes after each 2q gate.
                mps.normalize(form="B", decomposition="QR")
            vec_after = np.asarray(mps.to_vec(), dtype=np.complex128).copy()
            norm_delta = float(np.linalg.norm(vec_after - vec_before))
            overlap = float(abs(np.vdot(vec_before, vec_after)) ** 2)
        else:
            msg = f"Unsupported gate with {node.op.num_qubits} qubits: {node.op.name}"
            raise ValueError(msg)

        vec = np.asarray(mps.to_vec(), dtype=np.complex128)
        ref = np.asarray(Statevector(qiskit_prefix).data, dtype=np.complex128)
        fid = float(abs(np.vdot(ref, vec)) ** 2)

        obs_max, obs_l2, obs_mean, worst = _observable_errors_from_vector(qiskit_prefix, vec)

        print("step", k, node.op.name, q_indices)
        print("  fidelity_error", float(1.0 - fid))
        print("  obs_max", float(obs_max))
        print("  worst", worst)
        print("  bond_max", max_virtual_bond_dim(mps))

        if node.op.num_qubits == 2:
            print("  norm_delta", norm_delta)
            print("  overlap_before_after", overlap)

            # Endpoint panel (use Qiskit convention, on prefix state).
            for site in sorted(set(q_indices)):
                for letter in ("X", "Y", "Z"):
                    obs = Observable({"X": X, "Y": Y, "Z": Z}[letter](), int(site))
                    ref_ev = qiskit_expectation(qiskit_prefix, obs)
                    sim_ev = qiskit_expectation_from_vector(vec, qiskit_prefix, obs)
                    print(f"  <{letter}> q{site}: sim={sim_ev:+.6f} ref={ref_ev:+.6f} err={abs(sim_ev-ref_ev):.2e}")


def run_simulator_hybrid(
    qc: QuantumCircuit,
    *,
    gate_mode: GateMode = "hybrid",
    tdvp_sweeps: int = 4,
    tdvp_circuit_full_sweep: bool = False,
) -> tuple[np.ndarray, Simulator, State, int]:
    params = StrongSimParams(
        svd_threshold=SVD_THRESHOLD,
        preset="exact",
        get_state=True,
        gate_mode=gate_mode,
        tdvp_sweeps=tdvp_sweeps,
        tdvp_circuit_full_sweep=tdvp_circuit_full_sweep,
    )
    sim = Simulator()
    initial_state = State(qc.num_qubits, initial="zeros", representation="mps")
    result = sim.run(initial_state, qc, params)
    if result.output_state is None:
        msg = "Simulator did not return an output_state (get_state must be True)."
        raise RuntimeError(msg)
    vec = np.asarray(result.output_state.mps.to_vec(), dtype=np.complex128)
    init_bond = max_virtual_bond_dim(result.output_state.mps)
    return vec, sim, result.output_state, int(init_bond)


def run_direct_hybrid(
    qc: QuantumCircuit,
    *,
    initial_pad: int | None,
    tdvp_sweeps: int = 4,
    tdvp_circuit_full_sweep: bool = False,
) -> tuple[np.ndarray, MPS, int]:
    # Direct MPS path: no simulator auto-pad; initial_pad=None means chi=1.
    params = StrongSimParams(
        svd_threshold=SVD_THRESHOLD,
        preset="exact",
        gate_mode="hybrid",
        tdvp_sweeps=tdvp_sweeps,
        tdvp_circuit_full_sweep=tdvp_circuit_full_sweep,
    )
    mps = State(qc.num_qubits, initial="zeros", representation="mps", pad=initial_pad).mps
    init_bond = max_virtual_bond_dim(mps)

    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        if node.op.num_qubits == 1:
            apply_single_qubit_gate(mps, node)
        elif node.op.num_qubits == 2:
            apply_two_qubit_gate(mps, node, params)
        else:
            msg = f"Unsupported gate with {node.op.num_qubits} qubits: {node.op.name}"
            raise ValueError(msg)

    vec = np.asarray(mps.to_vec(), dtype=np.complex128)
    return vec, mps, int(init_bond)


def _passes(row: BenchmarkRow) -> bool:
    return abs(row.fidelity_error) < PASS_TOL and row.obs_max_error < PASS_TOL


def _best_by(rows: Iterable[BenchmarkRow], key: str) -> BenchmarkRow:
    return min(rows, key=lambda r: getattr(r, key))


def classify_probe(
    *,
    probe_id: str,
    qc: QuantumCircuit,
    rows_for_probe: list[BenchmarkRow],
    sweep_counts: tuple[int, ...],
    pad_values: tuple[int | None, ...],
) -> dict[str, Any]:
    """Classify a probe from its rows."""
    default_rows = [
        r
        for r in rows_for_probe
        if r.path == "simulator"
        and r.gate_mode == "hybrid"
        and r.tdvp_sweeps == 4
        and (not r.tdvp_circuit_full_sweep)
    ]
    default = default_rows[0] if default_rows else None
    passes_default = bool(default and _passes(default))

    # Padding: direct MPS, fixed sweeps=4
    pad_rows = {
        r.initial_pad: r
        for r in rows_for_probe
        if r.path == "direct_mps" and r.gate_mode == "hybrid" and r.tdvp_sweeps == 4
    }
    pad_none = pad_rows.get(None)
    pad2 = pad_rows.get(2)

    needs_padding = bool(pad_none and not _passes(pad_none) and pad2 and _passes(pad2))
    padding_does_not_help = False
    if pad_rows:
        if any(not _passes(r) for r in pad_rows.values()) and all(not _passes(r) for r in pad_rows.values()):
            padding_does_not_help = True

    # Sweeps: simulator hybrid, varying tdvp_sweeps
    sweep_rows = [
        r
        for r in rows_for_probe
        if r.path == "simulator"
        and r.gate_mode == "hybrid"
        and (not r.tdvp_circuit_full_sweep)
        and r.tdvp_sweeps in sweep_counts
    ]
    sweep_by = {r.tdvp_sweeps: r for r in sweep_rows}
    needs_more_sweeps = False
    sweeps_do_not_help = False
    likely_convergence_issue = False

    if sweep_by:
        s1 = sweep_by.get(1)
        if s1 and not _passes(s1):
            for s in sweep_counts:
                if s in sweep_by and _passes(sweep_by[s]):
                    needs_more_sweeps = True
                    break

        # 100x improvement heuristic (either obs_max or fidelity_error)
        if s1:
            best_obs = min(sweep_by[s].obs_max_error for s in sweep_by)
            best_fid = min(abs(sweep_by[s].fidelity_error) for s in sweep_by)
            if s1.obs_max_error > 0 and best_obs <= s1.obs_max_error / 100:
                likely_convergence_issue = True
            if abs(s1.fidelity_error) > 0 and best_fid <= abs(s1.fidelity_error) / 100:
                likely_convergence_issue = True

            # "do not help": all fail and best is within WARN_TOL-relative band
            if all(not _passes(sweep_by[s]) for s in sweep_by):
                best = _best_by(sweep_by.values(), "obs_max_error")
                if s1.obs_max_error > 0:
                    rel = best.obs_max_error / s1.obs_max_error
                    if rel > 0.5:  # basically unchanged
                        sweeps_do_not_help = True

    structural_or_impl_suspect = False
    if default and not _passes(default):
        if padding_does_not_help and (sweeps_do_not_help or not sweep_by):
            structural_or_impl_suspect = True
        if (not needs_padding) and (not needs_more_sweeps) and padding_does_not_help and sweeps_do_not_help:
            structural_or_impl_suspect = True

    likely_padding_issue = bool(needs_padding)

    likely_safe_gate = bool(passes_default and not needs_padding and not needs_more_sweeps)

    unresolved_failure = bool(default and not _passes(default) and not needs_padding and not needs_more_sweeps)

    # Gate listing for unresolved failures (only if TEBD passes).
    tebd_rows = [r for r in rows_for_probe if r.path == "simulator" and r.gate_mode == "tebd"]
    tebd_passes = bool(tebd_rows and _passes(tebd_rows[0]))
    failing_circuit_text = None
    if unresolved_failure and tebd_passes:
        try:
            failing_circuit_text = qc.draw(output="text").single_string()
        except Exception:
            failing_circuit_text = _as_gate_list_text(qc)

    if likely_safe_gate:
        likely_reason = "safe isolated LR rotation"
    elif likely_padding_issue:
        likely_reason = "padding/tangent-space issue"
    elif likely_convergence_issue:
        likely_reason = "sweep convergence issue"
    elif structural_or_impl_suspect:
        likely_reason = "mixed-axis/stack structural or implementation issue"
    else:
        likely_reason = "unexpected gate-specific failure"

    return {
        "probe": probe_id,
        "passes_default": passes_default,
        "needs_padding": needs_padding,
        "needs_more_sweeps": needs_more_sweeps,
        "sweeps_do_not_help": sweeps_do_not_help,
        "padding_does_not_help": padding_does_not_help,
        "structural_or_impl_suspect": structural_or_impl_suspect,
        "likely_padding_issue": likely_padding_issue,
        "likely_convergence_issue": likely_convergence_issue,
        "likely_safe_gate": likely_safe_gate,
        "unresolved_failure": unresolved_failure,
        "likely_reason": likely_reason,
        "failing_circuit_text": failing_circuit_text,
    }


def _run_baseline_rows(probe_map: dict[str, QuantumCircuit]) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for probe_id, qc in probe_map.items():
        vec, _, _, init_bond = run_simulator_hybrid(qc, gate_mode="hybrid", tdvp_sweeps=4, tdvp_circuit_full_sweep=False)
        rows.append(
            evaluate_row(
                probe_id,
                qc,
                vec,
                gate_mode="hybrid",
                tdvp_sweeps=4,
                tdvp_circuit_full_sweep=False,
                initial_pad=None,
                path="simulator",
                initial_bond_dim=init_bond,
            )
        )
        vec_t, _, _, init_bond_t = run_simulator_hybrid(qc, gate_mode="tebd", tdvp_sweeps=4)
        rows.append(
            evaluate_row(
                probe_id,
                qc,
                vec_t,
                gate_mode="tebd",
                tdvp_sweeps=4,
                tdvp_circuit_full_sweep=False,
                initial_pad=None,
                path="simulator",
                initial_bond_dim=init_bond_t,
            )
        )
    return rows


def _run_sweep_rows(
    probe_map: dict[str, QuantumCircuit],
    probe_ids: Iterable[str],
    sweep_counts: Iterable[int],
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for probe_id in probe_ids:
        qc = probe_map[probe_id]
        for sweeps in sweep_counts:
            vec, _, _, init_bond = run_simulator_hybrid(qc, gate_mode="hybrid", tdvp_sweeps=int(sweeps), tdvp_circuit_full_sweep=False)
            rows.append(
                evaluate_row(
                    probe_id,
                    qc,
                    vec,
                    gate_mode="hybrid",
                    tdvp_sweeps=int(sweeps),
                    tdvp_circuit_full_sweep=False,
                    initial_pad=None,
                    path="simulator",
                    initial_bond_dim=init_bond,
                    meta={"study": "sweeps"},
                )
            )
    return rows


def _run_padding_rows(
    probe_map: dict[str, QuantumCircuit],
    probe_ids: Iterable[str],
    pad_values: Iterable[int | None],
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    for probe_id in probe_ids:
        qc = probe_map[probe_id]
        for pad in pad_values:
            vec, _, init_bond = run_direct_hybrid(qc, initial_pad=pad, tdvp_sweeps=4, tdvp_circuit_full_sweep=False)
            rows.append(
                evaluate_row(
                    probe_id,
                    qc,
                    vec,
                    gate_mode="hybrid",
                    tdvp_sweeps=4,
                    tdvp_circuit_full_sweep=False,
                    initial_pad=pad,
                    path="direct_mps",
                    initial_bond_dim=init_bond,
                    meta={"study": "padding"},
                )
            )
    return rows


def _diagnostic_probe_ids(probe_map: dict[str, QuantumCircuit]) -> tuple[str, ...]:
    return tuple(probe_map.keys())


def generate_rotation_diagnostic_report(output_dir: Path) -> dict[str, Any]:
    """Generate the minimal diagnostic suite report and write JSON + Markdown."""
    config = ROTATION_DIAGNOSTIC_CONFIG
    probe_map = config.probes()
    probe_ids = _diagnostic_probe_ids(probe_map)

    baseline = _run_baseline_rows(probe_map)
    sweep_rows = _run_sweep_rows(probe_map, probe_ids, config.sweep_counts)
    padding_rows = _run_padding_rows(probe_map, probe_ids, config.pad_values)

    all_rows = [*baseline, *sweep_rows, *padding_rows]
    by_probe: dict[str, list[BenchmarkRow]] = defaultdict(list)
    for r in all_rows:
        by_probe[r.probe].append(r)

    classifications: list[dict[str, Any]] = []
    for probe_id in probe_ids:
        classifications.append(
            classify_probe(
                probe_id=probe_id,
                qc=probe_map[probe_id],
                rows_for_probe=by_probe[probe_id],
                sweep_counts=config.sweep_counts,
                pad_values=config.pad_values,
            )
        )

    report: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "title": config.title,
            "description": config.description,
            "basename": config.basename,
            "probe_ids": list(probe_ids),
            "settings": {
                "gate_family": config.gate_family_note,
                "pass_tol": PASS_TOL,
                "warn_tol": WARN_TOL,
                "study_sweep_counts": list(config.sweep_counts),
                "study_pad_values": ["None" if v is None else int(v) for v in config.pad_values],
                "required_paths_per_probe": [
                    "simulator/hybrid (auto-padding, 4 sweeps)",
                    "direct_mps/hybrid (pad=None, 4 sweeps)",
                    "direct_mps/hybrid (pad=2, 4 sweeps)",
                    "direct_mps/hybrid (pad=4, 4 sweeps)",
                    "simulator/tebd reference",
                ],
            },
        },
        "rows": [row_to_dict(r) for r in all_rows],
        "classification": classifications,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{config.basename}.json"
    md_path = output_dir / f"{config.basename}.md"
    json_path.write_text(_json_dumps(report), encoding="utf-8")
    md_path.write_text(render_rotation_diagnostic_markdown(report, probe_map), encoding="utf-8")
    report["meta"]["output_files"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def _json_dumps(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=False)


def render_rotation_diagnostic_markdown(report: dict[str, Any], probe_map: dict[str, QuantumCircuit]) -> str:
    title = report["meta"]["title"]
    lines: list[str] = [
        f"# {title}",
        "",
        f"Generated: {report['meta']['generated_at']}",
        "",
        report["meta"]["description"],
        "",
        "## Settings",
        "",
    ]
    for key, value in report["meta"]["settings"].items():
        if isinstance(value, list):
            lines.append(f"- **{key}**:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"- **{key}**: `{value}`")

    lines.extend(["", "## Diagnostic classification", ""])
    lines.append("| probe | default | needs_padding | needs_more_sweeps | unresolved_failure | likely_reason |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for c in report["classification"]:
        default = "pass" if c["passes_default"] else "FAIL"
        lines.append(
            f"| {c['probe']} | {default} | {c['needs_padding']} | {c['needs_more_sweeps']} | "
            f"{c['unresolved_failure']} | {c['likely_reason']} |"
        )

    # Print failing circuits for unresolved failures
    unresolved = [c for c in report["classification"] if c["unresolved_failure"] and c.get("failing_circuit_text")]
    if unresolved:
        lines.extend(["", "## Unresolved failures (hybrid fails, TEBD passes)", ""])
        for c in unresolved:
            probe_id = c["probe"]
            lines.append(f"### `{probe_id}`")
            lines.append("")
            lines.append("Circuit:")
            lines.append("")
            lines.append("```text")
            lines.append(str(c["failing_circuit_text"]).rstrip())
            lines.append("```")
            lines.append("")

    # Include the minimal required path rows per probe (compact list)
    lines.extend(["", "## Required path comparisons (per probe)", ""])
    rows = report["rows"]
    by_probe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_probe[r["probe"]].append(r)

    def pick(rows_for_probe: list[dict[str, Any]], *, path: str, gate_mode: str, tdvp_sweeps: int, initial_pad: Any) -> dict[str, Any] | None:
        for r in rows_for_probe:
            if (
                r["path"] == path
                and r["gate_mode"] == gate_mode
                and r["tdvp_sweeps"] == tdvp_sweeps
                and r["initial_pad"] == initial_pad
            ):
                return r
        return None

    for probe_id in report["meta"]["probe_ids"]:
        rset = by_probe[probe_id]
        sim_h = pick(rset, path="simulator", gate_mode="hybrid", tdvp_sweeps=4, initial_pad=None)
        tebd = pick(rset, path="simulator", gate_mode="tebd", tdvp_sweeps=4, initial_pad=None)
        pad_none = pick(rset, path="direct_mps", gate_mode="hybrid", tdvp_sweeps=4, initial_pad=None)
        pad2 = pick(rset, path="direct_mps", gate_mode="hybrid", tdvp_sweeps=4, initial_pad=2)
        pad4 = pick(rset, path="direct_mps", gate_mode="hybrid", tdvp_sweeps=4, initial_pad=4)

        lines.append(f"### `{probe_id}`")
        lines.append("")
        lines.append("| path | fidelity_error | obs_max | init_bond |")
        lines.append("|---|---:|---:|---:|")
        for label, row in (
            ("simulator/hybrid", sim_h),
            ("direct/pad=None", pad_none),
            ("direct/pad=2", pad2),
            ("direct/pad=4", pad4),
            ("simulator/tebd", tebd),
        ):
            if row is None:
                lines.append(f"| {label} | (missing) | (missing) | (missing) |")
            else:
                lines.append(
                    f"| {label} | {row['fidelity_error']:.2e} | {row['obs_max_error']:.2e} | {row['initial_bond_dim']} |"
                )
        lines.append("")

    return "\n".join(lines)


def generate_report(output_dir: Path, *, config: BenchmarkSuiteConfig = DEFAULT_BENCHMARK_CONFIG) -> dict[str, Any]:
    """Generate the legacy grid-style benchmark report (JSON + Markdown).

    This is kept for continuity, but the preferred debugging workflow is
    `generate_rotation_diagnostic_report()`.
    """
    probe_map = config.probes()
    sweep_probes = list(config.sweep_probes) if config.sweep_probes else list(probe_map.keys())
    padding_probes = list(config.padding_probes) if config.padding_probes else list(probe_map.keys())
    sweep_counts = list(config.sweep_counts)
    pad_values = list(config.pad_values)

    baseline = _run_baseline_rows(probe_map)
    sweep_rows = _run_sweep_rows(probe_map, sweep_probes, sweep_counts)
    padding_rows = _run_padding_rows(probe_map, padding_probes, pad_values)

    report: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "title": config.title,
            "description": config.description,
            "probe_ids": list(probe_map.keys()),
            "settings": {
                "gate_mode_default": "hybrid",
                "tdvp_sweeps_default": 4,
                "svd_threshold": SVD_THRESHOLD,
                "observable_panel": "Pauli X, Y, Z on each qubit",
                "simulator_auto_pad": "product states padded to chi >= 2 before gates",
                "gate_family": config.gate_family_note,
                "study_sweep_counts": sweep_counts,
                "study_pad_values": ["None" if v is None else int(v) for v in pad_values],
            },
        },
        "baseline_hybrid_vs_tebd": [row_to_dict(r) for r in baseline],
        "sweep_count_study": [row_to_dict(r) for r in sweep_rows],
        "initial_padding_study": [row_to_dict(r) for r in padding_rows],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{config.basename}.json"
    md_path = output_dir / f"{config.basename}.md"
    json_path.write_text(_json_dumps(report), encoding="utf-8")
    md_path.write_text(render_markdown(report, config), encoding="utf-8")
    report["meta"]["output_files"] = {"json": str(json_path), "markdown": str(md_path)}
    return report


def render_markdown(report: dict[str, Any], config: BenchmarkSuiteConfig) -> str:
    lines: list[str] = [
        f"# {report['meta'].get('title', config.title)}",
        "",
        f"Generated: {report['meta']['generated_at']}",
        "",
        report["meta"]["description"],
        "",
        "## Settings",
        "",
    ]
    for key, value in report["meta"]["settings"].items():
        if isinstance(value, list):
            lines.append(f"- **{key}**:")
            lines.extend(f"  - {item}" for item in value)
        else:
            lines.append(f"- **{key}**: `{value}`")

    lines.extend(["", "## 1. Baseline: hybrid vs TEBD (4 sweeps, simulator path)", ""])
    lines.append(_markdown_table(report["baseline_hybrid_vs_tebd"]))
    lines.extend(["", "## 2. Effect of `tdvp_sweeps` (hybrid, simulator)", ""])
    lines.append(_markdown_sweep_table(report["sweep_count_study"]))
    lines.extend(["", "## 3. Effect of initial bond padding (hybrid, direct MPS, 4 sweeps)", ""])
    lines.append(_markdown_padding_table(report["initial_padding_study"]))
    lines.extend(["", "## Key takeaways for downstream analysis", ""])
    for item in config.takeaways:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = "| probe | mode | fidelity | 1-fidelity | obs_max | obs_l2 | worst_obs |"
    sep = "|---|---:|---:|---:|---:|---:|---|"
    body: list[str] = []
    for row in rows:
        worst = row.get("worst_observable") or {}
        worst_str = f"{worst.get('name','?')}@{worst.get('site','?')} err={worst.get('abs_error', 0):.4f}"
        body.append(
            f"| {row['probe']} | {row['gate_mode']} | {row['fidelity']:.6f} | "
            f"{row['fidelity_error']:.2e} | {row['obs_max_error']:.4f} | "
            f"{row['obs_l2_error']:.4f} | {worst_str} |"
        )
    return "\n".join([header, sep, *body])


def _markdown_sweep_table(rows: list[dict[str, Any]]) -> str:
    header = "| probe | sweeps | fidelity | 1-fidelity | obs_max | worst_obs |"
    sep = "|---|---:|---:|---:|---:|---|"
    body: list[str] = []
    for row in rows:
        worst = row.get("worst_observable") or {}
        worst_str = f"{worst.get('name','?')}@{worst.get('site','?')} ({worst.get('abs_error', 0):.4f})"
        body.append(
            f"| {row['probe']} | {row['tdvp_sweeps']} | {row['fidelity']:.6f} | "
            f"{row['fidelity_error']:.2e} | {row['obs_max_error']:.4f} | {worst_str} |"
        )
    return "\n".join([header, sep, *body])


def _markdown_padding_table(rows: list[dict[str, Any]]) -> str:
    header = "| probe | initial_pad | init_bond | fidelity | 1-fidelity | obs_max | worst_obs |"
    sep = "|---|---:|---:|---:|---:|---:|---|"
    body: list[str] = []
    for row in rows:
        pad = "None" if row["initial_pad"] is None else str(row["initial_pad"])
        worst = row.get("worst_observable") or {}
        worst_str = f"{worst.get('name','?')}@{worst.get('site','?')} ({worst.get('abs_error', 0):.4f})"
        body.append(
            f"| {row['probe']} | {pad} | {row['initial_bond_dim']} | {row['fidelity']:.6f} | "
            f"{row['fidelity_error']:.2e} | {row['obs_max_error']:.4f} | {worst_str} |"
        )
    return "\n".join([header, sep, *body])

