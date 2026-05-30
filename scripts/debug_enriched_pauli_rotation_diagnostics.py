#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Minimal diagnostics for enriched Pauli-product long-range rotations.

Run:

    uv run python -m scripts.debug_enriched_pauli_rotation_diagnostics
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate, apply_two_qubit_gate


def fidelity_error(qc: QuantumCircuit, vec: np.ndarray) -> float:
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    return float(1.0 - abs(np.vdot(ref, vec)) ** 2)


def max_single_qubit_pauli_error(qc: QuantumCircuit, vec: np.ndarray) -> float:
    """Max abs error over single-qubit X/Y/Z expectations vs Qiskit."""
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    n = qc.num_qubits

    def exp_pauli(state: np.ndarray, qubit: int, which: str) -> float:
        # Apply Pauli to qubit via basis-index manipulation (dense, but n<=12 in diagnostics).
        out = state.copy()
        bit = 1 << (n - 1 - qubit)
        if which == "Z":
            for idx in range(out.size):
                if idx & bit:
                    out[idx] *= -1
        elif which == "X":
            swapped = out.copy()
            for idx in range(out.size):
                j = idx ^ bit
                swapped[j] = out[idx]
            out = swapped
        elif which == "Y":
            swapped = np.zeros_like(out)
            for idx in range(out.size):
                j = idx ^ bit
                swapped[j] = (1j if (idx & bit) == 0 else -1j) * out[idx]
            out = swapped
        else:
            raise ValueError(which)
        return float(np.vdot(state, out).real)

    err = 0.0
    for q in range(n):
        for p in ("X", "Y", "Z"):
            err = max(err, abs(exp_pauli(vec, q, p) - exp_pauli(ref, q, p)))
    return err


def run_case(name: str, qc: QuantumCircuit) -> None:
    params = StrongSimParams(
        preset="exact",
        svd_threshold=1e-14,
        gate_mode="hybrid",
        tangent_blindness_tol=1e-12,
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_tol=1e-10,
        tdvp_pauli_consistency_check=True,
    )
    init = State(qc.num_qubits, initial="zeros", representation="mps")
    mps = init.mps
    dag = circuit_to_dag(qc)

    print(f"\n=== {name} ===")
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
            continue
        if len(node.qargs) == 2:
            apply_two_qubit_gate(mps, node, params)
            if hasattr(params, "last_lr_visibility"):
                vis = params.last_lr_visibility
                decision = getattr(params, "last_lr_decision", None)
                print(
                    "lr_gate",
                    node.op.name,
                    "qargs",
                    [qc.find_bit(q).index for q in node.qargs],
                    "route",
                    getattr(params, "last_lr_route", None),
                    "is_blind",
                    getattr(vis, "is_blind", None),
                    "is_weakly_visible",
                    getattr(vis, "is_weakly_visible", None),
                    "projected_norm",
                    getattr(vis, "projected_norm", None),
                    "generator_norm",
                    getattr(vis, "generator_norm", None),
                    "projected_ratio",
                    getattr(vis, "projected_ratio", None),
                    "update_delta_ratio",
                    getattr(vis, "update_delta_ratio", None),
                    "candidate_fidelity_error",
                    None if decision is None else getattr(decision, "candidate_fidelity_error", None),
                    "reason",
                    None if decision is None else getattr(decision, "reason", None),
                )
            continue
        raise ValueError(f"Unsupported operation with {len(node.qargs)} qubits: {node.op.name}")

    vec = np.asarray(mps.to_vec(), dtype=np.complex128)

    print("num_qubits", qc.num_qubits)
    print("fidelity_error", fidelity_error(qc, vec))
    print("max_pauli_error", max_single_qubit_pauli_error(qc, vec))
    print("max_bond", mps.get_max_bond())
    stats = getattr(mps, "route_stats", None)
    print("lr_pauli_tdvp_count", None if stats is None else stats.get("tdvp_lr_pauli"))
    print("lr_pauli_enriched_count", None if stats is None else stats.get("enriched_lr_pauli"))


def main() -> None:
    # Isolated position sweeps
    for gate in ("rxx", "ryy", "rzz"):
        for i, j in ((0, 5), (1, 6), (2, 7), (3, 8), (4, 9)):
            qc = QuantumCircuit(10)
            if gate == "rzz":
                qc.ry(np.pi / 4, i)
                qc.ry(np.pi / 5, j)
            getattr(qc, gate)(0.25, i, j)
            run_case(f"{gate} sweep ({i},{j})", qc)

    # Previously failing mixed-stack circuits
    qc = QuantumCircuit(10)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    run_case("stack_core_rxx_ryy_vacuum_10q", qc)

    qc = QuantumCircuit(8)
    qc.ry(np.pi / 4, 0)
    qc.ry(np.pi / 5, 7)
    qc.rzz(0.19, 0, 7)
    qc.rxx(0.21, 1, 6)
    qc.ryy(0.25, 2, 5)
    run_case("mixed_axes_disjoint_pairs_8q", qc)

    qc = QuantumCircuit(10)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 5, 4)
    qc.rzz(0.19, 1, 8)
    qc.rzz(0.27, 4, 9)
    qc.rxx(0.21, 2, 7)
    qc.ryy(0.25, 3, 6)
    run_case("minimal_mixed_stack_10q", qc)

    qc = QuantumCircuit(12)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 4, 4)
    qc.ry(np.pi / 4, 7)
    qc.rzz(0.19, 1, 10)
    qc.rzz(0.27, 4, 11)
    qc.rzz(0.33, 0, 7)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    run_case("lr_stack_mixed_12q", qc)


if __name__ == "__main__":
    main()

