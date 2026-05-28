#!/usr/bin/env python
# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Run the minimal TDVP projector debug matrix.

This script relies on `YAQS_DEBUG_TDVP_PROJECTOR=1` to emit per-pair TDVP traces
from `src/mqt/yaqs/core/methods/tdvp.py`.

Run:

    $env:YAQS_DEBUG_TDVP_PROJECTOR = '1'
    uv run python -m benchmarks.debug_tdvp_projector_matrix
"""

from __future__ import annotations

import os

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.methods.tdvp import two_site_tdvp
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate
from mqt.yaqs.digital.digital_tjm import construct_generator_mpo
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm


def _gate_from_single_gate_circuit(qc: QuantumCircuit):
    nodes = list(circuit_to_dag(qc).topological_op_nodes())
    if not nodes:
        raise ValueError("Circuit has no operations.")
    # We want the *two-qubit* long-range gate node, not any preparation 1q rotations.
    twoq = [n for n in nodes if n.op.num_qubits == 2]
    if len(twoq) != 1:
        raise ValueError(f"Expected exactly one 2-qubit gate, got {len(twoq)}.")
    return convert_dag_to_tensor_algorithm(twoq[0])[0]


def run_one(*, gate_name: str, theta: float, L: int, sites: tuple[int, int], prep: list[tuple[str, float, int]] | None = None) -> None:
    qc = QuantumCircuit(L)
    if prep:
        for g, ang, q in prep:
            getattr(qc, g)(ang, q)
    getattr(qc, gate_name)(theta, sites[0], sites[1])

    gate = _gate_from_single_gate_circuit(qc)
    mpo, *_ = construct_generator_mpo(gate, L)
    mps = MPS(length=L, state="zeros", pad=None)

    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=1e-14,
        gate_mode="hybrid",
        tdvp_sweeps=1,
        tdvp_circuit_full_sweep=False,
    )

    # Apply the preparation gates to the MPS so the TDVP test matches the Qiskit reference.
    if prep:
        for g, ang, q in prep:
            qc_p = QuantumCircuit(L)
            getattr(qc_p, g)(ang, q)
            node = list(circuit_to_dag(qc_p).topological_op_nodes())[0]
            apply_single_qubit_gate(mps, node)

    v0 = np.asarray(mps.to_vec(), dtype=np.complex128)
    two_site_tdvp(mps, mpo, params)
    v1 = np.asarray(mps.to_vec(), dtype=np.complex128)
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    fid_err = float(1.0 - abs(np.vdot(ref, v1)) ** 2)

    print("")
    print("case", f"{gate_name} theta={theta} L={L} sites={sites} prep={prep}")
    print("final_state_delta", float(np.linalg.norm(v1 - v0)))
    print("fidelity_error", fid_err)


def main() -> None:
    if os.environ.get("YAQS_DEBUG_TDVP_PROJECTOR", "") == "":
        print("NOTE: set YAQS_DEBUG_TDVP_PROJECTOR=1 to get per-pair traces.")

    theta = 0.25

    # Minimal pass/fail
    run_one(gate_name="ryy", theta=theta, L=6, sites=(0, 5))
    run_one(gate_name="ryy", theta=theta, L=8, sites=(1, 6))

    # RXX/RZZ comparisons
    run_one(gate_name="rxx", theta=theta, L=6, sites=(0, 5))
    run_one(gate_name="rxx", theta=theta, L=8, sites=(1, 6))
    run_one(gate_name="rzz", theta=theta, L=6, sites=(0, 5))
    run_one(gate_name="rzz", theta=theta, L=8, sites=(1, 6))

    # Endpoint-preparation tests (interior)
    run_one(gate_name="rxx", theta=theta, L=8, sites=(1, 6), prep=[("ry", float(np.pi / 2), 6)])
    run_one(gate_name="rxx", theta=theta, L=8, sites=(1, 6), prep=[("ry", float(np.pi / 2), 1)])
    run_one(gate_name="ryy", theta=theta, L=8, sites=(1, 6), prep=[("rx", float(np.pi / 2), 6)])
    run_one(gate_name="ryy", theta=theta, L=8, sites=(1, 6), prep=[("rx", float(np.pi / 2), 1)])

    # Geometry sweep at L=8
    for sites in [(0, 7), (0, 6), (1, 7), (1, 6)]:
        run_one(gate_name="rxx", theta=theta, L=8, sites=sites)
    for sites in [(0, 7), (0, 6), (1, 7), (1, 6)]:
        run_one(gate_name="ryy", theta=theta, L=8, sites=sites)
    for sites in [(0, 7), (0, 6), (1, 7), (1, 6)]:
        run_one(gate_name="rzz", theta=theta, L=8, sites=sites)


if __name__ == "__main__":
    main()

