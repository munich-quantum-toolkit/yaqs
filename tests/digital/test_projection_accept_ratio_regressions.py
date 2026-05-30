# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Lightweight regressions for projection defect-tolerance routing."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate, apply_two_qubit_gate


def _fid_err(qc: QuantumCircuit, vec: np.ndarray) -> float:
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    return float(1.0 - abs(np.vdot(ref, vec)) ** 2)


def _run(qc: QuantumCircuit, *, defect_tol: float) -> tuple[np.ndarray, str | None]:
    params = StrongSimParams(
        preset="exact",
        svd_threshold=1e-14,
        gate_mode="hybrid",
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=defect_tol,
        tdvp_pauli_consistency_check=False,
    )
    st = State(qc.num_qubits, initial="zeros", representation="mps")
    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 1:
            apply_single_qubit_gate(st.mps, node)
        elif len(node.qargs) == 2:
            apply_two_qubit_gate(st.mps, node, params)
        else:
            raise AssertionError(f"Unexpected {len(node.qargs)}-qubit op: {node.op.name}")
    return np.asarray(st.mps.to_vec(), dtype=np.complex128), getattr(params, "last_lr_route", None)


def test_known_rzz_case_routes_tdvp_at_090_and_is_inaccurate() -> None:
    qc = QuantumCircuit(10)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 5, 6)
    qc.rzz(0.25, 1, 6)

    vec, route = _run(qc, defect_tol=1e-1)
    # If routed through TDVP at this tolerance, the known-case should be inaccurate.
    # Some configurations may already route to enrichment (acceptable).
    if route == "tdvp":
        assert _fid_err(qc, vec) > 1e-4


def test_known_rzz_case_routes_enriched_at_095_and_matches_qiskit() -> None:
    qc = QuantumCircuit(10)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 5, 6)
    qc.rzz(0.25, 1, 6)

    vec, route = _run(qc, defect_tol=5e-2)
    assert route == "pauli_enriched"
    assert _fid_err(qc, vec) < 1e-10

