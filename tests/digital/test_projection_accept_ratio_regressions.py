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
from mqt.yaqs.digital.digital_tjm import (
    apply_pauli_product_rotation_enriched,
    apply_single_qubit_gate,
    apply_two_qubit_gate,
    decide_long_range_pauli_route,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm


def _fid_err(qc: QuantumCircuit, vec: np.ndarray) -> float:
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    return float(1.0 - abs(np.vdot(ref, vec)) ** 2)


def _params(*, defect_tol: float) -> StrongSimParams:
    return StrongSimParams(
        preset="exact",
        svd_threshold=1e-14,
        gate_mode="hybrid",
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=defect_tol,
        tdvp_pauli_consistency_check=False,
    )


def _long_range_gate(qc: QuantumCircuit):
    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 2:
            return convert_dag_to_tensor_algorithm(node)[0]
    msg = "No two-qubit gate found"
    raise AssertionError(msg)


def _run_hybrid(qc: QuantumCircuit, *, defect_tol: float) -> tuple[np.ndarray, str | None]:
    params = _params(defect_tol=defect_tol)
    st = State(qc.num_qubits, initial="zeros", representation="mps")
    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 1:
            apply_single_qubit_gate(st.mps, node)
        elif len(node.qargs) == 2:
            apply_two_qubit_gate(st.mps, node, params)
        else:
            msg = f"Unexpected {len(node.qargs)}-qubit op: {node.op.name}"
            raise AssertionError(msg)
    return np.asarray(st.mps.to_vec(), dtype=np.complex128), getattr(params, "last_lr_route", None)


def _run_enriched(qc: QuantumCircuit, *, defect_tol: float) -> np.ndarray:
    params = _params(defect_tol=defect_tol)
    st = State(qc.num_qubits, initial="zeros", representation="mps")
    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 1:
            apply_single_qubit_gate(st.mps, node)
            continue
        if len(node.qargs) != 2:
            continue
        gate = convert_dag_to_tensor_algorithm(node)[0]
        i, j = int(gate.sites[0]), int(gate.sites[1])
        if abs(i - j) > 1 and gate.name in {"rxx", "ryy", "rzz"}:
            apply_pauli_product_rotation_enriched(st.mps, gate, params)
        else:
            apply_two_qubit_gate(st.mps, node, params)
    return np.asarray(st.mps.to_vec(), dtype=np.complex128)


def test_known_rzz_case_routes_tdvp_at_090_and_is_inaccurate() -> None:
    qc = QuantumCircuit(10)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 5, 6)
    qc.rzz(0.25, 1, 6)

    vec, route = _run_hybrid(qc, defect_tol=1e-1)
    assert route == "tdvp"
    assert _fid_err(qc, vec) > 1e-4


def test_known_rzz_case_router_recommends_enriched_at_095_and_enriched_matches_qiskit() -> None:
    qc = QuantumCircuit(10)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 5, 6)
    qc.rzz(0.25, 1, 6)

    params = _params(defect_tol=5e-2)
    st = State(qc.num_qubits, initial="zeros", representation="mps")
    for node in circuit_to_dag(qc).topological_op_nodes():
        if len(node.qargs) == 1:
            apply_single_qubit_gate(st.mps, node)
    gate = _long_range_gate(qc)
    decision = decide_long_range_pauli_route(st.mps, gate, params)
    assert decision.route == "pauli_enriched"

    vec = _run_enriched(qc, defect_tol=5e-2)
    assert _fid_err(qc, vec) < 1e-10
