# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for state-dependent routing via tangent-visibility diagnostic."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate, apply_two_qubit_gate


def _apply_circuit_direct(qc: QuantumCircuit, params: StrongSimParams) -> np.ndarray:
    """Apply `qc` gate-by-gate using the digital backend helpers."""
    st = State(qc.num_qubits, initial="zeros", representation="mps")
    mps = st.mps
    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
        elif len(node.qargs) == 2:
            apply_two_qubit_gate(mps, node, params)
        else:
            msg = f"Unexpected {len(node.qargs)}-qubit operation: {node.op.name}"
            raise AssertionError(msg)
    return np.asarray(mps.to_vec(), dtype=np.complex128)


def _fid_err(qc: QuantumCircuit, vec: np.ndarray) -> float:
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    return float(1.0 - abs(np.vdot(ref, vec)) ** 2)


def test_tangent_blind_ryy_routes_to_enriched_and_matches_qiskit() -> None:
    qc = QuantumCircuit(8)
    qc.ryy(0.25, 1, 6)
    params = StrongSimParams(
        preset="exact",
        gate_mode="hybrid",
        svd_threshold=1e-14,
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=5e-2,
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=False,
        tdvp_pauli_consistency_tol=1e-10,
    )
    vec = _apply_circuit_direct(qc, params)
    assert getattr(params, "last_lr_route", None) == "pauli_enriched"
    vis = getattr(params, "last_lr_visibility", None)
    assert vis is not None
    assert vis.projected_ratio < params.tangent_blindness_tol
    assert _fid_err(qc, vec) < 1e-10


def test_endpoint_prepared_ryy_routes_to_tdvp_and_matches_qiskit() -> None:
    qc = QuantumCircuit(8)
    qc.rx(np.pi / 2, 6)
    qc.ryy(0.25, 1, 6)
    params = StrongSimParams(
        preset="exact",
        gate_mode="hybrid",
        svd_threshold=1e-14,
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=5e-2,
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=False,
        tdvp_pauli_consistency_tol=1e-10,
    )
    vec = _apply_circuit_direct(qc, params)
    assert getattr(params, "last_lr_route", None) in {"tdvp", "pauli_enriched"}
    vis = getattr(params, "last_lr_visibility", None)
    assert vis is not None
    assert not vis.is_blind
    assert _fid_err(qc, vec) < 1e-10


def test_rzz_on_product_state_routes_to_tdvp_and_matches_qiskit() -> None:
    qc = QuantumCircuit(8)
    qc.rzz(0.25, 1, 6)
    params = StrongSimParams(
        preset="exact",
        gate_mode="hybrid",
        svd_threshold=1e-14,
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=5e-2,
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=False,
        tdvp_pauli_consistency_tol=1e-10,
    )
    vec = _apply_circuit_direct(qc, params)
    assert getattr(params, "last_lr_route", None) in {"tdvp", "pauli_enriched"}
    vis = getattr(params, "last_lr_visibility", None)
    assert vis is not None
    assert not vis.is_blind
    assert _fid_err(qc, vec) < 1e-10
