# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for experimental TDVP helpers (diagnostics only)."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.digital.digital_tjm import (
    apply_single_qubit_gate,
    apply_two_qubit_gate_tdvp,
    apply_two_qubit_gate_tdvp_experimental,
    copy_pauli_rotation_with_angle,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm


def _fid_err(qc: QuantumCircuit, vec: np.ndarray) -> float:
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    return float(max(0.0, 1.0 - abs(np.vdot(ref, vec)) ** 2))


def test_copy_pauli_rotation_preserves_sites_and_rescales_angle() -> None:
    qc = QuantumCircuit(4)
    qc.rzz(0.5, 1, 3)
    dag = circuit_to_dag(qc)
    node = next(dag.topological_op_nodes())
    gate = convert_dag_to_tensor_algorithm(node)[0]
    half = copy_pauli_rotation_with_angle(gate, 0.25)
    assert half.name == "rzz"
    assert tuple(half.sites) == (1, 3)
    assert abs(float(half.theta) - 0.25) < 1e-15


def test_experimental_tdvp_matches_plain_for_defaults_on_short_range() -> None:
    qc = QuantumCircuit(6)
    qc.ry(np.pi / 4, 2)
    qc.rzz(0.2, 1, 4)
    dag = circuit_to_dag(qc)

    params = StrongSimParams(
        preset="exact",
        svd_threshold=1e-12,
        max_bond_dim=None,
        krylov_tol=1e-12,
        gate_mode="hybrid",
    )

    def run_plain() -> np.ndarray:
        mps = State(6, initial="zeros", representation="mps").mps
        for node in dag.topological_op_nodes():
            if node.op.name == "rzz":
                gate = convert_dag_to_tensor_algorithm(node)[0]
                apply_two_qubit_gate_tdvp(mps, gate, params)
                return np.asarray(mps.to_vec(), dtype=np.complex128)
            if len(node.qargs) == 1:
                apply_single_qubit_gate(mps, node)
        msg = "missing rzz"
        raise AssertionError(msg)

    def run_experimental() -> np.ndarray:
        mps = State(6, initial="zeros", representation="mps").mps
        for node in dag.topological_op_nodes():
            if node.op.name == "rzz":
                gate = convert_dag_to_tensor_algorithm(node)[0]
                apply_two_qubit_gate_tdvp_experimental(
                    mps,
                    gate,
                    params,
                    window_size=1,
                    tdvp_sweeps=1,
                    substeps=1,
                    tdvp_circuit_full_sweep=False,
                )
                return np.asarray(mps.to_vec(), dtype=np.complex128)
            if len(node.qargs) == 1:
                apply_single_qubit_gate(mps, node)
        msg = "missing rzz"
        raise AssertionError(msg)

    vec_plain = run_plain()
    vec_exp = run_experimental()
    assert 1.0 - abs(np.vdot(vec_plain, vec_exp)) ** 2 < 1e-12


def test_substepping_composes_to_full_angle_reference() -> None:
    qc = QuantumCircuit(6)
    qc.ry(np.pi / 3, 1)
    qc.ry(np.pi / 5, 4)
    qc.rzz(0.35, 1, 4)

    params = StrongSimParams(
        preset="exact",
        svd_threshold=1e-12,
        max_bond_dim=None,
        krylov_tol=1e-12,
        gate_mode="hybrid",
    )
    dag = circuit_to_dag(qc)

    st = State(6, initial="zeros", representation="mps")
    mps = st.mps
    for node in dag.topological_op_nodes():
        if node.op.name == "rzz":
            gate = convert_dag_to_tensor_algorithm(node)[0]
            apply_two_qubit_gate_tdvp_experimental(
                mps,
                gate,
                params,
                window_size=1,
                tdvp_sweeps=1,
                substeps=4,
                tdvp_circuit_full_sweep=False,
            )
            break
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
    vec = np.asarray(mps.to_vec(), dtype=np.complex128)
    assert _fid_err(qc, vec) < 1e-2
