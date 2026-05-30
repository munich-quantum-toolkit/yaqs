# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regression tests for previously failing mixed long-range Pauli rotation stacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.digital.digital_tjm import (
    apply_pauli_product_rotation_enriched,
    apply_single_qubit_gate,
    apply_two_qubit_gate,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm

if TYPE_CHECKING:
    from collections.abc import Callable


def _run_hybrid(qc: QuantumCircuit) -> tuple[np.ndarray, int]:
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        svd_threshold=1e-14,
        gate_mode="hybrid",
        tangent_blindness_tol=1e-12,
        tdvp_projection_defect_tol=5e-2,
        tdvp_visibility_safety_tol=None,
        tdvp_pauli_consistency_check=False,
        tdvp_pauli_consistency_tol=1e-10,
    )
    st = State(qc.num_qubits, initial="zeros", representation="mps")
    mps = st.mps
    dag = circuit_to_dag(qc)
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 1:
            apply_single_qubit_gate(mps, node)
            continue
        if len(node.qargs) != 2:
            continue
        gate = convert_dag_to_tensor_algorithm(node)[0]
        i, j = int(gate.sites[0]), int(gate.sites[1])
        if abs(i - j) > 1 and gate.name in {"rxx", "ryy", "rzz"}:
            apply_pauli_product_rotation_enriched(mps, gate, params)
        else:
            apply_two_qubit_gate(mps, node, params)
    vec = np.asarray(mps.to_vec(), dtype=np.complex128)
    return vec, int(mps.get_max_bond())


def _fid_err(qc: QuantumCircuit, vec: np.ndarray) -> float:
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    return float(1.0 - abs(np.vdot(ref, vec)) ** 2)


def _qc_stack_core_rxx_ryy_vacuum_10q() -> QuantumCircuit:
    qc = QuantumCircuit(10)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    return qc


def _qc_mixed_axes_disjoint_pairs_8q() -> QuantumCircuit:
    qc = QuantumCircuit(8)
    qc.ry(np.pi / 4, 0)
    qc.ry(np.pi / 5, 7)
    qc.rzz(0.19, 0, 7)
    qc.rxx(0.21, 1, 6)
    qc.ryy(0.25, 2, 5)
    return qc


def _qc_minimal_mixed_stack_10q() -> QuantumCircuit:
    qc = QuantumCircuit(10)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 5, 4)
    qc.rzz(0.19, 1, 8)
    qc.rzz(0.27, 4, 9)
    qc.rxx(0.21, 2, 7)
    qc.ryy(0.25, 3, 6)
    return qc


def _qc_lr_stack_mixed_12q() -> QuantumCircuit:
    qc = QuantumCircuit(12)
    qc.ry(np.pi / 4, 1)
    qc.ry(np.pi / 4, 4)
    qc.ry(np.pi / 4, 7)
    qc.rzz(0.19, 1, 10)
    qc.rzz(0.27, 4, 11)
    qc.rzz(0.33, 0, 7)
    qc.rxx(0.21, 2, 9)
    qc.ryy(0.25, 3, 8)
    return qc


@pytest.mark.parametrize(
    "qc_builder",
    [
        pytest.param(
            _qc_stack_core_rxx_ryy_vacuum_10q,
            id="stack_core_rxx_ryy_vacuum_10q",
        ),
        pytest.param(
            _qc_mixed_axes_disjoint_pairs_8q,
            id="mixed_axes_disjoint_pairs_8q",
        ),
        pytest.param(
            _qc_minimal_mixed_stack_10q,
            id="minimal_mixed_stack_10q",
        ),
        pytest.param(
            _qc_lr_stack_mixed_12q,
            id="lr_stack_mixed_12q",
        ),
    ],
)
def test_enriched_mixed_long_range_stacks_match_qiskit(qc_builder: Callable[[], QuantumCircuit]) -> None:
    """Previously failing mixed LR stacks match Qiskit within tolerance."""
    qc = qc_builder()
    vec, max_bond = _run_hybrid(qc)
    assert _fid_err(qc, vec) < 1e-10
    # Loose guardrail: ensure we didn't keep doubling bonds without truncation.
    assert max_bond <= 128
