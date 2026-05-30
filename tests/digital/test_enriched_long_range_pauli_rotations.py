# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for enriched long-range Pauli-product rotation updates (RXX/RYY)."""

from __future__ import annotations

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


def _enriched_statevector(qc: QuantumCircuit) -> np.ndarray:
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
    return np.asarray(mps.to_vec(), dtype=np.complex128)


def _fidelity_error(qc: QuantumCircuit, vec: np.ndarray) -> float:
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    return float(1.0 - abs(np.vdot(ref, vec)) ** 2)


def test_enriched_rxx_strictly_interior_long_range_matches_qiskit() -> None:
    """Enriched LR `rxx` on a product state matches Qiskit."""
    qc = QuantumCircuit(8)
    qc.rxx(0.25, 1, 6)
    vec = _enriched_statevector(qc)
    assert _fidelity_error(qc, vec) < 1e-10


def test_enriched_ryy_strictly_interior_long_range_matches_qiskit() -> None:
    """Enriched LR `ryy` on a product state matches Qiskit."""
    qc = QuantumCircuit(8)
    qc.ryy(0.25, 1, 6)
    vec = _enriched_statevector(qc)
    assert _fidelity_error(qc, vec) < 1e-10


@pytest.mark.parametrize("gate_name", ["rxx", "ryy"])
@pytest.mark.parametrize(("i", "j"), [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)])
def test_enriched_long_range_pauli_rotations_position_sweep_match_qiskit(gate_name: str, i: int, j: int) -> None:
    """Enriched LR `rxx/ryy` position sweeps match Qiskit."""
    qc = QuantumCircuit(10)
    getattr(qc, gate_name)(0.25, i, j)
    vec = _enriched_statevector(qc)
    assert _fidelity_error(qc, vec) < 1e-10


@pytest.mark.parametrize(("i", "j"), [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)])
def test_enriched_long_range_rzz_position_sweep_match_qiskit(i: int, j: int) -> None:
    """Enriched LR `rzz` position sweeps match Qiskit."""
    qc = QuantumCircuit(10)
    qc.ry(np.pi / 4, i)
    qc.ry(np.pi / 5, j)
    qc.rzz(0.25, i, j)
    vec = _enriched_statevector(qc)
    assert _fidelity_error(qc, vec) < 1e-10
