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
from qiskit.quantum_info import Statevector

from mqt.yaqs import Simulator
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State


def _hybrid_statevector(qc: QuantumCircuit) -> np.ndarray:
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
    sim = Simulator()
    init = State(qc.num_qubits, initial="zeros", representation="mps")
    result = sim.run(init, qc, params)
    assert result.output_state is not None
    return np.asarray(result.output_state.mps.to_vec(), dtype=np.complex128)


def _fidelity_error(qc: QuantumCircuit, vec: np.ndarray) -> float:
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    return float(1.0 - abs(np.vdot(ref, vec)) ** 2)


def test_enriched_rxx_strictly_interior_long_range_matches_qiskit() -> None:
    """Enriched LR `rxx` on a product state matches Qiskit."""
    qc = QuantumCircuit(8)
    qc.rxx(0.25, 1, 6)
    vec = _hybrid_statevector(qc)
    assert _fidelity_error(qc, vec) < 1e-10


def test_enriched_ryy_strictly_interior_long_range_matches_qiskit() -> None:
    """Enriched LR `ryy` on a product state matches Qiskit."""
    qc = QuantumCircuit(8)
    qc.ryy(0.25, 1, 6)
    vec = _hybrid_statevector(qc)
    assert _fidelity_error(qc, vec) < 1e-10


@pytest.mark.parametrize("gate_name", ["rxx", "ryy"])
@pytest.mark.parametrize(("i", "j"), [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)])
def test_enriched_long_range_pauli_rotations_position_sweep_match_qiskit(gate_name: str, i: int, j: int) -> None:
    """Enriched LR `rxx/ryy` position sweeps match Qiskit."""
    qc = QuantumCircuit(10)
    getattr(qc, gate_name)(0.25, i, j)
    vec = _hybrid_statevector(qc)
    assert _fidelity_error(qc, vec) < 1e-10


@pytest.mark.parametrize(("i", "j"), [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)])
def test_enriched_long_range_rzz_position_sweep_match_qiskit(i: int, j: int) -> None:
    """Enriched LR `rzz` position sweeps match Qiskit."""
    qc = QuantumCircuit(10)
    qc.ry(np.pi / 4, i)
    qc.ry(np.pi / 5, j)
    qc.rzz(0.25, i, j)
    vec = _hybrid_statevector(qc)
    assert _fidelity_error(qc, vec) < 1e-10
