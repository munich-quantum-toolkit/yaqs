# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regression: window-local TDVP renorm must not inflate global norm after grafting."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.methods.tdvp.sweep_utils import _global_mps_norm
from mqt.yaqs.digital.digital_tjm import apply_single_qubit_gate, apply_two_qubit_gate
from tests.core.methods.tdvp.conftest import _fidelity

pytestmark = pytest.mark.tdvp_regression

ISING_LENGTH = 6
ISING_CHI = 16
RZZ_ANGLE = 0.3
TARGET_GATE = 12
NORM_TOL = 1e-6
FID_TOL = 1e-3
SQRT2 = float(np.sqrt(2.0))


def _grid_shape(num_qubits: int) -> tuple[int, int]:
    for nrow in range(int(math.sqrt(num_qubits)), 0, -1):
        if num_qubits % nrow == 0:
            return nrow, num_qubits // nrow
    return 1, num_qubits


def _grid_index(row: int, col: int, ncol: int) -> int:
    return row * ncol + col


def _ising_2d_mapped_circuit(length: int = ISING_LENGTH) -> QuantumCircuit:
    qc = QuantumCircuit(length)
    nrow, ncol = _grid_shape(length)
    for r in range(nrow):
        for c in range(ncol):
            qc.h(_grid_index(r, c, ncol))
    for r in range(nrow):
        for c in range(ncol - 1):
            qc.rzz(RZZ_ANGLE, _grid_index(r, c, ncol), _grid_index(r, c + 1, ncol))
    for r in range(nrow - 1):
        for c in range(ncol):
            qc.rzz(RZZ_ANGLE, _grid_index(r, c, ncol), _grid_index(r + 1, c, ncol))
    return qc


def _tdvp_params(*, tdvp_mode: str) -> StrongSimParams:
    return StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=ISING_CHI,
        tdvp_sweeps=1,
        tdvp_mode=tdvp_mode,  # type: ignore[arg-type]
        svd_threshold=1e-10,
        krylov_tol=1e-12,
    )


def _exact_reference(qc: QuantumCircuit, num_gates: int) -> np.ndarray:
    sub = QuantumCircuit(qc.num_qubits)
    for inst in qc.data[:num_gates]:
        sub.append(inst)
    return np.asarray(Statevector(sub).data, dtype=np.complex128)


def _replay_through_gate(num_gates: int, *, tdvp_mode: str = "dynamic") -> State:
    qc = _ising_2d_mapped_circuit()
    params = _tdvp_params(tdvp_mode=tdvp_mode)
    state = State(ISING_LENGTH, initial="zeros")
    dag = circuit_to_dag(qc)
    for gate_index, node in enumerate(dag.topological_op_nodes()):
        if gate_index >= num_gates:
            break
        if len(node.qargs) == 1:
            apply_single_qubit_gate(state.mps, node)
        else:
            apply_two_qubit_gate(state.mps, node, params)
    return state


def test_ising_geometry_gate12_norm_stays_unit() -> None:
    """Gate 12 LR RZZ must not jump global norm to sqrt(2) after window graft."""
    state = _replay_through_gate(TARGET_GATE + 1)
    norm = _global_mps_norm(state.mps)
    vec_norm = float(np.linalg.norm(state.mps.to_vec()))
    assert abs(norm - 1.0) < NORM_TOL, f"scalar norm {norm}"
    assert abs(vec_norm - 1.0) < NORM_TOL, f"vec norm {vec_norm}"
    assert abs(vec_norm - SQRT2) > 0.1


def test_ising_geometry_gate12_matches_main_legacy() -> None:
    """Branch dynamic TDVP should match main-style 2site TDVP at gate 12."""
    qc = _ising_2d_mapped_circuit()
    ref = _exact_reference(qc, TARGET_GATE + 1)
    branch = _replay_through_gate(TARGET_GATE + 1, tdvp_mode="dynamic")
    main = _replay_through_gate(TARGET_GATE + 1, tdvp_mode="2site")
    f_branch = _fidelity(ref, branch.mps.to_vec())
    f_main = _fidelity(ref, main.mps.to_vec())
    f_cross = _fidelity(branch.mps.to_vec(), main.mps.to_vec())
    assert f_branch > 0.94
    assert f_main > 0.94
    assert abs(f_branch - f_main) < FID_TOL
    assert f_cross > 1.0 - FID_TOL


def test_ising_geometry_full_circuit_observables() -> None:
    """Full ising_2d_mapped zeros circuit stays near exact after all gates."""
    qc = _ising_2d_mapped_circuit()
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    num_gates = len(list(circuit_to_dag(qc).topological_op_nodes()))
    state = _replay_through_gate(num_gates)
    assert abs(_global_mps_norm(state.mps) - 1.0) < NORM_TOL
    assert _fidelity(ref, state.mps.to_vec()) > 0.94
