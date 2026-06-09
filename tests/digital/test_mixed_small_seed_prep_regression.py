# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Regression: mixed_small L=10 zeros must not lose fidelity at LR CX gate 2."""

from __future__ import annotations

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

LENGTH = 10
CHI = 16
TARGET_GATE = 2
RZZ_ANGLE = 0.3
NORM_TOL = 1e-6
FID_TOL = 1e-3


def _mixed_small_circuit(length: int = LENGTH) -> QuantumCircuit:
    qc = QuantumCircuit(length)
    qc.h(0)
    qc.cx(4, 5)
    qc.cx(0, length - 1)
    qc.rzz(RZZ_ANGLE, 0, length - 1)
    return qc


def _tdvp_params(*, tdvp_mode: str) -> StrongSimParams:
    return StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=CHI,
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
    qc = _mixed_small_circuit()
    params = _tdvp_params(tdvp_mode=tdvp_mode)
    state = State(LENGTH, initial="zeros")
    dag = circuit_to_dag(qc)
    for gate_index, node in enumerate(dag.topological_op_nodes()):
        if gate_index >= num_gates:
            break
        if len(node.qargs) == 1:
            apply_single_qubit_gate(state.mps, node)
        else:
            apply_two_qubit_gate(state.mps, node, params)
    return state


def test_mixed_small_gate2_norm_stays_unit() -> None:
    """Gate 2 LR CX must preserve global norm."""
    state = _replay_through_gate(TARGET_GATE + 1)
    norm = _global_mps_norm(state.mps)
    vec_norm = float(np.linalg.norm(state.mps.to_vec()))
    assert abs(norm - 1.0) < NORM_TOL, f"scalar norm {norm}"
    assert abs(vec_norm - 1.0) < NORM_TOL, f"vec norm {vec_norm}"


def test_mixed_small_gate2_matches_main_legacy() -> None:
    """Branch dynamic TDVP should match main-style 2site TDVP at gate 2 LR CX."""
    qc = _mixed_small_circuit()
    ref = _exact_reference(qc, TARGET_GATE + 1)
    branch = _replay_through_gate(TARGET_GATE + 1, tdvp_mode="dynamic")
    main = _replay_through_gate(TARGET_GATE + 1, tdvp_mode="2site")
    f_branch = _fidelity(ref, branch.mps.to_vec())
    f_main = _fidelity(ref, main.mps.to_vec())
    f_cross = _fidelity(branch.mps.to_vec(), main.mps.to_vec())
    assert f_branch > 0.99
    assert f_main > 0.99
    assert abs(f_branch - f_main) < FID_TOL
    assert f_cross > 1.0 - FID_TOL


@pytest.mark.parametrize("chi", (16, 32))
def test_mixed_small_full_circuit(chi: int) -> None:
    """Full mixed_small zeros circuit matches exact reference."""
    qc = _mixed_small_circuit()
    ref = np.asarray(Statevector(qc).data, dtype=np.complex128)
    num_gates = len(list(circuit_to_dag(qc).topological_op_nodes()))
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=chi,
        tdvp_sweeps=1,
        tdvp_mode="dynamic",
        svd_threshold=1e-10,
        krylov_tol=1e-12,
    )
    state = State(LENGTH, initial="zeros")
    for node in circuit_to_dag(qc).topological_op_nodes():
        if len(node.qargs) == 1:
            apply_single_qubit_gate(state.mps, node)
        else:
            apply_two_qubit_gate(state.mps, node, params)
    assert abs(_global_mps_norm(state.mps) - 1.0) < NORM_TOL
    assert _fidelity(ref, state.mps.to_vec()) > 0.99
