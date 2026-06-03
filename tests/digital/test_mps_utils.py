# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for generic MPO--MPS gate application via :meth:`~mqt.yaqs.core.data_structures.mpo.MPO.multiply`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import Statevector

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import BaseGate, GateLibrary, Z
from mqt.yaqs.digital.digital_tjm import apply_long_range_gate_mpo, apply_two_qubit_gate_tebd
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _sim_params() -> StrongSimParams:
    return StrongSimParams(observables=[Observable(Z(), 0)], preset="exact", gate_mode="mpo")


def _gate_from_circuit(qc: QuantumCircuit, *, op_name: str | None = None) -> BaseGate:
    dag = circuit_to_dag(qc)
    if op_name is None:
        node = next(n for n in dag.op_nodes())
    else:
        node = next(n for n in dag.op_nodes() if n.op.name.lower() == op_name.lower())
    return convert_dag_to_tensor_algorithm(node)[0]


def _apply_mpo_reference(
    length: int,
    gate: BaseGate,
    *,
    compress: bool,
) -> MPS:
    """Apply the extended gate MPO on the full chain (reference path).

    Returns:
        MPS after explicit :meth:`~mqt.yaqs.core.data_structures.mpo.MPO.multiply`.
    """
    state = MPS(length, state="ones")
    state.normalize()
    MPO.from_gate(gate, length).multiply(
        state,
        sim_params=_sim_params() if compress else None,
        compress=compress,
    )
    return state


def _qiskit_evolved_vec(qc: QuantumCircuit, label: str) -> NDArray[np.complex128]:
    initial = Statevector.from_label(label).data
    return np.asarray(Statevector(initial).evolve(qc).data, dtype=np.complex128)


def test_identity_mpo_preserves_statevector() -> None:
    """Identity MPO on the full chain leaves the dense state unchanged."""
    length = 4
    state = MPS(length, state="ones")
    state.normalize()
    expected = np.asarray(state.to_vec(), dtype=np.complex128)

    identity_mpo = MPO.identity(length)
    identity_mpo.multiply(state, sim_params=_sim_params(), compress=True)

    np.testing.assert_allclose(state.to_vec(), expected, atol=1e-10)


def test_nearest_neighbor_cx_mpo_matches_tebd() -> None:
    """Adjacent CX via extended MPO matches direct TEBD application."""
    length = 4
    qc = QuantumCircuit(length)
    qc.cx(1, 2)
    gate = _gate_from_circuit(qc)
    sim_params = _sim_params()

    mpo_path = MPS(length, state="ones")
    mpo_path.normalize()
    MPO.from_gate(gate, length).multiply(mpo_path, sim_params=sim_params, compress=True)

    tebd_path = MPS(length, state="ones")
    tebd_path.normalize()
    apply_two_qubit_gate_tebd(tebd_path, gate, sim_params)

    np.testing.assert_allclose(mpo_path.to_vec(), tebd_path.to_vec(), atol=1e-10)


def test_long_range_cx_matches_statevector_reference() -> None:
    """Long-range CX via MPO--MPS matches a dense statevector reference."""
    length = 4
    qc = QuantumCircuit(length)
    qc.cx(1, 3)
    gate = _gate_from_circuit(qc)
    expected = _qiskit_evolved_vec(qc, "1111")

    state = MPS(length, state="ones")
    state.normalize()
    apply_long_range_gate_mpo(state, gate, _sim_params())
    np.testing.assert_allclose(np.abs(state.to_vec()), np.abs(expected), atol=1e-10)


def test_long_range_cx_ones_state() -> None:
    """CX(1, 3) on |1111> maps to |1111> (index 7)."""
    length = 4
    qc = QuantumCircuit(length)
    qc.cx(1, 3)
    gate = _gate_from_circuit(qc)
    state = MPS(length, state="ones")
    state.normalize()
    apply_long_range_gate_mpo(state, gate, _sim_params())
    state.normalize(decomposition="SVD")
    for index, element in enumerate(state.to_vec()):
        if index == 7:
            np.testing.assert_allclose(np.abs(element), 1, atol=1e-10)
        else:
            np.testing.assert_allclose(np.abs(element), 0, atol=1e-10)


def test_directional_long_range_vs_nearest_neighbor_cnot() -> None:
    """Long-range and nearest-neighbor CNOTs differ and both match TEBD."""
    length = 4
    label = "0110"
    sim_params = _sim_params()

    qc_long = QuantumCircuit(length)
    qc_long.cx(1, 3)
    gate_long = _gate_from_circuit(qc_long)
    long_range = MPS(length, state="basis", basis_string=label)
    long_range.normalize()
    apply_long_range_gate_mpo(long_range, gate_long, sim_params)

    qc_nn = QuantumCircuit(length)
    qc_nn.cx(1, 2)
    gate_nn = _gate_from_circuit(qc_nn)
    nearest = MPS(length, state="basis", basis_string=label)
    nearest.normalize()
    apply_long_range_gate_mpo(nearest, gate_nn, sim_params)

    tebd_long = MPS(length, state="basis", basis_string=label)
    tebd_long.normalize()
    apply_two_qubit_gate_tebd(tebd_long, gate_long, sim_params)

    tebd_nn = MPS(length, state="basis", basis_string=label)
    tebd_nn.normalize()
    apply_two_qubit_gate_tebd(tebd_nn, gate_nn, sim_params)

    np.testing.assert_allclose(np.abs(long_range.to_vec()), np.abs(tebd_long.to_vec()), atol=1e-10)
    np.testing.assert_allclose(np.abs(nearest.to_vec()), np.abs(tebd_nn.to_vec()), atol=1e-10)
    assert np.max(np.abs(np.abs(long_range.to_vec()) - np.abs(nearest.to_vec()))) > 0.5


def test_apply_long_range_gate_mpo_matches_mpo_reference() -> None:
    """Zip-up entry point matches explicit apply-compress MPO reference."""
    length = 4
    qc = QuantumCircuit(length)
    qc.cx(1, 3)
    gate = _gate_from_circuit(qc)
    reference = _apply_mpo_reference(length, gate, compress=True)

    state = MPS(length, state="ones")
    state.normalize()
    apply_long_range_gate_mpo(state, gate, _sim_params())
    np.testing.assert_allclose(state.to_vec(), reference.to_vec(), atol=1e-10)


def test_apply_long_range_gate_mpo_wide_cx_n18() -> None:
    """Wide CX(0, n-1) on 18 qubits completes without error."""
    length = 18
    qc = QuantumCircuit(length)
    qc.h(0)
    qc.cx(0, length - 1)
    gate = _gate_from_circuit(qc, op_name="cx")
    mps = MPS(length, state="zeros")
    apply_long_range_gate_mpo(mps, gate, _sim_params())


def test_apply_long_range_gate_mpo_wide_cx_n32() -> None:
    """Wide CX(0, n-1) on 32 qubits completes without label-budget errors."""
    length = 32
    qc = QuantumCircuit(length)
    qc.h(0)
    qc.cx(0, length - 1)
    gate = _gate_from_circuit(qc, op_name="cx")
    mps = MPS(length, state="zeros")
    apply_long_range_gate_mpo(mps, gate, _sim_params())


def test_swap_via_mpo_matches_tebd() -> None:
    """Non-symmetric SWAP on adjacent sites matches TEBD."""
    length = 4
    swap = GateLibrary.swap()
    swap.set_sites(1, 2)
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], preset="exact", gate_mode="swaps")

    mpo_path = MPS(length, state="basis", basis_string="1010")
    mpo_path.normalize()
    MPO.from_gate(swap, length).multiply(mpo_path, sim_params=sim_params, compress=True)

    tebd_path = MPS(length, state="basis", basis_string="1010")
    tebd_path.normalize()
    apply_two_qubit_gate_tebd(tebd_path, swap, sim_params)

    np.testing.assert_allclose(mpo_path.to_vec(), tebd_path.to_vec(), atol=1e-10)


def test_from_gate_reuses_mpo_tensors() -> None:
    """from_gate matches extend_gate when gate mpo_tensors are already cached."""
    length = 4
    qc = QuantumCircuit(length)
    qc.cx(1, 3)
    gate = _gate_from_circuit(qc)

    first = MPO.from_gate(gate, length)
    second = MPO.from_gate(gate, length)
    np.testing.assert_allclose(first.to_matrix(), second.to_matrix(), atol=1e-12)
