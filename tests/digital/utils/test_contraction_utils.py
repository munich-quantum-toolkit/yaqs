# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for contraction utilities used in the equivalence checking framework.

This module contains unit tests for the digital contraction utilities used in the equivalence checking framework.
It verifies the correct functionality of tensor operations including:
  - SVD-based splitting of MPS tensors (decompose_theta)
  - Gate application routines (apply_gate, apply_temporal_zone)
  - MPO tensor merging (merge_two_site for MPS, merge_mpo_tensors for MPO)
  - Environment updates for MPOs (update_mpo, update_right_environment, update_left_environment)
  - Layer and long-range updates (apply_layer, apply_long_range_layer)
  - Generator MPO construction (construct_generator_mpo)
  - Grouping of DAG nodes (process_layer) and starting point selection (select_starting_point).

These tests ensure that the tensor network manipulations and gate applications required
for simulating quantum circuits are performed correctly.
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mpo_utils import decompose_theta
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import BaseGate, GateLibrary
from mqt.yaqs.digital.utils.contraction_utils import (
    MIN_QUBITS_FOR_MPO_PARALLEL,
    apply_gate,
    apply_layer,
    apply_long_range_layer,
    apply_temporal_zone,
    compute_pair_update,
    iterate,
    update_mpo,
)
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm, get_temporal_zone, select_starting_point

if TYPE_CHECKING:
    from numpy.typing import NDArray

##############################################################################
# Helper Functions
##############################################################################


def random_theta_6d() -> NDArray[np.complex128]:
    """Helper function to create random 8D theta for nearest-neighbor gates in equivalence checking.

    Create a random 6D tensor, e.g. for two-qubit local blocks.

    Returns:
        A random 6-dimensional tensor of shape (2,2,2,2,2,2).
    """
    rng = np.random.default_rng()
    return rng.random(size=(2, 2, 2, 2, 2, 2)) + 1j * rng.random(size=(2, 2, 2, 2, 2, 2))


def random_theta_8d() -> NDArray[np.complex128]:
    """Helper function to create random 8D theta for long-range gates in equivalence checking.

    Returns:
        A random 8-dimensional tensor of shape (2,2,2,2,2,2,2,2).
    """
    rng = np.random.default_rng()
    return rng.random(size=(2, 2, 2, 2, 2, 2, 2, 2)) + 1j * rng.random(size=(2, 2, 2, 2, 2, 2, 2, 2))


def approximate_reconstruction(
    u_tensor: NDArray[np.complex128],
    m_tensor: NDArray[np.complex128],
    original: NDArray[np.complex128],
    atol: float = 1e-10,
) -> None:
    """Helper function to reconstruct tensor.

    Check if the decomposition U * diag(S) * V (reconstructed from U and M)
    approximates 'original' within a given tolerance.

    This function re-applies the reshaping/transpose logic used in decompose_theta,
    reconstructs the matrix, and asserts that it is close to the flattened version of the original tensor.

    Args:
        u_tensor: The left factor from the SVD decomposition.
        m_tensor: The reshaped product of the singular values and right factor.
        original: The original tensor before decomposition.
        atol: Absolute tolerance for the reconstruction check. Defaults to 1e-10.
    """
    dims = original.shape
    # Reorder original to match the permutation used in decompose_theta: (0,3,2,1,4,5)
    original_reordered = np.transpose(original, (0, 3, 2, 1, 4, 5))
    original_mat = np.reshape(original_reordered, (dims[0] * dims[1] * dims[2], dims[3] * dims[4] * dims[5]))

    # Rebuild from U and M
    rank = u_tensor.shape[-1]
    u_mat = np.reshape(u_tensor, (-1, rank))  # Flatten U
    # Reorder and flatten M: from shape (dims[3], dims[4], rank, dims[5]) to (rank, dims[3]*dims[4]*dims[5])
    m_reordered = np.transpose(m_tensor, (2, 0, 1, 3))
    m_mat = np.reshape(m_reordered, (rank, dims[3] * dims[4] * dims[5]))

    reconstruction = u_mat @ m_mat
    assert np.allclose(reconstruction, original_mat, atol=atol), "Decomposition does not reconstruct original"


##############################################################################
# Tests
##############################################################################


def test_decompose_theta() -> None:
    """Test the SVD-based decomposition of a 6D tensor using decompose_theta.

    The test creates a random 6D tensor, decomposes it with a specified threshold,
    checks that the resulting tensors have the expected number of dimensions, and
    verifies that the reconstruction approximates the original tensor.
    """
    theta = random_theta_6d()
    threshold = 1e-5

    tensor1, tensor2 = decompose_theta(theta, threshold)

    # Basic shape checks: U should be rank-4 and M should be rank-4.
    assert tensor1.ndim == 4, "U should be a 4D tensor (including the rank dimension)."
    assert tensor2.ndim == 4, "M should be a 4D tensor (including the rank dimension)."

    # Check if the original tensor is approximately reconstructed.
    approximate_reconstruction(tensor1, tensor2, theta, atol=1e-5)


def test_apply_gate_identity_is_noop() -> None:
    """Identity gates skip contraction and return ``theta`` unchanged."""
    theta = random_theta_6d()
    gate = cast(
        "BaseGate",
        type(
            "IdentityGate",
            (),
            {"name": "I", "interaction": 1, "sites": [0], "matrix": np.eye(2, dtype=np.complex128)},
        )(),
    )
    updated = apply_gate(gate, theta, site0=0, site1=1, conjugate=False)
    np.testing.assert_allclose(updated, theta)


@pytest.mark.parametrize("conjugate", [False, True])
def test_apply_single_qubit_gate(*, conjugate: bool) -> None:
    """Test applying a single-qubit gate (X gate) to a tensor using apply_gate.

    The test creates a single-qubit gate from GateLibrary, sets its site,
    applies it to a random 6D tensor, and verifies that the output shape matches the input.

    Args:
        conjugate (bool): Whether to apply the conjugated version of the gate.
    """
    gate = GateLibrary.x()  # Single-qubit gate.
    gate.set_sites(0)
    theta = random_theta_6d()
    updated = apply_gate(gate, theta, site0=0, site1=1, conjugate=conjugate)
    assert updated.shape == theta.shape, "Shape should remain consistent after apply_gate."


@pytest.mark.parametrize("conjugate", [False, True])
def test_apply_two_qubit_gate(*, conjugate: bool) -> None:
    """Test applying a two-qubit gate (Rzz gate) to a tensor using apply_gate.

    The test sets up a two-qubit gate with a rotation parameter, applies it to a random 6D tensor,
    and asserts that the output tensor has the same shape as the input.

    Args:
        conjugate (bool): Whether to apply the conjugated version of the gate.
    """
    gate = GateLibrary.rzz([np.pi / 2])
    gate.set_sites(0, 1)
    theta = random_theta_6d()
    updated = apply_gate(gate, theta, site0=0, site1=1, conjugate=conjugate)
    assert updated.shape == theta.shape, "Shape should remain consistent after apply_gate."


def test_apply_temporal_zone_no_op_nodes() -> None:
    """Test that apply_temporal_zone returns the original tensor when there are no operation nodes in the DAG.

    This test constructs an empty QuantumCircuit, converts it to a DAG, and applies the temporal zone.
    The result should be identical to the input tensor.
    """
    circuit = QuantumCircuit()
    dag = circuit_to_dag(circuit)

    theta = random_theta_6d()
    qubits = [0, 1]

    updated = apply_temporal_zone(theta, dag, qubits, conjugate=False)
    assert np.allclose(updated, theta), "If no gates exist, theta should be unchanged."


def test_apply_temporal_zone_single_qubit_gates() -> None:
    """Test that apply_temporal_zone correctly applies single-qubit gates from the temporal zone.

    Constructs an Ising circuit with only single-qubit gates and verifies that applying the temporal zone
    returns a tensor with the same shape as the input.
    """
    circuit = create_ising_circuit(L=5, J=0, g=1, dt=0.1, timesteps=1)
    dag = circuit_to_dag(circuit)

    theta = random_theta_6d()
    updated = apply_temporal_zone(theta, dag, [0, 1], conjugate=False)
    assert updated.shape == theta.shape


def test_apply_temporal_zone_two_qubit_gates() -> None:
    """Test that apply_temporal_zone correctly applies two-qubit gates from the temporal zone.

    Constructs an Ising circuit with only two-qubit gates and verifies that the tensor shape remains unchanged
    after applying the temporal zone.
    """
    circuit = create_ising_circuit(L=5, J=1, g=0, dt=0.1, timesteps=1)
    dag = circuit_to_dag(circuit)

    theta = random_theta_6d()
    updated = apply_temporal_zone(theta, dag, [0, 1], conjugate=False)
    assert updated.shape == theta.shape


def test_apply_temporal_zone_mixed_qubit_gates() -> None:
    """Test that apply_temporal_zone correctly applies a mix of single- and two-qubit gates.

    Constructs an Ising circuit with both J and g nonzero, applies the temporal zone, and checks that
    the output tensor has the same shape as the input.
    """
    circuit = create_ising_circuit(L=5, J=1, g=1, dt=0.1, timesteps=1)
    dag = circuit_to_dag(circuit)

    theta = random_theta_6d()
    updated = apply_temporal_zone(theta, dag, [0, 1], conjugate=False)
    assert updated.shape == theta.shape


def test_update_mpo() -> None:
    """Test the update_mpo function on a small 2-qubit MPO.

    This test initializes an identity MPO for 2 qubits, creates an Ising circuit,
    and applies update_mpo. It then checks that each tensor in the updated MPO is a rank-4 tensor.
    """
    length = 2
    mpo = MPO.identity(length)
    circuit = create_ising_circuit(L=5, J=1, g=1, dt=0.1, timesteps=1)
    dag1 = circuit_to_dag(circuit)
    dag2 = copy.deepcopy(dag1)
    qubits = [0, 1]
    threshold = 1e-5

    update_mpo(mpo, dag1, dag2, qubits, threshold)

    # Each MPO tensor should be a 4-dimensional tensor.
    for site_tensor in mpo.tensors:
        assert site_tensor.ndim == 4, "Each MPO tensor should have 4 indices."


def test_apply_layer() -> None:
    """Test the apply_layer function by confirming that update_mpo is applied over both iterators.

    This test initializes an identity MPO for 3 qubits and applies a layer update using two sweeps.
    It then checks if the final MPO is (approximately) the identity.
    """
    length = 3
    mpo = MPO.identity(length)
    circuit = create_ising_circuit(L=5, J=1, g=1, dt=0.1, timesteps=1)
    dag1 = circuit_to_dag(circuit)
    dag2 = copy.deepcopy(dag1)
    threshold = 1e-5

    first_iterator, second_iterator = select_starting_point(length, dag1)
    apply_layer(mpo, dag1, dag2, first_iterator, second_iterator, threshold)

    assert mpo.check_if_identity(1 - 1e-13), "MPO should approximate identity after applying layer."


def test_apply_long_range_layer() -> None:
    """Test the apply_long_range_layer function for handling long-range gates.

    Initializes an identity MPO for 3 qubits and a circuit with a long-range CX gate,
    then applies the long-range layer with both conjugated and non-conjugated settings.
    Checks that the final MPO approximates the identity.
    """
    num_qubits = 3
    mpo = MPO.identity(num_qubits)
    circuit = QuantumCircuit(num_qubits)
    circuit.cx(0, 2)
    dag1 = circuit_to_dag(circuit)
    dag2 = copy.deepcopy(dag1)
    threshold = 1e-12
    apply_long_range_layer(mpo, dag1, dag2, conjugate=False, threshold=threshold)
    apply_long_range_layer(mpo, dag1, dag2, conjugate=True, threshold=threshold)

    assert mpo.check_if_identity(1 - 1e-6), "MPO should approximate identity after long-range layer."


def test_apply_long_range_layer_skips_leading_single_qubit_gate() -> None:
    """The first layer may begin with a single-qubit gate before a long-range CX."""
    mpo = MPO.identity(3)
    circuit = QuantumCircuit(3)
    circuit.h(1)
    circuit.cx(0, 2)
    dag1 = circuit_to_dag(circuit)
    dag2 = copy.deepcopy(dag1)
    apply_long_range_layer(mpo, dag1, dag2, conjugate=False, threshold=1e-12)
    assert mpo.check_if_valid_mpo()


def test_apply_long_range_layer_on_wider_mpo() -> None:
    """Long-range updates embed gate support inside a longer identity MPO."""
    mpo = MPO.identity(5)
    circuit = QuantumCircuit(5)
    circuit.cx(0, 2)
    dag1 = circuit_to_dag(circuit)
    dag2 = copy.deepcopy(dag1)
    apply_long_range_layer(mpo, dag1, dag2, conjugate=False, threshold=1e-12)
    assert mpo.length == 5
    assert mpo.check_if_valid_mpo()


def test_iterate_serial_and_parallel() -> None:
    """``iterate`` drives checkerboard sweeps with and without a thread pool."""
    qc = QuantumCircuit(MIN_QUBITS_FOR_MPO_PARALLEL)
    qc.cx(0, 2)
    dag1 = circuit_to_dag(qc)

    mpo_serial = MPO.identity(MIN_QUBITS_FOR_MPO_PARALLEL)
    iterate(mpo_serial, copy.deepcopy(dag1), copy.deepcopy(dag1), threshold=1e-12, parallel=False)
    assert mpo_serial.check_if_identity(1 - 1e-6)

    mpo_parallel = MPO.identity(MIN_QUBITS_FOR_MPO_PARALLEL)
    iterate(
        mpo_parallel,
        circuit_to_dag(qc),
        circuit_to_dag(qc),
        threshold=1e-12,
        parallel=True,
        max_workers=2,
    )
    assert mpo_parallel.check_if_identity(1 - 1e-6)


def test_apply_layer_parallel_single_pair_uses_serial_path() -> None:
    """Parallel sweeps with one pair fall back to the serial worker path."""
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    dag1 = circuit_to_dag(qc)
    dag2 = circuit_to_dag(qc)
    mpo = MPO.identity(2)
    first_iterator, second_iterator = select_starting_point(2, dag1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        apply_layer(
            mpo,
            dag1,
            dag2,
            first_iterator,
            second_iterator,
            1e-12,
            parallel=True,
            thread_pool=pool,
        )
    assert mpo.check_if_valid_mpo()


def _make_n_by_n_circuit(num_qubits: int) -> QuantumCircuit:
    """Build an ``n`` x ``n`` layered circuit (``n`` qubits, ``n`` repetitions).

    Returns:
        A layered circuit with all-qubit ``h`` gates and linear ``cx`` chains.
    """
    qc = QuantumCircuit(num_qubits)
    for _ in range(num_qubits):
        for q in range(num_qubits):
            qc.h(q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
    return qc


def test_compute_pair_update_matches_update_mpo_step() -> None:
    """Pure pair kernel reproduces a single update_mpo step on two sites."""
    qc1 = QuantumCircuit(2)
    qc1.h(0)
    qc1.cx(0, 1)
    qc2 = qc1.copy()

    mpo_ref = MPO.identity(2)
    dag1 = circuit_to_dag(qc1)
    dag2 = circuit_to_dag(qc2)
    update_mpo(mpo_ref, dag1, dag2, [0, 1], 1e-12)

    mpo_test = MPO.identity(2)

    dag1b = circuit_to_dag(qc1)
    dag2b = circuit_to_dag(qc2)
    zone1 = get_temporal_zone(dag1b, [0, 1])
    zone2 = get_temporal_zone(dag2b, [0, 1])
    gates1 = convert_dag_to_tensor_algorithm(zone1)
    gates2 = convert_dag_to_tensor_algorithm(zone2)
    t0, t1 = compute_pair_update(
        mpo_test.tensors[0],
        mpo_test.tensors[1],
        gates1,
        gates2,
        1e-12,
        [0, 1],
        apply_conjugate_on_second=True,
    )
    mpo_test.tensors[0] = t0
    mpo_test.tensors[1] = t1

    for a, b in zip(mpo_ref.tensors, mpo_test.tensors, strict=True):
        assert np.allclose(a, b, atol=1e-10)


def test_compute_pair_update_conjugates_second_zone() -> None:
    """``gates2`` are conjugated whenever the second zone is non-empty."""
    rz_gate = GateLibrary.rz([np.pi / 4])
    rz_gate.set_sites(0)

    tensor_n = MPO.identity(2).tensors[0]
    tensor_n1 = MPO.identity(2).tensors[1]

    with_conjugate = compute_pair_update(
        tensor_n,
        tensor_n1,
        [],
        [rz_gate],
        1e-12,
        [0, 1],
        apply_conjugate_on_second=True,
    )
    without_conjugate = compute_pair_update(
        tensor_n,
        tensor_n1,
        [],
        [rz_gate],
        1e-12,
        [0, 1],
        apply_conjugate_on_second=False,
    )

    assert not np.allclose(with_conjugate[0], without_conjugate[0], atol=1e-10)


def test_apply_layer_parallel_requires_thread_pool() -> None:
    """Parallel layer updates without a thread pool raise at runtime."""
    num_qubits = MIN_QUBITS_FOR_MPO_PARALLEL
    qc = _make_n_by_n_circuit(num_qubits)
    dag1 = circuit_to_dag(qc)
    dag2 = circuit_to_dag(qc)
    mpo = MPO.identity(num_qubits)
    first_iterator, second_iterator = select_starting_point(num_qubits, dag1)

    with pytest.raises(RuntimeError, match="thread pool"):
        apply_layer(
            mpo,
            dag1,
            dag2,
            first_iterator,
            second_iterator,
            1e-6,
            parallel=True,
            max_workers=2,
            thread_pool=None,
        )


def test_apply_layer_parallel_with_thread_pool() -> None:
    """Parallel apply_layer succeeds when a thread pool is provided."""
    num_qubits = MIN_QUBITS_FOR_MPO_PARALLEL
    qc = _make_n_by_n_circuit(num_qubits)
    dag1 = circuit_to_dag(qc)
    dag2 = circuit_to_dag(qc)
    mpo_serial = MPO.identity(num_qubits)
    mpo_parallel = MPO.identity(num_qubits)
    first_iterator, second_iterator = select_starting_point(num_qubits, dag1)

    apply_layer(
        mpo_serial,
        dag1,
        dag2,
        first_iterator,
        second_iterator,
        1e-6,
        parallel=False,
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        apply_layer(
            mpo_parallel,
            dag1,
            dag2,
            first_iterator,
            second_iterator,
            1e-6,
            parallel=True,
            max_workers=2,
            thread_pool=pool,
        )

    for serial_tensor, parallel_tensor in zip(mpo_serial.tensors, mpo_parallel.tensors, strict=True):
        assert np.allclose(serial_tensor, parallel_tensor, atol=1e-8)
