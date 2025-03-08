# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the MPO utility functions used in the equivalence checking framework.

This module contains unit tests for the MPO utility functions used in the equivalence checking framework.
It verifies the correct functionality of tensor operations including:
  - SVD-based splitting of MPS tensors (decompose_theta)
  - Gate application routines (apply_gate, apply_temporal_zone)
  - MPO tensor merging (merge_mps_tensors, merge_mpo_tensors)
  - Environment updates for MPOs (update_mpo, update_right_environment, update_left_environment)
  - Layer and long-range updates (apply_layer, apply_long_range_layer)
  - Generator MPO construction (construct_generator_mpo)
  - Grouping of DAG nodes (process_layer) and starting point selection (select_starting_point).

These tests ensure that the tensor network manipulations and gate applications required
for simulating quantum circuits are performed correctly.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from mqt.yaqs.circuits.utils.dag_utils import select_starting_point
from mqt.yaqs.circuits.utils.mpo_utils import (
    apply_gate,
    apply_layer,
    apply_long_range_layer,
    apply_temporal_zone,
    decompose_theta,
    update_mpo,
)
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import GateLibrary

if TYPE_CHECKING:
    from numpy.typing import NDArray

##############################################################################
# Helper Functions
##############################################################################

rng = np.random.default_rng()


def random_theta_6d() -> NDArray[np.float64]:
    """Create a random 6D tensor, e.g. for two-qubit local blocks.

    Returns:
        NDArray[np.float64]: A random 6-dimensional tensor of shape (2,2,2,2,2,2).
    """
    return rng.random(size=(2, 2, 2, 2, 2, 2))


def random_theta_8d() -> NDArray[np.float64]:
    """Create a random 8D tensor, e.g. for 'long-range' gates.

    Returns:
        NDArray[np.float64]: A random 8-dimensional tensor of shape (2,2,2,2,2,2,2,2).
    """
    rng = np.random.default_rng()
    return rng.random(size=(2, 2, 2, 2, 2, 2, 2, 2))


def approximate_reconstruction(
    U: NDArray[np.float64], M: NDArray[np.float64], original: NDArray[np.float64], atol: float = 1e-10
) -> None:
    """Check if the decomposition U * diag(S) * V (reconstructed from U and M)
    approximates 'original' within a given tolerance.

    This function re-applies the reshaping/transpose logic used in decompose_theta,
    reconstructs the matrix, and asserts that it is close to the flattened version of the original tensor.

    Args:
        U (NDArray[np.float64]): The left factor from the SVD decomposition.
        M (NDArray[np.float64]): The reshaped product of the singular values and right factor.
        original (NDArray[np.float64]): The original tensor before decomposition.
        atol (float, optional): Absolute tolerance for the reconstruction check. Defaults to 1e-10.

    Raises:
        AssertionError: If the reconstructed matrix does not match the original within the tolerance.
    """
    dims = original.shape
    # Reorder original to match the permutation used in decompose_theta: (0,3,2,1,4,5)
    original_reordered = np.transpose(original, (0, 3, 2, 1, 4, 5))
    original_mat = np.reshape(original_reordered, (dims[0] * dims[1] * dims[2], dims[3] * dims[4] * dims[5]))

    # Rebuild from U and M
    rank = U.shape[-1]
    U_mat = np.reshape(U, (-1, rank))  # Flatten U
    # Reorder and flatten M: from shape (dims[3], dims[4], rank, dims[5]) to (rank, dims[3]*dims[4]*dims[5])
    M_reordered = np.transpose(M, (2, 0, 1, 3))
    M_mat = np.reshape(M_reordered, (rank, dims[3] * dims[4] * dims[5]))

    reconstruction = U_mat @ M_mat
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

    U, M = decompose_theta(theta, threshold)

    # Basic shape checks: U should be rank-4 and M should be rank-4.
    assert U.ndim == 4, "U should be a 4D tensor (including the rank dimension)."
    assert M.ndim == 4, "M should be a 4D tensor (including the rank dimension)."

    # Check if the original tensor is approximately reconstructed.
    approximate_reconstruction(U, M, theta, atol=1e-5)


@pytest.mark.parametrize("conjugate", [False, True])
def test_apply_single_qubit_gate(conjugate: bool) -> None:
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
def test_apply_two_qubit_gate(conjugate: bool) -> None:
    """Test applying a two-qubit gate (Rzz gate) to a tensor using apply_gate.

    The test sets up a two-qubit gate with a rotation parameter, applies it to a random 6D tensor,
    and asserts that the output tensor has the same shape as the input.

    Args:
        conjugate (bool): Whether to apply the conjugated version of the gate.
    """
    gate = GateLibrary.rzz()  # Two-qubit gate.
    gate.set_params([np.pi / 2])
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
    mpo = MPO()
    length = 2
    mpo.init_identity(length)
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
    mpo = MPO()
    length = 3
    mpo.init_identity(length)
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
    mpo = MPO()
    num_qubits = 3
    mpo.init_identity(num_qubits)
    circuit = QuantumCircuit(num_qubits)
    circuit.cx(0, 2)
    dag1 = circuit_to_dag(circuit)
    dag2 = copy.deepcopy(dag1)
    threshold = 1e-12
    apply_long_range_layer(mpo, dag1, dag2, conjugate=False, threshold=threshold)
    apply_long_range_layer(mpo, dag1, dag2, conjugate=True, threshold=threshold)

    assert mpo.check_if_identity(1 - 1e-6), "MPO should approximate identity after long-range layer."
