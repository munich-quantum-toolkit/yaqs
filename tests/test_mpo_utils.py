import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from yaqs.circuits.equivalence_checking.mpo_utils import (
    decompose_theta,
    apply_gate,
    apply_temporal_zone,
    update_MPO,
    apply_layer
)
from yaqs.general.data_structures.MPO import MPO
from yaqs.general.libraries.tensor_library import TensorLibrary


##############################################################################
# Helper Functions
##############################################################################

def random_theta_6d():
    """Create a random 6D tensor, e.g. for two-qubit local blocks."""
    return np.random.rand(2, 2, 2, 2, 2, 2)

def random_theta_8d():
    """Create a random 8D tensor, e.g. for 'long-range' gates."""
    return np.random.rand(2, 2, 2, 2, 2, 2, 2, 2)

def approximate_reconstruction(U, M, original, atol=1e-10):
    """
    Check if U * diag(S) * V reconstructs 'original' (up to a tolerance).
    Re-applies the reshaping/transpose logic used in decompose_theta.
    """
    dims = original.shape
    # Reorder original to the same permutation used in decompose_theta => (0,3,2,1,4,5)
    original_reordered = np.transpose(original, (0, 3, 2, 1, 4, 5))
    original_mat = np.reshape(original_reordered, (dims[0]*dims[1]*dims[2], dims[3]*dims[4]*dims[5]))

    # Rebuild from U and M
    rank = U.shape[-1]
    U_mat = np.reshape(U, (-1, rank))  # Flatten U
    # M is shape (dims[3], dims[4], rank, dims[5]), so reorder to (rank, dims[3], dims[4], dims[5])
    M_reordered = np.transpose(M, (2, 0, 1, 3))
    M_mat = np.reshape(M_reordered, (rank, dims[3]*dims[4]*dims[5]))

    reconstruction = U_mat @ M_mat
    assert np.allclose(reconstruction, original_mat, atol=atol), "Decomposition does not reconstruct original"


##############################################################################
# Tests
##############################################################################

def test_decompose_theta():
    """Test SVD-based decomposition of a 6D tensor."""
    theta = random_theta_6d()
    threshold = 1e-5

    U, M = decompose_theta(theta, threshold)

    # Basic shape checks
    assert U.ndim == 4  # e.g. (dims[0], dims[1], dims[2], rank)
    assert M.ndim == 4  # e.g. (dims[3], dims[4], rank, dims[5])

    # Check reconstruction
    approximate_reconstruction(U, M, theta, atol=1e-5)


@pytest.mark.parametrize("interaction,conjugate", [(1, False), (1, True), (2, False), (2, True)])
def test_apply_gate(interaction, conjugate):
    """
    Test applying single- or two-qubit gates to a 6D (or 8D) tensor,
    with or without conjugation.
    """
    gate_mock = MagicMock()
    gate_mock.interaction = interaction
    gate_mock.name = "X" if interaction == 1 else "CNOT"
    gate_mock.sites = [0, 1] if interaction == 2 else [0]

    if interaction == 1:
        # Single-qubit gate (2x2)
        gate_mock.tensor = getattr(TensorLibrary, "x")().tensor
    else:
        gate_mock = getattr(TensorLibrary, "cx")()
        gate_mock.set_sites(0, 1)

    # For simplicity, we'll just use a 6D tensor (2-qubit nearest neighbor scenario).
    theta = random_theta_6d()

    updated = apply_gate(gate_mock, theta, site0=0, site1=1, conjugate=conjugate)
    assert updated.shape == theta.shape, "Shape should remain consistent after apply_gate."


def test_apply_temporal_zone_no_op_nodes():
    """
    If the DAG has no op_nodes, apply_temporal_zone should return theta unchanged.
    """
    dag_mock = MagicMock()
    dag_mock.op_nodes.return_value = []  # No gates
    theta = random_theta_6d()
    qubits = [0, 1]

    updated = apply_temporal_zone(theta, dag_mock, qubits, conjugate=False)
    assert np.allclose(updated, theta), "If no gates, theta should be unchanged."


@patch("yaqs.circuits.equivalence_checking.mpo_utils.get_temporal_zone")
@patch("yaqs.circuits.equivalence_checking.mpo_utils.convert_dag_to_tensor_algorithm")
def test_apply_temporal_zone_one_gate(mock_convert, mock_get_zone):
    """
    If the DAG has one gate in the 'temporal zone', it should be applied.
    """
    dag_mock = MagicMock()
    dag_mock.op_nodes.return_value = ["some_node"]  # So the code sees we have gates

    mock_get_zone.return_value = MagicMock()  # Just a placeholder zone
    # Return a single gate from convert_dag_to_tensor_algorithm
    gate_mock = MagicMock()
    gate_mock.interaction = 1
    gate_mock.sites = [0]
    gate_mock.name = "X"
    gate_mock.tensor = TensorLibrary.x().matrix
    mock_convert.return_value = [gate_mock]

    theta = random_theta_6d()
    updated = apply_temporal_zone(theta, dag_mock, [0,1], conjugate=False)

    # shape should remain the same
    assert updated.shape == theta.shape
    # We could also check that apply_gate was effectively used,
    # but since we didn't patch 'apply_gate', no direct check is possible here.


@patch("yaqs.circuits.equivalence_checking.mpo_utils.apply_temporal_zone", side_effect=lambda t, *_, **__: t)
def test_update_MPO_basic(mock_apply_temporal_zone):
    """
    Test update_MPO with a small 2-qubit MPO. We'll intercept apply_temporal_zone calls.
    """
    mpo = MPO()
    length = 2
    pdim = 2
    mpo.init_identity(length, pdim)

    dag1 = MagicMock()
    dag2 = MagicMock()
    qubits = [0,1]
    threshold = 1e-5

    update_MPO(mpo, dag1, dag2, qubits, threshold)

    # Called twice: once for circuit1 (conjugate=False), once for circuit2 (conjugate=True)
    assert mock_apply_temporal_zone.call_count == 2

    # Check final shapes
    for site_tensor in mpo.tensors:
        # Each MPO site is rank-4 for a 1D chain: (pdim, pdim, bond_in, bond_out)
        assert site_tensor.ndim == 4


@patch("yaqs.circuits.equivalence_checking.mpo_utils.update_MPO")
def test_apply_layer(mock_update_mpo):
    """
    Basic test for apply_layer. We'll confirm update_MPO is called for first and second iterators.
    """
    mpo = MPO()
    mpo.init_identity(4, physical_dimension=2)

    circuit1_dag = MagicMock()
    circuit2_dag = MagicMock()
    # Example: even then odd layering
    first_iterator = range(0, 3, 2)   # e.g. n = 0, 2 => [0,1], [2,3]
    second_iterator = range(1, 3, 2)  # e.g. n = 1 => [1,2]
    threshold = 1e-5

    apply_layer(mpo, circuit1_dag, circuit2_dag, first_iterator, second_iterator, threshold)

    # We expect update_MPO calls for each n in both iterators
    assert mock_update_mpo.call_count == (len(list(first_iterator)) + len(list(second_iterator)))

