import pytest

import copy
import numpy as np

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
    from yaqs.circuits.equivalence_checking.mpo_utils import decompose_theta
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
    from yaqs.core.libraries.gate_library import GateLibrary
    from yaqs.circuits.equivalence_checking.mpo_utils import apply_gate

    if interaction == 1:
        attr = getattr(GateLibrary, 'x')
        gate = attr()
        gate.set_sites(0)
    else:
        attr = getattr(GateLibrary, "rzz")
        gate = attr()
        gate.set_params([np.pi/2])
        gate.set_sites(0, 1)

    theta = random_theta_6d()

    updated = apply_gate(gate, theta, site0=0, site1=1, conjugate=conjugate)
    assert updated.shape == theta.shape, "Shape should remain consistent after apply_gate."


def test_apply_temporal_zone_no_op_nodes():
    """
    If the DAG has no op_nodes, apply_temporal_zone should return theta unchanged.
    """
    from qiskit.circuit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from yaqs.circuits.equivalence_checking.mpo_utils import apply_temporal_zone
    circuit = QuantumCircuit()
    dag = circuit_to_dag(circuit)

    theta = random_theta_6d()
    qubits = [0, 1]

    updated = apply_temporal_zone(theta, dag, qubits, conjugate=False)
    assert np.allclose(updated, theta), "If no gates, theta should be unchanged."


def test_apply_temporal_zone_single_qubit_gates():
    """
    If the DAG has one gate in the 'temporal zone', it should be applied.
    """
    from qiskit.converters import circuit_to_dag
    from yaqs.circuits.equivalence_checking.mpo_utils import apply_temporal_zone
    from yaqs.core.libraries.circuit_library import create_Ising_circuit
    model = {'name': 'Ising', 'L': 5, 'J': 0, 'g': 1}
    circuit = create_Ising_circuit(model, dt=0.1, timesteps=1)
    dag = circuit_to_dag(circuit)

    theta = random_theta_6d()
    updated = apply_temporal_zone(theta, dag, [0,1], conjugate=False)

    assert updated.shape == theta.shape


def test_apply_temporal_zone_two_qubit_gates():
    """
    If the DAG has one gate in the 'temporal zone', it should be applied.
    """
    from qiskit.converters import circuit_to_dag
    from yaqs.circuits.equivalence_checking.mpo_utils import apply_temporal_zone
    from yaqs.core.libraries.circuit_library import create_Ising_circuit
    model = {'name': 'Ising', 'L': 5, 'J': 1, 'g': 0}
    circuit = create_Ising_circuit(model, dt=0.1, timesteps=1)
    dag = circuit_to_dag(circuit)

    theta = random_theta_6d()
    updated = apply_temporal_zone(theta, dag, [0,1], conjugate=False)

    assert updated.shape == theta.shape


def test_apply_temporal_zone_mixed_qubit_gates():
    """
    If the DAG has one gate in the 'temporal zone', it should be applied.
    """
    from qiskit.converters import circuit_to_dag
    from yaqs.circuits.equivalence_checking.mpo_utils import apply_temporal_zone
    from yaqs.core.libraries.circuit_library import create_Ising_circuit
    model = {'name': 'Ising', 'L': 5, 'J': 1, 'g': 1}
    circuit = create_Ising_circuit(model, dt=0.1, timesteps=1)
    dag = circuit_to_dag(circuit)

    theta = random_theta_6d()
    updated = apply_temporal_zone(theta, dag, [0,1], conjugate=False)

    assert updated.shape == theta.shape


def test_update_MPO():
    """
    Test update_MPO with a small 2-qubit MPO. We'll intercept apply_temporal_zone calls.
    """
    from qiskit.converters import circuit_to_dag
    from yaqs.core.data_structures.networks import MPO
    from yaqs.core.libraries.circuit_library import create_Ising_circuit
    from yaqs.circuits.equivalence_checking.mpo_utils import update_MPO
    mpo = MPO()
    length = 2
    mpo.init_identity(length)
    model = {'name': 'Ising', 'L': 5, 'J': 1, 'g': 1}
    circuit = create_Ising_circuit(model, dt=0.1, timesteps=1)
    dag1 = circuit_to_dag(circuit)
    dag2 = copy.deepcopy(dag1)
    qubits = [0, 1]
    threshold = 1e-5

    update_MPO(mpo, dag1, dag2, qubits, threshold)

    # Check final shapes
    for site_tensor in mpo.tensors:
        # Each MPO site is rank-4 for a 1D chain: (pdim, pdim, bond_in, bond_out)
        assert site_tensor.ndim == 4


def test_apply_layer():
    """
    Basic test for apply_layer. We'll confirm update_MPO is called for first and second iterators.
    """
    from qiskit.converters import circuit_to_dag
    from yaqs.core.data_structures.networks import MPO
    from yaqs.core.libraries.circuit_library import create_Ising_circuit
    from yaqs.circuits.equivalence_checking.mpo_utils import apply_layer, select_starting_point
    mpo = MPO()
    length = 3
    mpo.init_identity(length)
    model = {'name': 'Ising', 'L': 5, 'J': 1, 'g': 1}
    circuit = create_Ising_circuit(model, dt=0.1, timesteps=1)
    dag1 = circuit_to_dag(circuit)
    dag2 = copy.deepcopy(dag1)
    threshold = 1e-5

    first_iterator, second_iterator = select_starting_point(length, dag1)
    apply_layer(mpo, dag1, dag2, first_iterator, second_iterator, threshold)

    assert mpo.check_if_identity(1-1e-13)


def test_apply_long_range_layer():
    from qiskit.circuit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from yaqs.core.data_structures.networks import MPO
    from yaqs.circuits.equivalence_checking.mpo_utils import apply_long_range_layer, select_starting_point
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

    assert mpo.check_if_identity(1-1e-6)
