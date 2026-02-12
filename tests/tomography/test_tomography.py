
import numpy as np
import pytest
from mqt.yaqs.tomography.tomography import ProcessTensor, _vec_to_rho, _get_basis_states
from qiskit.quantum_info import DensityMatrix, entropy

def test_vec_to_rho():
    """Test the vector to density matrix conversion."""
    # Test with |0><0|
    psi0 = np.array([1, 0], dtype=complex)
    rho0 = np.outer(psi0, psi0.conj())
    vec0 = rho0.reshape(-1)
    rho_out = _vec_to_rho(vec0)
    np.testing.assert_allclose(rho_out, rho0, atol=1e-15)

    # Test with |+><+|
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho_plus = np.outer(psi_plus, psi_plus.conj())
    vec_plus = rho_plus.reshape(-1)
    rho_out = _vec_to_rho(vec_plus)
    np.testing.assert_allclose(rho_out, rho_plus, atol=1e-15)


def get_standard_basis():
    """Returns the standard 6-state Pauli basis for testing."""
    psi_0 = np.array([1, 0], dtype=complex)
    psi_1 = np.array([0, 1], dtype=complex)
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    psi_i_minus = np.array([1, -1j], dtype=complex) / np.sqrt(2)

    states = [
        ("zeros", psi_0),
        ("ones", psi_1),
        ("x+", psi_plus),
        ("x-", psi_minus),
        ("y+", psi_i_plus),
        ("y-", psi_i_minus),
    ]
    basis_set = []
    for name, psi in states:
        rho = np.outer(psi, psi.conj())
        basis_set.append((name, psi, rho))
    return basis_set

def test_holevo_information_identity():
    """Test Holevo information for an identity channel."""
    # Use standard basis to ensure source entropy is 1
    basis = get_standard_basis()
    num_frames = len(basis)
    
    # Create tensor for 1 timestep
    tensor = np.zeros((4, num_frames), dtype=complex)
    
    for i, (_, _, rho) in enumerate(basis):
        tensor[:, i] = rho.reshape(-1)
        
    pt = ProcessTensor(tensor, [1.0])
    
    # For identity channel on single qubit with uniform input distribution over 2-design:
    # Average output is Identity/2
    # S(rho_avg) = S(I/2) = 1 bit (base 2)
    # S(rho_i) = S(pure state) = 0
    # Holevo = 1 - 0 = 1
    
    holevo = pt.holevo_information(base=2)
    assert np.isclose(holevo, 1.0, atol=1e-10)

def test_holevo_information_fully_depolarizing():
    """Test Holevo information for a fully depolarizing channel."""
    # Maps everything to I/2
    basis = get_standard_basis()
    num_frames = len(basis)
    
    tensor = np.zeros((4, num_frames), dtype=complex)
    
    rho_mixed = 0.5 * np.eye(2, dtype=complex)
    vec_mixed = rho_mixed.reshape(-1)
    
    for i in range(num_frames):
        tensor[:, i] = vec_mixed
        
    pt = ProcessTensor(tensor, [1.0])
    
    # S(rho_avg) = S(I/2) = 1
    # S(rho_i) = S(I/2) = 1
    # Holevo = 1 - 1 = 0
    
    holevo = pt.holevo_information(base=2)
    assert np.isclose(holevo, 0.0, atol=1e-10)

def test_holevo_information_conditional():
    """Test conditional Holevo information."""
    # Construct a 2-step process where output depends ONLY on step 0.
    # T[out, i, j] = rho_i
    basis = get_standard_basis()
    num_frames = len(basis)
    
    # Shape: (4, N, N)
    tensor = np.zeros((4, num_frames, num_frames), dtype=complex)
    
    for i in range(num_frames):
        rho_i = basis[i][2]
        vec_i = rho_i.reshape(-1)
        for j in range(num_frames):
            tensor[:, i, j] = vec_i
            
    pt = ProcessTensor(tensor, [1.0, 1.0])
    
    # CASE 1: Fix step 0 to index 0 (state |0><0|)
    # The output is always |0><0|, regardless of step 1.
    # S(rho_avg) = S(|0><0|) = 0
    # S(rho_seq) = 0
    # Holevo = 0
    h_cond_0 = pt.holevo_information_conditional(fixed_step=0, fixed_idx=0, base=2)
    assert np.isclose(h_cond_0, 0.0, atol=1e-10)
    
    # CASE 2: Fix step 1 to index 0.
    # The output still varies with step 0.
    # It acts like an identity channel from step 0 to output.
    # S(rho_avg) = 1 (maximally mixed average of basis)
    # S(rho_seq) = 0 (pure outputs)
    # Holevo = 1
    h_cond_1 = pt.holevo_information_conditional(fixed_step=1, fixed_idx=0, base=2)
    assert np.isclose(h_cond_1, 1.0, atol=1e-10)
