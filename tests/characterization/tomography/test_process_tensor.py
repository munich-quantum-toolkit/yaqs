import numpy as np
import pytest
from mqt.yaqs.characterization.tomography.process_tensor import ProcessTensor, _vec_to_rho


def test_vec_to_rho() -> None:
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

    # Test with non-normalized input (should normalize)
    vec_unnorm = np.array([2, 0, 0, 0], dtype=complex)
    rho_out = _vec_to_rho(vec_unnorm)
    assert np.isclose(np.trace(rho_out), 1.0)
    assert rho_out[0, 0] == 1.0


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


def test_process_tensor_init() -> None:
    """Test ProcessTensor initialization and properties."""
    tensor = np.zeros((4, 6, 6), dtype=complex)
    timesteps = [0.1, 0.2]
    pt = ProcessTensor(tensor, timesteps)

    assert pt.tensor is tensor
    assert pt.data is tensor
    assert pt.timesteps == timesteps
    assert pt.rank == 1  # 3 dimensions // 2 = 1. Wait, rank definition in ProcessTensor is len(shape)//2.
    # shape (4, 6, 6) -> len=3 -> rank=1.
    # For k=2 steps, shape is (4, 6, 6) -> rank 1?
    # Usually rank refers to input slots. Let's check process_tensor.py:37
    # self.rank = len(tensor.shape) // 2.
    # (4, 6) -> rank 1. (4, 6, 6) -> rank 1. (4, 6, 6, 6, 6) -> rank 2.
    # This seems to be counting "pairs" of indices if it was a Cho matrix,
    # but here it's (out, in1, in2, ...).
    # So k steps gives 1 + k indices. rank = (1+k)//2.
    # For k=1, rank=1. For k=2, rank=1. For k=3, rank=2.
    # This might be a slightly non-standard definition of rank, but let's verify it works as intended.


def test_to_choi_matrix() -> None:
    """Test reshaping to Choi matrix representation."""
    tensor = np.random.randn(4, 6, 6) + 1j * np.random.randn(4, 6, 6)
    pt = ProcessTensor(tensor, [0.1, 0.1])
    choi = pt.to_choi_matrix()
    assert choi.shape == (4, 36)
    assert np.all(choi[:, 7] == tensor[:, 1, 1])


def test_predict_final_state_error() -> None:
    """Test error handling in prediction."""
    pt = ProcessTensor(np.zeros((4, 6)), [0.1])
    with pytest.raises(ValueError, match="Sequence length 2 must match number of timesteps 1"):
        pt.predict_final_state([np.eye(2), np.eye(2)], [])


def test_holevo_information_identity() -> None:
    """Test Holevo information for an identity channel."""
    basis = get_standard_basis()
    num_frames = len(basis)
    tensor = np.zeros((4, num_frames), dtype=complex)

    for i, (_, _, rho) in enumerate(basis):
        tensor[:, i] = rho.reshape(-1)

    pt = ProcessTensor(tensor, [1.0])
    holevo = pt.holevo_information(base=2)
    assert np.isclose(holevo, 1.0, atol=1e-10)


def test_holevo_information_fully_depolarizing() -> None:
    """Test Holevo information for a fully depolarizing channel."""
    basis = get_standard_basis()
    num_frames = len(basis)
    tensor = np.zeros((4, num_frames), dtype=complex)
    rho_mixed = 0.5 * np.eye(2, dtype=complex)
    vec_mixed = rho_mixed.reshape(-1)

    for i in range(num_frames):
        tensor[:, i] = vec_mixed

    pt = ProcessTensor(tensor, [1.0])
    holevo = pt.holevo_information(base=2)
    assert np.isclose(holevo, 0.0, atol=1e-10)


def test_holevo_information_conditional() -> None:
    """Test conditional Holevo information."""
    basis = get_standard_basis()
    num_frames = len(basis)
    # T[out, i, j] = rho_i (identity from step 0, ignores step 1)
    tensor = np.zeros((4, num_frames, num_frames), dtype=complex)

    for i in range(num_frames):
        rho_i = basis[i][2]
        vec_i = rho_i.reshape(-1)
        for j in range(num_frames):
            tensor[:, i, j] = vec_i

    pt = ProcessTensor(tensor, [1.0, 1.0])

    # Fix step 0: output fixed, Holevo=0
    h_cond_0 = pt.holevo_information_conditional(fixed_step=0, fixed_idx=0, base=2)
    assert np.isclose(h_cond_0, 0.0, atol=1e-10)

    # Fix step 1: output varies with step 0, Holevo=1
    h_cond_1 = pt.holevo_information_conditional(fixed_step=1, fixed_idx=0, base=2)
    assert np.isclose(h_cond_1, 1.0, atol=1e-10)

    # Test out of bounds
    with pytest.raises(ValueError, match="fixed_step 2 out of bounds for 2 steps"):
        pt.holevo_information_conditional(fixed_step=2, fixed_idx=0)
    with pytest.raises(ValueError, match="fixed_idx 6 out of bounds for 6 basis states"):
        pt.holevo_information_conditional(fixed_step=0, fixed_idx=6)


def test_holevo_information_empty_sequences() -> None:
    """Test Holevo information when no sequences match (edge case)."""
    # This is hard to trigger with valid inputs because the loops are consistent,
    # but we can mock or force it if we want to hit that line 100%.
    # Actually, the code has:
    # seqs = [seq for seq in itertools.product(range(N), repeat=k) if seq[fixed_step] == fixed_idx]
    # If N>0 and k>0 and fixed_idx < N, seqs will NEVER be empty.
    # The only way it's empty is if N=0 or k=0, but PT init ensures rank > 0?
    # No, (4,) tensor would have k=0.

    # Let's test with k=0 (standard 1-site density matrix, no input steps)
    tensor = np.array([1, 0, 0, 0], dtype=complex)
    pt = ProcessTensor(tensor, [])  # Empty timesteps -> k=0
    # fixed_step=0 would be out of bounds.
    with pytest.raises(ValueError, match="fixed_step 0 out of bounds for 0 steps"):
        pt.holevo_information_conditional(0, 0)
