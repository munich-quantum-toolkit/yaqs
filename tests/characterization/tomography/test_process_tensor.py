# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the ProcessTensor class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.characterization.tomography.process_tensor import ProcessTensor, _vec_to_rho  # noqa: PLC2701

if TYPE_CHECKING:
    from numpy.typing import NDArray


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
    assert np.isclose(rho_out[0, 0], 1.0)


def get_standard_basis() -> list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]]:
    """Returns the standard 4-state Pauli basis for testing.

    Returns:
        Standard 4-state basis.
    """
    psi_0 = np.array([1, 0], dtype=complex)
    psi_1 = np.array([0, 1], dtype=complex)
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j], dtype=complex) / np.sqrt(2)

    states = [
        ("zeros", psi_0),
        ("ones", psi_1),
        ("x+", psi_plus),
        ("y+", psi_i_plus),
    ]
    basis_set = []
    for name, psi in states:
        rho = np.outer(psi, psi.conj())
        basis_set.append((name, psi, rho))
    return basis_set


def test_process_tensor_init() -> None:
    """Test ProcessTensor initialization and properties."""
    tensor = np.zeros((4, 4, 4), dtype=complex)
    weights = np.ones((4, 4), dtype=float) / 16.0
    timesteps = [0.1, 0.2]
    pt = ProcessTensor(tensor, weights, timesteps)

    assert pt.tensor is tensor
    assert pt.weights is weights
    assert pt.timesteps == timesteps


def test_to_choi_matrix() -> None:
    """Test reshaping to Choi matrix representation."""
    rng = np.random.default_rng(42)
    tensor = rng.standard_normal((4, 4, 4)) + 1j * rng.standard_normal((4, 4, 4))
    weights = np.ones((4, 4), dtype=float) / 16.0
    pt = ProcessTensor(tensor, weights, [0.1, 0.1])
    choi = pt.to_choi_matrix()
    assert choi.shape == (4, 16)
    assert np.all(choi[:, 5] == tensor[:, 1, 1])


def test_predict_final_state_error() -> None:
    """Test error handling in prediction."""
    pt = ProcessTensor(np.zeros((4, 4), dtype=complex), np.ones(4), [0.1])
    with pytest.raises(ValueError, match="Expected 1 interventions \\(including t=0 prep\\), got 2."):
        pt.predict_final_state([lambda x: x, lambda x: x], [])


def test_qmi_identity() -> None:
    """Test Quantum Mutual Information for an identity channel."""
    basis = get_standard_basis()
    num_frames = len(basis)
    tensor = np.zeros((4, num_frames), dtype=complex)
    weights = np.ones(num_frames, dtype=float) / num_frames

    for i, (_, _, rho) in enumerate(basis):
        tensor[:, i] = rho.reshape(-1)

    pt = ProcessTensor(tensor, weights, [1.0])
    qmi = pt.quantum_mutual_information(base=2)
    # The 4 basis states (0, 1, x+, y+) are not symmetrically distributed on the Bloch sphere
    # (they do not form a regular polyhedron summing exactly to the maximally mixed state I/2).
    # Their average density matrix has a slight bias, meaning the marginal entropy is less than 1.
    # Therefore, the uncorrected QMI for this asymmetric basis mapping is ~0.907 bits.
    assert np.isclose(qmi, 0.90785, atol=1e-3)


def test_qmi_fully_depolarizing() -> None:
    """Test QMI for a fully depolarizing channel."""
    basis = get_standard_basis()
    num_frames = len(basis)
    tensor = np.zeros((4, num_frames), dtype=complex)
    weights = np.ones(num_frames, dtype=float) / num_frames
    
    rho_mixed = 0.5 * np.eye(2, dtype=complex)
    vec_mixed = rho_mixed.reshape(-1)

    for i in range(num_frames):
        tensor[:, i] = vec_mixed

    pt = ProcessTensor(tensor, weights, [1.0])
    qmi = pt.quantum_mutual_information(base=2)
    assert np.isclose(qmi, 0.0, atol=1e-10)
