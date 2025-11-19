# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for decompositions.

This module tests the left and right qr and svd decompositions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.methods.decompositions import left_qr, right_qr, right_svd, truncated_right_svd

if TYPE_CHECKING:
    from numpy.typing import NDArray


def crandn(
    size: int | tuple[int, ...], *args: int, seed: np.random.Generator | int | None = None
) -> NDArray[np.complex128]:
    """Draw random samples from the standard complex normal distribution.

    Args:
        size (int |Tuple[int,...]): The size/shape of the output array.
        *args (int): Additional dimensions for the output array.
        seed (Generator | int): The seed for the random number generator.

    Returns:
        NDArray[np.complex128]: The array of random complex numbers.
    """
    if isinstance(size, int) and len(args) > 0:
        size = (size, *list(args))
    elif isinstance(size, int):
        size = (size,)
    rng = np.random.default_rng(seed)
    # 1 / sqrt(2) is a normalization factor
    return np.asarray((rng.standard_normal(size) + 1j * rng.standard_normal(size)) / np.sqrt(2), dtype=np.complex128)


def test_right_qr() -> None:
    """Tests the right qr decomposition.

    Ensures that it produces tensors of the correct shape and a unitary tensor.
    Also checks that the decomposition is actually the original tensor.
    """
    shape = (2, 3, 4)
    tensor = crandn(shape)
    q_tensor, r_matrix = right_qr(tensor)
    assert q_tensor.ndim == 3
    assert r_matrix.ndim == 2
    assert q_tensor.shape[0] == shape[0]
    assert q_tensor.shape[1] == shape[1]
    assert r_matrix.shape[1] == shape[2]
    assert q_tensor.shape[2] == r_matrix.shape[0]
    # Check that q_tensor is unitary
    iden = np.eye(q_tensor.shape[2])
    q_matrix = q_tensor.reshape(q_tensor.shape[0] * q_tensor.shape[1], -1)
    assert np.allclose(q_matrix.conj().T @ q_matrix, iden)
    # Check that qr = tensor
    contr = np.tensordot(q_tensor, r_matrix, axes=(2, 0))
    assert np.allclose(contr, tensor)


def test_left_qr() -> None:
    """Tests the left qr decomposition.

    Ensures that it produces tensors of the correct shape and a unitary tensor.
    Also checks that the decomposition is actually the original tensor.
    """
    shape = (2, 3, 4)
    tensor = crandn(shape)
    q_tensor, r_matrix = left_qr(tensor)
    assert q_tensor.ndim == 3
    assert r_matrix.ndim == 2
    assert q_tensor.shape[0] == shape[0]
    assert q_tensor.shape[2] == shape[2]
    assert r_matrix.shape[0] == shape[1]
    assert q_tensor.shape[1] == r_matrix.shape[1]
    # Check that q_tensor is unitary
    iden = np.eye(q_tensor.shape[1])
    q_matrix = q_tensor.transpose(0, 2, 1)
    q_matrix = q_matrix.reshape(-1, q_tensor.shape[1])
    assert np.allclose(q_matrix.T.conj() @ q_matrix, iden)
    # Check that qr = tensor
    contr = np.tensordot(q_tensor, r_matrix, axes=(1, 1))
    contr = contr.transpose(0, 2, 1)
    assert np.allclose(contr, tensor)


def test_right_svd() -> None:
    """Test that the svd produces the correct shapes and tensors."""
    tensor = crandn(2, 3, 4)
    u_tensor, s_vec, v_matrix = right_svd(tensor)
    # Check shapes
    assert u_tensor.shape[0] == 2
    assert u_tensor.shape[1] == 3
    assert v_matrix.shape[1] == 4
    assert u_tensor.shape[2] == s_vec.shape[0]
    assert s_vec.shape[0] == v_matrix.shape[0]
    # Check that u_tensor is unitary
    iden = np.eye(u_tensor.shape[2])
    result = np.tensordot(u_tensor, u_tensor.conj(), axes=([0, 1], [0, 1]))
    assert np.allclose(result, iden)
    # Check that v_matrix is unitary
    iden = np.eye(v_matrix.shape[1])
    result = v_matrix.conj().T @ v_matrix
    assert np.allclose(result, iden)
    # Check that svd = tensor
    contr = np.tensordot(u_tensor, np.diag(s_vec) @ v_matrix, axes=(2, 0))
    assert np.allclose(contr, tensor)


def test_truncated_right_svd_thresh() -> None:
    """Test that the tensor is correctly truncated."""
    # Placeholder
    sim_params = AnalogSimParams(
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        max_bond_dim=4,
        threshold=0.2,
        order=1,
        show_progress=False,
        get_state=True,
    )
    s_vector_i = np.array([1, 0.5, 0.1, 0.01])
    u_tensor_i, _ = right_qr(crandn(2, 3, 4))
    v_matrix_i, _ = np.linalg.qr(crandn(4, 4))
    tensor = np.tensordot(u_tensor_i, np.diag(s_vector_i) @ v_matrix_i, axes=(2, 0))

    # Thus the values 0.1 and 0.01 should be truncated
    u_tensor, s_vector, v_matrix = truncated_right_svd(tensor, sim_params.threshold, sim_params.max_bond_dim)
    # Check shapes
    assert u_tensor.shape[0] == 2
    assert u_tensor.shape[1] == 3
    assert v_matrix.shape[1] == 4
    assert u_tensor.shape[2] == 2
    assert v_matrix.shape[0] == 2
    assert s_vector.shape[0] == 2
    assert np.allclose(s_vector, s_vector_i[:2])


def test_truncated_right_svd_maxbd() -> None:
    """Test that the tensor is correctly truncated."""
    # Placeholder
    sim_params = AnalogSimParams(
        elapsed_time=0.2,
        dt=0.1,
        sample_timesteps=True,
        max_bond_dim=3,
        threshold=1e-4,
        order=1,
        show_progress=False,
        get_state=True,
    )

    s_vector_i = np.array([1, 0.5, 0.1, 0.01])
    u_tensor_i, _ = right_qr(crandn(2, 3, 4))
    v_matrix_i, _ = np.linalg.qr(crandn(4, 4))
    tensor = np.tensordot(u_tensor_i, np.diag(s_vector_i) @ v_matrix_i, axes=(2, 0))

    # Thus the value 0.01 should be truncated
    u_tensor, s_vector, v_matrix = truncated_right_svd(tensor, sim_params.threshold, sim_params.max_bond_dim)
    # Check shapes
    assert u_tensor.shape[0] == 2
    assert u_tensor.shape[1] == 3
    assert v_matrix.shape[1] == 4
    assert u_tensor.shape[2] == sim_params.max_bond_dim
    assert sim_params.max_bond_dim == v_matrix.shape[0]
    assert sim_params.max_bond_dim == s_vector.shape[0]
    assert np.allclose(s_vector, s_vector_i[: sim_params.max_bond_dim])
