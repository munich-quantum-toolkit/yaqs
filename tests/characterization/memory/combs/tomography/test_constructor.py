# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for construct_process_tensor entry point."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs import construct_process_tensor
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_construct_process_tensor_invalid_return_type_raises() -> None:
    """Unknown return_type values are rejected."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    with pytest.raises(ValueError, match="Unknown return_type"):
        construct_process_tensor(op, params, timesteps=[0.0], return_type="nope")  # ty: ignore[invalid-argument-type]


def test_construct_process_tensor_returns_dense_and_mpo_smoke() -> None:
    """construct_process_tensor returns dense and MPO comb wrappers."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)

    dense = construct_process_tensor(op, params, timesteps=[0.0], parallel=False, return_type="dense")
    assert dense.to_matrix().shape == (8, 8)

    mpo = construct_process_tensor(op, params, timesteps=[0.0], parallel=False, return_type="mpo", compress_every=1)
    mat = mpo.to_matrix()
    assert mat.shape == (8, 8)
    np.testing.assert_allclose(mat, dense.to_matrix(), atol=1e-8)
