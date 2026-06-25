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

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.combs.tomography import construct_process_tensor
from mqt.yaqs.core.data_structures.mpo import MPO


def test_construct_process_tensor_invalid_return_type_raises() -> None:
    """Unknown return_type values are rejected."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    with pytest.raises(ValueError, match="Unknown return_type"):
        construct_process_tensor(op, params, timesteps=[0.0], return_type="nope")  # ty: ignore[invalid-argument-type]


def test_build_comb_returns_dense_and_mpo_smoke() -> None:
    """build_comb returns dense and MPO comb wrappers."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    dense = mc.build_comb(ham, params, timesteps=[0.0], return_type="dense")
    assert dense.to_matrix().shape == (8, 8)

    mpo = mc.build_comb(ham, params, timesteps=[0.0], return_type="mpo", compress_every=1)
    mat = mpo.to_matrix()
    assert mat.shape == (8, 8)
    np.testing.assert_allclose(mat, dense.to_matrix(), atol=1e-8)


def test_build_comb_parallel_smoke() -> None:
    """build_comb runs with parallel execution enabled."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    dense = MemoryCharacterizer(parallel=True, max_workers=2, show_progress=False).build_comb(
        ham, params, timesteps=[0.0], return_type="dense"
    )
    assert dense.to_matrix().shape == (8, 8)
