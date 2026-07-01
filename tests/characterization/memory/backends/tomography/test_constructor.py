# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for build_process_tensor entry point."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.tomography import build_process_tensor
from mqt.yaqs.characterization.memory.backends.tomography.constructor import run_all_sequences
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.noise_model import NoiseModel


def test_build_process_tensor_invalid_return_type_raises() -> None:
    """Unknown return_type values are rejected."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    with pytest.raises(ValueError, match="Unknown return_type"):
        build_process_tensor(op, params, timesteps=[0.0, 0.0], return_type=cast("Any", "nope"))


def test_build_process_tensor_returns_dense_and_mpo_smoke() -> None:
    """build_process_tensor returns dense and MPO process-tensor wrappers."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    dense = mc.build_process_tensor(ham, params, timesteps=[0.0, 0.0], return_type="dense")
    assert dense.to_matrix().shape == (8, 8)

    mpo = mc.build_process_tensor(ham, params, timesteps=[0.0, 0.0], return_type="mpo", compress_every=1)
    mat = mpo.to_matrix()
    assert mat.shape == (8, 8)
    np.testing.assert_allclose(mat, dense.to_matrix(), atol=1e-8)


def test_build_process_tensor_rejects_k_zero() -> None:
    """Zero-step tomography is rejected before sequence enumeration."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    with pytest.raises(ValueError, match="No sequences for num_interventions=0"):
        build_process_tensor(op, params, timesteps=[])
    with pytest.raises(ValueError, match="No sequences for num_interventions=0"):
        build_process_tensor(op, params, timesteps=[0.1])


def test_build_process_tensor_parallel_smoke() -> None:
    """build_process_tensor runs with parallel execution enabled."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    dense = MemoryCharacterizer(parallel=True, max_workers=2, show_progress=False).build_process_tensor(
        ham, params, timesteps=[0.0, 0.0], return_type="dense"
    )
    assert dense.to_matrix().shape == (8, 8)


def test_build_process_tensor_stores_reference_initial_rho() -> None:
    """Built process tensors store the site-0 reference after U_0 evolution."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    pt = MemoryCharacterizer(parallel=False, show_progress=False).build_process_tensor(
        ham, params, timesteps=[0.0, 0.0], return_type="dense"
    )
    np.testing.assert_allclose(pt.initial_rho, np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128), atol=1e-10)


def test_build_process_tensor_validates_initial_rho_arg() -> None:
    """Optional initial_rho at build time is checked against the computed reference."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    ref = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    pt = mc.build_process_tensor(ham, params, timesteps=[0.0, 0.0], return_type="dense", initial_rho=ref)
    np.testing.assert_allclose(pt.initial_rho, ref, atol=1e-10)
    with pytest.raises(ValueError, match="rho0 does not match"):
        mc.build_process_tensor(
            ham,
            params,
            timesteps=[0.0, 0.0],
            return_type="dense",
            initial_rho=np.eye(2, dtype=np.complex128) / 2.0,
        )


def test_run_all_sequences_rejects_non_positive_num_trajectories_with_noise() -> None:
    """Noisy tomography rejects zero or negative trajectory counts before reference-state work."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    noise_model = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 0.1}])
    with pytest.raises(ValueError, match="num_trajectories must be positive"):
        run_all_sequences(
            op,
            params,
            [0.0, 0.0],
            parallel=False,
            num_trajectories=0,
            noise_model=noise_model,
            show_progress=False,
        )
    with pytest.raises(ValueError, match="num_trajectories must be non-negative"):
        run_all_sequences(
            op,
            params,
            [0.0, 0.0],
            parallel=False,
            num_trajectories=-3,
            noise_model=noise_model,
            show_progress=False,
        )
