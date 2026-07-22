# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for leg-by-leg direct process-tensor construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.tomography.constructor import build_process_tensor
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import MPOProcessTensor
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

if TYPE_CHECKING:
    from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import DenseProcessTensor


def test_build_process_tensor_defaults_to_mpo() -> None:
    """Default return_type is direct MPO construction."""
    ham = Hamiltonian.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    pt = MemoryCharacterizer(parallel=False, show_progress=False).build_process_tensor(
        ham, params, timesteps=[0.0, 0.0], compress_every=1
    )
    assert isinstance(pt, MPOProcessTensor)


def test_default_mpo_recreates_dense_process_tensor() -> None:
    """Uncapped default MPO path matches the dense Choi matrix on a small schedule."""
    ham = Hamiltonian.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    timesteps = [0.1, 0.1, 0.1]
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    pt_mpo = mc.build_process_tensor(ham, params, timesteps=timesteps, max_bond_dim=None, compress_every=1)
    pt_dense = mc.build_process_tensor(ham, params, timesteps=timesteps, return_type="dense")

    assert isinstance(pt_mpo, MPOProcessTensor)
    np.testing.assert_allclose(pt_mpo.to_matrix(), pt_dense.to_matrix(), atol=1e-8)
    np.testing.assert_allclose(pt_mpo.initial_rho, pt_dense.initial_rho, atol=1e-10)


@pytest.mark.parametrize("j_val", [0.0, 1.0])
@pytest.mark.parametrize("num_interventions", [1, 2])
def test_direct_mpo_matches_dense_temporal_entropy(j_val: float, num_interventions: int) -> None:
    """Direct MPO and dense tomography agree on temporal entanglement at small k."""
    ham = Hamiltonian.ising(length=2, J=float(j_val), g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    timesteps = [0.1] * (num_interventions + 1)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    pt_dense = cast(
        "DenseProcessTensor",
        mc.build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            return_type="dense",
            num_trajectories=12,
        ),
    )
    pt_mpo = cast(
        "MPOProcessTensor",
        build_process_tensor(ham.mpo, params, timesteps=timesteps, max_bond_dim=None, compress_every=1),
    )

    for cut in range(1, num_interventions + 1):
        dense = pt_dense.compute_temporal_entropy(cut)
        mpo = pt_mpo.compute_temporal_entropy(cut)
        assert float(cast("float", mpo["entropy"])) == pytest.approx(
            float(cast("float", dense["entropy"])),
            abs=1e-6,
        )
        assert cast("int", mpo["schmidt_rank"]) == cast("int", dense["schmidt_rank"])


def test_direct_j_zero_matches_dense_temporal_entropy() -> None:
    """Direct MPO matches dense tomography for a memoryless chain."""
    ham = Hamiltonian.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    timesteps = [0.0, 0.0, 0.0]
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    pt_dense = cast(
        "DenseProcessTensor",
        mc.build_process_tensor(ham, params, timesteps=timesteps, return_type="dense"),
    )
    pt_mpo = cast(
        "MPOProcessTensor",
        build_process_tensor(ham.mpo, params, timesteps=timesteps, max_bond_dim=None, compress_every=1),
    )
    for cut in (1, 2):
        dense = pt_dense.compute_temporal_entropy(cut)
        mpo = pt_mpo.compute_temporal_entropy(cut)
        assert float(cast("float", mpo["entropy"])) == pytest.approx(
            float(cast("float", dense["entropy"])),
            abs=1e-6,
        )


def test_mpo_rejects_noise_model() -> None:
    """Direct MPO construction rejects a noise model."""
    ham = Hamiltonian.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)

    with pytest.raises(ValueError, match="does not support noise_model"):
        build_process_tensor(ham.mpo, params, timesteps=[0.0, 0.0], noise_model=NoiseModel([]))


def test_direct_parallel_matches_serial() -> None:
    """Parallel and serial direct construction agree on the process-tensor matrix."""
    ham = Hamiltonian.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    timesteps = [0.1, 0.1]
    serial = cast(
        "MPOProcessTensor",
        MemoryCharacterizer(parallel=False, show_progress=False).build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            compress_every=1,
        ),
    )
    parallel = cast(
        "MPOProcessTensor",
        MemoryCharacterizer(parallel=True, max_workers=2, show_progress=False).build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            compress_every=1,
        ),
    )
    np.testing.assert_allclose(parallel.to_matrix(), serial.to_matrix(), atol=1e-8)


def test_direct_parallel_temporal_entropy_matches_dense() -> None:
    """Parallel direct MPO agrees with dense tomography on temporal entanglement."""
    ham = Hamiltonian.ising(length=2, J=0.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    timesteps = [0.1, 0.1, 0.1]
    mc = MemoryCharacterizer(parallel=True, max_workers=2, show_progress=False)
    pt_dense = cast(
        "DenseProcessTensor",
        mc.build_process_tensor(ham, params, timesteps=timesteps, return_type="dense"),
    )
    pt_mpo = cast(
        "MPOProcessTensor",
        mc.build_process_tensor(ham, params, timesteps=timesteps, max_bond_dim=None, compress_every=1),
    )
    for cut in (1, 2):
        dense = pt_dense.compute_temporal_entropy(cut)
        mpo = pt_mpo.compute_temporal_entropy(cut)
        assert float(cast("float", mpo["entropy"])) == pytest.approx(
            float(cast("float", dense["entropy"])),
            abs=1e-6,
        )


def test_direct_tjm_matches_mcwf() -> None:
    """Direct MPO construction preserves MPS states under TJM and matches MCWF."""
    ham = Hamiltonian.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    timesteps = [0.1, 0.1]
    mcwf = cast(
        "MPOProcessTensor",
        MemoryCharacterizer(representation="vector", parallel=False, show_progress=False).build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            max_bond_dim=4,
            compress_every=1,
        ),
    )
    tjm = cast(
        "MPOProcessTensor",
        MemoryCharacterizer(representation="mps", parallel=False, show_progress=False).build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            max_bond_dim=4,
            compress_every=1,
        ),
    )
    np.testing.assert_allclose(tjm.to_matrix(), mcwf.to_matrix(), atol=1e-6)
