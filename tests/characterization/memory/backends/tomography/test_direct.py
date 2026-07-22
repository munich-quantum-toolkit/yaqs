# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for leg-by-leg direct process-tensor construction."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.tomography.constructor import build_process_tensor
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

if TYPE_CHECKING:
    from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import (
        DenseProcessTensor,
        MPOProcessTensor,
    )


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
        build_process_tensor(
            ham.mpo,
            params,
            timesteps=timesteps,
            return_type="mpo",
            max_bond_dim=None,
            compress_every=1,
        ),
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
        mc.build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            return_type="dense",
        ),
    )
    pt_mpo = cast(
        "MPOProcessTensor",
        build_process_tensor(
            ham.mpo,
            params,
            timesteps=timesteps,
            return_type="mpo",
            max_bond_dim=None,
            compress_every=1,
        ),
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
        build_process_tensor(
            ham.mpo,
            params,
            timesteps=[0.0, 0.0],
            return_type="mpo",
            noise_model=NoiseModel([]),
        )
