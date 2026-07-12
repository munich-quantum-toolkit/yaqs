# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for leg-by-leg direct process-tensor construction."""

from __future__ import annotations

from typing import cast

import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.backends.tomography.constructor import build_process_tensor
from mqt.yaqs.characterization.memory.backends.tomography.process_tensors import MPOProcessTensor


@pytest.mark.parametrize("j_val", [0.0, 1.0])
@pytest.mark.parametrize("num_interventions", [1, 2])
def test_direct_matches_exhaustive_mpo_metrics(j_val: float, num_interventions: int) -> None:
    """Direct and exhaustive MPO builders agree on cut metrics at small k."""
    ham = Hamiltonian.ising(length=2, J=float(j_val), g=1.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    timesteps = [0.0] * (num_interventions + 1)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    pt_ex = cast(
        "MPOProcessTensor",
        mc.build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            return_type="mpo",
            method="exhaustive",
            compress_every=1,
            num_trajectories=12,
        ),
    )
    pt_dir = cast(
        "MPOProcessTensor",
        build_process_tensor(
            ham.mpo,
            params,
            timesteps=timesteps,
            return_type="mpo",
            method="direct",
            max_bond_dim=None,
            compress_every=1,
        ),
    )

    cuts = list(range(1, num_interventions + 1))
    for cut in cuts:
        chi_ex = pt_ex.temporal_bond_dimension(cut)
        chi_dir = pt_dir.temporal_bond_dimension(cut)
        s_ex = pt_ex.cut_entanglement_entropy(cut)
        s_dir = pt_dir.cut_entanglement_entropy(cut)
        assert chi_dir == pytest.approx(chi_ex, rel=0.0, abs=0)
        assert s_dir == pytest.approx(s_ex, rel=0.0, abs=1e-6)


def test_direct_j_zero_matches_exhaustive() -> None:
    """Direct builder matches exhaustive tomography for a memoryless chain."""
    ham = Hamiltonian.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8, order=1)
    timesteps = [0.0, 0.0, 0.0]
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    pt_ex = cast(
        "MPOProcessTensor",
        mc.build_process_tensor(
            ham,
            params,
            timesteps=timesteps,
            return_type="mpo",
            method="exhaustive",
            compress_every=1,
        ),
    )
    pt_dir = cast(
        "MPOProcessTensor",
        build_process_tensor(
            ham.mpo,
            params,
            timesteps=timesteps,
            return_type="mpo",
            method="direct",
            max_bond_dim=None,
            compress_every=1,
        ),
    )
    for cut in (1, 2):
        assert pt_dir.temporal_bond_dimension(cut) == pt_ex.temporal_bond_dimension(cut)
        assert pt_dir.cut_entanglement_entropy(cut) == pytest.approx(
            pt_ex.cut_entanglement_entropy(cut),
            abs=1e-6,
        )
