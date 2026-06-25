# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for CharacterizationResult."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer


def test_rank_equals_exp_entropy() -> None:
    """CharacterizationResult.rank is R(c)=exp(S_V(c))."""
    ham = Hamiltonian.ising(length=1, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    result = MemoryCharacterizer(parallel=False, show_progress=False).characterize(
        ham,
        params,
        cut=1,
        k=1,
        n_pasts=6,
        n_futures=6,
    )
    sv = result.entropy(1)
    r = result.rank(1)
    assert r == pytest.approx(math.exp(sv), rel=1e-9, abs=1e-9)


def test_probes_export_arrays() -> None:
    """characterize() stores probe arrays retrievable via result.probes(cut)."""
    ham = Hamiltonian.ising(length=1, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    result = MemoryCharacterizer(parallel=False, show_progress=False).characterize(
        ham,
        params,
        cut=1,
        k=1,
        n_pasts=4,
        n_futures=3,
        rng=np.random.default_rng(0),
    )
    probes = result.probes(1)
    assert probes["cut"] == 1
    assert probes["k"] == 1
    assert probes["past_features"].shape[0] == 4
    assert probes["future_features"].shape[0] == 3
