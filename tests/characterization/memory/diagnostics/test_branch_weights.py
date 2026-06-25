# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for causal cut branch weights."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.diagnostics.probe import (
    branch_weights_ij,
    probe_process,
    rollout_branch_weight,
    sample_split_cut_probes,
)
from mqt.yaqs.characterization.memory.reference.exact import evaluate_exact_probe_set_with_diagnostics
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

_PSI0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)


def test_branch_weights_constant_across_future_columns() -> None:
    rng = np.random.default_rng(3)
    probe_set = sample_split_cut_probes(cut=2, k=3, n_pasts=5, n_futures=4, rng=rng)
    w = branch_weights_ij(probe_set)
    assert np.allclose(w.std(axis=1), 0.0, atol=1e-14)


def test_rollout_branch_weight_from_steps() -> None:
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    steps = [
        {"type": "unitary", "U": np.eye(2, dtype=np.complex128)},
        (z, z),
    ]
    assert rollout_branch_weight(steps, cut=2) == pytest.approx(1.0)


def test_comb_probe_process_returns_cut_weights() -> None:
    from mqt.yaqs.characterization.memory.combs.tomography import construct_process_tensor

    rng = np.random.default_rng(0)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.05, max_bond_dim=8, order=1)
    comb = construct_process_tensor(
        op,
        params,
        timesteps=[0.05],
        num_trajectories=20,
        parallel=False,
        return_type="dense",
    )
    out = probe_process(process=comb, cut=1, k=1, n_pasts=4, n_futures=3, rng=rng)
    assert "weights_ij" in out
    assert out["weights_ij"].shape == (4, 3)
    assert np.all(out["weights_ij"] > 0.0)


def test_analytic_weights_match_exact_for_trivial_dynamics() -> None:
    """Local |0> branch rollout matches exact traces when intervening steps have prob 1."""
    rng = np.random.default_rng(11)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.05, max_bond_dim=8, order=1)
    probe_set = sample_split_cut_probes(
        cut=2,
        k=3,
        n_pasts=4,
        n_futures=3,
        rng=rng,
        intervention_mode="unitary_break_mp",
    )
    w_analytic = branch_weights_ij(probe_set)
    _, w_exact, _ = evaluate_exact_probe_set_with_diagnostics(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=_PSI0,
        parallel=False,
    )
    np.testing.assert_allclose(w_analytic, w_exact, rtol=1e-10, atol=1e-12)
