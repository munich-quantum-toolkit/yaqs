# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: PLR6301, PLC2701 -- protocol-style dummy backend; white-box rollout test

"""Tests for split-cut probes and _probe_process."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.combs.tomography import construct_process_tensor
from mqt.yaqs.characterization.memory.combs.tomography.combs import DenseComb, MPOComb
from mqt.yaqs.characterization.memory.diagnostics.memory_matrix import (
    _analyze_memory_matrix,
    _build_weighted_memory_matrix_from_probe,
)
from mqt.yaqs.characterization.memory.diagnostics.probe import (
    ProbeSet,
    _branch_weights_ij,
    _probe_process,
    _rollout_branch_weight,
    sample_split_cut_probes,
)
from mqt.yaqs.characterization.memory.reference.exact import (
    ExactProbeProcess,
    evaluate_exact_probe_set_with_diagnostics,
)
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

_PSI0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)


def _params() -> AnalogSimParams:
    return AnalogSimParams(dt=0.05, max_bond_dim=8, order=1)


def test__probe_process_uses_object_backend() -> None:
    """_probe_process delegates evaluation to a user-supplied process object."""

    class DummyProcess:
        def evaluate_probe_set(self, probe_set: ProbeSet) -> np.ndarray:
            n_p = len(probe_set.past_pairs)
            n_f = len(probe_set.future_pairs)
            return np.zeros((n_p, n_f, 4), dtype=np.float32)

    out = _probe_process(process=DummyProcess(), cut=1, k=1, n_pasts=2, n_futures=3, rng=np.random.default_rng(7))
    assert out["pauli_xyz_ij"].shape == (2, 3, 4)
    assert "entropy" in out


def test_branch_weights_constant_across_future_columns() -> None:
    rng = np.random.default_rng(3)
    probe_set = sample_split_cut_probes(cut=2, k=3, n_pasts=5, n_futures=4, rng=rng)
    w = _branch_weights_ij(probe_set)
    assert np.allclose(w.std(axis=1), 0.0, atol=1e-14)


def test_rollout_branch_weight_from_steps() -> None:
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    steps = [
        {"type": "unitary", "U": np.eye(2, dtype=np.complex128)},
        (z, z),
    ]
    assert _rollout_branch_weight(steps, cut=2) == pytest.approx(1.0)


def test_comb__probe_process_returns_cut_weights() -> None:
    rng = np.random.default_rng(0)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    comb = construct_process_tensor(
        op,
        _params(),
        timesteps=[0.05],
        num_trajectories=20,
        parallel=False,
        return_type="dense",
    )
    out = _probe_process(process=comb, cut=1, k=1, n_pasts=4, n_futures=3, rng=rng)
    assert "weights_ij" in out
    assert out["weights_ij"].shape == (4, 3)
    assert np.all(out["weights_ij"] > 0.0)


def test_analytic_weights_match_exact_for_trivial_dynamics() -> None:
    rng = np.random.default_rng(11)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    probe_set = sample_split_cut_probes(
        cut=2,
        k=3,
        n_pasts=4,
        n_futures=3,
        rng=rng,
        intervention_mode="unitary_break_mp",
    )
    w_analytic = _branch_weights_ij(probe_set)
    _, w_exact, _ = evaluate_exact_probe_set_with_diagnostics(
        probe_set=probe_set,
        operator=op,
        sim_params=_params(),
        initial_psi=_PSI0,
        parallel=False,
    )
    np.testing.assert_allclose(w_analytic, w_exact, rtol=1e-10, atol=1e-12)


def test_dense_comb_vs_exact_probe_entropy() -> None:
    """DenseComb weighted entropy agrees with exact rollout on small k."""
    rng = np.random.default_rng(42)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = _params()
    comb = construct_process_tensor(
        op,
        params,
        timesteps=[0.05, 0.05],
        num_trajectories=50,
        parallel=False,
        return_type="dense",
    )
    assert isinstance(comb, DenseComb)
    probe_set = sample_split_cut_probes(
        cut=2,
        k=2,
        n_pasts=5,
        n_futures=4,
        rng=rng,
        intervention_mode="unitary_break_mp",
    )
    exact = ExactProbeProcess(
        operator=op,
        sim_params=params,
        initial_psi=_PSI0,
        parallel=False,
    )
    pauli_e, weights_e, _ = evaluate_exact_probe_set_with_diagnostics(
        probe_set=probe_set,
        operator=exact.operator,
        sim_params=exact.sim_params,
        initial_psi=exact.initial_psi,
        parallel=exact.parallel,
    )
    _m_e_raw, memory_matrix_e = _build_weighted_memory_matrix_from_probe(pauli_e, weights_e)
    out_exact = _analyze_memory_matrix(memory_matrix_e)
    out_comb = _probe_process(process=comb, cut=2, k=2, probe_set=probe_set)
    assert out_comb["entropy"] == pytest.approx(out_exact["entropy"], rel=0.15, abs=0.05)


def test_mpo_comb_entropy_matches_dense() -> None:
    rng = np.random.default_rng(1)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = _params()
    mpo_comb = construct_process_tensor(
        op,
        params,
        timesteps=[0.05],
        num_trajectories=40,
        parallel=False,
        return_type="mpo",
        compress_every=1,
    )
    assert isinstance(mpo_comb, MPOComb)
    dense = mpo_comb.to_dense()
    probe_set = sample_split_cut_probes(cut=1, k=1, n_pasts=4, n_futures=3, rng=rng)
    out_mpo = _probe_process(process=mpo_comb, cut=1, k=1, probe_set=probe_set)
    out_dense = _probe_process(process=dense, cut=1, k=1, probe_set=probe_set)
    assert out_mpo["entropy"] == pytest.approx(out_dense["entropy"], rel=1e-10, abs=1e-10)
