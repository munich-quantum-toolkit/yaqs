# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for operational memory metrics (entropy / singular values)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mqt.yaqs import MemoryCharacterizer
from mqt.yaqs.characterization.memory.combs.tomography import construct_process_tensor
from mqt.yaqs.characterization.memory.combs.tomography.combs import DenseComb, MPOComb
from mqt.yaqs.characterization.memory.diagnostics.probe import (
    analyze_v_matrix,
    build_weighted_v_from_probe,
    probe_process,
    sample_split_cut_probes,
)
from mqt.yaqs.characterization.memory.reference.exact import (
    ExactProbeProcess,
    evaluate_exact_probe_set_with_diagnostics,
)
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def _params() -> AnalogSimParams:
    return AnalogSimParams(dt=0.05, max_bond_dim=8, order=1)


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
        initial_psi=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128),
        parallel=False,
    )
    pauli_e, weights_e, _ = evaluate_exact_probe_set_with_diagnostics(
        probe_set=probe_set,
        operator=exact.operator,
        sim_params=exact.sim_params,
        initial_psi=exact.initial_psi,
        parallel=exact.parallel,
    )
    v_e, v_c_e = build_weighted_v_from_probe(pauli_e, weights_e)
    out_exact = analyze_v_matrix(v_e, v_c_e)
    out_comb = probe_process(
        process=comb,
        cut=2,
        k=2,
        probe_set=probe_set,
    )
    assert out_comb["entropy"] == pytest.approx(out_exact["entropy"], rel=0.15, abs=0.05)


def test_mpo_comb_entropy_matches_dense() -> None:
    """MPOComb memory metrics match dense fallback."""
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
    out_mpo = probe_process(process=mpo_comb, cut=1, k=1, probe_set=probe_set)
    out_dense = probe_process(process=dense, cut=1, k=1, probe_set=probe_set)
    assert out_mpo["entropy"] == pytest.approx(out_dense["entropy"], rel=1e-10, abs=1e-10)


def test_characterize_comb_default_cut() -> None:
    """characterize() uses interior default cut when cut is omitted."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = _params()
    comb = construct_process_tensor(
        op,
        params,
        timesteps=[0.05, 0.05],
        num_trajectories=30,
        parallel=False,
        return_type="dense",
    )
    assert isinstance(comb, DenseComb)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    rng = np.random.default_rng(0)
    default_cut = (2 + 1) // 2
    ent_default = mc.characterize(comb, k=2, n_pasts=4, n_futures=4, rng=rng).entropy(default_cut)
    ent_explicit = mc.characterize(
        comb,
        cut=default_cut,
        k=2,
        n_pasts=4,
        n_futures=4,
        rng=np.random.default_rng(0),
    ).entropy(default_cut)
    assert ent_default == pytest.approx(ent_explicit)
    result = mc.characterize(
        comb,
        cut=2,
        k=2,
        n_pasts=4,
        n_futures=4,
        rng=np.random.default_rng(0),
    )
    sv = result.singular_values(2)
    assert sv.ndim == 1
    assert sv.size >= 1
    assert math.isfinite(float(result.entropy(2)))


def test_transformercomb_characterize_singular_values_shape() -> None:
    """MemoryCharacterizer.characterize returns the full SVD spectrum for a surrogate."""
    pytest.importorskip("torch")

    from mqt.yaqs.characterization.memory.combs.surrogates.model import TransformerComb

    model = TransformerComb(
        d_e=32,
        d_rho=8,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_ff=64,
        dropout=0.0,
        sequence_length=3,
    )
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    sv = mc.characterize(
        model,
        cut=2,
        n_pasts=4,
        n_futures=3,
        rng=np.random.default_rng(0),
    ).singular_values(2)
    assert sv.shape == (min(4, 3 * 3),)
