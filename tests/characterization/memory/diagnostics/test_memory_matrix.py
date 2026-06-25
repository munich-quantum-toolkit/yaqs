# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: PLC2701 -- white-box tests import private memory-matrix helpers

"""Tests for memory-matrix construction and spectrum analysis."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.diagnostics.memory_matrix import (
    _analyze_memory_matrix,
    _build_weighted_memory_matrix,
    _build_weighted_memory_matrix_from_probe,
    _center_past_rows,
    _pauli_xyz_channels,
    _prepare_branch_weights,
)
from mqt.yaqs.characterization.memory.diagnostics.probe import sample_split_cut_probes
from mqt.yaqs.characterization.memory.reference.exact import evaluate_exact_probe_set_with_diagnostics
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_four_component_memory_metric_matches_xyz_only() -> None:
    """S_V is unchanged when storing (I,X,Y,Z) but using X,Y,Z for the memory matrix."""
    rng = np.random.default_rng(11)
    op = MPO.ising(length=1, J=0.5, g=0.3)
    params = AnalogSimParams(dt=0.05, max_bond_dim=8, order=1)
    probe_set = sample_split_cut_probes(cut=1, k=1, n_pasts=5, n_futures=4, rng=rng)
    psi0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    pauli4, weights, _ = evaluate_exact_probe_set_with_diagnostics(
        probe_set=probe_set,
        operator=op,
        sim_params=params,
        initial_psi=psi0,
        parallel=False,
    )
    pauli3 = _pauli_xyz_channels(pauli4)
    w_clean, _ = _prepare_branch_weights(weights)
    m4_raw, m4 = _build_weighted_memory_matrix_from_probe(pauli4, weights)
    m3_raw = _build_weighted_memory_matrix(pauli3, w_clean, beta=1.0)
    m3 = _center_past_rows(m3_raw)
    np.testing.assert_allclose(m4_raw, m3_raw, atol=1e-12)
    np.testing.assert_allclose(m4, m3, atol=1e-12)
    out4 = _analyze_memory_matrix(m4)
    out3 = _analyze_memory_matrix(m3)
    assert out4["entropy"] == pytest.approx(out3["entropy"])
    assert out4["rank"] == pytest.approx(out3["rank"])


def test_analyze_memory_matrix_rank_equals_exp_entropy() -> None:
    """analyze_memory_matrix reports R(c)=exp(S_V(c))."""
    m = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64)
    memory_matrix = m - m.mean(axis=0, keepdims=True)
    out = _analyze_memory_matrix(memory_matrix)
    assert out["rank"] == pytest.approx(math.exp(out["entropy"]), rel=1e-12, abs=1e-12)
