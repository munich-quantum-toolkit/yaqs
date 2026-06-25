# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer`."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.diagnostics.probe import sample_split_cut_probes
from mqt.yaqs.characterization.memory.diagnostics.results import ProbeResult


@pytest.fixture
def ham_and_params() -> tuple[Hamiltonian, AnalogSimParams]:
    ham = Hamiltonian.ising(length=1, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    return ham, params


def test_probe_exact_smoke(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """probe_exact returns a ProbeResult with V-matrix diagnostics."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    out = mc.probe_exact(ham, params, cut=1, k=1, n_pasts=3, n_futures=3, rng=np.random.default_rng(0))
    assert isinstance(out, ProbeResult)
    assert out.cut == 1
    assert out.entropy(1) >= 0.0
    assert out.rank(1) >= 1


def test_probe_from_responses_matches_probe_exact(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
) -> None:
    """probe_from_responses reproduces probe_exact on the same sampled grid."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    rng = np.random.default_rng(0)
    probe_set = sample_split_cut_probes(cut=1, k=1, n_pasts=3, n_futures=3, rng=rng)
    exact = mc.probe_exact(ham, params, cut=1, k=1, probe_set=probe_set)
    cut = exact.by_cut[1]
    assert cut.weights is not None
    rebuilt = mc.probe_from_responses(cut.pauli_xyz_ij, cut.weights, probe_set, cut=1)
    assert rebuilt.entropy(1) == pytest.approx(exact.entropy(1))
    assert rebuilt.rank(1) == exact.rank(1)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_train_then_probe(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """train returns a model; probe returns ProbeResult diagnostics."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    model = mc.train(
        ham,
        params,
        k=1,
        n=8,
        train_kwargs={"epochs": 1, "batch_size": 4},
        model_kwargs={"d_model": 32, "nhead": 2, "num_layers": 1, "dim_ff": 64},
    )
    out = mc.probe(model, cut=1, k=1, n_pasts=4, n_futures=4)
    assert isinstance(out, ProbeResult)
    assert out.entropy(1) >= 0.0


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_characterize_smoke(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """Characterize trains and returns multi-cut ProbeResult with model attached."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    out = mc.characterize(
        ham,
        params,
        k=1,
        n=8,
        n_pasts=4,
        n_futures=4,
        train_kwargs={"epochs": 1, "batch_size": 4},
        model_kwargs={"d_model": 32, "nhead": 2, "num_layers": 1, "dim_ff": 64},
    )
    assert isinstance(out, ProbeResult)
    assert out.model is not None
    assert 1 in out.by_cut


def test_build_comb_then_probe(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """build_comb returns a comb; probe returns ProbeResult diagnostics."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_comb(ham, params, timesteps=[0.1], num_trajectories=12, return_type="dense")
    out = mc.probe(comb, cut=1, k=1, n_pasts=3, n_futures=3)
    assert isinstance(out, ProbeResult)
    assert out.entropy(1) >= 0.0
