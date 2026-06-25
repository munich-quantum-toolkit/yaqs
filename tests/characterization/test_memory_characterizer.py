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
from mqt.yaqs.characterization.memory.diagnostics.results import CharacterizationResult


@pytest.fixture
def ham_and_params() -> tuple[Hamiltonian, AnalogSimParams]:
    ham = Hamiltonian.ising(length=1, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    return ham, params


def test_characterize_hamiltonian_smoke(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """``characterize(ham, params, ...)`` returns CharacterizationResult with memory metrics."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    out = mc.characterize(
        ham,
        params,
        k=1,
        cut=1,
        n_pasts=3,
        n_futures=3,
        rng=np.random.default_rng(0),
    )
    assert isinstance(out, CharacterizationResult)
    assert out.entropy(1) >= 0.0
    assert out.rank(1) >= 1
    assert out.memory_matrix(1).ndim == 2


def test_characterize_reuses_probe_set(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """Passing the same probe_set yields reproducible characterization."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    probe_set = sample_split_cut_probes(cut=1, k=1, n_pasts=3, n_futures=3, rng=np.random.default_rng(0))
    first = mc.characterize(ham, params, k=1, cut=1, probe_set=probe_set)
    second = mc.characterize(ham, params, k=1, cut=1, probe_set=probe_set)
    assert second.entropy(1) == pytest.approx(first.entropy(1))
    assert second.rank(1) == first.rank(1)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_train_then_characterize(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """Train returns a model; characterize returns CharacterizationResult diagnostics."""
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
    out = mc.characterize(model, cut=1, k=1, n_pasts=4, n_futures=4)
    assert isinstance(out, CharacterizationResult)
    assert out.entropy(1) >= 0.0


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_predict_surrogate_smoke(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """predict(model, rho0, sequence) returns a valid density matrix."""
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
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    rho_out = mc.predict(model, rho0, "measure_prepare", k=1)
    assert rho_out.shape == (2, 2)
    assert np.all(np.isfinite(rho_out))


def test_build_comb_then_characterize(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """build_comb returns a comb; characterize returns CharacterizationResult diagnostics."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_comb(ham, params, timesteps=[0.1], num_trajectories=12, return_type="dense")
    out = mc.characterize(comb, cut=1, k=1, n_pasts=3, n_futures=3)
    assert isinstance(out, CharacterizationResult)
    assert out.entropy(1) >= 0.0


def test_predict_comb_smoke(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """predict(comb, rho0, sequence, k=...) returns a valid density matrix."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_comb(ham, params, timesteps=[0.1], num_trajectories=12, return_type="dense")
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    rho_out = mc.predict(comb, rho0, "measure_prepare", k=1)
    assert rho_out.shape == (2, 2)
    assert np.all(np.isfinite(rho_out))


def test_predict_hamiltonian_removed(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """predict(ham, ...) is no longer supported."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    with pytest.raises(TypeError):
        mc.predict(ham, params, rho0, "measure_prepare", k=1)  # type: ignore[call-overload]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_predict_surrogate_different_k(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """Train at k=2; predict at k=1 and k=3 returns finite density matrices."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    model = mc.train(
        ham,
        params,
        k=2,
        n=8,
        train_kwargs={"epochs": 1, "batch_size": 4},
        model_kwargs={"d_model": 32, "nhead": 2, "num_layers": 1, "dim_ff": 64},
    )
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    for k_prime in (1, 3):
        rho_out = mc.predict(model, rho0, "measure_prepare", k=k_prime)
        assert rho_out.shape == (2, 2)
        assert np.all(np.isfinite(rho_out))
