# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer`."""

from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, MemoryCharacterizer


@pytest.fixture
def ham_and_params() -> tuple[Hamiltonian, AnalogSimParams]:
    ham = Hamiltonian.ising(length=1, J=1.0, g=0.5)
    params = AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)
    return ham, params


def test_characterize_hamiltonian_smoke(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """``characterize(ham, params, ...)`` returns diagnostics with memory metrics."""
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
    assert out.entropy(1) >= 0.0
    assert out.rank(1) >= 1
    assert out.memory_matrix(1).ndim == 2


def test_characterize_reuses_probe_set(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """Passing a prior characterize() result reuses the same probes."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    first = mc.characterize(
        ham,
        params,
        k=1,
        cut=1,
        n_pasts=3,
        n_futures=3,
        rng=np.random.default_rng(0),
    )
    second = mc.characterize(ham, params, k=1, cut=1, probe_set=first)
    assert second.entropy(1) == pytest.approx(first.entropy(1))
    assert second.rank(1) == first.rank(1)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_train_default_style_is_haar(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train() defaults to style='haar' when style is omitted."""
    ham, params = ham_and_params
    captured: dict[str, str] = {}

    def _fake_train(*_args: object, **kwargs: object) -> object:
        captured["style"] = str(kwargs["style"])
        from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb  # noqa: PLC0415

        return TransformerComb(d_e=32, d_rho=8, d_model=16, nhead=2, num_layers=1, dim_ff=32)

    import mqt.yaqs.characterization.memory.backends.surrogates.workflow as wf  # noqa: PLC0415

    monkeypatch.setattr(wf, "train_surrogate_model", _fake_train)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    mc.train(ham, params, k=1, n=4, train_kwargs={"epochs": 0})
    assert captured["style"] == "haar"


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
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
    assert out.entropy(1) >= 0.0


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
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
    rho_out = mc.predict(model, rho0, "haar", k=1)
    assert rho_out.shape == (2, 2)
    assert np.all(np.isfinite(rho_out))


def test_build_comb_then_characterize(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """build_comb returns a comb; characterize returns CharacterizationResult diagnostics."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_comb(ham, params, timesteps=[0.1], num_trajectories=12, return_type="dense")
    out = mc.characterize(comb, cut=1, k=1, n_pasts=3, n_futures=3)
    assert out.entropy(1) >= 0.0


def test_characterize_comb_default_cut(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """characterize() uses interior default cut when cut is omitted."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_comb(ham, params, timesteps=[0.1, 0.1], num_trajectories=30, return_type="dense")
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


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_transformercomb_characterize_singular_values_shape(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
) -> None:
    """Characterize returns the full SVD spectrum for a surrogate."""
    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb

    _ham, _params = ham_and_params
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


def test_predict_comb_smoke(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """predict(comb, rho0, sequence, k=...) returns a valid density matrix."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_comb(ham, params, timesteps=[0.1], num_trajectories=12, return_type="dense")
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    rho_out = mc.predict(comb, rho0, "haar", k=1)
    assert rho_out.shape == (2, 2)
    assert np.all(np.isfinite(rho_out))


def test_predict_hamiltonian_removed(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """predict(ham, ...) is no longer supported."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    with pytest.raises(TypeError):
        mc.predict(ham, params, rho0, "haar", k=1)  # type: ignore[call-overload]


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
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
        rho_out = mc.predict(model, rho0, "haar", k=k_prime)
        assert rho_out.shape == (2, 2)
        assert np.all(np.isfinite(rho_out))


@pytest.fixture
def paper_params() -> AnalogSimParams:
    """Analog parameters for L=2 paper-style benchmark geometry."""
    return AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)


def test_characterize_paper_geometry_finite_entropy(paper_params: AnalogSimParams) -> None:
    """L=2, k=8 characterize path yields finite S_V and R (quick benchmark geometry)."""
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    ham = Hamiltonian.ising(length=2, J=1.0, g=1.0)
    result = mc.characterize(
        ham,
        paper_params,
        k=8,
        cut=4,
        n_pasts=8,
        n_futures=8,
        rng=np.random.default_rng(0),
    )
    assert result.entropy(4) >= 0.0
    assert result.rank(4) >= 1.0


def test_characterize_markovian_at_zero_coupling(paper_params: AnalogSimParams) -> None:
    """With J=0 the process is Markovian: cross-cut memory entropy S_V is near zero."""
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    ham = Hamiltonian.ising(length=2, J=0.0, g=1.0)
    result = mc.characterize(
        ham,
        paper_params,
        k=8,
        cut=4,
        n_pasts=12,
        n_futures=12,
        rng=np.random.default_rng(11),
    )
    assert result.entropy(4) < 0.05
    assert result.rank(4) == pytest.approx(1.0, abs=0.05)


def test_characterize_entropy_monotone_in_coupling(paper_params: AnalogSimParams) -> None:
    """S_V at fixed cut increases monotonically with Ising coupling J."""
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    j_values = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    anchor = mc.characterize(
        Hamiltonian.ising(length=2, J=0.0, g=1.0),
        paper_params,
        k=8,
        cut=4,
        n_pasts=12,
        n_futures=12,
        rng=np.random.default_rng(42),
    )
    entropies = [anchor.entropy(4)]
    for jv in j_values[1:]:
        result = mc.characterize(
            Hamiltonian.ising(length=2, J=jv, g=1.0),
            paper_params,
            k=8,
            cut=4,
            probe_set=anchor,
        )
        entropies.append(result.entropy(4))
    assert entropies[0] < 0.05
    assert entropies[-1] > entropies[0] + 0.1
    assert all(entropies[i + 1] >= entropies[i] - 1e-4 for i in range(len(entropies) - 1))
