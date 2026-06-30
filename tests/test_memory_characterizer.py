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
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet, sample_probes
from mqt.yaqs.memory_characterizer import make_zero_psi

_PAPER_L = 6
_PAPER_K = 20
_PAPER_G = 1.0
_PAPER_SEED = 0


def _paper_params() -> AnalogSimParams:
    return AnalogSimParams(dt=0.1)


def _paper_mc() -> MemoryCharacterizer:
    return MemoryCharacterizer(parallel=False, show_progress=False)


def _sample_cut_probes(*, cut: int, n_pasts: int, n_futures: int, num_interventions: int = _PAPER_K) -> ProbeSet:
    rng = np.random.default_rng(_PAPER_SEED + 10_000 * int(cut))
    return sample_probes(
        cut=int(cut),
        num_interventions=int(num_interventions),
        n_pasts=int(n_pasts),
        n_futures=int(n_futures),
        rng=rng,
        intervention_mode="split_cut_unitary",
        unitary_ensemble="haar",
    )


def _entropy_at_j(
    mc: MemoryCharacterizer,
    *,
    cut: int,
    j: float,
    n_pasts: int,
    n_futures: int,
    probe_set: ProbeSet,
    length: int = _PAPER_L,
    num_interventions: int = _PAPER_K,
) -> float:
    ham = Hamiltonian.ising(length=length, J=float(j), g=_PAPER_G)
    result = mc.characterize(
        ham,
        _paper_params(),
        num_interventions=int(num_interventions),
        cut=int(cut),
        n_pasts=int(n_pasts),
        n_futures=int(n_futures),
        probe_set=probe_set,
        initial_psi=make_zero_psi(length),
    )
    return float(result.entropy(int(cut)))


@pytest.fixture
def ham_and_params() -> tuple[Hamiltonian, AnalogSimParams]:
    """Single-qubit Ising Hamiltonian and analog simulation parameters.

    Returns:
        Hamiltonian and :class:`~mqt.yaqs.AnalogSimParams` pair.
    """
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
        num_interventions=1,
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
        num_interventions=1,
        cut=1,
        n_pasts=3,
        n_futures=3,
        rng=np.random.default_rng(0),
    )
    second = mc.characterize(ham, params, num_interventions=1, cut=1, probe_set=first)
    assert second.entropy(1) == pytest.approx(first.entropy(1))
    assert second.rank(1) == first.rank(1)


def test_characterize_rejects_cut_and_cuts_together(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """cut= and cuts= are mutually exclusive."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    with pytest.raises(ValueError, match="Specify only one of cut="):
        mc.characterize(ham, params, num_interventions=2, cut=1, cuts=[1, 2])


def test_characterize_rejects_empty_cuts(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """An explicit empty cuts list is rejected up front."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    with pytest.raises(ValueError, match="cuts must be 'all' or a non-empty list"):
        mc.characterize(ham, params, num_interventions=2, cuts=[])


def test_characterize_rejects_probe_set_for_multi_cut(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """probe_set cannot be reused when characterize sweeps multiple cuts."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    first = mc.characterize(
        ham,
        params,
        num_interventions=2,
        cut=1,
        n_pasts=3,
        n_futures=3,
        rng=np.random.default_rng(0),
    )
    with pytest.raises(ValueError, match="probe_set cannot be reused across multiple cuts"):
        mc.characterize(ham, params, num_interventions=2, cuts="all", probe_set=first)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_train_default_style_is_haar(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train() defaults to intervention_style='haar' when style is omitted."""
    ham, params = ham_and_params
    captured: dict[str, str] = {}

    def _fake_train(*_args: object, **kwargs: object) -> object:
        captured["style"] = str(kwargs["style"])
        from mqt.yaqs.characterization.memory.backends.surrogates.model import ProcessTensorSurrogate  # noqa: PLC0415

        return ProcessTensorSurrogate(d_e=32, d_rho=8, d_model=16, nhead=2, num_layers=1, dim_ff=32)

    import mqt.yaqs.characterization.memory.backends.surrogates.workflow as wf  # noqa: PLC0415

    monkeypatch.setattr(wf, "train_surrogate_model", _fake_train)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    mc.train(ham, params, num_interventions=1, n=4, train_kwargs={"epochs": 0})
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
        num_interventions=1,
        n=8,
        train_kwargs={"epochs": 1, "batch_size": 4},
        model_kwargs={"d_model": 32, "nhead": 2, "num_layers": 1, "dim_ff": 64},
    )
    out = mc.characterize(model, cut=1, num_interventions=1, n_pasts=4, n_futures=4)
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
        num_interventions=1,
        n=8,
        train_kwargs={"epochs": 1, "batch_size": 4},
        model_kwargs={"d_model": 32, "nhead": 2, "num_layers": 1, "dim_ff": 64},
    )
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    rho_out = mc.predict(model, rho0, "haar", num_interventions=1)
    assert rho_out.shape == (2, 2)
    assert np.all(np.isfinite(rho_out))


def test_build_process_tensor_then_characterize(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """build_process_tensor returns a process tensor; characterize returns CharacterizationResult diagnostics."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_process_tensor(ham, params, timesteps=[0.1], num_trajectories=12, return_type="dense")
    out = mc.characterize(comb, cut=1, num_interventions=1, n_pasts=3, n_futures=3)
    assert out.entropy(1) >= 0.0


def test_characterize_process_tensor_default_cut(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """characterize() uses interior default cut when cut is omitted."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_process_tensor(ham, params, timesteps=[0.1, 0.1], num_trajectories=30, return_type="dense")
    rng = np.random.default_rng(0)
    default_cut = (2 + 1) // 2
    ent_default = mc.characterize(comb, num_interventions=2, n_pasts=4, n_futures=4, rng=rng).entropy(default_cut)
    ent_explicit = mc.characterize(
        comb,
        cut=default_cut,
        num_interventions=2,
        n_pasts=4,
        n_futures=4,
        rng=np.random.default_rng(0),
    ).entropy(default_cut)
    assert ent_default == pytest.approx(ent_explicit)
    result = mc.characterize(
        comb,
        cut=2,
        num_interventions=2,
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
    from mqt.yaqs.characterization.memory.backends.surrogates.model import ProcessTensorSurrogate  # noqa: PLC0415

    _ham, _params = ham_and_params
    model = ProcessTensorSurrogate(
        d_e=32,
        d_rho=8,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_ff=64,
        dropout=0.0,
        num_interventions=3,
    )
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    sv = mc.characterize(
        model,
        cut=2,
        n_pasts=4,
        n_futures=3,
        rng=np.random.default_rng(0),
    ).singular_values(2)
    assert sv.ndim == 1
    assert 1 <= sv.size <= min(4, 3 * 3)


def test_predict_process_tensor_smoke(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """predict(process_tensor, rho0, sequence, num_interventions=...) returns a valid density matrix."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_process_tensor(ham, params, timesteps=[0.1], num_trajectories=12, return_type="dense")
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    rho_out = mc.predict(comb, rho0, "haar", num_interventions=1)
    assert rho_out.shape == (2, 2)
    assert np.all(np.isfinite(rho_out))


def test_predict_hamiltonian_removed(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """predict(ham, ...) is no longer supported."""
    ham, _params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    with pytest.raises(TypeError, match="Unsupported predict target"):
        mc.predict(ham, rho0, "haar", num_interventions=1)


def test_predict_process_tensor_rejects_return_sequence(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """predict(process_tensor, ..., return_sequence=True) is not supported."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_process_tensor(ham, params, timesteps=[0.1], num_trajectories=12, return_type="dense")
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    with pytest.raises(ValueError, match="return_sequence=True"):
        mc.predict(comb, rho0, "haar", num_interventions=1, return_sequence=True)


def test_predict_process_tensor_ignores_invalid_rho0(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """Comb predict does not validate rho0 because it is unused."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_process_tensor(ham, params, timesteps=[0.1], num_trajectories=12, return_type="dense")
    rho_out = mc.predict(comb, np.array([99.0]), "haar", num_interventions=1)
    assert rho_out.shape == (2, 2)
    assert np.all(np.isfinite(rho_out))


def test_build_process_tensor_forwards_parallel_override(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_process_tensor passes the resolved parallel flag into build_process_tensor."""
    ham, params = ham_and_params
    seen: list[bool] = []

    def _capture(*_args: object, **kwargs: object) -> None:
        seen.append(bool(kwargs["parallel"]))
        msg = "stop"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        "mqt.yaqs.memory_characterizer._build_process_tensor",
        _capture,
    )
    mc = MemoryCharacterizer(parallel=True, show_progress=False)
    with pytest.raises(RuntimeError, match="stop"):
        mc.build_process_tensor(ham, params, timesteps=[0.1], parallel=False)
    assert seen == [False]


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_predict_surrogate_different_k(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """Train at num_interventions=2; predict at num_interventions=1 and num_interventions=3 returns finite density matrices."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    model = mc.train(
        ham,
        params,
        num_interventions=2,
        n=8,
        train_kwargs={"epochs": 1, "batch_size": 4},
        model_kwargs={"d_model": 32, "nhead": 2, "num_layers": 1, "dim_ff": 64},
    )
    rho0 = np.eye(2, dtype=np.complex128) / 2.0
    for k_prime in (1, 3):
        rho_out = mc.predict(model, rho0, "haar", num_interventions=k_prime)
        assert rho_out.shape == (2, 2)
        assert np.all(np.isfinite(rho_out))


@pytest.fixture
def paper_params() -> AnalogSimParams:
    """Analog parameters for L=2 paper-style benchmark geometry.

    Returns:
        Shared :class:`~mqt.yaqs.AnalogSimParams` for paper regression tests.
    """
    return AnalogSimParams(dt=0.1, max_bond_dim=12, order=1)


def test_characterize_paper_geometry_finite_entropy(paper_params: AnalogSimParams) -> None:
    """L=2, num_interventions=8 characterize path yields finite S_V and R (quick benchmark geometry)."""
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    ham = Hamiltonian.ising(length=2, J=1.0, g=1.0)
    result = mc.characterize(
        ham,
        paper_params,
        num_interventions=8,
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
        num_interventions=8,
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
        num_interventions=8,
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
            num_interventions=8,
            cut=4,
            probe_set=anchor,
        )
        entropies.append(result.entropy(4))
    assert entropies[0] < 0.05
    assert entropies[-1] > entropies[0] + 0.1
    assert all(entropies[i + 1] >= entropies[i] - 1e-4 for i in range(len(entropies) - 1))


def test_paper_cut_vs_j_entropy_rises_with_coupling() -> None:
    """Smoke cut x J benchmark: stronger coupling yields larger cross-cut memory."""
    mc = _paper_mc()
    cut = 2
    n_pasts = n_futures = 8
    probe_set = _sample_cut_probes(cut=cut, n_pasts=n_pasts, n_futures=n_futures)
    s_j0 = _entropy_at_j(mc, cut=cut, j=0.0, n_pasts=n_pasts, n_futures=n_futures, probe_set=probe_set)
    s_j05 = _entropy_at_j(mc, cut=cut, j=0.5, n_pasts=n_pasts, n_futures=n_futures, probe_set=probe_set)
    s_j2 = _entropy_at_j(mc, cut=cut, j=2.0, n_pasts=n_pasts, n_futures=n_futures, probe_set=probe_set)
    assert s_j0 < 0.01
    assert s_j2 > s_j05 + 0.005


def test_paper_finite_size_integrated_entropy_falls_with_bath() -> None:
    """Smoke finite-size benchmark: integrated memory weakens as the bath grows."""
    mc = _paper_mc()
    k = 4
    n_pasts = n_futures = 6
    cuts = list(range(1, k + 1))
    probe_sets = {c: _sample_cut_probes(cut=c, num_interventions=k, n_pasts=n_pasts, n_futures=n_futures) for c in cuts}
    jv = 1.0

    def integrated_entropy(length: int) -> float:
        ent = {
            c: _entropy_at_j(
                mc,
                cut=c,
                j=jv,
                n_pasts=n_pasts,
                n_futures=n_futures,
                probe_set=probe_sets[c],
                length=length,
                num_interventions=k,
            )
            for c in cuts
        }
        return float(sum(ent.values()))

    assert integrated_entropy(2) > integrated_entropy(3) + 0.0002


def test_paper_modes_rank_rises_with_coupling() -> None:
    """Smoke modes benchmark: effective rank grows with coupling at a fixed cut."""
    mc = _paper_mc()
    cut = 2
    m_spectrum = 8
    rank_tol = 1e-16

    def effective_rank(j: float) -> int:
        probe_seed = _PAPER_SEED + 900_000 + 100_000 * cut + 100 * round(100 * j)
        probe_set = sample_probes(
            cut=cut,
            num_interventions=_PAPER_K,
            n_pasts=m_spectrum,
            n_futures=m_spectrum,
            rng=np.random.default_rng(probe_seed),
            intervention_mode="split_cut_unitary",
            unitary_ensemble="haar",
        )
        result = mc.characterize(
            Hamiltonian.ising(length=_PAPER_L, J=float(j), g=_PAPER_G),
            _paper_params(),
            num_interventions=_PAPER_K,
            cut=cut,
            n_pasts=m_spectrum,
            n_futures=m_spectrum,
            probe_set=probe_set,
            initial_psi=make_zero_psi(_PAPER_L),
        )
        s = result.singular_values(cut)
        return int(np.sum(s > rank_tol))

    assert effective_rank(0.5) < effective_rank(2.0)
    assert effective_rank(2.0) >= 4


def test_paper_reset_delay_entropy_nondecreasing_at_unit_coupling() -> None:
    """Smoke reset-delay benchmark: memory grows with delay at moderate coupling."""
    mc = _paper_mc()
    cut = 4
    k = 6
    n_pasts = n_futures = 6
    probe_set = sample_probes(
        cut=cut,
        num_interventions=k,
        n_pasts=n_pasts,
        n_futures=n_futures,
        rng=np.random.default_rng(999_991),
        intervention_mode="split_cut_unitary",
        unitary_ensemble="haar",
    )
    ham = Hamiltonian.ising(length=_PAPER_L, J=1.0, g=_PAPER_G)
    entropies: list[float] = []
    for delay in (0, 1, 2):
        result = mc.characterize(
            ham,
            _paper_params(),
            num_interventions=k,
            cut=cut,
            delay=delay,
            n_pasts=n_pasts,
            n_futures=n_futures,
            probe_set=probe_set,
            initial_psi=make_zero_psi(_PAPER_L),
        )
        entropies.append(float(result.entropy(cut)))
    assert entropies[-1] > entropies[0] + 0.001
    assert all(entropies[i + 1] >= entropies[i] - 1e-4 for i in range(len(entropies) - 1))


def test_characterize_delay_rejects_negative() -> None:
    """Negative reset delay is rejected by characterize()."""
    mc = _paper_mc()
    ham = Hamiltonian.ising(length=_PAPER_L, J=1.0, g=_PAPER_G)
    with pytest.raises(ValueError, match="delay must be >= 0"):
        mc.characterize(ham, _paper_params(), num_interventions=6, cut=4, delay=-1)


def test_characterize_delay_rejects_process_tensor(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """Reset delay is supported for Hamiltonian characterize() only."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    comb = mc.build_process_tensor(ham, params, timesteps=[0.1, 0.1], return_type="dense")
    with pytest.raises(ValueError, match="delay > 0 is supported for Hamiltonian"):
        mc.characterize(comb, cut=1, num_interventions=2, delay=1)


def test_characterize_delay_reuses_prior_result_probes() -> None:
    """A prior characterize() result can anchor a delay sweep via probe_set=."""
    mc = _paper_mc()
    ham = Hamiltonian.ising(length=_PAPER_L, J=1.0, g=_PAPER_G)
    anchor = mc.characterize(
        ham,
        _paper_params(),
        num_interventions=6,
        cut=4,
        delay=0,
        n_pasts=4,
        n_futures=4,
        rng=np.random.default_rng(999_991),
    )
    delayed = mc.characterize(ham, _paper_params(), num_interventions=6, cut=4, delay=1, probe_set=anchor)
    assert np.isfinite(delayed.entropy(4))
