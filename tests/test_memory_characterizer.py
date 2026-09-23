# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :class:`~mqt.yaqs.memory_characterizer.MemoryCharacterizer`."""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pytest
from torch_support import requires_torch

from mqt.yaqs import MPO, AnalogSimParams, Hamiltonian, MemoryCharacterizer
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet, sample_probes
from mqt.yaqs.characterization.memory.shared.utils import make_zero_psi

with contextlib.suppress(ImportError):
    import mqt.yaqs.characterization.memory.backends.surrogates.workflow as wf
    from mqt.yaqs.characterization.memory.backends.surrogates.model import ProcessTensorSurrogate

_PAPER_L = 6
_PAPER_K = 20
_PAPER_G = 1.0
_PAPER_SEED = 0


@dataclass(kw_only=True)
class _DummySurrogateTarget:
    """Surrogate-like target with a test-controlled inferred intervention count."""

    num_interventions: object

    def _num_interventions_for_probe(self) -> object:
        """Return the test-controlled inferred intervention count."""
        return self.num_interventions


@dataclass(kw_only=True)
class _DummyProcessTensorTarget:
    """Process-tensor-like target with a test-controlled inferred intervention count."""

    num_interventions: object

    def _num_interventions_for_probe(self) -> object:
        """Return the test-controlled inferred intervention count."""
        return self.num_interventions


def _paper_params() -> AnalogSimParams:
    return AnalogSimParams(dt=0.1)


def _paper_mc() -> MemoryCharacterizer:
    return MemoryCharacterizer(parallel=False, show_progress=False)


def test_memory_characterizer_accepts_numpy_constructor_scalars() -> None:
    """NumPy scalar equivalents are normalized at the public constructor boundary."""
    characterizer = MemoryCharacterizer(
        parallel=np.zeros((), dtype=np.bool_)[()],  # ty: ignore[invalid-argument-type]
        max_workers=np.int64(2),  # ty: ignore[invalid-argument-type]
        vector_max_qubits=np.int64(8),  # ty: ignore[invalid-argument-type]
    )

    assert characterizer.parallel is False
    assert characterizer.max_workers == 2
    assert characterizer.vector_max_qubits == 8


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"parallel": "false"}, TypeError, "parallel must be a boolean"),
        ({"representation": 1}, TypeError, "representation must be a string"),
        ({"representation": "AUTO"}, ValueError, "representation must be one of"),
        ({"vector_max_qubits": 1.5}, TypeError, "vector_max_qubits must be an integer"),
        ({"vector_max_qubits": -1}, ValueError, "vector_max_qubits must be >= 0"),
    ],
)
def test_memory_characterizer_rejects_invalid_constructor_settings(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    """Constructor settings fail before a characterization workflow starts."""
    with pytest.raises(error, match=match):
        MemoryCharacterizer(**cast("Any", kwargs))


@pytest.mark.parametrize(
    ("attribute", "value", "error", "match"),
    [
        ("order", 3, ValueError, "order must be 1 or 2"),
        ("dt", "0.1", TypeError, "dt must be a real number"),
    ],
)
def test_memory_characterizer_revalidates_mutated_simulation_controls_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    """Memory workflows reject invalid post-construction controls before backend dispatch."""
    params = AnalogSimParams()
    setattr(params, attribute, value)
    monkeypatch.setattr("mqt.yaqs.memory_characterizer._require_hamiltonian", pytest.fail)
    monkeypatch.setattr("mqt.yaqs.memory_characterizer._build_process_tensor", pytest.fail)

    with pytest.raises(error, match=match):
        MemoryCharacterizer(parallel=False, show_progress=False).build_process_tensor(
            Hamiltonian.ising(1, J=0.0, g=0.0),
            params,
        )


def _sample_cut_probes(*, cut: int, n_pasts: int, n_futures: int, num_interventions: int = _PAPER_K) -> ProbeSet:
    rng = np.random.default_rng(_PAPER_SEED + 10_000 * int(cut))
    return sample_probes(
        cut=int(cut),
        num_interventions=int(num_interventions),
        n_pasts=int(n_pasts),
        n_futures=int(n_futures),
        rng=rng,
        intervention_style="haar",
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
    assert out.modes(1) >= 1
    assert out.response_matrix(1).ndim == 2


def test_characterize_hamiltonian_mps_matches_vector_site_order() -> None:
    """MPS and vector characterization agree for asymmetric site-0 dynamics."""
    identity = np.eye(2, dtype=np.complex128)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    hamiltonian = Hamiltonian(matrix=np.asarray(np.kron(identity, pauli_x), dtype=np.complex128))
    params = AnalogSimParams(dt=0.05, max_bond_dim=None, svd_threshold=1e-12, order=1)
    probe_set = sample_probes(
        cut=1,
        num_interventions=1,
        n_pasts=2,
        n_futures=2,
        rng=np.random.default_rng(52),
        intervention_style="haar",
    )
    initial_psi = np.zeros(4, dtype=np.complex128)
    initial_psi[2] = 1.0

    vector_result = MemoryCharacterizer(
        representation="vector",
        parallel=False,
        show_progress=False,
    ).characterize(
        hamiltonian,
        params,
        num_interventions=1,
        cut=1,
        probe_set=probe_set,
        initial_psi=initial_psi,
    )
    mps_result = MemoryCharacterizer(
        representation="mps",
        parallel=False,
        show_progress=False,
    ).characterize(
        hamiltonian,
        params,
        num_interventions=1,
        cut=1,
        probe_set=probe_set,
        initial_psi=initial_psi,
    )

    np.testing.assert_allclose(mps_result.response_matrix(1), vector_result.response_matrix(1), atol=1e-10)


@requires_torch
def test_sample_mps_matches_vector_representation() -> None:
    """Public MPS sampling preserves the same site-0 states as vector sampling."""
    hamiltonian = Hamiltonian.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)

    vector_dataset = MemoryCharacterizer(representation="vector", parallel=False, show_progress=False).sample(
        hamiltonian,
        params,
        num_interventions=2,
        n=2,
        seed=17,
        timesteps=[0.0, 0.0, 0.0],
        intervention_style="measure_prepare",
    )
    mps_dataset = MemoryCharacterizer(representation="mps", parallel=False, show_progress=False).sample(
        hamiltonian,
        params,
        num_interventions=2,
        n=2,
        seed=17,
        timesteps=[0.0, 0.0, 0.0],
        intervention_style="measure_prepare",
    )

    assert tuple(mps_dataset.tensors[0].shape) == (2, 2, 32)
    assert tuple(mps_dataset.tensors[1].shape) == (2, 8)
    assert tuple(mps_dataset.tensors[2].shape) == (2, 2, 8)
    for vector_tensor, mps_tensor in zip(vector_dataset.tensors, mps_dataset.tensors, strict=True):
        np.testing.assert_allclose(mps_tensor.numpy(), vector_tensor.numpy(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("representation", ["vector", "mps"])
def test_characterize_rejects_nonqubit_hamiltonian(representation: Literal["vector", "mps"]) -> None:
    """Memory characterization rejects nonqubit dimensions before backend work."""
    operator = MPO.from_local_ops([
        np.eye(3, dtype=np.complex128),
        np.eye(3, dtype=np.complex128),
    ])
    hamiltonian = Hamiltonian.from_mpo(operator)
    characterizer = MemoryCharacterizer(
        representation=representation,
        parallel=False,
        show_progress=False,
    )

    with pytest.raises(
        ValueError,
        match=r"supports qubit Hamiltonians only; got local physical dimensions \[3, 3\]",
    ):
        characterizer.characterize(
            hamiltonian,
            AnalogSimParams(dt=0.1),
            num_interventions=1,
            cut=1,
            n_pasts=1,
            n_futures=1,
        )


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
    assert second.modes(1) == first.modes(1)


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


@requires_torch
def test_train_default_style_is_haar(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """train() defaults to intervention_style='haar' when style is omitted."""
    ham, params = ham_and_params
    captured: dict[str, str] = {}

    def _fake_train(*_args: object, **kwargs: object) -> object:
        captured["intervention_style"] = str(kwargs["intervention_style"])
        return ProcessTensorSurrogate(d_e=32, d_rho=8, d_model=16, nhead=2, num_layers=1, dim_ff=32)

    monkeypatch.setattr(wf, "train_surrogate_model", _fake_train)
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    mc.train(ham, params, num_interventions=1, n=4, train_kwargs={"epochs": 0})
    assert captured["intervention_style"] == "haar"


@requires_torch
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
    out = mc.characterize(
        model,
        cut=1,
        num_interventions=1,
        n_pasts=4,
        n_futures=4,
        initial_rho=np.eye(2, dtype=np.complex128) / 2.0,
    )
    assert out.entropy(1) >= 0.0


@requires_torch
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
    pt = mc.build_process_tensor(ham, params, timesteps=[0.1, 0.1], num_trajectories=12, return_type="dense")
    out = mc.characterize(pt, cut=1, num_interventions=1, n_pasts=3, n_futures=3)
    assert out.entropy(1) >= 0.0
    with pytest.raises(ValueError, match="initial_rho is supported only for surrogate characterization"):
        mc.characterize(
            pt,
            cut=1,
            num_interventions=1,
            n_pasts=1,
            n_futures=1,
            initial_rho=pt.initial_rho,
        )


def test_characterize_process_tensor_default_cut(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """characterize() uses interior default cut when cut is omitted."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    pt = mc.build_process_tensor(ham, params, timesteps=[0.1, 0.1, 0.1], num_trajectories=30, return_type="dense")
    rng = np.random.default_rng(0)
    default_cut = (2 + 1) // 2
    ent_default = mc.characterize(pt, num_interventions=2, n_pasts=4, n_futures=4, rng=rng).entropy(default_cut)
    ent_explicit = mc.characterize(
        pt,
        cut=default_cut,
        num_interventions=2,
        n_pasts=4,
        n_futures=4,
        rng=np.random.default_rng(0),
    ).entropy(default_cut)
    assert ent_default == pytest.approx(ent_explicit)
    result = mc.characterize(
        pt,
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


@requires_torch
def test_process_tensor_surrogate_characterize_singular_values_shape() -> None:
    """Characterize returns the full SVD spectrum for a surrogate."""
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
        initial_rho=np.eye(2, dtype=np.complex128) / 2.0,
    ).singular_values(2)
    assert sv.ndim == 1
    assert 1 <= sv.size <= min(4, 3 * 3)


@requires_torch
def test_process_tensor_surrogate_characterize_requires_initial_rho() -> None:
    """Characterization does not guess the surrogate's post-evolution boundary state."""
    model = ProcessTensorSurrogate(
        d_e=32,
        d_rho=8,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_ff=64,
        dropout=0.0,
        num_interventions=1,
    )
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    with pytest.raises(ValueError, match="initial_rho is required for surrogate characterization"):
        mc.characterize(model, cut=1, n_pasts=1, n_futures=1)


def test_predict_process_tensor_smoke(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """predict(process_tensor, rho0, sequence, num_interventions=...) returns a valid density matrix."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    pt = mc.build_process_tensor(ham, params, timesteps=[0.1, 0.1], num_trajectories=12, return_type="dense")
    rho_out = mc.predict(pt, pt.initial_rho, "haar", num_interventions=1)
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
    pt = mc.build_process_tensor(ham, params, timesteps=[0.1, 0.1], num_trajectories=12, return_type="dense")
    with pytest.raises(ValueError, match="return_sequence=True"):
        mc.predict(pt, pt.initial_rho, "haar", num_interventions=1, return_sequence=True)


def test_predict_process_tensor_rejects_mismatched_rho0(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """Process-tensor predict validates rho0 against the stored reference initial state."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    pt = mc.build_process_tensor(ham, params, timesteps=[0.1, 0.1], num_trajectories=12, return_type="dense")
    with pytest.raises(ValueError, match="rho0 must be shape"):
        mc.predict(pt, np.array([99.0]), "haar", num_interventions=1)
    with pytest.raises(ValueError, match="rho0 does not match"):
        mc.predict(pt, np.eye(2, dtype=np.complex128) / 2.0, "haar", num_interventions=1)


def test_compute_qmi_and_cmi_process_tensor(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """compute_qmi and compute_cmi delegate to reference process tensors."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    pt = mc.build_process_tensor(ham, params, timesteps=[0.1, 0.1], return_type="dense")
    assert mc.compute_qmi(pt, past="all") == pt.qmi(past="all")
    assert mc.compute_cmi(pt) == pt.cmi()


def test_compute_qmi_rejects_non_process_tensor(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """compute_qmi and compute_cmi require reference process tensor targets."""
    ham, _params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    with pytest.raises(TypeError, match="compute_qmi requires"):
        mc.compute_qmi(cast("Any", ham))
    with pytest.raises(TypeError, match="compute_cmi requires"):
        mc.compute_cmi(cast("Any", ham))


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
        mc.build_process_tensor(ham, params, timesteps=[0.1, 0.1], parallel=False)
    assert seen == [False]


@requires_torch
def test_predict_surrogate_different_k(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """Train at k=2; predict at k=1 and k=3 returns finite density matrices."""
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
    assert result.modes(4) >= 1.0


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
    assert result.modes(4) == pytest.approx(1.0, abs=0.05)


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
    assert entropies[-1] > entropies[0] + 0.02
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
    assert s_j2 > 10.0 * s_j05


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

    small_bath = integrated_entropy(2)
    large_bath = integrated_entropy(3)
    assert small_bath > 1.01 * large_bath


def test_paper_modes_and_spectrum_plot_data_shift_with_coupling() -> None:
    """The public result provides the mode and spectrum data plotted in the paper."""
    mc = _paper_mc()
    cut = 2
    m_spectrum = 8

    def plot_data(j: float) -> tuple[float, float]:
        probe_seed = _PAPER_SEED + 900_000 + 100_000 * cut + 100 * round(100 * j)
        probe_set = sample_probes(
            cut=cut,
            num_interventions=_PAPER_K,
            n_pasts=m_spectrum,
            n_futures=m_spectrum,
            rng=np.random.default_rng(probe_seed),
            intervention_style="haar",
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
        response_matrix = result.response_matrix(cut)
        singular_values = result.singular_values_full(cut)
        mode_weights = singular_values**2 / np.sum(singular_values**2)
        retained_values = result.singular_values(cut)
        retained_weights = retained_values**2 / np.sum(retained_values**2)
        positive = retained_weights > 0.0
        expected_entropy = float(-np.sum(retained_weights[positive] * np.log(retained_weights[positive])))

        assert response_matrix.shape == (4 * m_spectrum, m_spectrum)
        assert np.all(np.isfinite(response_matrix))
        assert np.all(np.isfinite(singular_values))
        assert np.all(singular_values >= 0.0)
        assert np.all(np.diff(singular_values) <= 0.0)
        assert np.sum(mode_weights) == pytest.approx(1.0, abs=1e-12)
        assert result.entropy(cut) == pytest.approx(expected_entropy, abs=1e-12)
        assert result.modes(cut) == pytest.approx(math.exp(expected_entropy), abs=1e-12)
        return result.modes(cut), float(np.sum(mode_weights[1:]))

    weak_modes, weak_tail_weight = plot_data(0.5)
    strong_modes, strong_tail_weight = plot_data(2.0)
    assert strong_modes > weak_modes
    assert strong_tail_weight > weak_tail_weight


def test_paper_reset_delay_entropy_decreases_at_strong_coupling() -> None:
    """A reduced Figure 5 sweep loses memory under a longer reset at strong coupling."""
    mc = _paper_mc()
    cut = 16
    k = 21
    n_pasts = n_futures = 8
    probe_set = sample_probes(
        cut=cut,
        num_interventions=k,
        n_pasts=n_pasts,
        n_futures=n_futures,
        rng=np.random.default_rng(999_991),
        intervention_style="haar",
    )
    ham = Hamiltonian.ising(length=_PAPER_L, J=2.0, g=_PAPER_G)
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
    assert all(entropies[i + 1] < entropies[i] for i in range(len(entropies) - 1))


def test_characterize_delay_rejects_negative() -> None:
    """Negative reset delay is rejected by characterize()."""
    mc = _paper_mc()
    ham = Hamiltonian.ising(length=_PAPER_L, J=1.0, g=_PAPER_G)
    with pytest.raises(ValueError, match="delay must be >= 0"):
        mc.characterize(ham, _paper_params(), num_interventions=6, cut=4, delay=-1)


@pytest.mark.parametrize("delay", [0, 1])
def test_characterize_delay_rejects_process_tensor(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    delay: int,
) -> None:
    """Reset delay is supported for Hamiltonian characterize() only."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    pt = mc.build_process_tensor(ham, params, timesteps=[0.1, 0.1, 0.1], return_type="dense")
    with pytest.raises(ValueError, match="delay is supported for Hamiltonian"):
        mc.characterize(pt, cut=1, num_interventions=2, delay=delay)


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


def test_characterize_rejects_unknown_probe_kwargs(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """Stale probe_kwargs keys fail fast instead of being ignored."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    with pytest.raises(ValueError, match="Unsupported probe_kwargs"):
        mc.characterize(ham, params, num_interventions=2, cut=1, typo_style="haar")  # ty: ignore[no-matching-overload]


def test_characterize_accepts_intervention_style_keyword(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
) -> None:
    """intervention_style is set via the explicit keyword argument."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    result = mc.characterize(
        ham,
        params,
        num_interventions=2,
        cut=1,
        n_pasts=4,
        n_futures=4,
        intervention_style="clifford",
    )
    assert np.isfinite(result.entropy(1))


def test_characterize_rejects_invalid_probe_set(ham_and_params: tuple[Hamiltonian, AnalogSimParams]) -> None:
    """probe_set must be None, CharacterizationResult, or ProbeSet."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    with pytest.raises(TypeError, match="probe_set must be None, CharacterizationResult, or ProbeSet"):
        mc.characterize(ham, params, num_interventions=2, cut=1, probe_set={"bad": 1})


@pytest.mark.parametrize("method_name", ["sample", "train"])
@pytest.mark.parametrize(
    ("parameter", "value", "error", "match"),
    [
        ("num_interventions", 0, ValueError, "num_interventions must be >= 1"),
        ("num_interventions", -1, ValueError, "num_interventions must be >= 1"),
        ("num_interventions", False, TypeError, "num_interventions must be an integer"),
        ("num_interventions", 1.5, TypeError, "num_interventions must be an integer"),
        ("num_interventions", "1", TypeError, "num_interventions must be an integer"),
        ("n", 0, ValueError, "n must be >= 1"),
        ("n", -1, ValueError, "n must be >= 1"),
        ("n", False, TypeError, "n must be an integer"),
        ("n", 1.5, TypeError, "n must be an integer"),
        ("n", "1", TypeError, "n must be an integer"),
    ],
)
def test_sample_and_train_validate_counts_before_hamiltonian_conversion(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    parameter: str,
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    """Invalid sampling sizes fail before Hamiltonian conversion and optional imports."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    def _fail_conversion(_hamiltonian: Hamiltonian) -> None:
        pytest.fail("Hamiltonian conversion must not run for invalid counts")

    monkeypatch.setattr("mqt.yaqs.memory_characterizer._require_hamiltonian", _fail_conversion)
    kwargs: dict[str, Any] = {"num_interventions": 1, "n": 1}
    kwargs[parameter] = value
    method = getattr(mc, method_name)
    with pytest.raises(error, match=match):
        method(ham, params, **kwargs)


@pytest.mark.parametrize(
    ("parameter", "value", "error", "match"),
    [
        ("n_pasts", 0, ValueError, "n_pasts must be >= 1"),
        ("n_pasts", -1, ValueError, "n_pasts must be >= 1"),
        ("n_pasts", False, TypeError, "n_pasts must be an integer"),
        ("n_pasts", 1.5, TypeError, "n_pasts must be an integer"),
        ("n_pasts", "1", TypeError, "n_pasts must be an integer"),
        ("n_futures", 0, ValueError, "n_futures must be >= 1"),
        ("n_futures", -1, ValueError, "n_futures must be >= 1"),
        ("n_futures", True, TypeError, "n_futures must be an integer"),
        ("n_futures", 1.5, TypeError, "n_futures must be an integer"),
        ("n_futures", "1", TypeError, "n_futures must be an integer"),
    ],
)
def test_characterize_validates_probe_counts_before_backend_setup(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
    parameter: str,
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    """Invalid probe counts fail before converting or constructing a backend."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    def _fail_conversion(_hamiltonian: Hamiltonian) -> None:
        pytest.fail("Hamiltonian conversion must not run for invalid probe counts")

    monkeypatch.setattr("mqt.yaqs.memory_characterizer._require_hamiltonian", _fail_conversion)
    kwargs: dict[str, Any] = {
        "num_interventions": 2,
        "cut": 1,
        "n_pasts": 1,
        "n_futures": 1,
    }
    kwargs[parameter] = value
    with pytest.raises(error, match=match):
        mc.characterize(ham, params, **kwargs)


@pytest.mark.parametrize(
    ("value", "error"),
    [(0, ValueError), (-1, ValueError), (False, TypeError), (1.5, TypeError), ("1", TypeError)],
)
def test_characterize_validates_explicit_num_interventions_before_backend_setup(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    error: type[Exception],
) -> None:
    """Invalid explicit intervention counts fail before backend setup."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    def _fail_conversion(_hamiltonian: Hamiltonian) -> None:
        pytest.fail("Hamiltonian conversion must not run for invalid intervention counts")

    monkeypatch.setattr("mqt.yaqs.memory_characterizer._require_hamiltonian", _fail_conversion)
    with pytest.raises(error, match="num_interventions must"):
        mc.characterize(ham, params, num_interventions=cast("Any", value), cut=1)


@pytest.mark.parametrize("target_type", [_DummyProcessTensorTarget, _DummySurrogateTarget])
@pytest.mark.parametrize(
    ("value", "error"),
    [(0, ValueError), (-1, ValueError), (False, TypeError), (1.5, TypeError), ("1", TypeError)],
)
def test_characterize_validates_inferred_num_interventions(
    monkeypatch: pytest.MonkeyPatch,
    target_type: type[_DummyProcessTensorTarget | _DummySurrogateTarget],
    value: object,
    error: type[Exception],
) -> None:
    """Invalid counts inferred from dummy process-tensor and surrogate targets are rejected."""
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    monkeypatch.setattr(
        "mqt.yaqs.characterization.memory.operational_memory.run.sample_probes",
        pytest.fail,
    )
    with pytest.raises(error, match="num_interventions must"):
        mc.characterize(target_type(num_interventions=value), cut=1, n_pasts=1, n_futures=1)


@pytest.mark.parametrize(
    ("cut_kwargs", "error"),
    [
        ({"cut": 0}, ValueError),
        ({"cut": -1}, ValueError),
        ({"cut": 3}, ValueError),
        ({"cut": 1.5}, TypeError),
        ({"cut": True}, TypeError),
        ({"cut": "1"}, TypeError),
        ({"cuts": [1, 3]}, ValueError),
    ],
)
def test_characterize_validates_single_and_listed_cuts_before_backend_setup(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
    cut_kwargs: dict[str, Any],
    error: type[Exception],
) -> None:
    """Every requested cut is an integer within the intervention range."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    def _fail_conversion(_hamiltonian: Hamiltonian) -> None:
        pytest.fail("Hamiltonian conversion must not run for invalid cuts")

    monkeypatch.setattr("mqt.yaqs.memory_characterizer._require_hamiltonian", _fail_conversion)
    with pytest.raises(error, match="cut must"):
        mc.characterize(ham, params, num_interventions=2, **cut_kwargs)


@pytest.mark.parametrize(
    ("delay", "error"),
    [(-1, ValueError), (False, TypeError), (1.5, TypeError), ("0", TypeError)],
)
def test_characterize_validates_delay_before_backend_setup(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
    delay: object,
    error: type[Exception],
) -> None:
    """A reset delay must be a non-negative integer before backend setup."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    def _fail_conversion(_hamiltonian: Hamiltonian) -> None:
        pytest.fail("Hamiltonian conversion must not run for an invalid delay")

    monkeypatch.setattr("mqt.yaqs.memory_characterizer._require_hamiltonian", _fail_conversion)
    with pytest.raises(error, match="delay must"):
        mc.characterize(ham, params, num_interventions=2, cut=1, delay=cast("Any", delay))


@pytest.mark.parametrize(
    ("target", "num_interventions", "error"),
    [
        (object(), 0, ValueError),
        (object(), -1, ValueError),
        (object(), False, TypeError),
        (object(), 1.5, TypeError),
        (object(), "1", TypeError),
        (_DummyProcessTensorTarget(num_interventions=0), None, ValueError),
        (_DummyProcessTensorTarget(num_interventions=-1), None, ValueError),
        (_DummyProcessTensorTarget(num_interventions=False), None, TypeError),
        (_DummyProcessTensorTarget(num_interventions=1.5), None, TypeError),
        (_DummyProcessTensorTarget(num_interventions="1"), None, TypeError),
        (_DummySurrogateTarget(num_interventions=0), None, ValueError),
        (_DummySurrogateTarget(num_interventions=-1), None, ValueError),
        (_DummySurrogateTarget(num_interventions=False), None, TypeError),
        (_DummySurrogateTarget(num_interventions=1.5), None, TypeError),
        (_DummySurrogateTarget(num_interventions="1"), None, TypeError),
    ],
)
def test_predict_validates_num_interventions_before_rho_conversion(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    num_interventions: object,
    error: type[Exception],
) -> None:
    """Explicit and inferred prediction lengths fail before density-matrix conversion."""
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    def _fail_rho_conversion(_rho: object) -> None:
        pytest.fail("rho0 conversion must not run for an invalid intervention count")

    monkeypatch.setattr("mqt.yaqs.memory_characterizer.coerce_rho_matrix", _fail_rho_conversion)
    with pytest.raises(error, match="num_interventions must"):
        mc.predict(
            target,
            np.array([99.0]),
            "haar",
            num_interventions=cast("Any", num_interventions),
        )


@pytest.mark.parametrize("method_name", ["sample", "train"])
def test_sample_and_train_accept_numpy_integer_counts_before_conversion(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    """NumPy integer sample and training counts pass validation unchanged."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    def _stop_conversion(_hamiltonian: Hamiltonian) -> None:
        msg = "validated counts"
        raise RuntimeError(msg)

    monkeypatch.setattr("mqt.yaqs.memory_characterizer._require_hamiltonian", _stop_conversion)
    with pytest.raises(RuntimeError, match="validated counts"):
        getattr(mc, method_name)(
            ham,
            params,
            num_interventions=np.int64(1),
            n=np.int64(1),
        )


@pytest.mark.parametrize("invalid_index", [0, 1, 2])
@pytest.mark.parametrize(
    ("invalid_cut", "error", "match"),
    [
        (0, ValueError, r"cut must be >= 1"),
        (-1, ValueError, r"cut must be >= 1"),
        (1.5, TypeError, r"cut must be an integer"),
        (False, TypeError, r"cut must be an integer"),
        ("1", TypeError, r"cut must be an integer"),
    ],
)
def test_characterize_validates_every_explicit_cut_list_entry_before_conversion(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
    invalid_index: int,
    invalid_cut: object,
    error: type[Exception],
    match: str,
) -> None:
    """Each position in an explicit cut list receives the same range validation."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    cuts: list[object] = [1, 2, 3]
    cuts[invalid_index] = invalid_cut
    monkeypatch.setattr("mqt.yaqs.memory_characterizer._require_hamiltonian", pytest.fail)

    with pytest.raises(error, match=match):
        mc.characterize(ham, params, num_interventions=3, cuts=cast("Any", cuts))


@pytest.mark.parametrize(
    "cut_kwargs",
    [
        {"cut": np.int64(2)},
        {"cuts": [np.int64(1), np.int64(2), np.int64(3)]},
    ],
)
def test_characterize_accepts_numpy_integer_sizes_and_zero_delay_before_conversion(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
    cut_kwargs: dict[str, Any],
) -> None:
    """NumPy integer probe sizes, cuts, and the zero-delay boundary pass validation."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    def _stop_conversion(_hamiltonian: Hamiltonian) -> None:
        msg = "validated characterization sizes"
        raise RuntimeError(msg)

    monkeypatch.setattr("mqt.yaqs.memory_characterizer._require_hamiltonian", _stop_conversion)
    with pytest.raises(RuntimeError, match="validated characterization sizes"):
        mc.characterize(
            ham,
            params,
            num_interventions=np.int64(3),  # ty: ignore[invalid-argument-type]
            n_pasts=np.int64(1),  # ty: ignore[invalid-argument-type]
            n_futures=np.int64(1),  # ty: ignore[invalid-argument-type]
            delay=np.int64(0),  # ty: ignore[invalid-argument-type]
            **cut_kwargs,
        )


@pytest.mark.parametrize(
    ("return_type", "parameter", "value", "error", "match"),
    [
        ("dense", "num_trajectories", 0, ValueError, r"num_trajectories must be >= 1"),
        ("dense", "num_trajectories", -1, ValueError, r"num_trajectories must be >= 1"),
        ("dense", "num_trajectories", 1.5, TypeError, r"num_trajectories must be an integer"),
        ("dense", "num_trajectories", False, TypeError, r"num_trajectories must be an integer"),
        ("dense", "num_trajectories", "1", TypeError, r"num_trajectories must be an integer"),
        ("mpo", "max_bond_dim", 0, ValueError, r"max_bond_dim must be >= 1"),
        ("mpo", "max_bond_dim", -1, ValueError, r"max_bond_dim must be >= 1"),
        ("mpo", "max_bond_dim", 1.5, TypeError, r"max_bond_dim must be an integer"),
        ("mpo", "max_bond_dim", False, TypeError, r"max_bond_dim must be an integer"),
        ("mpo", "max_bond_dim", "1", TypeError, r"max_bond_dim must be an integer"),
        ("mpo", "compress_every", 0, ValueError, r"compress_every must be >= 1"),
        ("mpo", "compress_every", -1, ValueError, r"compress_every must be >= 1"),
        ("mpo", "compress_every", 1.5, TypeError, r"compress_every must be an integer"),
        ("mpo", "compress_every", False, TypeError, r"compress_every must be an integer"),
        ("mpo", "compress_every", "1", TypeError, r"compress_every must be an integer"),
        ("mpo", "n_sweeps", -1, ValueError, r"n_sweeps must be >= 0"),
        ("mpo", "n_sweeps", 1.5, TypeError, r"n_sweeps must be an integer"),
        ("mpo", "n_sweeps", False, TypeError, r"n_sweeps must be an integer"),
        ("mpo", "n_sweeps", "0", TypeError, r"n_sweeps must be an integer"),
    ],
)
def test_build_process_tensor_validates_selected_path_sizes_before_conversion(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
    return_type: Literal["dense", "mpo"],
    parameter: str,
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    """Invalid tomography sizes fail before Hamiltonian conversion."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)
    monkeypatch.setattr("mqt.yaqs.memory_characterizer._require_hamiltonian", pytest.fail)
    kwargs: dict[str, Any] = {"return_type": return_type, parameter: value}

    with pytest.raises(error, match=match):
        mc.build_process_tensor(ham, params, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"return_type": "dense", "num_trajectories": np.int64(1)},
        {
            "return_type": "mpo",
            "max_bond_dim": np.int64(2),
            "compress_every": np.int64(1),
            "n_sweeps": np.int64(0),
        },
    ],
)
def test_build_process_tensor_accepts_numpy_integer_sizes_before_conversion(
    ham_and_params: tuple[Hamiltonian, AnalogSimParams],
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
) -> None:
    """NumPy integer tomography sizes and zero sweeps pass validation."""
    ham, params = ham_and_params
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    def _stop_conversion(_hamiltonian: Hamiltonian) -> None:
        msg = "validated tomography sizes"
        raise RuntimeError(msg)

    monkeypatch.setattr("mqt.yaqs.memory_characterizer._require_hamiltonian", _stop_conversion)
    with pytest.raises(RuntimeError, match="validated tomography sizes"):
        mc.build_process_tensor(ham, params, **kwargs)


@pytest.mark.parametrize(
    ("target", "num_interventions"),
    [
        (object(), np.int64(2)),
        (_DummyProcessTensorTarget(num_interventions=np.int64(2)), None),
        (_DummySurrogateTarget(num_interventions=np.int64(2)), None),
    ],
)
def test_predict_accepts_numpy_integer_intervention_counts_before_rho_conversion(
    monkeypatch: pytest.MonkeyPatch,
    target: object,
    num_interventions: object,
) -> None:
    """Explicit and inferred NumPy integer prediction lengths pass validation."""
    mc = MemoryCharacterizer(parallel=False, show_progress=False)

    def _stop_rho_conversion(_rho: object) -> None:
        msg = "validated prediction size"
        raise RuntimeError(msg)

    monkeypatch.setattr("mqt.yaqs.memory_characterizer.coerce_rho_matrix", _stop_rho_conversion)
    with pytest.raises(RuntimeError, match="validated prediction size"):
        mc.predict(
            target,
            np.eye(2),
            "haar",
            num_interventions=cast("Any", num_interventions),
        )
