# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for surrogate data generation and training workflows."""

from __future__ import annotations

import contextlib
import sys
from typing import Literal

import numpy as np
import pytest
from torch_support import import_torch

import mqt.yaqs.characterization.memory.backends.surrogates.workflow as workflow_module
from mqt.yaqs import MPS
from mqt.yaqs.characterization.memory.backends.surrogates.data import SequenceRecord
from mqt.yaqs.characterization.memory.backends.surrogates.workflow import (
    build_training_dataset,
    pack_dataset,
    train_surrogate_model,
)
from mqt.yaqs.characterization.memory.shared.encoding import extract_ket, unpack_rho8
from mqt.yaqs.characterization.memory.shared.metrics import (
    compute_trace_distance,
    mean_frobenius_mse_rho8,
    mean_trace_distance_rho8,
)
from mqt.yaqs.characterization.memory.shared.utils import extract_site0_rho
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

with contextlib.suppress(ImportError):
    from torch.utils.data import TensorDataset

    from mqt.yaqs.characterization.memory.backends.surrogates.model import (
        ProcessTensorSurrogate,
    )


def test_extract_ket_fallback_for_zero_projector() -> None:
    """Zero projectors fall back to the |0> state vector."""
    psi = extract_ket(np.zeros((2, 2), dtype=np.complex128))
    np.testing.assert_allclose(psi, np.array([1.0 + 0.0j, 0.0 + 0.0j]))


def test_pack_dataset_shapes() -> None:
    """Rollout arrays convert to a three-tensor TensorDataset."""
    torch = import_torch()

    rho0 = np.zeros((2, 8), dtype=np.float32)
    e_features = np.zeros((2, 3, 32), dtype=np.float32)
    rho_seq = np.zeros((2, 3, 8), dtype=np.float32)
    ds = pack_dataset(rho0, e_features, rho_seq)
    assert len(ds.tensors) == 3
    assert tuple(ds.tensors[0].shape) == (2, 3, 32)
    assert tuple(ds.tensors[1].shape) == (2, 8)
    assert tuple(ds.tensors[2].shape) == (2, 3, 8)
    assert ds.tensors[0].dtype == torch.float32


def test_build_training_dataset_and_train_surrogate_model_tiny_smoke() -> None:
    """TJM data generation and surrogate training run end to end on a tiny Ising chain."""
    torch = import_torch()

    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)

    ds = build_training_dataset(
        op,
        params,
        num_interventions=1,
        n=2,
        seed=0,
        parallel=False,
        show_progress=False,
        timesteps=[0.0, 0.0],
        solver="TJM",
    )
    assert len(ds.tensors) == 3

    model = train_surrogate_model(
        op,
        params,
        num_interventions=1,
        n=2,
        seed=0,
        parallel=False,
        show_progress=False,
        timesteps=[0.0, 0.0],
        solver="TJM",
        model_kwargs={"d_model": 32, "nhead": 4, "num_layers": 1, "dim_ff": 64, "dropout": 0.0},
        train_kwargs={"epochs": 1, "batch_size": 2, "lr": 1e-3, "device": "cpu"},
    )
    e_features, rho0, tgt = ds.tensors
    dev = next(model.parameters()).device
    out = model(e_features.to(device=dev, dtype=torch.float32), rho0.to(device=dev, dtype=torch.float32))
    assert tuple(out.shape) == tuple(tgt.shape)


def test_sample_initial_tjm_state_uses_only_local_dense_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Long-chain TJM initialization never asks for a full dense state vector."""
    rho = np.array([[0.6, 0.1 + 0.2j], [0.1 - 0.2j, 0.4]], dtype=np.complex128)
    sampled_lengths: list[int] = []
    captured_states: list[MPS] = []
    original = workflow_module.sample_initial_psi

    def record_length(
        rho_in: np.ndarray,
        *,
        length: int,
        rng: np.random.Generator,
        init_mode: str,
        return_eig_sample: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, int, float]:
        sampled_lengths.append(length)
        return original(
            rho_in,
            length=length,
            rng=rng,
            init_mode=init_mode,
            return_eig_sample=return_eig_sample,
        )

    monkeypatch.setattr(workflow_module, "sample_initial_psi", record_length)
    monkeypatch.setattr(workflow_module, "sample_density_matrix", lambda _rng: rho)

    def capture_sequences(**kwargs: object) -> list[SequenceRecord]:
        initial_psis = kwargs["initial_psis"]
        assert isinstance(initial_psis, list)
        state = initial_psis[0]
        assert isinstance(state, MPS)
        captured_states.append(state)
        return [
            SequenceRecord(
                rho_0=np.zeros(8, dtype=np.float32),
                E_features=np.zeros((1, 32), dtype=np.float32),
                rho_seq=np.zeros((1, 8), dtype=np.float32),
                context=None,
                weight=1.0,
            )
        ]

    monkeypatch.setattr(workflow_module, "simulate_sequences", capture_sequences)
    build_training_dataset(
        MPO.ising(length=20, J=0.0, g=0.0),
        AnalogSimParams(dt=0.1),
        num_interventions=1,
        n=1,
        rng=np.random.default_rng(5),
        init_mode="purified",
        solver="TJM",
        parallel=False,
        show_progress=False,
        timesteps=[0.0, 0.0],
    )

    assert sampled_lengths == [2]
    assert len(captured_states) == 1
    state = captured_states[0]
    assert state.length == 20
    assert state.bond_dimensions() == [2] + [1] * 18
    np.testing.assert_allclose(extract_site0_rho(state), rho, atol=1e-12)


@pytest.mark.parametrize("solver", ["MCWF", "TJM"])
def test_build_training_dataset_rejects_nonqubit_operator(solver: Literal["MCWF", "TJM"]) -> None:
    """The direct surrogate entry point enforces the qubit-only memory contract."""
    operator = MPO.from_local_ops([np.eye(3, dtype=np.complex128)])

    with pytest.raises(ValueError, match=r"supports qubit Hamiltonians only.*\[3\]"):
        build_training_dataset(
            operator,
            AnalogSimParams(dt=0.1),
            num_interventions=1,
            n=1,
            timesteps=[0.0, 0.0],
            solver=solver,
            parallel=False,
            show_progress=False,
        )


def test_build_training_dataset_timesteps_length_mismatch_raises() -> None:
    """build_training_dataset enforces process-tensor schedule length num_interventions+1."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)

    with pytest.raises(ValueError, match="Process-tensor schedule: timesteps length must be num_interventions\\+1"):
        build_training_dataset(op, params, num_interventions=2, n=1, timesteps=[0.1])


def test_surrogate_end_to_end_accuracy_regression_tiny() -> None:
    """Trained surrogate achieves modest error on held-out rollout samples."""
    torch = import_torch()

    torch.manual_seed(0)

    # Two sites: system qubit + environment qubit, non-trivial Ising dynamics.
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1)

    # Small but non-trivial dataset: num_interventions=2 sequences, enough samples to evaluate generalization.
    k = 2
    n = 60
    # Fixed rank-1 leg intervention for a stable accuracy benchmark (independent of train default).
    ds = build_training_dataset(
        op,
        params,
        num_interventions=k,
        n=n,
        seed=123,
        intervention_style="measure_prepare",
        parallel=False,
        show_progress=False,
        timesteps=[0.0, 0.0, 0.0],
    )
    e_features, rho0, tgt = ds.tensors

    # Split deterministically: 45 train, 15 test.
    train = TensorDataset(e_features[:45], rho0[:45], tgt[:45])
    e_test, rho0_test, tgt_test = e_features[45:], rho0[45:], tgt[45:]

    model = ProcessTensorSurrogate(
        d_e=int(e_features.shape[-1]),
        d_rho=8,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_ff=64,
        dropout=0.0,
    )
    model.fit(train, epochs=120, batch_size=16, lr=2e-3, prefix_loss="full", device=torch.device("cpu"))

    pred = np.asarray(model.predict(e_test.numpy(), rho0_test.numpy(), return_numpy=True))
    assert pred.shape == tgt_test.numpy().shape

    # Accuracy on many samples and both time steps (flatten across steps).
    pred_flat = pred.reshape(-1, 8)
    tgt_flat = tgt_test.numpy().reshape(-1, 8)
    mse = mean_frobenius_mse_rho8(pred_flat, tgt_flat)
    td = mean_trace_distance_rho8(pred_flat, tgt_flat)

    # Stricter absolute thresholds: ensure the surrogate is actually predictive on held-out data.
    assert mse < 0.05
    assert td < 0.25

    # Also sanity check at matrix level for the first test sample.
    rho_pred = unpack_rho8(pred[0, -1, :])
    rho_true = unpack_rho8(tgt_test.numpy()[0, -1, :])
    assert compute_trace_distance(rho_pred, rho_true) < 0.5


def test_build_training_dataset_requires_torch_before_simulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_training_dataset fails fast when torch is unavailable."""
    import mqt.yaqs.characterization.memory.backends.surrogates.workflow as wf  # ruff:ignore[import-outside-top-level]

    def _raise_import() -> None:
        msg = "no torch"
        raise ImportError(msg)

    monkeypatch.setattr(wf, "_require_torch", _raise_import)
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    with pytest.raises(ImportError):
        build_training_dataset(op, params, num_interventions=1, n=1, parallel=False, show_progress=False)


@pytest.mark.parametrize("n", [0, -2])
def test_build_training_dataset_rejects_non_positive_n(n: int) -> None:
    """build_training_dataset rejects non-positive batch sizes before simulation."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    with pytest.raises(ValueError, match=r"n must be >= 1"):
        build_training_dataset(op, params, num_interventions=1, n=n, parallel=False, show_progress=False)


@pytest.mark.parametrize("n", [False, 1.5, "1"])
def test_build_training_dataset_rejects_non_integer_n(n: object) -> None:
    """build_training_dataset rejects booleans and non-integral batch sizes before simulation."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    with pytest.raises(TypeError, match=r"n must be an integer"):
        build_training_dataset(
            op,
            params,
            num_interventions=1,
            n=n,  # ty: ignore[invalid-argument-type]
            parallel=False,
            show_progress=False,
        )


@pytest.mark.parametrize(
    ("num_interventions", "error", "match"),
    [
        (0, ValueError, r"num_interventions must be >= 1"),
        (-1, ValueError, r"num_interventions must be >= 1"),
        (False, TypeError, r"num_interventions must be an integer"),
        (1.5, TypeError, r"num_interventions must be an integer"),
        ("1", TypeError, r"num_interventions must be an integer"),
    ],
)
def test_build_training_dataset_rejects_invalid_intervention_counts_before_torch(
    monkeypatch: pytest.MonkeyPatch,
    num_interventions: object,
    error: type[Exception],
    match: str,
) -> None:
    """Dataset count validation runs before the optional PyTorch check."""
    monkeypatch.setattr(workflow_module, "_require_torch", pytest.fail)

    with pytest.raises(error, match=match):
        build_training_dataset(
            MPO.ising(length=1, J=0.0, g=0.0),
            AnalogSimParams(dt=0.1),
            num_interventions=num_interventions,  # ty: ignore[invalid-argument-type]
            n=1,
            parallel=False,
            show_progress=False,
        )


@pytest.mark.parametrize(
    ("num_interventions", "n", "error", "match"),
    [
        (0, 1, ValueError, r"num_interventions must be >= 1"),
        (-1, 1, ValueError, r"num_interventions must be >= 1"),
        (1, False, TypeError, r"n must be an integer"),
        (1, "1", TypeError, r"n must be an integer"),
    ],
)
def test_train_surrogate_model_validates_counts_before_optional_imports(
    monkeypatch: pytest.MonkeyPatch,
    num_interventions: object,
    n: object,
    error: type[Exception],
    match: str,
) -> None:
    """Training rejects invalid counts before importing the optional model stack."""
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(error, match=match):
        train_surrogate_model(
            MPO.ising(length=1, J=0.0, g=0.0),
            AnalogSimParams(dt=0.1),
            num_interventions=num_interventions,  # ty: ignore[invalid-argument-type]
            n=n,  # ty: ignore[invalid-argument-type]
            parallel=False,
            show_progress=False,
        )


def test_surrogate_workflows_accept_numpy_integer_counts_before_optional_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NumPy integer counts pass lower-level validation before optional imports."""
    operator = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)

    def _stop_torch_check() -> None:
        msg = "validated dataset counts"
        raise RuntimeError(msg)

    monkeypatch.setattr(workflow_module, "_require_torch", _stop_torch_check)
    with pytest.raises(RuntimeError, match="validated dataset counts"):
        build_training_dataset(
            operator,
            params,
            num_interventions=np.int64(1),  # ty: ignore[invalid-argument-type]
            n=np.int64(1),  # ty: ignore[invalid-argument-type]
            parallel=False,
            show_progress=False,
        )

    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ModuleNotFoundError):
        train_surrogate_model(
            operator,
            params,
            num_interventions=np.int64(1),  # ty: ignore[invalid-argument-type]
            n=np.int64(1),  # ty: ignore[invalid-argument-type]
            parallel=False,
            show_progress=False,
        )
