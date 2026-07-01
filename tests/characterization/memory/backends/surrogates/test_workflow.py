# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for surrogate data generation and training workflows."""

from __future__ import annotations

import numpy as np
import pytest

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
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_extract_ket_fallback_for_zero_projector() -> None:
    """Zero projectors fall back to the |0> state vector."""
    psi = extract_ket(np.zeros((2, 2), dtype=np.complex128))
    np.testing.assert_allclose(psi, np.array([1.0 + 0.0j, 0.0 + 0.0j]))


def test_pack_dataset_shapes() -> None:
    """Rollout arrays convert to a three-tensor TensorDataset."""
    torch = pytest.importorskip("torch")

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
    """End-to-end build_training_dataset and train_surrogate_model run on a tiny Ising chain."""
    torch = pytest.importorskip("torch")

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
        model_kwargs={"d_model": 32, "nhead": 4, "num_layers": 1, "dim_ff": 64, "dropout": 0.0},
        train_kwargs={"epochs": 1, "batch_size": 2, "lr": 1e-3, "device": "cpu"},
    )
    e_features, rho0, tgt = ds.tensors
    dev = next(model.parameters()).device
    out = model(e_features.to(device=dev, dtype=torch.float32), rho0.to(device=dev, dtype=torch.float32))
    assert tuple(out.shape) == tuple(tgt.shape)


def test_build_training_dataset_timesteps_length_mismatch_raises() -> None:
    """build_training_dataset enforces process-tensor schedule length num_interventions+1."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)

    with pytest.raises(ValueError, match="Process-tensor schedule: timesteps length must be num_interventions\\+1"):
        build_training_dataset(op, params, num_interventions=2, n=1, timesteps=[0.1])


def test_surrogate_end_to_end_accuracy_regression_tiny() -> None:
    """Trained surrogate achieves modest error on held-out rollout samples."""
    torch = pytest.importorskip("torch")

    from torch.utils.data import TensorDataset  # noqa: PLC0415

    from mqt.yaqs.characterization.memory.backends.surrogates.model import ProcessTensorSurrogate  # noqa: PLC0415

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
    import mqt.yaqs.characterization.memory.backends.surrogates.workflow as wf  # noqa: PLC0415

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
    with pytest.raises(ValueError, match=r"n must be positive"):
        build_training_dataset(op, params, num_interventions=1, n=n, parallel=False, show_progress=False)
