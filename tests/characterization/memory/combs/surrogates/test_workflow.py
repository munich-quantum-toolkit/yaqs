# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: PLC2701 -- surrogate workflow tests use E tensors and private helpers

"""Tests for surrogate data generation and training workflows."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.combs.core.encoding import unpack_rho8
from mqt.yaqs.characterization.memory.combs.core.metrics import (
    _mean_frobenius_mse_rho8,
    _mean_trace_distance_rho8,
    _trace_distance,
)
from mqt.yaqs.characterization.memory.combs.core.utils import make_mcwf_static_context
from mqt.yaqs.characterization.memory.combs.surrogates.workflow import (
    _psi_from_rank1_projector,
    _rollout_arrays_to_tensor_dataset,
    _simulate_sequences,
    create_surrogate,
    generate_data,
)
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_psi_from_rank1_projector_fallback_for_zero_projector() -> None:
    """Zero projectors fall back to the |0> state vector."""
    psi = _psi_from_rank1_projector(np.zeros((2, 2), dtype=np.complex128))
    np.testing.assert_allclose(psi, np.array([1.0 + 0.0j, 0.0 + 0.0j]))


def test_simulate_sequences_input_validation_errors() -> None:
    """_simulate_sequences validates comb schedule and rollout feature inputs."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)

    with pytest.raises(ValueError, match="psi_pairs_list and initial_psis must have equal length"):
        _simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.1],
            psi_pairs_list=[],
            initial_psis=[np.array([1.0, 0.0], dtype=np.complex128)],
            static_ctx=None,
            parallel=False,
        )

    with pytest.raises(ValueError, match="record_step_states=True requires e_features_rows"):
        _simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.1, 0.1],
            psi_pairs_list=[[(np.array([1.0, 0.0]), np.array([1.0, 0.0]))]],
            initial_psis=[np.array([1.0, 0.0], dtype=np.complex128)],
            static_ctx=None,
            parallel=False,
            record_step_states=True,
            e_features_rows=None,
        )

    with pytest.raises(ValueError, match="e_features_rows is only used when record_step_states=True"):
        _simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.1, 0.1],
            psi_pairs_list=[[(np.array([1.0, 0.0]), np.array([1.0, 0.0]))]],
            initial_psis=[np.array([1.0, 0.0], dtype=np.complex128)],
            static_ctx=None,
            parallel=False,
            record_step_states=False,
            e_features_rows=[np.zeros((1, 32), dtype=np.float32)],
        )


def test_rollout_arrays_to_tensor_dataset_shapes() -> None:
    """Rollout arrays convert to a three-tensor TensorDataset."""
    torch = pytest.importorskip("torch")

    rho0 = np.zeros((2, 8), dtype=np.float32)
    e_features = np.zeros((2, 3, 32), dtype=np.float32)
    rho_seq = np.zeros((2, 3, 8), dtype=np.float32)
    ds = _rollout_arrays_to_tensor_dataset(rho0, e_features, rho_seq)
    assert len(ds.tensors) == 3
    assert tuple(ds.tensors[0].shape) == (2, 3, 32)
    assert tuple(ds.tensors[1].shape) == (2, 8)
    assert tuple(ds.tensors[2].shape) == (2, 3, 8)
    assert ds.tensors[0].dtype == torch.float32


def test_simulate_sequences_mcwf_final_states_and_rollouts_smoke() -> None:
    """MCWF simulation returns final packed states or rollout samples."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)

    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    psi_pairs_list = [[(psi0, psi0)]]
    initial_psis = [psi0.copy()]
    timesteps = [0.0, 0.0]

    finals = _simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=timesteps,
        psi_pairs_list=psi_pairs_list,
        initial_psis=initial_psis,
        static_ctx=static_ctx,
        parallel=False,
        show_progress=False,
        record_step_states=False,
    )
    assert isinstance(finals, np.ndarray)
    assert finals.shape == (1, 8)

    samples = _simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=timesteps,
        psi_pairs_list=psi_pairs_list,
        initial_psis=initial_psis,
        static_ctx=static_ctx,
        parallel=False,
        show_progress=False,
        record_step_states=True,
        e_features_rows=[np.zeros((1, 32), dtype=np.float32)],
    )
    assert isinstance(samples, list)
    assert len(samples) == 1
    s0 = samples[0]
    assert s0.rho_0.shape == (8,)
    assert s0.E_features.shape == (1, 32)
    assert s0.rho_seq.shape == (1, 8)


def test_generate_data_and_create_surrogate_tiny_smoke() -> None:
    """End-to-end generate_data and create_surrogate run on a tiny Ising chain."""
    torch = pytest.importorskip("torch")

    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)

    ds = generate_data(
        op,
        params,
        k=1,
        n=2,
        seed=0,
        parallel=False,
        show_progress=False,
        timesteps=[0.0, 0.0],
    )
    assert len(ds.tensors) == 3

    model = create_surrogate(
        op,
        params,
        k=1,
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


def test_generate_data_timesteps_length_mismatch_raises() -> None:
    """generate_data enforces comb schedule length k+1."""
    pytest.importorskip("torch")

    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)

    with pytest.raises(ValueError, match="Comb schedule: timesteps length must be k\\+1"):
        generate_data(op, params, k=2, n=1, timesteps=[0.1])


def test_surrogate_end_to_end_accuracy_regression_tiny() -> None:
    """Trained surrogate achieves modest error on held-out rollout samples."""
    torch = pytest.importorskip("torch")

    from torch.utils.data import TensorDataset  # noqa: PLC0415

    from mqt.yaqs.characterization.memory.combs.surrogates.model import TransformerComb  # noqa: PLC0415

    torch.manual_seed(0)

    # Two sites: system qubit + environment qubit, non-trivial Ising dynamics.
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1)

    # Small but non-trivial dataset: k=2 sequences, enough samples to evaluate generalization.
    k = 2
    n = 60
    ds = generate_data(
        op,
        params,
        k=k,
        n=n,
        seed=123,
        parallel=False,
        show_progress=False,
        timesteps=[0.0, 0.0, 0.0],
    )
    e_features, rho0, tgt = ds.tensors

    # Split deterministically: 45 train, 15 test.
    train = TensorDataset(e_features[:45], rho0[:45], tgt[:45])
    e_test, rho0_test, tgt_test = e_features[45:], rho0[45:], tgt[45:]

    model = TransformerComb(
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
    mse = _mean_frobenius_mse_rho8(pred_flat, tgt_flat)
    td = _mean_trace_distance_rho8(pred_flat, tgt_flat)

    # Stricter absolute thresholds: ensure the surrogate is actually predictive on held-out data.
    assert mse < 0.05
    assert td < 0.25

    # Also sanity check at matrix level for the first test sample.
    rho_pred = unpack_rho8(pred[0, -1, :])
    rho_true = unpack_rho8(tgt_test.numpy()[0, -1, :])
    assert _trace_distance(rho_pred, rho_true) < 0.5
