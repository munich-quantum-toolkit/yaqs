from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.process_tensors.core.encoding import unpack_rho8
from mqt.yaqs.characterization.process_tensors.core.metrics import (
    _mean_frobenius_mse_rho8,
    _mean_trace_distance_rho8,
    _trace_distance,
)
from mqt.yaqs.characterization.process_tensors.core.utils import make_mcwf_static_context
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_psi_from_rank1_projector_fallback_for_zero_projector() -> None:
    from mqt.yaqs.characterization.process_tensors.surrogates.workflow import _psi_from_rank1_projector  # noqa: PLC0415

    psi = _psi_from_rank1_projector(np.zeros((2, 2), dtype=np.complex128))
    np.testing.assert_allclose(psi, np.array([1.0 + 0.0j, 0.0 + 0.0j]))


def test_simulate_sequences_input_validation_errors() -> None:
    from mqt.yaqs.characterization.process_tensors.surrogates.workflow import _simulate_sequences  # noqa: PLC0415

    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, solver="TJM", show_progress=False, max_bond_dim=8)

    with pytest.raises(ValueError):
        _simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.1],
            psi_pairs_list=[],
            initial_psis=[np.array([1.0, 0.0], dtype=np.complex128)],
            static_ctx=None,
            parallel=False,
        )

    with pytest.raises(ValueError):
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

    with pytest.raises(ValueError):
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
    torch = pytest.importorskip("torch")

    from mqt.yaqs.characterization.process_tensors.surrogates.workflow import (
        _rollout_arrays_to_tensor_dataset,
    )

    rho0 = np.zeros((2, 8), dtype=np.float32)
    E = np.zeros((2, 3, 32), dtype=np.float32)
    rho_seq = np.zeros((2, 3, 8), dtype=np.float32)
    ds = _rollout_arrays_to_tensor_dataset(rho0, E, rho_seq)
    assert len(ds.tensors) == 3
    assert tuple(ds.tensors[0].shape) == (2, 3, 32)
    assert tuple(ds.tensors[1].shape) == (2, 8)
    assert tuple(ds.tensors[2].shape) == (2, 3, 8)
    assert ds.tensors[0].dtype == torch.float32


def test_simulate_sequences_mcwf_final_states_and_rollouts_smoke() -> None:
    from mqt.yaqs.characterization.process_tensors.surrogates.workflow import _simulate_sequences  # noqa: PLC0415

    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)
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
    torch = pytest.importorskip("torch")

    from mqt.yaqs.characterization.process_tensors.surrogates.workflow import (  # noqa: PLC0415
        create_surrogate,
        generate_data,
    )

    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)

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
        train_kwargs={"epochs": 1, "batch_size": 2, "lr": 1e-3},
    )
    E, rho0, tgt = ds.tensors
    out = model(E.to(torch.float32), rho0.to(torch.float32))
    assert tuple(out.shape) == tuple(tgt.shape)


def test_generate_data_timesteps_length_mismatch_raises() -> None:
    pytest.importorskip("torch")

    from mqt.yaqs.characterization.process_tensors.surrogates.workflow import generate_data  # noqa: PLC0415

    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)

    with pytest.raises(ValueError):
        generate_data(op, params, k=2, n=1, timesteps=[0.1])


def test_surrogate_end_to_end_accuracy_regression_tiny() -> None:
    torch = pytest.importorskip("torch")

    from torch.utils.data import TensorDataset  # noqa: PLC0415

    from mqt.yaqs.characterization.process_tensors.surrogates.model import TransformerComb  # noqa: PLC0415
    from mqt.yaqs.characterization.process_tensors.surrogates.workflow import generate_data  # noqa: PLC0415

    torch.manual_seed(0)
    np.random.seed(0)

    # Two sites: system qubit + environment qubit, non-trivial Ising dynamics.
    op = MPO.ising(length=2, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, solver="MCWF", show_progress=False)

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
    E, rho0, tgt = ds.tensors

    # Split deterministically: 45 train, 15 test.
    train = TensorDataset(E[:45], rho0[:45], tgt[:45])
    test_E, test_rho0, test_tgt = E[45:], rho0[45:], tgt[45:]

    model = TransformerComb(
        d_e=int(E.shape[-1]),
        d_rho=8,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_ff=64,
        dropout=0.0,
    )
    model.fit(train, epochs=120, batch_size=16, lr=2e-3, prefix_loss="full", device=torch.device("cpu"))

    pred = model.predict(test_E.numpy(), test_rho0.numpy(), return_numpy=True)
    assert pred.shape == test_tgt.numpy().shape

    # Accuracy on many samples and both time steps (flatten across steps).
    pred_flat = pred.reshape(-1, 8)
    tgt_flat = test_tgt.numpy().reshape(-1, 8)
    mse = _mean_frobenius_mse_rho8(pred_flat, tgt_flat)
    td = _mean_trace_distance_rho8(pred_flat, tgt_flat)

    # Stricter absolute thresholds: ensure the surrogate is actually predictive on held-out data.
    assert mse < 0.05
    assert td < 0.25

    # Also sanity check at matrix level for the first test sample.
    rho_pred = unpack_rho8(pred[0, -1, :])
    rho_true = unpack_rho8(test_tgt.numpy()[0, -1, :])
    assert _trace_distance(rho_pred, rho_true) < 0.5
