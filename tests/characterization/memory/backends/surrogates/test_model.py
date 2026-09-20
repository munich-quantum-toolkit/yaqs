# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff:file-ignore[non-lowercase-variable-in-function, import-private-name, private-member-access] -- surrogate tests use E tensors and private helpers

"""Tests for the ProcessTensorSurrogate surrogate model."""

from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pytest
from torch_support import import_torch

from mqt.yaqs.characterization.memory.backends.surrogates.model import ProcessTensorSurrogate
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet
from mqt.yaqs.characterization.memory.shared.interventions import (
    _sample_random_intervention,
    encode_choi_features,
    sample_intervention_parts,
)

with contextlib.suppress(ImportError):
    from torch.utils.data import TensorDataset

    from mqt.yaqs.characterization.memory.backends.surrogates.model import (
        ProcessTensorSurrogate,
    )
    from mqt.yaqs.characterization.memory.shared.encoding import (
        pack_rho8,
    )


def _tiny_model(*, layernorm_in: bool = False, num_interventions: int | None = None) -> ProcessTensorSurrogate:
    """Build a small CPU ProcessTensorSurrogate for unit tests.

    Returns:
        A tiny untrained :class:`ProcessTensorSurrogate` on CPU.
    """
    import_torch()

    return ProcessTensorSurrogate(
        d_e=32,
        d_rho=8,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_ff=64,
        dropout=0.0,
        layernorm_in=layernorm_in,
        num_interventions=num_interventions,
    )


def _make_probe_set(*, cut: int = 1, num_interventions: int = 1, n_p: int = 2, n_f: int = 3) -> ProbeSet:
    """Minimal ProbeSet compatible with ProcessTensorSurrogate.evaluate_probes.

    Returns:
        A probe set with zero feature rows and |0> cut kets.
    """
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    identity_step = {"type": "unitary", "U": np.eye(2, dtype=np.complex128)}
    return ProbeSet(
        cut=cut,
        num_interventions=num_interventions,
        past_features=np.zeros((n_p, cut, 32), dtype=np.float32),
        future_features=np.zeros((n_f, num_interventions - cut + 1, 32), dtype=np.float32),
        past_pairs=[[identity_step.copy() for _ in range(cut - 1)] for _ in range(n_p)],
        past_cut_meas=[z.copy() for _ in range(n_p)],
        future_prep_cut=[z.copy() for _ in range(n_f)],
        future_pairs=[[identity_step.copy() for _ in range(num_interventions - cut)] for _ in range(n_f)],
    )


def test_process_tensor_surrogate_forward_shape_cpu() -> None:
    """Forward pass returns one rho8 vector per sequence step."""
    torch = import_torch()

    model = ProcessTensorSurrogate(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E = torch.zeros((2, 3, 32), dtype=torch.float32)
    rho0 = torch.zeros((2, 8), dtype=torch.float32)
    out = model(E, rho0)
    assert tuple(out.shape) == (2, 3, 8)


def test_process_tensor_surrogate_predict_numpy_roundtrip() -> None:
    """Predict with return_numpy=True yields a float32 ndarray."""
    import_torch()

    model = ProcessTensorSurrogate(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E = np.zeros((1, 2, 32), dtype=np.float32)
    rho0 = np.zeros((1, 8), dtype=np.float32)
    y = model.predict(E, rho0, device="cpu", return_numpy=True)
    assert isinstance(y, np.ndarray)
    assert y.shape == (1, 2, 8)


def test_process_tensor_surrogate_predict_tensor_return_and_restores_mode() -> None:
    """Predict can return torch tensors and preserves train/eval mode."""
    torch = import_torch()

    model = ProcessTensorSurrogate(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    model.train()

    E = np.zeros((1, 2, 32), dtype=np.float32)
    rho0 = np.zeros((1, 8), dtype=np.float32)
    out_t = model.predict(E, rho0, device="cpu", return_numpy=False)
    assert isinstance(out_t, torch.Tensor)
    assert tuple(out_t.shape) == (1, 2, 8)
    assert model.training is True

    # Tiny fit run to cover training loop + val checkpoint path
    E_t = torch.zeros((4, 2, 32), dtype=torch.float32)
    rho0_t = torch.zeros((4, 8), dtype=torch.float32)
    tgt_t = torch.zeros((4, 2, 8), dtype=torch.float32)
    train_ds = TensorDataset(E_t, rho0_t, tgt_t)
    val_ds = TensorDataset(E_t[:2], rho0_t[:2], tgt_t[:2])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="CUDA initialization", category=UserWarning)
        model.fit(
            train_ds,
            val_dataset=val_ds,
            epochs=1,
            batch_size=2,
            prefix_loss="random",
            device=torch.device("cpu"),
        )


def test_process_tensor_surrogate_fit_invalid_prefix_loss_raises() -> None:
    """Fit rejects unknown prefix_loss modes."""
    torch = import_torch()

    model = ProcessTensorSurrogate(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E_t = torch.zeros((2, 2, 32), dtype=torch.float32)
    rho0_t = torch.zeros((2, 8), dtype=torch.float32)
    tgt_t = torch.zeros((2, 2, 8), dtype=torch.float32)
    ds = TensorDataset(E_t, rho0_t, tgt_t)
    with pytest.raises(ValueError, match="Unknown prefix_loss"):
        model.fit(ds, epochs=1, prefix_loss="nope")  # type: ignore[arg-type]


def test_process_tensor_surrogate_predict_final_state_batch_matches_forward_last_step() -> None:
    """predict_final_state_batch agrees with the last forward-pass output."""
    torch = import_torch()

    model = ProcessTensorSurrogate(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E = torch.randn((5, 4, 32), dtype=torch.float32)
    rho0 = torch.randn((8,), dtype=torch.float32)
    last = model(E, rho0.unsqueeze(0).expand(5, -1))[:, -1, :]
    batched = model.predict_final_state_batch(rho0, E)
    assert torch.allclose(batched, last, atol=1e-6, rtol=1e-6)


def test_process_tensor_surrogate_fit_sets_num_interventions() -> None:
    """Fit infers num_interventions from training data."""
    torch = import_torch()

    k = 4
    model = ProcessTensorSurrogate(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E_t = torch.zeros((2, k, 32), dtype=torch.float32)
    rho0_t = torch.zeros((2, 8), dtype=torch.float32)
    tgt_t = torch.zeros((2, k, 8), dtype=torch.float32)
    model.fit(TensorDataset(E_t, rho0_t, tgt_t), epochs=1, batch_size=2, device=torch.device("cpu"))
    assert model.num_interventions == k


def test_intervention_parts_reassemble_to_same_choi_features() -> None:
    """Measurement/preparation parts must reassemble into the standard fused Choi feature row."""
    rng = np.random.default_rng(0)

    _emap, rho_prep, effect, _ = _sample_random_intervention(rng)
    feat_from_choi = encode_choi_features(rho_prep, effect)
    rho2, eff2, feat2 = sample_intervention_parts(rng)
    feat_from_parts = encode_choi_features(rho2, eff2)

    assert feat_from_choi.shape == (32,)
    assert feat_from_parts.shape == (32,)
    np.testing.assert_allclose(feat_from_parts, feat2, atol=0.0)


def test_process_tensor_surrogate_init_rejects_incompatible_head_width() -> None:
    """d_model must be divisible by nhead."""
    import_torch()

    with pytest.raises(ValueError, match="d_model=33 must be divisible by nhead=4"):
        ProcessTensorSurrogate(d_e=32, d_rho=8, d_model=33, nhead=4)


def test_process_tensor_surrogate_rejects_non_positive_nhead() -> None:
    """nhead=0 raises ValueError instead of ZeroDivisionError."""
    import_torch()

    with pytest.raises(ValueError, match="nhead must be positive"):
        ProcessTensorSurrogate(d_e=32, d_rho=8, d_model=32, nhead=0)


def test_process_tensor_surrogate_d_e_property_matches_input_projection() -> None:
    """d_e reports the intervention feature width excluding the rho side channel."""
    model = _tiny_model()
    assert model.d_e == 32


def test_process_tensor_surrogate_layernorm_in_forward() -> None:
    """layernorm_in=True applies input LayerNorm before the encoder."""
    torch = import_torch()

    model = _tiny_model(layernorm_in=True)
    e_features = torch.zeros((1, 2, 32), dtype=torch.float32)
    rho0 = torch.zeros((1, 8), dtype=torch.float32)
    out = model(e_features, rho0)
    assert tuple(out.shape) == (1, 2, 8)


def test_process_tensor_surrogate_forward_rejects_bad_rho0_shape() -> None:
    """Forward validates rho0 batch shape."""
    torch = import_torch()

    model = _tiny_model()
    e_features = torch.zeros((2, 3, 32), dtype=torch.float32)
    rho0 = torch.zeros((3, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="rho0 mode expects rho0"):
        model(e_features, rho0)


def test_process_tensor_surrogate_predict_final_state_batch_validation_errors() -> None:
    """predict_final_state_batch validates feature tensor ranks and rho0 shapes."""
    torch = import_torch()

    model = _tiny_model()
    e_features = torch.zeros((2, 3, 32), dtype=torch.float32)
    with pytest.raises(ValueError, match="e_features must be"):
        model.predict_final_state_batch(torch.zeros(8), e_features[:, 0, :])
    with pytest.raises(ValueError, match="rho0 \\(d_rho,\\) expected length"):
        model.predict_final_state_batch(torch.zeros(7), e_features)
    with pytest.raises(ValueError, match="rho0 must be"):
        model.predict_final_state_batch(torch.zeros((3, 8)), e_features)


def test_process_tensor_surrogate_num_interventions_for_probe_requires_num_interventions() -> None:
    """_num_interventions_for_probe raises when num_interventions was never set."""
    model = _tiny_model()
    with pytest.raises(ValueError, match="num_interventions is unset"):
        model._num_interventions_for_probe()


@pytest.mark.parametrize(
    ("num_interventions", "error", "match"),
    [
        (0, ValueError, r"num_interventions must be >= 1"),
        (False, TypeError, r"num_interventions must be an integer"),
        (1.5, TypeError, r"num_interventions must be an integer"),
    ],
)
def test_process_tensor_surrogate_rejects_invalid_num_interventions(
    num_interventions: object,
    error: type[Exception],
    match: str,
) -> None:
    """The surrogate constructor does not coerce invalid sequence lengths."""
    with pytest.raises(error, match=match):
        _tiny_model(num_interventions=num_interventions)  # ty: ignore[invalid-argument-type]


def test_process_tensor_surrogate_accepts_numpy_num_interventions() -> None:
    """The surrogate constructor accepts NumPy integer sequence lengths."""
    model = _tiny_model(num_interventions=np.int64(2))  # ty: ignore[invalid-argument-type]
    assert model.num_interventions == 2


def test_process_tensor_surrogate_rho_to_features_casts_to_float64() -> None:
    """_rho_to_features preserves shape and promotes to float64."""
    torch = import_torch()

    model = _tiny_model()
    rho = torch.zeros((2, 8), dtype=torch.float32)
    feats = model._rho_to_features(rho)
    assert feats.dtype == torch.float64
    assert tuple(feats.shape) == (2, 8)
    with pytest.raises(ValueError, match="Expected last dim d_rho"):
        model._rho_to_features(torch.zeros((2, 7)))


def test_sinusoidal_positional_encoding_rejects_nonpositive_width() -> None:
    """Positional encoding requires a positive model width."""
    torch = import_torch()

    from mqt.yaqs.characterization.memory.backends.surrogates.model import (  # ruff:ignore[import-outside-top-level]
        _sinusoidal_positional_encoding,
    )

    with pytest.raises(ValueError, match="d_model must be positive"):
        _sinusoidal_positional_encoding(3, 0, device=torch.device("cpu"), dtype=torch.float32)


def test_process_tensor_surrogate_fit_prefix_loss_modes() -> None:
    """Fit supports full and all prefix-loss horizons."""
    torch = import_torch()

    from torch.utils.data import TensorDataset  # ruff:ignore[import-outside-top-level]

    model = _tiny_model()
    e_t = torch.zeros((3, 3, 32), dtype=torch.float32)
    rho0_t = torch.zeros((3, 8), dtype=torch.float32)
    tgt_t = torch.zeros((3, 3, 8), dtype=torch.float32)
    ds = TensorDataset(e_t, rho0_t, tgt_t)

    model.fit(ds, epochs=1, batch_size=2, prefix_loss="full", grad_clip=0.0, device=torch.device("cpu"))
    model.fit(ds, epochs=1, batch_size=2, prefix_loss="all", device=torch.device("cpu"))


def test_process_tensor_surrogate_evaluate_probes_shape_and_restores_mode() -> None:
    """evaluate_probes returns Pauli tomography rows and restores train/eval mode."""
    import_torch()

    model = _tiny_model(num_interventions=1)
    model.train()
    probe_set = _make_probe_set(cut=1, num_interventions=1, n_p=2, n_f=3)
    out = model.evaluate_probes(probe_set, initial_rho=np.eye(2, dtype=np.complex128) / 2.0)
    assert out.shape == (2, 3, 4)
    assert out.dtype == np.float32
    assert model.training is True


def test_process_tensor_surrogate_evaluate_probes_requires_boundary_state() -> None:
    """Surrogate probing cannot infer the state after the initial evolution segment."""
    import_torch()

    model = _tiny_model(num_interventions=1)
    probe_set = _make_probe_set(cut=1, num_interventions=1, n_p=1, n_f=1)
    with pytest.raises(ValueError, match="initial_rho is required for surrogate characterization"):
        model.evaluate_probes(probe_set)


def test_process_tensor_surrogate_evaluate_probes_requires_rho8_output() -> None:
    """Probe evaluation rejects a model whose output is not a rho8 encoding."""
    import_torch()

    model = ProcessTensorSurrogate(
        d_e=32,
        d_rho=7,
        d_model=28,
        nhead=4,
        num_layers=1,
        dim_ff=28,
        num_interventions=1,
    )
    probe_set = _make_probe_set(cut=1, num_interventions=1, n_p=1, n_f=1)

    with pytest.raises(ValueError, match="rho8 packing length 8 does not match d_rho=7"):
        model.evaluate_probes(probe_set, initial_rho=np.eye(2, dtype=np.complex128) / 2.0)


def test_process_tensor_surrogate_estimates_selected_future_probability(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surrogate complete weights use its predicted state before each retained outcome."""
    torch = import_torch()
    model = _tiny_model(num_interventions=2)
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    rho_z = np.outer(z, z.conj())
    rho_plus = np.outer(plus, plus.conj())
    received_rho0: list[np.ndarray] = []

    def _fake_forward(e_features: object, rho0: object) -> object:
        _ = e_features
        received_rho0.append(np.asarray(rho0, dtype=np.float32))
        packed_z = pack_rho8(rho_z)
        packed_plus = pack_rho8(rho_plus)
        return torch.as_tensor(
            np.array([[packed_z, packed_z], [packed_plus, packed_z]], dtype=np.float32),
            dtype=torch.float32,
        )

    monkeypatch.setattr(model, "forward", _fake_forward)
    future_features = np.empty((2, 2, 32), dtype=np.float32)
    future_features[0, 0] = encode_choi_features(rho_z, np.eye(2, dtype=np.complex128))
    future_features[1, 0] = encode_choi_features(rho_plus, np.eye(2, dtype=np.complex128))
    future_features[:, 1] = encode_choi_features(rho_z, rho_z)
    probe_set = ProbeSet(
        cut=1,
        num_interventions=2,
        past_features=np.zeros((1, 1, 32), dtype=np.float32),
        future_features=future_features,
        past_pairs=[[]],
        past_cut_meas=[z.copy()],
        future_prep_cut=[z.copy(), plus.copy()],
        future_pairs=[[(z, z)], [(z, z)]],
    )
    _pauli, weights = model.evaluate_probes_with_weights(probe_set, initial_rho=rho_plus)
    np.testing.assert_allclose(weights, np.array([[0.5, 0.25]], dtype=np.float64), atol=1e-7)
    np.testing.assert_allclose(received_rho0[0], np.broadcast_to(pack_rho8(rho_plus), (2, 8)))


def test_process_tensor_surrogate_evaluate_probes_with_past_and_future_segments() -> None:
    """evaluate_probes stitches non-empty past and future feature segments."""
    import_torch()

    model = _tiny_model(num_interventions=3)
    probe_set = _make_probe_set(cut=2, num_interventions=3, n_p=1, n_f=2)
    out = model.evaluate_probes(probe_set, initial_rho=np.eye(2, dtype=np.complex128) / 2.0)
    assert out.shape == (1, 2, 4)


def test_process_tensor_surrogate_evaluate_probes_rejects_k_mismatch() -> None:
    """evaluate_probes rejects ProbeSet num_interventions values that differ from training horizon."""
    import_torch()

    model = _tiny_model(num_interventions=2)
    probe_set = _make_probe_set(cut=1, num_interventions=3, n_p=1, n_f=1)
    with pytest.raises(ValueError, match="ProbeSet num_interventions=3 does not match model num_interventions=2"):
        model.evaluate_probes(probe_set)
