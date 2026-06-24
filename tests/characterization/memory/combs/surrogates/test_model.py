# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: N806, PLC2701, SLF001 -- surrogate tests use E tensors and private helpers

"""Tests for the TransformerComb surrogate model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.combs.surrogates.utils import (
    _choi_features_from_parts,
    _sample_random_intervention,
    _sample_random_intervention_parts,
)


def test_transformercomb_forward_shape_cpu() -> None:
    """Forward pass returns one rho8 vector per sequence step."""
    torch = pytest.importorskip("torch")

    from mqt.yaqs import TransformerComb  # noqa: PLC0415

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E = torch.zeros((2, 3, 32), dtype=torch.float32)
    rho0 = torch.zeros((2, 8), dtype=torch.float32)
    out = model(E, rho0)
    assert tuple(out.shape) == (2, 3, 8)


def test_transformercomb_predict_numpy_roundtrip() -> None:
    """Predict with return_numpy=True yields a float32 ndarray."""
    pytest.importorskip("torch")

    from mqt.yaqs import TransformerComb  # noqa: PLC0415

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E = np.zeros((1, 2, 32), dtype=np.float32)
    rho0 = np.zeros((1, 8), dtype=np.float32)
    y = model.predict(E, rho0, device="cpu", return_numpy=True)
    assert isinstance(y, np.ndarray)
    assert y.shape == (1, 2, 8)


def test_transformercomb_predict_tensor_return_and_restores_mode() -> None:
    """Predict can return torch tensors and preserves train/eval mode."""
    torch = pytest.importorskip("torch")

    from torch.utils.data import TensorDataset  # noqa: PLC0415

    from mqt.yaqs import TransformerComb  # noqa: PLC0415

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
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
    model.fit(
        train_ds,
        val_dataset=val_ds,
        epochs=1,
        batch_size=2,
        prefix_loss="random",
        device=torch.device("cpu"),
    )


def test_transformercomb_fit_invalid_prefix_loss_raises() -> None:
    """Fit rejects unknown prefix_loss modes."""
    torch = pytest.importorskip("torch")

    from torch.utils.data import TensorDataset  # noqa: PLC0415

    from mqt.yaqs import TransformerComb  # noqa: PLC0415

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E_t = torch.zeros((2, 2, 32), dtype=torch.float32)
    rho0_t = torch.zeros((2, 8), dtype=torch.float32)
    tgt_t = torch.zeros((2, 2, 8), dtype=torch.float32)
    ds = TensorDataset(E_t, rho0_t, tgt_t)
    with pytest.raises(ValueError, match="Unknown prefix_loss"):
        model.fit(ds, epochs=1, prefix_loss="nope")  # type: ignore[arg-type]


def test_transformercomb_predict_final_state_batch_matches_forward_last_step() -> None:
    """predict_final_state_batch agrees with the last forward-pass output."""
    torch = pytest.importorskip("torch")

    from mqt.yaqs import TransformerComb  # noqa: PLC0415

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E = torch.randn((5, 4, 32), dtype=torch.float32)
    rho0 = torch.randn((8,), dtype=torch.float32)
    last = model(E, rho0.unsqueeze(0).expand(5, -1))[:, -1, :]
    batched = model.predict_final_state_batch(rho0, E)
    assert torch.allclose(batched, last, atol=1e-6, rtol=1e-6)


def test_transformercomb_entropy_shapes_and_batched_futures() -> None:
    """Entropy returns a finite scalar for a configured sequence length."""
    pytest.importorskip("torch")

    from mqt.yaqs import TransformerComb  # noqa: PLC0415

    d_e = 32
    cut = 2
    k_total = 5
    model = TransformerComb(
        d_e=d_e,
        d_rho=8,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_ff=64,
        dropout=0.0,
        sequence_length=k_total,
    )
    ent = model.entropy(cut)
    assert isinstance(ent, float)
    assert math.isfinite(ent)


def test_transformercomb_entropy_restores_training_mode() -> None:
    """Entropy leaves the model in its prior train/eval mode."""
    pytest.importorskip("torch")

    from mqt.yaqs import TransformerComb  # noqa: PLC0415

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
    model.train()
    model.entropy(1)
    assert model.training is True


def test_transformercomb_default_rho0_is_ground_state_rho8() -> None:
    """Default initial state matches the normalized |0> density matrix."""
    torch = pytest.importorskip("torch")

    from mqt.yaqs import TransformerComb  # noqa: PLC0415
    from mqt.yaqs.characterization.memory.combs.core.encoding import (  # noqa: PLC0415
        normalize_rho_from_backend_output,
        pack_rho8,
    )

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    rho0 = model._default_rho0(device=torch.device("cpu"), dtype=torch.float32)
    rho_ground = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    expected = pack_rho8(normalize_rho_from_backend_output(rho_ground)).astype(np.float32)
    np.testing.assert_array_almost_equal(rho0.cpu().numpy(), expected)


def test_intervention_parts_reassemble_to_same_choi_features() -> None:
    """Measurement/preparation parts must reassemble into the standard fused Choi feature row."""
    rng = np.random.default_rng(0)

    _emap, rho_prep, effect, _ = _sample_random_intervention(rng)
    feat_from_choi = _choi_features_from_parts(rho_prep, effect)
    # Also cover the direct parts sampler path.
    rho2, eff2, feat2 = _sample_random_intervention_parts(rng)
    feat_from_parts = _choi_features_from_parts(rho2, eff2)

    assert feat_from_choi.shape == (32,)
    assert feat_from_parts.shape == (32,)
    np.testing.assert_allclose(feat_from_parts, feat2, atol=0.0)


def test_transformercomb_entropy_rejects_out_of_range_cut() -> None:
    """Entropy validates the cut index against sequence_length."""
    pytest.importorskip("torch")

    from mqt.yaqs import TransformerComb  # noqa: PLC0415

    model = TransformerComb(
        d_e=32,
        d_rho=8,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_ff=64,
        dropout=0.0,
        sequence_length=4,
    )
    with pytest.raises(ValueError, match="cut must satisfy"):
        model.entropy(0)
    with pytest.raises(ValueError, match="cut must satisfy"):
        model.entropy(5)
    with pytest.raises(ValueError, match="cut must satisfy"):
        TransformerComb(
            d_e=32,
            d_rho=8,
            d_model=32,
            nhead=4,
            num_layers=1,
            dim_ff=64,
            dropout=0.0,
            sequence_length=1,
        ).entropy(2)


def test_transformercomb_entropy_requires_sequence_length() -> None:
    """Entropy requires sequence_length to be configured on the model."""
    pytest.importorskip("torch")

    from mqt.yaqs import TransformerComb  # noqa: PLC0415

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    with pytest.raises(ValueError, match="sequence_length"):
        model.entropy(1)


def test_transformercomb_entropy_sets_sequence_length_from_fit() -> None:
    """Fit infers sequence_length from training data for subsequent entropy calls."""
    torch = pytest.importorskip("torch")

    from torch.utils.data import TensorDataset  # noqa: PLC0415

    from mqt.yaqs import TransformerComb  # noqa: PLC0415

    k = 4
    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E_t = torch.zeros((2, k, 32), dtype=torch.float32)
    rho0_t = torch.zeros((2, 8), dtype=torch.float32)
    tgt_t = torch.zeros((2, k, 8), dtype=torch.float32)
    model.fit(TensorDataset(E_t, rho0_t, tgt_t), epochs=1, batch_size=2, device=torch.device("cpu"))
    assert model.sequence_length == k
    ent = model.entropy(2)
    assert isinstance(ent, float)
    assert math.isfinite(ent)
