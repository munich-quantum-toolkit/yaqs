# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: N806, PLC2701, SLF001 -- surrogate tests use E tensors and private helpers

"""Tests for the TransformerComb surrogate model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.surrogates.utils import (
    _sample_random_intervention,
    encode_choi_features,
    sample_intervention_parts,
)
from mqt.yaqs.characterization.memory.operational_memory.samples import ProbeSet

if TYPE_CHECKING:
    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb


def _tiny_model(*, layernorm_in: bool = False, sequence_length: int | None = None) -> TransformerComb:
    """Build a small CPU TransformerComb for unit tests.

    Returns:
        A tiny untrained :class:`TransformerComb` on CPU.
    """
    pytest.importorskip("torch")

    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb  # noqa: PLC0415

    return TransformerComb(
        d_e=32,
        d_rho=8,
        d_model=32,
        nhead=4,
        num_layers=1,
        dim_ff=64,
        dropout=0.0,
        layernorm_in=layernorm_in,
        sequence_length=sequence_length,
    )


def _make_probe_set(*, cut: int = 1, k: int = 1, n_p: int = 2, n_f: int = 3) -> ProbeSet:
    """Minimal ProbeSet compatible with TransformerComb.evaluate_probes.

    Returns:
        A probe set with zero feature rows and |0> cut kets.
    """
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return ProbeSet(
        cut=cut,
        k=k,
        past_features=np.zeros((n_p, cut, 32), dtype=np.float32),
        future_features=np.zeros((n_f, k - cut + 1, 32), dtype=np.float32),
        past_pairs=[[] for _ in range(n_p)],
        past_cut_meas=[z.copy() for _ in range(n_p)],
        future_prep_cut=[z.copy() for _ in range(n_f)],
        future_pairs=[[] for _ in range(n_f)],
    )


def test_transformercomb_forward_shape_cpu() -> None:
    """Forward pass returns one rho8 vector per sequence step."""
    torch = pytest.importorskip("torch")

    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb  # noqa: PLC0415

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E = torch.zeros((2, 3, 32), dtype=torch.float32)
    rho0 = torch.zeros((2, 8), dtype=torch.float32)
    out = model(E, rho0)
    assert tuple(out.shape) == (2, 3, 8)


def test_transformercomb_predict_numpy_roundtrip() -> None:
    """Predict with return_numpy=True yields a float32 ndarray."""
    pytest.importorskip("torch")

    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb  # noqa: PLC0415

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

    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb  # noqa: PLC0415

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

    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb  # noqa: PLC0415

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

    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb  # noqa: PLC0415

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E = torch.randn((5, 4, 32), dtype=torch.float32)
    rho0 = torch.randn((8,), dtype=torch.float32)
    last = model(E, rho0.unsqueeze(0).expand(5, -1))[:, -1, :]
    batched = model.predict_final_state_batch(rho0, E)
    assert torch.allclose(batched, last, atol=1e-6, rtol=1e-6)


def test_transformercomb_fit_sets_sequence_length() -> None:
    """Fit infers sequence_length from training data."""
    torch = pytest.importorskip("torch")

    from torch.utils.data import TensorDataset  # noqa: PLC0415

    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb  # noqa: PLC0415

    k = 4
    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E_t = torch.zeros((2, k, 32), dtype=torch.float32)
    rho0_t = torch.zeros((2, 8), dtype=torch.float32)
    tgt_t = torch.zeros((2, k, 8), dtype=torch.float32)
    model.fit(TensorDataset(E_t, rho0_t, tgt_t), epochs=1, batch_size=2, device=torch.device("cpu"))
    assert model.sequence_length == k


def test_transformercomb_default_rho0_is_ground_state_rho8() -> None:
    """Default initial state matches the normalized |0> density matrix."""
    torch = pytest.importorskip("torch")

    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb  # noqa: PLC0415
    from mqt.yaqs.characterization.memory.shared.encoding import (  # noqa: PLC0415
        normalize_backend_rho,
        pack_rho8,
    )

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    rho0 = model._default_rho0(device=torch.device("cpu"), dtype=torch.float32)
    rho_ground = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    expected = pack_rho8(normalize_backend_rho(rho_ground)).astype(np.float32)
    np.testing.assert_array_almost_equal(rho0.cpu().numpy(), expected)


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


def test_transformercomb_init_rejects_incompatible_head_width() -> None:
    """d_model must be divisible by nhead."""
    pytest.importorskip("torch")

    from mqt.yaqs.characterization.memory.backends.surrogates.model import TransformerComb  # noqa: PLC0415

    with pytest.raises(ValueError, match="d_model=33 must be divisible by nhead=4"):
        TransformerComb(d_e=32, d_rho=8, d_model=33, nhead=4)


def test_transformercomb_d_e_property_matches_input_projection() -> None:
    """d_e reports the intervention feature width excluding the rho side channel."""
    model = _tiny_model()
    assert model.d_e == 32


def test_transformercomb_layernorm_in_forward() -> None:
    """layernorm_in=True applies input LayerNorm before the encoder."""
    torch = pytest.importorskip("torch")

    model = _tiny_model(layernorm_in=True)
    e_features = torch.zeros((1, 2, 32), dtype=torch.float32)
    rho0 = torch.zeros((1, 8), dtype=torch.float32)
    out = model(e_features, rho0)
    assert tuple(out.shape) == (1, 2, 8)


def test_transformercomb_forward_rejects_bad_rho0_shape() -> None:
    """Forward validates rho0 batch shape."""
    torch = pytest.importorskip("torch")

    model = _tiny_model()
    e_features = torch.zeros((2, 3, 32), dtype=torch.float32)
    rho0 = torch.zeros((3, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="rho0 mode expects rho0"):
        model(e_features, rho0)


def test_transformercomb_predict_final_state_batch_validation_errors() -> None:
    """predict_final_state_batch validates feature tensor ranks and rho0 shapes."""
    torch = pytest.importorskip("torch")

    model = _tiny_model()
    e_features = torch.zeros((2, 3, 32), dtype=torch.float32)
    with pytest.raises(ValueError, match="e_features must be"):
        model.predict_final_state_batch(torch.zeros(8), e_features[:, 0, :])
    with pytest.raises(ValueError, match="rho0 \\(d_rho,\\) expected length"):
        model.predict_final_state_batch(torch.zeros(7), e_features)
    with pytest.raises(ValueError, match="rho0 must be"):
        model.predict_final_state_batch(torch.zeros((3, 8)), e_features)


def test_transformercomb_k_for_probe_requires_sequence_length() -> None:
    """_k_for_probe raises when sequence_length was never set."""
    model = _tiny_model()
    with pytest.raises(ValueError, match="sequence_length is unset"):
        model._k_for_probe()


def test_transformercomb_rho_to_features_casts_to_float64() -> None:
    """_rho_to_features preserves shape and promotes to float64."""
    torch = pytest.importorskip("torch")

    model = _tiny_model()
    rho = torch.zeros((2, 8), dtype=torch.float32)
    feats = model._rho_to_features(rho)
    assert feats.dtype == torch.float64
    assert tuple(feats.shape) == (2, 8)
    with pytest.raises(ValueError, match="Expected last dim d_rho"):
        model._rho_to_features(torch.zeros((2, 7)))


def test_sinusoidal_positional_encoding_rejects_nonpositive_width() -> None:
    """Positional encoding requires a positive model width."""
    torch = pytest.importorskip("torch")

    from mqt.yaqs.characterization.memory.backends.surrogates.model import (  # noqa: PLC0415
        _sinusoidal_positional_encoding,
    )

    with pytest.raises(ValueError, match="d_model must be positive"):
        _sinusoidal_positional_encoding(3, 0, device=torch.device("cpu"), dtype=torch.float32)


def test_transformercomb_fit_prefix_loss_modes() -> None:
    """Fit supports full and all prefix-loss horizons."""
    torch = pytest.importorskip("torch")

    from torch.utils.data import TensorDataset  # noqa: PLC0415

    model = _tiny_model()
    e_t = torch.zeros((3, 3, 32), dtype=torch.float32)
    rho0_t = torch.zeros((3, 8), dtype=torch.float32)
    tgt_t = torch.zeros((3, 3, 8), dtype=torch.float32)
    ds = TensorDataset(e_t, rho0_t, tgt_t)

    model.fit(ds, epochs=1, batch_size=2, prefix_loss="full", grad_clip=0.0, device=torch.device("cpu"))
    model.fit(ds, epochs=1, batch_size=2, prefix_loss="all", device=torch.device("cpu"))


def test_transformercomb_evaluate_probes_shape_and_restores_mode() -> None:
    """evaluate_probes returns Pauli tomography rows and restores train/eval mode."""
    pytest.importorskip("torch")

    model = _tiny_model(sequence_length=1)
    model.train()
    probe_set = _make_probe_set(cut=1, k=1, n_p=2, n_f=3)
    out = model.evaluate_probes(probe_set)
    assert out.shape == (2, 3, 4)
    assert out.dtype == np.float32
    assert model.training is True


def test_transformercomb_evaluate_probes_with_past_and_future_segments() -> None:
    """evaluate_probes stitches non-empty past and future feature segments."""
    pytest.importorskip("torch")

    model = _tiny_model(sequence_length=3)
    probe_set = _make_probe_set(cut=2, k=3, n_p=1, n_f=2)
    out = model.evaluate_probes(probe_set)
    assert out.shape == (1, 2, 4)
