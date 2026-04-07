from __future__ import annotations

import numpy as np
import pytest


def test_transformercomb_forward_shape_cpu() -> None:
    torch = pytest.importorskip("torch")

    from mqt.yaqs.characterization.process_tensors.surrogates.model import TransformerComb  # noqa: PLC0415

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E = torch.zeros((2, 3, 32), dtype=torch.float32)
    rho0 = torch.zeros((2, 8), dtype=torch.float32)
    out = model(E, rho0)
    assert tuple(out.shape) == (2, 3, 8)


def test_transformercomb_predict_numpy_roundtrip() -> None:
    torch = pytest.importorskip("torch")

    from mqt.yaqs.characterization.process_tensors.surrogates.model import TransformerComb  # noqa: PLC0415

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E = np.zeros((1, 2, 32), dtype=np.float32)
    rho0 = np.zeros((1, 8), dtype=np.float32)
    y = model.predict(E, rho0, device="cpu", return_numpy=True)
    assert isinstance(y, np.ndarray)
    assert y.shape == (1, 2, 8)


def test_transformercomb_predict_tensor_return_and_restores_mode() -> None:
    torch = pytest.importorskip("torch")

    from torch.utils.data import TensorDataset  # noqa: PLC0415

    from mqt.yaqs.characterization.process_tensors.surrogates.model import TransformerComb  # noqa: PLC0415

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
    model.fit(train_ds, val_dataset=val_ds, epochs=1, batch_size=2, prefix_loss="random", device=torch.device("cpu"))


def test_transformercomb_fit_invalid_prefix_loss_raises() -> None:
    torch = pytest.importorskip("torch")

    from torch.utils.data import TensorDataset  # noqa: PLC0415

    from mqt.yaqs.characterization.process_tensors.surrogates.model import TransformerComb  # noqa: PLC0415

    model = TransformerComb(d_e=32, d_rho=8, d_model=32, nhead=4, num_layers=1, dim_ff=64, dropout=0.0)
    E_t = torch.zeros((2, 2, 32), dtype=torch.float32)
    rho0_t = torch.zeros((2, 8), dtype=torch.float32)
    tgt_t = torch.zeros((2, 2, 8), dtype=torch.float32)
    ds = TensorDataset(E_t, rho0_t, tgt_t)
    with pytest.raises(ValueError):
        model.fit(ds, epochs=1, prefix_loss="nope")  # type: ignore[arg-type]

