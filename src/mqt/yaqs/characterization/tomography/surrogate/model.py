# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Neural surrogate module: :class:`TransformerComb` only.

Training data containers (:class:`~mqt.yaqs.characterization.tomography.surrogate.data.SequenceRolloutSample`,
:func:`~mqt.yaqs.characterization.tomography.surrogate.data.stack_rollouts`) live in
:mod:`mqt.yaqs.characterization.tomography.surrogate.data`.

Batch metrics on packed rho8 vectors live in :mod:`mqt.yaqs.characterization.tomography.core.metrics`.

**Naming** — A **sequence** is the chosen interventions (Choi / features) at each step. A **trajectory**
(in the noise sense) is one MCWF/TJM stochastic realization; see :mod:`mqt.yaqs.characterization.tomography.estimate.sampling`.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _sinusoidal_positional_encoding(
    seq_len: int,
    d_model: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """(1, T, d_model) sinusoidal encoding (Vaswani et al.)."""
    if d_model <= 0:
        raise ValueError("d_model must be positive.")
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # (T,1)
    half = d_model // 2
    div = torch.exp(
        torch.arange(half, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(half, 1))
    )
    ang = pos * div.unsqueeze(0)
    pe = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    pe[:, 0 : 2 * half : 2] = torch.sin(ang)
    pe[:, 1 : 2 * half : 2] = torch.cos(ang)
    if d_model % 2 == 1:
        pe[:, -1] = 0.0
    return pe.unsqueeze(0)


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Bool mask with True = **blocked** for :class:`nn.TransformerEncoder` (PyTorch convention)."""
    m = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                m[i, j] = True
    return m


class TransformerComb(nn.Module):
    """Causal transformer over per-step features ``(E_t, rho_0)``."""

    def __init__(
        self,
        d_e: int,
        d_rho: int,
        *,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.0,
        layernorm_in: bool = False,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            msg = f"d_model={d_model} must be divisible by nhead={nhead}."
            raise ValueError(msg)
        self.d_model = int(d_model)
        self.d_rho = int(d_rho)
        self._d_side = d_rho
        self.layernorm_in = bool(layernorm_in)
        self.in_proj = nn.Sequential(
            nn.Linear(d_e + self._d_side, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.in_ln = nn.LayerNorm(d_model) if self.layernorm_in else nn.Identity()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            batch_first=True,
            dropout=float(dropout),
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, d_rho)

    def forward(self, E: torch.Tensor, rho0: torch.Tensor) -> torch.Tensor:
        b, t, _de = E.shape
        if rho0.shape != (b, self.d_rho):
            msg = f"rho0 mode expects rho0 (B,d_rho), got {rho0.shape}."
            raise ValueError(msg)
        side = rho0[:, None, :].expand(b, t, self._d_side)
        x = torch.cat([E, side], dim=-1)
        pe = _sinusoidal_positional_encoding(t, self.d_model, device=x.device, dtype=x.dtype)
        h = self.in_ln(self.in_proj(x)) + pe
        mask = _causal_mask(t, h.device)
        h = self.encoder(h, mask=mask)
        return self.head(h)

    def predict(
        self,
        E: torch.Tensor | np.ndarray,
        rho0: torch.Tensor | np.ndarray,
        *,
        device: torch.device | str | None = None,
        return_numpy: bool = True,
    ) -> torch.Tensor | np.ndarray:
        """Run inference (eval mode, no gradients). Matches :meth:`forward` tensor shapes.

        ``E``: ``(B, T, d_e)``, ``rho0``: ``(B, d_rho)``. Returns predictions ``(B, T, d_rho)`` as NumPy
        float32 by default (set ``return_numpy=False`` for PyTorch tensors on ``device``).
        """
        if device is None:
            dev = next(self.parameters()).device
        else:
            dev = torch.device(device) if isinstance(device, str) else device
        was_training = self.training
        self.eval()
        E_t = torch.as_tensor(E, dtype=torch.float32, device=dev)
        r0_t = torch.as_tensor(rho0, dtype=torch.float32, device=dev)
        with torch.no_grad():
            out = self.forward(E_t, r0_t)
        if was_training:
            self.train()
        if return_numpy:
            return out.detach().cpu().numpy().astype(np.float32)
        return out

    def fit(
        self,
        train_dataset: TensorDataset,
        *,
        val_dataset: TensorDataset | None = None,
        epochs: int = 100,
        lr: float = 2e-3,
        batch_size: int = 64,
        grad_clip: float = 1.0,
        prefix_loss: str = "full",
        device: torch.device | None = None,
    ) -> TransformerComb:
        """Fit with MSE on packed ``rho`` targets (same layout as :meth:`forward`).

        ``train_dataset`` / ``val_dataset`` must be :class:`~torch.utils.data.TensorDataset` with tensors
        ``(E, rho0, target)`` in that order — same layout as :func:`~mqt.yaqs.characterization.tomography.surrogate.workflow.generate_data`.

        ``prefix_loss``: ``"full"`` = loss on full sequence; ``"random"`` / ``"all"`` vary the
        training horizon (see experiment scripts). If ``val_dataset`` is given, restores the best
        validation checkpoint (full-sequence MSE).
        """
        if device is None:
            device = next(self.parameters()).device
        self.to(device)

        E_train, rho0_train, target_train = train_dataset.tensors
        E_train = E_train.to(device)
        rho0_train = rho0_train.to(device)
        target_train = target_train.to(device)
        train_ds = TensorDataset(E_train, rho0_train, target_train)

        has_val = val_dataset is not None
        if has_val:
            E_val, rho0_val, target_val = val_dataset.tensors
            E_val = cast(torch.Tensor, E_val).to(device)
            rho0_val = cast(torch.Tensor, rho0_val).to(device)
            target_val = cast(torch.Tensor, target_val).to(device)

        opt = torch.optim.Adam(self.parameters(), lr=float(lr))
        loss_fn = nn.MSELoss()
        bs = min(int(batch_size), max(1, int(E_train.shape[0])))
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        k_max = int(target_train.shape[1])
        best = float("inf")
        best_state: dict[str, Any] | None = None

        for _ep in range(int(epochs)):
            self.train()
            for E_b, rho0_b, tgt_b in loader:
                opt.zero_grad(set_to_none=True)
                if prefix_loss == "full" or k_max <= 1:
                    pred = self(E_b, rho0_b)
                    loss = loss_fn(pred, tgt_b)
                elif prefix_loss == "random":
                    Ls = int(torch.randint(low=1, high=k_max + 1, size=(1,), device=E_b.device).item())
                    pred = self(E_b[:, :Ls, :], rho0_b)
                    loss = loss_fn(pred, tgt_b[:, :Ls, :])
                elif prefix_loss == "all":
                    losses = []
                    for Ls in range(1, k_max + 1):
                        pred_L = self(E_b[:, :Ls, :], rho0_b)
                        losses.append(loss_fn(pred_L, tgt_b[:, :Ls, :]))
                    loss = torch.stack(losses, dim=0).mean()
                else:
                    msg = f"Unknown prefix_loss: {prefix_loss!r}"
                    raise ValueError(msg)
                loss.backward()
                if grad_clip and float(grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), float(grad_clip))
                opt.step()

            if has_val:
                self.eval()
                with torch.no_grad():
                    pred_va = self(cast(torch.Tensor, E_val), cast(torch.Tensor, rho0_val))
                    val = float(loss_fn(pred_va, cast(torch.Tensor, target_val)).detach().cpu().item())
                if val < best:
                    best = val
                    best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

        if best_state is not None:
            self.load_state_dict(best_state)
        return self
