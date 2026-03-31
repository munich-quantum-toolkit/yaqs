# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Neural surrogate models and trajectory-level utilities.

This module consolidates:

* Low-level metrics on packed rho8 encodings (Frobenius / trace distance).
* TrajectoryCombSample + helpers for MCWF trajectory datasets.
* Sequence model ``TransformerComb`` over intervention features and rho0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
import torch.nn as nn

from .predictor_encoding import unpack_rho8


def mean_trace_distance_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Mean trace distance over batches of 8-float encodings (helper for benchmarks)."""
    from .ml_dataset import trace_distance  # local import to avoid cycles

    assert pred_rho8.shape == tgt_rho8.shape
    tds: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        tds.append(trace_distance(rp, rt))
    return float(np.mean(tds))


def mean_frobenius_mse_rho8(pred_rho8: np.ndarray, tgt_rho8: np.ndarray) -> float:
    """Mean squared Frobenius error for 8-float encodings (same Hilbert-Schmidt square as Frobenius)."""
    assert pred_rho8.shape == tgt_rho8.shape
    diffs: list[float] = []
    for i in range(pred_rho8.shape[0]):
        rp = unpack_rho8(pred_rho8[i])
        rt = unpack_rho8(tgt_rho8[i])
        d = rp - rt
        diffs.append(float(np.real(np.vdot(d, d))))
    return float(np.mean(diffs))


@dataclass(frozen=True)
class TrajectoryCombSample:
    """One simulated trajectory for sequential reduced-state learning.

    ``rho_seq[t]`` is the reduced state on site 0 **after** intervention ``t`` and the
    subsequent evolution segment (aligned with ``timesteps[t]``).

    ``alphas[t]`` indexes the Choi / intervention label at step ``t`` (same convention
    as the flat NN-comb predictor benchmarks). ``E_features`` rows are length 32 (Choi flat).
    """

    rho_0: np.ndarray  # shape (8,), float32 — packed 2x2 rho before first intervention
    alphas: np.ndarray  # shape (K,), int64
    E_features: np.ndarray  # shape (K, 32), float32
    rho_seq: np.ndarray  # shape (K, 8), float32
    context: np.ndarray | None  # optional static features (e.g. dt, J, g), shape (d_ctx,)
    weight: float


def trajectory_batch_to_tensors(
    samples: list[TrajectoryCombSample],
    *,
    append_context_to_E: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Stack samples into ``rho_0``, ``E``, ``rho_seq``, optional ``context`` arrays."""
    rho_0 = np.stack([s.rho_0 for s in samples], axis=0).astype(np.float32)
    E = np.stack([s.E_features for s in samples], axis=0).astype(np.float32)
    rho_seq = np.stack([s.rho_seq for s in samples], axis=0).astype(np.float32)
    ctx = None
    if samples[0].context is not None:
        ctx = np.stack([cast(np.ndarray, s.context) for s in samples], axis=0).astype(np.float32)
    if append_context_to_E and ctx is not None:
        k = E.shape[1]
        ctx_b = np.broadcast_to(ctx[:, None, :], (E.shape[0], k, ctx.shape[1])).astype(np.float32)
        E = np.concatenate([E, ctx_b], axis=-1)
        ctx = None
    return rho_0, E, rho_seq, ctx


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
        max_len: int | None = None,
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


__all__ = [
    "TrajectoryCombSample",
    "trajectory_batch_to_tensors",
    "mean_frobenius_mse_rho8",
    "mean_trace_distance_rho8",
    "TransformerComb",
]

