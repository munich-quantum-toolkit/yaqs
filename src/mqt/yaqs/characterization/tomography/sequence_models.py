# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Sequence models: (intervention features, rho0) -> rho_t.

We keep only a single input convention:

* ``input_mode="rho0"``: concatenate ``(E_t, rho_0)`` at every step so the network is not fed
  its own previous predictions; the transformer still builds long-range dependence on
  ``E_1..E_t`` via attention (process-tensor-style).

Optional ``memory_window`` restricts causal self-attention to the last ``m`` positions
(including the current step), for memory-stress experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _sinusoidal_positional_encoding(seq_len: int, d_model: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """(1, T, d_model) sinusoidal encoding (Vaswani et al.)."""
    if d_model <= 0:
        raise ValueError("d_model must be positive.")
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # (T,1)
    half = d_model // 2
    div = torch.exp(
        torch.arange(half, device=device, dtype=dtype) * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(half, 1))
    )  # (half,)
    ang = pos * div.unsqueeze(0)  # (T,half)
    pe = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    pe[:, 0 : 2 * half : 2] = torch.sin(ang)
    pe[:, 1 : 2 * half : 2] = torch.cos(ang)
    if d_model % 2 == 1:
        pe[:, -1] = 0.0
    return pe.unsqueeze(0)


def _causal_window_mask(seq_len: int, memory_window: int | None, device: torch.device) -> torch.Tensor:
    """Bool mask with True = **blocked** for :class:`nn.TransformerEncoder` (PyTorch convention)."""
    m = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    for i in range(seq_len):
        for j in range(seq_len):
            if j > i:
                m[i, j] = True
            elif memory_window is not None and memory_window > 0 and j < i - memory_window + 1:
                m[i, j] = True
    return m


class StateSequenceTransformer(nn.Module):
    """Causal transformer over per-step features (intervention encoding + state side).

    Alias: :data:`TransformerComb`.
    """

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
        memory_window: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            msg = f"d_model={d_model} must be divisible by nhead={nhead}."
            raise ValueError(msg)
        self.d_model = int(d_model)
        self.d_rho = int(d_rho)
        self.memory_window = memory_window
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
        mask = _causal_window_mask(t, self.memory_window, h.device)
        h = self.encoder(h, mask=mask)
        return self.head(h)


class StateSequenceGRU(nn.Module):
    """GRU over per-step concatenations ``(E_t, rho_0)`` (rho0 mode only)."""

    def __init__(
        self,
        d_e: int,
        d_rho: int,
        *,
        d_model: int = 128,
        num_layers: int = 1,
        summary_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.d_rho = int(d_rho)
        if summary_dim is not None and int(summary_dim) > 0:
            self._d_side = int(summary_dim)
            self.rho_compress = nn.Linear(d_rho, self._d_side)
        else:
            self._d_side = d_rho
            self.rho_compress = nn.Identity()
        self.in_proj = nn.Sequential(
            nn.Linear(d_e + self._d_side, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.rnn = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(d_model, d_rho)

    def _state_side_batch(self, rho_0: torch.Tensor, t: int) -> torch.Tensor:
        b = rho_0.shape[0]
        s = self.rho_compress(rho_0)
        return s[:, None, :].expand(b, t, s.shape[-1])

    def forward(self, E: torch.Tensor, rho0: torch.Tensor) -> torch.Tensor:
        b, t, _ = E.shape
        if rho0.shape != (b, self.d_rho):
            raise ValueError(f"rho0 mode expects rho0 (B,d_rho), got {rho0.shape}.")
        side = self._state_side_batch(rho0, t)
        x = torch.cat([E, side], dim=-1)
        h = self.in_proj(x)
        out, _ = self.rnn(h)
        return self.head(out)


TransformerComb = StateSequenceTransformer
