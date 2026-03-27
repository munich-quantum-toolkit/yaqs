# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Sequence models: (intervention features, state side) -> rho_t.

* ``input_mode="markov"``: concatenate ``(E_t, rho_{t-1})`` (short-memory baseline).
* ``input_mode="rho0"``: concatenate ``(E_t, rho_0)`` at every step so the RNN is not fed
  the immediate predecessor; the transformer still builds long-range dependence on
  ``E_1..E_t`` via attention (process-tensor-style).
* ``input_mode="compressed"``: ``(E_t, W rho_0)`` with a learned linear bottleneck (optional
  summary of the initial state).

Optional ``memory_window`` restricts causal self-attention to the last ``m`` positions
(including the current step), for memory-stress experiments.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

InputMode = Literal["markov", "rho0", "compressed"]


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
        input_mode: InputMode = "markov",
        summary_dim: int | None = None,
        memory_window: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            msg = f"d_model={d_model} must be divisible by nhead={nhead}."
            raise ValueError(msg)
        self.d_model = int(d_model)
        self.d_rho = int(d_rho)
        self.input_mode: InputMode = input_mode
        self.memory_window = memory_window
        if input_mode == "compressed":
            if summary_dim is None or int(summary_dim) <= 0:
                raise ValueError("compressed mode requires summary_dim > 0.")
            self._d_side = int(summary_dim)
            self.rho_compress = nn.Linear(d_rho, self._d_side)
        else:
            self._d_side = d_rho
            self.rho_compress = nn.Identity()
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

    def _state_side(self, rho_0: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "compressed":
            return self.rho_compress(rho_0)
        return rho_0

    def forward(self, E: torch.Tensor, rho_side: torch.Tensor) -> torch.Tensor:
        b, t, _de = E.shape
        if self.input_mode == "markov":
            if rho_side.shape != (b, t, self.d_rho):
                msg = f"markov mode expects rho_side (B,T,d_rho), got {rho_side.shape}."
                raise ValueError(msg)
            side = rho_side
        else:
            if rho_side.shape != (b, self.d_rho):
                msg = f"rho0/compressed mode expects rho_side (B,d_rho), got {rho_side.shape}."
                raise ValueError(msg)
            s = self._state_side(rho_side)
            side = s[:, None, :].expand(b, t, s.shape[-1])
        x = torch.cat([E, side], dim=-1)
        pe = _sinusoidal_positional_encoding(t, self.d_model, device=x.device, dtype=x.dtype)
        h = self.in_ln(self.in_proj(x)) + pe
        mask = _causal_window_mask(t, self.memory_window, h.device)
        h = self.encoder(h, mask=mask)
        return self.head(h)

    def forward_rollout(self, E: torch.Tensor, rho_0: torch.Tensor) -> torch.Tensor:
        """Autoregressive rollout (predicted rhos fed back only in markov mode)."""
        b, k, _ = E.shape
        dr = rho_0.shape[-1]
        if self.input_mode != "markov":
            preds = torch.empty(b, k, dr, device=E.device, dtype=E.dtype)
            for ell in range(k):
                L = ell + 1
                sub = self.forward(E[:, :L, :], rho_0)
                preds[:, ell, :] = sub[:, ell, :]
            return preds

        preds = torch.empty(b, k, dr, device=E.device, dtype=E.dtype)
        for ell in range(k):
            L = ell + 1
            E_sub = E[:, :L, :]
            rho_prev_sub = torch.zeros(b, L, dr, device=E.device, dtype=E.dtype)
            rho_prev_sub[:, 0, :] = rho_0
            if L > 1:
                rho_prev_sub[:, 1:L, :] = preds[:, :ell, :]
            out = self.forward(E_sub, rho_prev_sub)
            preds[:, ell, :] = out[:, ell, :]
        return preds


class StateSequenceGRU(nn.Module):
    """GRU over the same per-step concatenations as :class:`StateSequenceTransformer`."""

    def __init__(
        self,
        d_e: int,
        d_rho: int,
        *,
        d_model: int = 128,
        num_layers: int = 1,
        input_mode: InputMode = "markov",
        summary_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.d_rho = int(d_rho)
        self.input_mode: InputMode = input_mode
        if input_mode == "compressed":
            if summary_dim is None or int(summary_dim) <= 0:
                raise ValueError("compressed mode requires summary_dim > 0.")
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
        if self.input_mode == "compressed":
            s = self.rho_compress(rho_0)
        else:
            s = rho_0
        return s[:, None, :].expand(b, t, s.shape[-1])

    def forward(self, E: torch.Tensor, rho_side: torch.Tensor) -> torch.Tensor:
        b, t, _ = E.shape
        if self.input_mode == "markov":
            if rho_side.shape != (b, t, self.d_rho):
                raise ValueError(f"markov expects rho_side (B,T,d_rho), got {rho_side.shape}.")
            side = rho_side
        else:
            if rho_side.shape != (b, self.d_rho):
                raise ValueError(f"rho0/compressed expects rho_side (B,d_rho), got {rho_side.shape}.")
            side = self._state_side_batch(rho_side, t)
        x = torch.cat([E, side], dim=-1)
        h = self.in_proj(x)
        out, _ = self.rnn(h)
        return self.head(out)

    def forward_rollout(self, E: torch.Tensor, rho_0: torch.Tensor) -> torch.Tensor:
        b, k, _ = E.shape
        dr = rho_0.shape[-1]
        if self.input_mode != "markov":
            side = self._state_side_batch(rho_0, k)
            x = torch.cat([E, side], dim=-1)
            h = self.in_proj(x)
            out, _ = self.rnn(h)
            return self.head(out)

        preds = torch.empty(b, k, dr, device=E.device, dtype=E.dtype)
        rho_c = rho_0
        h = torch.zeros(self.num_layers, b, self.d_model, device=E.device, dtype=E.dtype)
        for t in range(k):
            x = torch.cat([E[:, t, :], rho_c], dim=-1)
            z = self.in_proj(x).unsqueeze(1)
            o, h = self.rnn(z, h)
            preds[:, t, :] = self.head(o.squeeze(1))
            rho_c = preds[:, t, :]
        return preds


TransformerComb = StateSequenceTransformer
