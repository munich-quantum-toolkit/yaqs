# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Neural surrogate module: :class:`TransformerComb` only.

Training data containers (:class:`~mqt.yaqs.characterization.process_tensors.surrogates.data.SequenceRolloutSample`,
:func:`~mqt.yaqs.characterization.process_tensors.surrogates.data.stack_rollouts`) live in
:mod:`mqt.yaqs.characterization.process_tensors.surrogates.data`.

Batch metrics on packed rho8 vectors live in :mod:`mqt.yaqs.characterization.process_tensors.core.metrics`.

**Naming** — A **sequence** is the chosen interventions (Choi / features) at each step. A **trajectory**
(in the noise sense) is one MCWF/TJM stochastic realization; see :mod:`mqt.yaqs.characterization.process_tensors.tomography.data`.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..core.encoding import normalize_rho_from_backend_output, pack_rho8
from .utils import _sample_random_intervention_sequence

# Computational-basis |0⟩⟨0| (same convention as :func:`~mqt.yaqs.characterization.process_tensors.surrogates.workflow.generate_data` rollouts).
_RHO0_PROJECTOR = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)


def _sinusoidal_positional_encoding(
    seq_len: int,
    d_model: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build sinusoidal positional encodings.

    Args:
        seq_len: Sequence length ``T``.
        d_model: Model dimension.
        device: Target device for the returned tensor.
        dtype: Target dtype for the returned tensor.

    Returns:
        Positional encoding tensor of shape ``(1, T, d_model)``.

    Raises:
        ValueError: If ``d_model`` is not positive.
    """
    if d_model <= 0:
        msg = "d_model must be positive."
        raise ValueError(msg)
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
    """Create a causal attention mask for a transformer encoder.

    Args:
        seq_len: Sequence length.
        device: Target device for the returned tensor.

    Returns:
        Boolean mask of shape ``(seq_len, seq_len)`` where ``True`` indicates blocked attention.
    """
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
        sequence_length: int | None = None,
    ) -> None:
        """Initialize the transformer surrogate.

        Args:
            d_e: Per-step feature dimension.
            d_rho: Output dimension per step (rho8 uses 8).
            d_model: Transformer model width.
            nhead: Number of attention heads.
            num_layers: Number of encoder layers.
            dim_ff: Feed-forward dimension inside encoder layers.
            dropout: Dropout rate.
            layernorm_in: Whether to apply a LayerNorm after the input projection.
            sequence_length: Total rollout length ``k`` for :meth:`entropy`. Set automatically by
                :meth:`fit` from training targets; may be set here before training.

        Raises:
            ValueError: If ``d_model`` is not divisible by ``nhead``.
        """
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
        self.sequence_length: int | None = int(sequence_length) if sequence_length is not None else None

    @property
    def d_e(self) -> int:
        """Per-step intervention feature dimension (excluding the initial-state side channel)."""
        return int(self.in_proj[0].in_features) - self._d_side

    def _default_rho0(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Packed rho8 for the physical |0⟩⟨0| reduced state (same path as training data).

        Uses :func:`~mqt.yaqs.characterization.process_tensors.core.encoding.pack_rho8` on the
        trace-1 density matrix for :math:`|0\\rangle\\langle 0|`, after
        :func:`~mqt.yaqs.characterization.process_tensors.core.encoding.normalize_rho_from_backend_output`,
        matching surrogate rollouts in
        :mod:`~mqt.yaqs.characterization.process_tensors.surrogates.workflow`.
        """
        packed = pack_rho8(normalize_rho_from_backend_output(_RHO0_PROJECTOR)).astype(np.float32)
        if packed.shape[0] != self.d_rho:
            msg = f"rho8 packing length {packed.shape[0]} does not match d_rho={self.d_rho}."
            raise ValueError(msg)
        return torch.as_tensor(packed, device=device, dtype=dtype)

    def _rho_to_features(self, rho: torch.Tensor) -> torch.Tensor:
        """Map predicted final density encodings to real feature vectors for the cut matrix."""
        if rho.shape[-1] != self.d_rho:
            msg = f"Expected last dim d_rho={self.d_rho}, got {rho.shape}."
            raise ValueError(msg)
        return rho.to(dtype=torch.float64)

    def _sample_past_future_sequences(
        self,
        cut_time: int,
        n_past: int,
        n_future: int,
        *,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw random Choi feature rows for past and future segments (same distribution as training data)."""
        if self.sequence_length is None:
            msg = "sequence_length must be set."
            raise ValueError(msg)
        k_total = int(self.sequence_length)
        t = int(cut_time)
        k_fut = k_total - t
        d_e = self.d_e
        if t > 0:
            past = np.empty((n_past, t, d_e), dtype=np.float32)
            for a in range(n_past):
                _, rows = _sample_random_intervention_sequence(t, rng)
                past[a] = rows
        else:
            past = np.zeros((n_past, 0, d_e), dtype=np.float32)
        if k_fut > 0:
            fut = np.empty((n_future, k_fut, d_e), dtype=np.float32)
            for b in range(n_future):
                _, rows = _sample_random_intervention_sequence(k_fut, rng)
                fut[b] = rows
        else:
            fut = np.zeros((n_future, 0, d_e), dtype=np.float32)
        return past, fut

    def predict_final_state_batch(
        self,
        rho0: torch.Tensor,
        E: torch.Tensor,
        *,
        restore_training: bool = True,
    ) -> torch.Tensor:
        """Batched inference: predicted reduced state at the **last** timestep (eval mode, no gradients).

        Args:
            rho0: Initial encoding of shape ``(d_rho,)`` or ``(B, d_rho)``. A single row is broadcast to
                the batch size of ``E``.
            E: Per-step features of shape ``(B, T, d_e)``.
            restore_training: If ``False``, do not restore ``.train()`` after inference (for callers that
                run many batched predictions in a loop, e.g. :meth:`entropy`).

        Returns:
            Tensor of shape ``(B, d_rho)``.

        Raises:
            ValueError: If shapes are inconsistent.
        """
        if E.dim() != 3:
            msg = f"E must be (B, T, d_e), got {E.shape}."
            raise ValueError(msg)
        b = int(E.shape[0])
        rho0_t = torch.as_tensor(rho0, dtype=E.dtype, device=E.device)
        if rho0_t.dim() == 1:
            if rho0_t.shape[0] != self.d_rho:
                msg = f"rho0 (d_rho,) expected length {self.d_rho}, got {rho0_t.shape}."
                raise ValueError(msg)
            rho0_t = rho0_t.unsqueeze(0).expand(b, -1)
        elif rho0_t.shape != (b, self.d_rho):
            msg = f"rho0 must be (d_rho,) or (B, d_rho) with B={b}, got {rho0_t.shape}."
            raise ValueError(msg)
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self.forward(E, rho0_t)
        if restore_training and was_training:
            self.train()
        return out[:, -1, :]

    def entropy(self, timestep: int, past_samples: int, future_samples: int) -> float:
        """Cut-entropy diagnostic from random past/future Choi sequences.

        Analogous to the entropy of the normalized squared singular values of an MPS/MPO bond matrix
        (spectrum of the reduced density operator across the cut): build a future-response matrix at
        the cut, take its singular values, form weights ``s_i^2 / \\sum_j s_j^2``, and return the Shannon
        entropy ``-\\sum_i w_i \\log w_i`` in **nats**.

        Uses a fixed **interior** cut only: past length ``t`` and future length ``k - t`` must both be
        positive, i.e. ``1 <= t < k`` where ``k`` is :attr:`sequence_length`. Cuts at ``t = 0`` (no
        past) or ``t = k`` (no future) are rejected.

        For such ``t``, draws ``past_samples`` independent past prefixes of length ``t`` and
        ``future_samples`` independent future suffixes of length ``k - t`` (same random-intervention
        distribution as :func:`~mqt.yaqs.characterization.process_tensors.surrogates.workflow.generate_data`).
        Inference starts from the packed :math:`|0\\rangle\\langle 0|` state. The matrix is column-centered
        before SVD.

        Args:
            timestep: Interior cut index ``t`` (past length). Must satisfy ``1 <= t < k`` where ``k`` is
                :attr:`sequence_length`.
            past_samples: Number ``N_p`` of random past prefixes.
            future_samples: Number ``N_f`` of random future suffixes.

        Returns:
            Entropy as a Python ``float`` (CPU scalar).

        Raises:
            ValueError: If :attr:`sequence_length` is unset or ``< 2``, counts are invalid, or ``timestep``
                is not an interior cut.
        """
        if self.sequence_length is None:
            msg = "sequence_length is unset: call fit() or pass sequence_length=... to __init__."
            raise ValueError(msg)
        k_total = int(self.sequence_length)
        t = int(timestep)
        n_p = int(past_samples)
        n_f = int(future_samples)
        if n_p <= 0 or n_f <= 0:
            msg = f"past_samples and future_samples must be positive, got {n_p} and {n_f}."
            raise ValueError(msg)
        if k_total < 2:
            msg = f"entropy requires sequence_length >= 2 for an interior past/future cut; got {k_total}."
            raise ValueError(msg)
        if not (1 <= t < k_total):
            msg = (
                f"timestep must be an interior cut with 1 <= timestep < sequence_length ({k_total}); "
                f"got {t}. Use 1 <= t <= k-1 (not t=0 or t=k)."
            )
            raise ValueError(msg)

        rng = np.random.default_rng()
        past_np, fut_np = self._sample_past_future_sequences(t, n_p, n_f, rng=rng)

        dev = next(self.parameters()).device
        past = torch.from_numpy(past_np).to(device=dev, dtype=torch.float32)
        fut = torch.from_numpy(fut_np).to(device=dev, dtype=torch.float32)

        d_y = self.d_rho
        v = torch.empty((n_p, n_f * d_y), device=dev, dtype=torch.float64)
        rho0 = self._default_rho0(device=dev, dtype=torch.float32)

        was_training = self.training
        self.eval()
        try:
            for a in range(n_p):
                past_row = past[a]
                past_batch = past_row.unsqueeze(0).expand(n_f, -1, -1)
                full_seq_batch = torch.cat([past_batch, fut], dim=1)
                rho_pred_batch = self.predict_final_state_batch(rho0, full_seq_batch, restore_training=False)
                feat_batch = self._rho_to_features(rho_pred_batch)
                v[a] = feat_batch.reshape(-1)
        finally:
            if was_training:
                self.train()

        v = v - v.mean(dim=0, keepdim=True)

        s = torch.linalg.svdvals(v)
        sq = s * s
        denom = sq.sum()
        if denom.item() <= 0.0:
            return 0.0
        weights = sq / denom
        ent = -(weights * torch.log(weights.clamp(min=1e-30))).sum()
        return float(ent.detach().cpu().item())

    def forward(self, E: torch.Tensor, rho0: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            E: Per-step features of shape ``(B, T, d_e)``.
            rho0: Initial reduced state encoding of shape ``(B, d_rho)``.

        Returns:
            Predicted packed reduced states of shape ``(B, T, d_rho)``.

        Raises:
            ValueError: If input shapes are inconsistent.
        """
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
        """Run inference (eval mode, no gradients).

        Args:
            E: Per-step features of shape ``(B, T, d_e)``.
            rho0: Initial reduced state encoding of shape ``(B, d_rho)``.
            device: Device for inference. Defaults to the model's current device.
            return_numpy: If ``True``, return a NumPy array on CPU; otherwise return a tensor on ``device``.

        Returns:
            Predictions of shape ``(B, T, d_rho)``.
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
        """Fit the model on rollout data using MSE loss.

        Args:
            train_dataset: TensorDataset containing tensors ``(E, rho0, target)``.
            val_dataset: Optional validation dataset with the same tensor layout.
            epochs: Number of epochs.
            lr: Learning rate.
            batch_size: Batch size.
            grad_clip: Gradient clipping norm (0 disables clipping).
            prefix_loss: Loss horizon mode: ``"full"``, ``"random"``, or ``"all"``.
            device: Training device. Defaults to the model's current device.

        Returns:
            Self (for chaining).

        Raises:
            ValueError: If ``prefix_loss`` is invalid.
        """
        if device is None:
            device = next(self.parameters()).device
        self.to(device)

        E_train, rho0_train, target_train = train_dataset.tensors
        E_train = E_train.to(device)
        rho0_train = rho0_train.to(device)
        target_train = target_train.to(device)
        self.sequence_length = int(target_train.shape[1])
        train_ds = TensorDataset(E_train, rho0_train, target_train)

        has_val = val_dataset is not None
        if has_val:
            E_val, rho0_val, target_val = val_dataset.tensors
            E_val = cast("torch.Tensor", E_val).to(device)
            rho0_val = cast("torch.Tensor", rho0_val).to(device)
            target_val = cast("torch.Tensor", target_val).to(device)

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
                    pred_va = self(cast("torch.Tensor", E_val), cast("torch.Tensor", rho0_val))
                    val = float(loss_fn(pred_va, cast("torch.Tensor", target_val)).detach().cpu().item())
                if val < best:
                    best = val
                    best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

        if best_state is not None:
            self.load_state_dict(best_state)
        return self
