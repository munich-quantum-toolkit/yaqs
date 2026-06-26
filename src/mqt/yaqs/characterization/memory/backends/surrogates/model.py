# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Neural surrogate module: :class:`TransformerComb` only.

Training data containers (:class:`~mqt.yaqs.characterization.memory.backends.surrogates.data.SeqTrace`,
:func:`~mqt.yaqs.characterization.memory.backends.surrogates.data.stack_traces`) live in
:mod:`mqt.yaqs.characterization.memory.backends.surrogates.data`.

Batch metrics on packed rho8 vectors live in :mod:`mqt.yaqs.characterization.memory.shared.metrics`.

**Naming** — A **sequence** is the chosen interventions (Choi / features) at each step. A **trajectory**
(in the noise sense) is one MCWF/TJM stochastic realization; see
:mod:`mqt.yaqs.characterization.memory.backends.tomography.data`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ...shared.encoding import decode_packed_pauli_batch, normalize_backend_rho, pack_rho8
from .utils import encode_choi_features

if TYPE_CHECKING:
    from ...operational_memory.samples import ProbeSet

# Computational-basis |0><0| (same convention as workflow sequence simulation).
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
            sequence_length: Total sequence length ``k`` for :meth:`entropy`. Set automatically by
                :meth:`fit` from training targets; may be set here before training.

        Raises:
            ValueError: If ``d_model`` is not divisible by ``nhead``.
        """
        super().__init__()
        if nhead <= 0:
            msg = f"nhead must be positive, got {nhead}."
            raise ValueError(msg)
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
        """Per-step intervention feature dimension (excluding the initial-state side channel).

        Raises:
            TypeError: If the input projection is not a linear layer.
        """
        in_proj = self.in_proj[0]
        if not isinstance(in_proj, nn.Linear):
            msg = "Expected a Linear input projection layer."
            raise TypeError(msg)
        return int(in_proj.in_features) - self._d_side

    def _default_rho0(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        r"""Packed rho8 for the physical |0⟩⟨0| reduced state (same path as training data).

        Uses :func:`~mqt.yaqs.characterization.memory.shared.encoding.pack_rho8` on the
        trace-1 density matrix for :math:`|0\\rangle\\langle 0|`, after
        :func:`~mqt.yaqs.characterization.memory.shared.encoding.normalize_backend_rho`,
        matching surrogate sequences in
        :mod:`~mqt.yaqs.characterization.memory.backends.surrogates.workflow`.

        Returns:
            Packed rho8 tensor on ``device`` with dtype ``dtype``.

        Raises:
            ValueError: If packed length does not match ``d_rho``.
        """
        packed = pack_rho8(normalize_backend_rho(_RHO0_PROJECTOR)).astype(np.float32)
        if packed.shape[0] != self.d_rho:
            msg = f"rho8 packing length {packed.shape[0]} does not match d_rho={self.d_rho}."
            raise ValueError(msg)
        return torch.as_tensor(packed, device=device, dtype=dtype)

    def _rho_to_features(self, rho: torch.Tensor) -> torch.Tensor:
        """Map predicted final density encodings to real feature vectors for the cut matrix.

        Returns:
            Real feature tensor with the same shape as ``rho``.

        Raises:
            ValueError: If the last dimension does not match ``d_rho``.
        """
        if rho.shape[-1] != self.d_rho:
            msg = f"Expected last dim d_rho={self.d_rho}, got {rho.shape}."
            raise ValueError(msg)
        return rho.to(dtype=torch.float64)

    def predict_final_state_batch(
        self,
        rho0: torch.Tensor,
        e_features: torch.Tensor,
        *,
        restore_training: bool = True,
    ) -> torch.Tensor:
        """Batched inference: predicted reduced state at the **last** timestep (eval mode, no gradients).

        Args:
            rho0: Initial encoding of shape ``(d_rho,)`` or ``(B, d_rho)``. A single row is broadcast to
                the batch size of ``e_features``.
            e_features: Per-step features of shape ``(B, T, d_e)``.
            restore_training: If ``False``, do not restore ``.train()`` after inference (for callers that
                run many batched predictions in a loop, e.g. :meth:`entropy`).

        Returns:
            Tensor of shape ``(B, d_rho)``.

        Raises:
            ValueError: If shapes are inconsistent.
        """
        if e_features.dim() != 3:
            msg = f"e_features must be (B, T, d_e), got {e_features.shape}."
            raise ValueError(msg)
        b = int(e_features.shape[0])
        rho0_t = torch.as_tensor(rho0, dtype=e_features.dtype, device=e_features.device)
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
            out = self.forward(e_features, rho0_t)
        if restore_training and was_training:
            self.train()
        return out[:, -1, :]

    def _k_for_probe(self) -> int:
        """Return the trained sequence length ``k`` used for probe evaluation.

        Returns:
            Total intervention steps ``k`` inferred from training data or ``__init__``.

        Raises:
            ValueError: If ``sequence_length`` was never set.
        """
        if self.sequence_length is None:
            msg = "sequence_length is unset: call fit() or pass sequence_length=... to __init__."
            raise ValueError(msg)
        return int(self.sequence_length)

    def evaluate_probes(self, probe_set: ProbeSet) -> np.ndarray:
        """Evaluate split-cut probe responses for :func:`run_operational_memory`.

        Returns:
            Array of shape ``(n_pasts, n_futures, 4)`` with Pauli tomography ``(I, X, Y, Z)``.
        """
        n_p = len(probe_set.past_pairs)
        n_f = len(probe_set.future_pairs)
        past_len = int(probe_set.cut) - 1
        suffix_len = int(probe_set.k) - int(probe_set.cut)
        v_rows = np.empty((n_p, n_f, 4), dtype=np.float32)
        dev = next(self.parameters()).device
        rho0 = self._default_rho0(device=dev, dtype=torch.float32)
        was_training = self.training
        self.eval()
        try:
            for i in range(n_p):
                past_prefix = (
                    probe_set.past_features[i, :past_len, :]
                    if past_len > 0
                    else np.zeros((0, self.d_e), dtype=np.float32)
                )
                past_batch = np.broadcast_to(past_prefix[None, :, :], (n_f, past_len, self.d_e)).copy()
                eff_ket = np.asarray(probe_set.past_cut_meas[i], dtype=np.complex128)
                eff_dm = np.outer(eff_ket, eff_ket.conj())
                cut_rows = []
                for j in range(n_f):
                    prep_ket = np.asarray(probe_set.future_prep_cut[j], dtype=np.complex128)
                    prep_dm = np.outer(prep_ket, prep_ket.conj())
                    cut_rows.append(encode_choi_features(prep_dm, eff_dm))
                cut_step = np.asarray(cut_rows, dtype=np.float32).reshape(n_f, 1, self.d_e)
                future_suffix = (
                    probe_set.future_features[:, 1:, :]
                    if suffix_len > 0
                    else np.zeros((n_f, 0, self.d_e), dtype=np.float32)
                )
                seq = np.concatenate([past_batch, cut_step, future_suffix], axis=1)
                seq_t = torch.from_numpy(seq).to(device=dev, dtype=torch.float32)
                rho_pred_batch = self.predict_final_state_batch(rho0, seq_t, restore_training=False)
                packed_np = rho_pred_batch.detach().cpu().numpy().astype(np.float32)
                v_rows[i] = decode_packed_pauli_batch(packed_np).astype(np.float32)
        finally:
            if was_training:
                self.train()
        return v_rows

    def forward(self, e_features: torch.Tensor, rho0: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            e_features: Per-step features of shape ``(B, T, d_e)``.
            rho0: Initial reduced state encoding of shape ``(B, d_rho)``.

        Returns:
            Predicted packed reduced states of shape ``(B, T, d_rho)``.

        Raises:
            ValueError: If input shapes are inconsistent.
        """
        b, t, _de = e_features.shape
        if rho0.shape != (b, self.d_rho):
            msg = f"rho0 mode expects rho0 (B,d_rho), got {rho0.shape}."
            raise ValueError(msg)
        side = rho0[:, None, :].expand(b, t, self._d_side)
        x = torch.cat([e_features, side], dim=-1)
        pe = _sinusoidal_positional_encoding(t, self.d_model, device=x.device, dtype=x.dtype)
        h = self.in_ln(self.in_proj(x)) + pe
        mask = _causal_mask(t, h.device)
        h = self.encoder(h, mask=mask)
        return self.head(h)

    def predict(
        self,
        e_features: torch.Tensor | np.ndarray,
        rho0: torch.Tensor | np.ndarray,
        *,
        device: torch.device | str | None = None,
        return_numpy: bool = True,
    ) -> torch.Tensor | np.ndarray:
        """Run inference (eval mode, no gradients).

        Args:
            e_features: Per-step features of shape ``(B, T, d_e)``.
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
        e_features_t = torch.as_tensor(e_features, dtype=torch.float32, device=dev)
        r0_t = torch.as_tensor(rho0, dtype=torch.float32, device=dev)
        with torch.no_grad():
            out = self.forward(e_features_t, r0_t)
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
        """Fit the model on sequence-trace data using MSE loss.

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

        e_train, rho0_train, target_train = train_dataset.tensors
        self.sequence_length = int(target_train.shape[1])
        train_ds = TensorDataset(e_train, rho0_train, target_train)

        has_val = val_dataset is not None
        if has_val:
            e_val, rho0_val, target_val = val_dataset.tensors

        opt = torch.optim.Adam(self.parameters(), lr=float(lr))
        loss_fn = nn.MSELoss()
        bs = min(int(batch_size), max(1, int(e_train.shape[0])))
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        k_max = int(target_train.shape[1])
        best = float("inf")
        best_state: dict[str, Any] | None = None

        for _ep in range(int(epochs)):
            self.train()
            for e_cpu, rho0_cpu, tgt_cpu in loader:
                e_batch = e_cpu.to(device)
                rho0_b = rho0_cpu.to(device)
                tgt_b = tgt_cpu.to(device)
                opt.zero_grad(set_to_none=True)
                if prefix_loss == "full" or k_max <= 1:
                    pred = self(e_batch, rho0_b)
                    loss = loss_fn(pred, tgt_b)
                elif prefix_loss == "random":
                    prefix_len = int(torch.randint(low=1, high=k_max + 1, size=(1,), device=e_batch.device).item())
                    pred = self(e_batch[:, :prefix_len, :], rho0_b)
                    loss = loss_fn(pred, tgt_b[:, :prefix_len, :])
                elif prefix_loss == "all":
                    losses = []
                    for prefix_len in range(1, k_max + 1):
                        pred_prefix = self(e_batch[:, :prefix_len, :], rho0_b)
                        losses.append(loss_fn(pred_prefix, tgt_b[:, :prefix_len, :]))
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
                    pred_va = self(e_val.to(device), rho0_val.to(device))
                    val = float(loss_fn(pred_va, target_val.to(device)).detach().cpu().item())
                if val < best:
                    best = val
                    best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

        if best_state is not None:
            self.load_state_dict(best_state)
        return self
