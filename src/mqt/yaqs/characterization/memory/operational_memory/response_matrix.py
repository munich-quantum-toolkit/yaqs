# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Response matrix construction and spectrum analysis."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def sanitize_branch_weights(
    weights_ij: np.ndarray,
    *,
    log_warnings: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sanitize branch weights for weighted matrix assembly.

    Clamps negative values to zero for ``w**beta`` construction. Does **not**
    renormalize weights across grid entries.

    Args:
        weights_ij: Branch weights of shape ``(n_pasts, n_futures)``.
        log_warnings: Whether to emit warnings for negative weights.

    Returns:
        Tuple ``(weights_clean, meta)`` with diagnostic metadata in ``meta``.
    """
    w = np.asarray(weights_ij, dtype=np.float64)
    meta: dict[str, Any] = {
        "weight_data_invalid": False,
        "nan_count": int(np.isnan(w).sum()),
        "posinf_count": int(np.isposinf(w).sum()),
        "neginf_count": int(np.isneginf(w).sum()),
        "negative_count": int((w < 0).sum()),
        "warnings": [],
    }
    if meta["nan_count"] or meta["posinf_count"] or meta["neginf_count"]:
        meta["weight_data_invalid"] = True
        meta["warnings"].append("Non-finite weights detected; replaced with 0 for response-matrix construction.")
    if meta["negative_count"]:
        meta["warnings"].append("Negative weights clamped to 0.")
        if log_warnings:
            warnings.warn(
                "sanitize_branch_weights: clamped negative cumulative weights to 0.",
                stacklevel=2,
            )
    w_clean = w.copy()
    w_clean[w_clean < 0] = 0.0
    w_clean = np.nan_to_num(w_clean, nan=0.0, posinf=0.0, neginf=0.0)
    return w_clean, meta


def assemble_response_matrix(
    pauli_ij: np.ndarray,
    weights_ij: np.ndarray,
    *,
    beta: float = 1.0,
    log_weight_warnings: bool = True,
) -> np.ndarray:
    r"""Build the raw weighted response matrix.

    Computes :math:`V^{(\beta)}_{(j,\alpha),i} = w_{ij}^{\beta} f_{ij,\alpha}` from Pauli
    tomography in ``(I, X, Y, Z)`` order. Rows label future-probe response channels and columns
    label conditioned histories. At the paper-facing default :math:`\beta=1`, normalized
    tomography has identity entries :math:`V_{(j,I),i}=w_{ij}`.

    Args:
        pauli_ij: Pauli tomography with shape ``(n_histories, n_futures, 4)`` and channel order
            ``(I, X, Y, Z)``.
        weights_ij: Branch weights with shape ``(n_histories, n_futures)``.
        beta: Weight exponent applied to branch weights.
        log_weight_warnings: Passed to :func:`sanitize_branch_weights`.

    Returns:
        Raw branch-weighted response matrix with shape
        ``(4 * n_futures, n_histories)``. Within each future probe, channels vary fastest in
        ``(I, X, Y, Z)`` order.

    Raises:
        ValueError: If ``pauli_ij`` is not a three-dimensional four-channel array, or if
            ``weights_ij`` does not match its history and future dimensions.
    """
    features = np.asarray(pauli_ij, dtype=np.float64)
    if features.ndim != 3 or features.shape[-1] != 4:
        msg = f"pauli_ij must have shape (n_histories, n_futures, 4), got {features.shape}."
        raise ValueError(msg)
    n_p, n_f, d_out = features.shape
    weights = np.asarray(weights_ij, dtype=np.float64)
    if weights.shape != (n_p, n_f):
        msg = f"weights_ij must have shape {(n_p, n_f)}, got {weights.shape}."
        raise ValueError(msg)
    w_clean, _ = sanitize_branch_weights(weights, log_warnings=log_weight_warnings)
    scale = np.power(w_clean, float(beta))
    weighted = features * scale[:, :, np.newaxis]
    return weighted.transpose(1, 2, 0).reshape(n_f * d_out, n_p)


def compute_spectrum(
    response_matrix: np.ndarray,
    *,
    discarded_weight_threshold: float | None = 1e-12,
    min_keep: int = 1,
) -> dict[str, Any]:
    r"""Cross-cut memory spectrum: :math:`S_V(c)` and :math:`R(c)=\exp(S_V(c))`.

    Args:
        response_matrix: Raw branch-weighted response matrix with future-response rows and
            history columns. Left singular vectors describe future-response directions; right
            singular vectors describe combinations of histories.
        discarded_weight_threshold: Relative tail weight above which singular values are
            discarded when computing entropy. ``None`` keeps the full spectrum.
        min_keep: Minimum number of singular values to retain after tail truncation.

    Returns:
        Dictionary with ``entropy``, ``modes`` (:math:`R(c)`), ``singular_values``, and
        ``singular_values_full``.
    """
    s_full = np.linalg.svd(response_matrix, compute_uv=False).astype(np.float64)
    s = s_full.copy()
    total_weight = float(np.sum(s_full**2))

    if s.size and discarded_weight_threshold is not None and total_weight > 0.0:
        the = max(float(discarded_weight_threshold), 0.0)
        min_keep_eff = max(1, min(int(min_keep), int(s.size)))
        tail_cumsum = np.cumsum(s_full[::-1] ** 2)
        keep = s_full.size
        for idx, tail_weight in enumerate(tail_cumsum):
            if float(tail_weight / total_weight) > the:
                keep = max(s_full.size - idx, min_keep_eff)
                break
        else:
            keep = s_full.size
        s = s_full[:keep]

    kept_weight = float(np.sum(s**2))
    if kept_weight <= 0.0:
        entropy = 0.0
        effective_modes = 1.0
    else:
        q = np.clip((s**2) / kept_weight, 1e-30, 1.0)
        entropy = float(-np.sum(q * np.log(q)))
        effective_modes = float(np.exp(entropy))

    return {
        "entropy": entropy,
        "modes": effective_modes,
        "singular_values": s,
        "singular_values_full": s_full,
    }
