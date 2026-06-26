# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Operational memory matrix construction and spectrum analysis."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np


def center_rows(memory_matrix: np.ndarray) -> np.ndarray:
    """Subtract the mean over past rows (axis 0).

    Returns:
        Past-row-centered matrix with the same shape as ``memory_matrix``.
    """
    m = np.asarray(memory_matrix, dtype=np.float64)
    return m - m.mean(axis=0, keepdims=True)


def sanitize_branch_weights(
    weights_ij: np.ndarray,
    *,
    log_warnings: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate branch weights; clamp negatives to 0 for ``w**beta`` construction.

    Does **not** renormalize weights across entries.

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
        meta["warnings"].append("Non-finite weights detected; replaced with 0 for memory-matrix construction.")
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


def extract_xyz_channels(pauli_ij: np.ndarray) -> np.ndarray:
    """Use :math:`X,Y,Z` response channels; identity is stored but fixed for physical states.

    Returns:
        Array with shape ``(..., 3)`` containing X, Y, Z expectations.

    Raises:
        ValueError: If the last dimension is not 4.
    """
    p = np.asarray(pauli_ij, dtype=np.float64)
    if p.shape[-1] != 4:
        msg = f"Expected Pauli tomography with last dim 4, got shape {p.shape}."
        raise ValueError(msg)
    return p[..., 1:4]


def assemble_weighted_matrix(
    pauli_ij: np.ndarray,
    weights_ij: np.ndarray,
    beta: float,
) -> np.ndarray:
    r"""Construct weighted memory matrix :math:`M^{(\beta)}_{i,(j,\alpha)} = w_{ij}^{\beta} f_{ij,\alpha}`.

    ``pauli_ij`` is ``(n_pasts, n_futures, 3|4)``; ``weights_ij`` is ``(n_pasts, n_futures)``.

    Returns:
        Matrix of shape ``(n_pasts, n_futures * d_out)`` with ``d_out=3``.
    """
    xyz = extract_xyz_channels(pauli_ij) if np.asarray(pauli_ij).shape[-1] == 4 else pauli_ij
    n_p, n_f, d_out = xyz.shape
    w = np.asarray(weights_ij, dtype=np.float64).reshape(n_p, n_f)
    features = np.asarray(xyz, dtype=np.float64).reshape(n_p, n_f, d_out)
    scale = np.power(w, float(beta))
    scale_exp = np.repeat(scale[:, :, np.newaxis], d_out, axis=2)
    return (features * scale_exp).reshape(n_p, n_f * d_out)


def assemble_weighted_matrix_from_probe(
    pauli_ij: np.ndarray,
    weights_ij: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build paper-weighted memory matrix with past centering (Eq. 14, beta=1).

    Returns:
        Tuple ``(memory_matrix_raw, memory_matrix)`` where the second entry is past-centered.
    """
    w_clean, _ = sanitize_branch_weights(weights_ij)
    m_raw = assemble_weighted_matrix(extract_xyz_channels(pauli_ij), w_clean, beta=1.0)
    return m_raw, center_rows(m_raw)


def assemble_memory_matrix(pauli_ij: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten Pauli features ``(n_p, n_f, 4)`` into a memory matrix (order preserved).

    Returns:
        Tuple ``(memory_matrix_raw, memory_matrix)`` where the second entry is past-centered.
    """
    xyz = extract_xyz_channels(pauli_ij)
    n_p, n_f, d_out = xyz.shape
    m_raw = xyz.reshape(n_p, n_f * d_out).astype(np.float64)
    return m_raw, center_rows(m_raw)


def compute_spectrum(
    memory_matrix: np.ndarray,
    *,
    discarded_weight_threshold: float | None = 1e-12,
    min_keep: int = 1,
) -> dict[str, Any]:
    r"""Cross-cut memory spectrum: :math:`S_V(c)` and :math:`R(c)=\exp(S_V(c))`.

    Args:
        memory_matrix: Past-row-centered memory matrix.
        discarded_weight_threshold: Relative tail weight above which singular values are
            discarded when computing entropy. ``None`` keeps the full spectrum.
        min_keep: Minimum number of singular values to retain after tail truncation.

    Returns:
        Dictionary with ``entropy``, ``rank`` (:math:`R(c)`), ``singular_values``, and
        ``singular_values_full``.
    """
    s_full = np.linalg.svd(memory_matrix, compute_uv=False).astype(np.float64)
    s = s_full.copy()
    total_weight = float(np.sum(s_full**2))

    if s.size and discarded_weight_threshold is not None and total_weight > 0.0:
        the = max(float(discarded_weight_threshold), 0.0)
        min_keep_eff = max(1, min(int(min_keep), int(s.size)))
        tail_cumsum = np.cumsum(s_full[::-1] ** 2)
        keep = s_full.size
        for idx, tail_weight in enumerate(tail_cumsum, start=1):
            if float(tail_weight / total_weight) > the:
                keep = max(s_full.size - idx, min_keep_eff)
                break
        else:
            keep = min_keep_eff
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
        "rank": effective_modes,
        "singular_values": s,
        "singular_values_full": s_full,
    }
