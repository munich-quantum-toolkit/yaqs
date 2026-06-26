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
    """Center memory-matrix rows by subtracting the past mean.

    Args:
        memory_matrix: Weighted response matrix with past index along axis 0.

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
    """Extract :math:`X,Y,Z` response channels from Pauli tomography.

    The identity component is stored but omitted here because it is fixed for
    physical states.

    Args:
        pauli_ij: Array with last dimension 4 (I, X, Y, Z).

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


def assemble_memory_matrix(
    pauli_ij: np.ndarray,
    weights_ij: np.ndarray,
    *,
    beta: float = 1.0,
    center: bool = True,
    log_weight_warnings: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Build the weighted memory matrix and optionally center past rows.

    Computes :math:`M^{(\beta)}_{i,(j,\alpha)} = w_{ij}^{\beta} f_{ij,\alpha}` from Pauli
    tomography ``(I,X,Y,Z)`` or XYZ channels, then subtracts the past-row mean when
    ``center=True`` (paper Eq. 14 at ``beta=1``).

    Args:
        pauli_ij: Pauli tomography ``(n_pasts, n_futures, 4)`` or XYZ ``(..., 3)``.
        weights_ij: Branch weights ``(n_pasts, n_futures)``.
        beta: Weight exponent applied to branch weights.
        center: If ``True``, return past-row-centered matrix as the second element.
        log_weight_warnings: Passed to :func:`sanitize_branch_weights`.

    Returns:
        Tuple ``(memory_matrix_raw, memory_matrix)`` where ``memory_matrix`` is centered
        when ``center=True``, otherwise equal to ``memory_matrix_raw``.
    """
    w_clean, _ = sanitize_branch_weights(weights_ij, log_warnings=log_weight_warnings)
    xyz = extract_xyz_channels(pauli_ij) if np.asarray(pauli_ij).shape[-1] == 4 else pauli_ij
    n_p, n_f, d_out = np.asarray(xyz, dtype=np.float64).shape
    w = np.asarray(w_clean, dtype=np.float64).reshape(n_p, n_f)
    features = np.asarray(xyz, dtype=np.float64).reshape(n_p, n_f, d_out)
    scale = np.power(w, float(beta))
    m_raw = (features * np.repeat(scale[:, :, np.newaxis], d_out, axis=2)).reshape(n_p, n_f * d_out)
    memory_matrix = center_rows(m_raw) if center else m_raw
    return m_raw, memory_matrix


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
        for idx, tail_weight in enumerate(tail_cumsum):
            if float(tail_weight / total_weight) > the:
                # ``tail_cumsum[idx]`` sums the ``idx + 1`` smallest modes. When
                # ``idx == n - 1`` the largest mode is included, so ``keep = n - idx``
                # would discard a significant mode even though the actual discarded
                # tail (``n - idx`` smallest only) is below the budget.
                if idx == s_full.size - 1:
                    keep = max(s_full.size - idx + 1, min_keep_eff)
                else:
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
        "rank": effective_modes,
        "singular_values": s,
        "singular_values_full": s_full,
    }
