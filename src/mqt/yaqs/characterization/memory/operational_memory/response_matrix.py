# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Response matrix construction and spectrum analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..shared.probabilities import PROBABILITY_ATOL


def assemble_response_matrix(
    pauli_ij: np.ndarray,
    weights_ij: np.ndarray,
) -> np.ndarray:
    r"""Build the raw weighted response matrix.

    Computes :math:`V_{(j,\alpha),i} = p_{ij} f_{ij,\alpha}` from Pauli tomography in
    ``(I, X, Y, Z)`` order, where :math:`p_{ij}` is the probability of every retained outcome
    in the complete history and future record. Rows label future-probe response channels and
    columns label conditioned histories. Normalized tomography has identity entries
    :math:`V_{(j,I),i}=p_{ij}`.

    Args:
        pauli_ij: Pauli tomography with shape ``(n_histories, n_futures, 4)`` and channel order
            ``(I, X, Y, Z)``.
        weights_ij: Complete retained-record probabilities with shape
            ``(n_histories, n_futures)``.

    Returns:
        Raw branch-weighted response matrix with shape
        ``(4 * n_futures, n_histories)``. Within each future probe, channels vary fastest in
        ``(I, X, Y, Z)`` order.

    Raises:
        ValueError: If the tomography shape or identity channel is invalid, or if
            ``weights_ij`` has the wrong shape, contains negative or non-finite values,
            or exceeds one beyond numerical tolerance.
    """
    features = np.asarray(pauli_ij, dtype=np.float64)
    if features.ndim != 3 or features.shape[-1] != 4:
        msg = f"pauli_ij must have shape (n_histories, n_futures, 4), got {features.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(features)):
        msg = "pauli_ij must contain only finite values."
        raise ValueError(msg)
    n_p, n_f, d_out = features.shape
    weights = np.asarray(weights_ij, dtype=np.float64)
    if weights.shape != (n_p, n_f):
        msg = f"weights_ij must have shape {(n_p, n_f)}, got {weights.shape}."
        raise ValueError(msg)
    if not np.allclose(features[..., 0], 1.0, rtol=0.0, atol=1e-8):
        msg = "pauli_ij identity expectations must equal 1 for normalized conditional states."
        raise ValueError(msg)
    if not np.all(np.isfinite(weights)) or np.any((weights < 0.0) | (weights > 1.0 + PROBABILITY_ATOL)):
        msg = (
            "weights_ij must contain finite complete-record probabilities in [0, 1]; "
            "only numerical roundoff above 1 is tolerated."
        )
        raise ValueError(msg)
    weights = np.clip(weights, 0.0, 1.0)
    weighted = features * weights[:, :, np.newaxis]
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
        ``singular_values_full``. ``left_singular_vectors`` and ``right_singular_vectors``
        contain the compact SVD directions as columns, so that
        ``response_matrix = left @ diag(singular_values_full) @ right.conj().T``. Left vectors
        span future responses and right vectors span combinations of histories.

    Raises:
        ValueError: If the response matrix has zero Frobenius norm, for which the normalized
            modal weights and their entropy are undefined.
    """
    left_singular_vectors, s_full, right_adjoint = np.linalg.svd(response_matrix, full_matrices=False)
    s_full = s_full.astype(np.float64)
    right_singular_vectors = right_adjoint.conj().T
    s = s_full.copy()
    if not s_full.size or s_full[0] <= 0.0:
        msg = "Response matrix must have nonzero Frobenius norm to define a normalized spectrum."
        raise ValueError(msg)
    scaled_squared = (s_full / s_full[0]) ** 2
    total_weight = float(np.sum(scaled_squared))

    if s.size and discarded_weight_threshold is not None:
        the = max(float(discarded_weight_threshold), 0.0)
        min_keep_eff = max(1, min(int(min_keep), int(s.size)))
        tail_cumsum = np.cumsum(scaled_squared[::-1])
        keep = s_full.size
        for idx, tail_weight in enumerate(tail_cumsum):
            if float(tail_weight / total_weight) > the:
                keep = max(s_full.size - idx, min_keep_eff)
                break
        else:
            keep = s_full.size
        s = s_full[:keep]

    kept_squared = scaled_squared[: s.size]
    q = np.clip(kept_squared / np.sum(kept_squared), 1e-30, 1.0)
    entropy = float(-np.sum(q * np.log(q)))
    effective_modes = float(np.exp(entropy))

    return {
        "entropy": entropy,
        "modes": effective_modes,
        "singular_values": s,
        "singular_values_full": s_full,
        "left_singular_vectors": left_singular_vectors,
        "right_singular_vectors": right_singular_vectors,
    }
