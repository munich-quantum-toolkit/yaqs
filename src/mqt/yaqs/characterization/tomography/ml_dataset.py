# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Convert :class:`TomographyEstimate` into a small supervised dataset (sequence → Bloch vector).

This does **not** reconstruct a comb; it treats the estimate as tabular data
``alphas ↦ ρ_out`` for proof-of-concept surrogate models.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import cast

import numpy as np

from .estimator_class import TomographyEstimate
from .predictor_encoding import unpack_rho8
from .surrogates import (
    TrajectoryCombSample,
    mean_frobenius_mse_rho8,
    mean_trace_distance_rho8,
    trajectory_batch_to_tensors,
)

# Pauli matrices (same convention as standard qubit Bloch sphere)
_SIGMAX = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_SIGMAY = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_SIGMAZ = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_ID = np.eye(2, dtype=np.complex128)


def bloch_vector_from_rho(rho: np.ndarray) -> np.ndarray:
    """Map 2×2 density matrix to Bloch coordinates ``(x, y, z)`` with ``x,y,z = Re Tr(ρ σ)``."""
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
    return np.array(
        [
            float(np.real(np.trace(r @ _SIGMAX))),
            float(np.real(np.trace(r @ _SIGMAY))),
            float(np.real(np.trace(r @ _SIGMAZ))),
        ],
        dtype=np.float32,
    )


def density_matrix_from_bloch(bloch: np.ndarray) -> np.ndarray:
    """``ρ = ½ (I + x σ_x + y σ_y + z σ_z)`` from Bloch vector ``(x,y,z)``."""
    x, y, z = (float(t) for t in np.asarray(bloch, dtype=np.float64).reshape(3))
    rho = 0.5 * (_ID + x * _SIGMAX + y * _SIGMAY + z * _SIGMAZ)
    return np.asarray(rho, dtype=np.complex128)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Trace distance ½ Σ |λᵢ| for Hermitian ``ρ − σ``."""
    d = np.asarray(rho, dtype=np.complex128) - np.asarray(sigma, dtype=np.complex128)
    d = 0.5 * (d + d.conj().T)
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(d))))


def clip_bloch_to_unit_ball(bloch: np.ndarray) -> np.ndarray:
    """``r = v / max(1, ‖v‖)`` (keep inside the closed unit ball)."""
    v = np.asarray(bloch, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n <= 1.0 or n < 1e-15:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def _weighted_rho_to_real8(rho: np.ndarray, w: float) -> np.ndarray:
    """8 floats: Re/Im of each entry of ``w * rho`` (row-major)."""
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2) * float(w)
    return np.array(
        [
            r[0, 0].real,
            r[0, 0].imag,
            r[0, 1].real,
            r[0, 1].imag,
            r[1, 0].real,
            r[1, 0].imag,
            r[1, 1].real,
            r[1, 1].imag,
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class TomographyMLDataset:
    """Rows are observed sequences only (exhaustive or positive-weight cells).

    Attributes:
        sequence_indices: Shape ``(N, k)``, ``int64``, entries in ``0..15``.
        bloch_target: Shape ``(N, 3)``, ``float32``, unweighted conditional state Bloch vector.
        sample_weight: Shape ``(N,)``, ``float32``, training loss weight (``1.0`` or estimator weight).
        estimator_weight: Shape ``(N,)``, ``float32``, always the estimate's ``weights[alphas]``.
        weighted_rho8: Optional shape ``(N, 8)`` — real/imag of ``w * rho`` (extension target).
        k: Number of intervention steps.
    """

    sequence_indices: np.ndarray
    bloch_target: np.ndarray
    sample_weight: np.ndarray
    estimator_weight: np.ndarray
    k: int
    weighted_rho8: np.ndarray | None = None


def tomography_estimate_to_ml_dataset(
    estimate: TomographyEstimate,
    *,
    weight_tol: float = 1e-30,
    use_estimator_weight: bool = True,
    include_weighted_rho8: bool = False,
) -> TomographyMLDataset:
    """Build numpy arrays from ``estimate.tensor`` / ``estimate.weights``.

    For each ``alphas`` in ``product(range(16), repeat=k)``, if
    ``estimate.weights[alphas] > weight_tol``, take
    ``ρ = tensor[(slice(None), *alphas)].reshape(2, 2)`` **without** multiplying by the weight.
    """
    if estimate.tensor is None or estimate.weights is None:
        msg = "TomographyEstimate must have tensor and weights."
        raise ValueError(msg)

    k = int(estimate.tensor.ndim - 1)
    seq_list: list[tuple[int, ...]] = []
    bloch_rows: list[np.ndarray] = []
    w_train: list[float] = []
    w_est: list[float] = []
    wr8_rows: list[np.ndarray] = []

    for alphas in itertools.product(range(16), repeat=k):
        w = float(estimate.weights[alphas])
        if w <= weight_tol:
            continue
        rho = np.asarray(
            estimate.tensor[(slice(None), *alphas)].reshape(2, 2),
            dtype=np.complex128,
        )
        rho_h = 0.5 * (rho + rho.conj().T)
        b = bloch_vector_from_rho(rho_h)
        seq_list.append(tuple(int(a) for a in alphas))
        bloch_rows.append(b)
        w_est.append(float(w))
        w_train.append(float(w) if use_estimator_weight else 1.0)
        if include_weighted_rho8:
            wr8_rows.append(_weighted_rho_to_real8(rho_h, w))

    if not seq_list:
        msg = "No sequences passed weight_tol filter."
        raise ValueError(msg)

    sequence_indices = np.array(seq_list, dtype=np.int64)
    bloch_target = np.stack(bloch_rows, axis=0).astype(np.float32)
    sample_weight = np.array(w_train, dtype=np.float32)
    estimator_weight = np.array(w_est, dtype=np.float32)
    wr8_arr = np.stack(wr8_rows, axis=0) if include_weighted_rho8 else None

    return TomographyMLDataset(
        sequence_indices=sequence_indices,
        bloch_target=bloch_target,
        sample_weight=sample_weight,
        estimator_weight=estimator_weight,
        k=k,
        weighted_rho8=wr8_arr,
    )


def build_rho_prev_rho_target(
    rho_0: np.ndarray,
    rho_seq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forcing tensors: ``rho_prev[:,0]=rho_0``, ``rho_prev[:,t]=rho_seq[:,t-1]``."""
    rho_0 = np.asarray(rho_0, dtype=np.float32)
    rho_seq = np.asarray(rho_seq, dtype=np.float32)
    n, k, d = rho_seq.shape
    if rho_0.shape != (n, d):
        msg = f"rho_0 must be (N, d_rho), got {rho_0.shape} vs rho_seq {rho_seq.shape}."
        raise ValueError(msg)
    rho_prev = np.zeros((n, k, d), dtype=np.float32)
    rho_prev[:, 0, :] = rho_0
    if k > 1:
        rho_prev[:, 1:, :] = rho_seq[:, :-1, :]
    return rho_prev, rho_seq


@dataclass(frozen=True)
class GridCoverageSummary:
    """How many of the ``16**k`` discrete sequences appear in the estimate."""

    k: int
    n_grid: int
    n_weight_positive: int
    n_weight_at_or_below_tol: int
    weight_tol: float
    n_positive_but_vanishing_rho: int


def summarize_grid_coverage(
    estimate: TomographyEstimate,
    *,
    k: int,
    weight_tol: float = 1e-30,
    rho_tol: float = 1e-14,
) -> GridCoverageSummary:
    """Count sequences on the full ``16^k`` grid vs ``weight_tol`` filtering.

    Rows are included in :func:`tomography_estimate_to_ml_dataset` iff
    ``weights[alphas] > weight_tol``. ``n_positive_but_vanishing_rho`` counts
    cells with ``weight > tol`` but ``‖ρ‖_F < rho_tol`` (unexpected data issue).
    """
    if estimate.tensor is None or estimate.weights is None:
        msg = "TomographyEstimate must have tensor and weights."
        raise ValueError(msg)
    k_est = int(estimate.tensor.ndim - 1)
    if k_est != k:
        msg = f"Expected tensor with k={k}, got k={k_est}."
        raise ValueError(msg)

    n_grid = 16**k
    n_pos = 0
    n_below = 0
    n_vanish = 0
    for alphas in itertools.product(range(16), repeat=k):
        w = float(estimate.weights[alphas])
        if w > weight_tol:
            n_pos += 1
            rho = np.asarray(
                estimate.tensor[(slice(None), *alphas)].reshape(2, 2),
                dtype=np.complex128,
            )
            if float(np.linalg.norm(rho, ord="fro")) < rho_tol:
                n_vanish += 1
        else:
            n_below += 1

    return GridCoverageSummary(
        k=k,
        n_grid=n_grid,
        n_weight_positive=n_pos,
        n_weight_at_or_below_tol=n_below,
        weight_tol=weight_tol,
        n_positive_but_vanishing_rho=n_vanish,
    )
