# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Sampling utilities and data containers for process tomography."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class SequenceData:
    """Internal container for Stage-A tomography estimation results (Discrete)."""

    sequences: list[tuple[int, ...]]
    outputs: list[np.ndarray]  # (2, 2) density matrices
    weights: list[float]
    choi_basis: list[np.ndarray]
    choi_indices: list[tuple[int, int]]
    choi_duals: list[np.ndarray]
    timesteps: list[float]


@dataclass
class SamplingData:
    """Internal container for Stage-A tomography estimation results (Continuous)."""

    outputs: list[np.ndarray]  # (2, 2) final density matrices
    dual_ops: list[list[np.ndarray]]  # [N][k] list of 4x4 dual frame operators
    weights: list[float]
    timesteps: list[float]


def _enumerate_sequences(k: int) -> list[tuple[int, ...]]:
    """Iterate over all 16^k basis sequences as a list of ``tuple[int, ...]`` (deterministic)."""
    import itertools
    return list(itertools.product(range(16), repeat=k))


def _sample_haar_pure_state(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Sample a 2-dim complex vector from the Haar measure (uniform on sphere)."""
    z = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    return z / np.linalg.norm(z)


def _sample_local_measurement_exact(
    rho_0: NDArray[np.complex128],
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Sample a qubit state from the exact local proposal density q(psi) propto <psi|rho_0|psi>."""
    tr_rho = float(np.trace(rho_0).real)
    if tr_rho < 1e-18:
        return _sample_haar_pure_state(rng)

    rho = rho_0 / tr_rho
    evals, evecs = np.linalg.eigh(rho)
    idx = np.argsort(evals.real)[::-1]
    lam1, lam2 = evals.real[idx]
    v1, v2 = evecs[:, idx[0]], evecs[:, idx[1]]
    diff = lam1 - lam2
    sum_l = lam1 + lam2

    u = rng.random()
    if abs(diff) < 1e-12:
        t = u
    else:
        a, b, c = 0.5 * diff, lam2, -0.5 * u * sum_l
        disc = b**2 - 4.0 * a * c
        t = (-b + np.sqrt(max(0.0, disc))) / (2.0 * a)

    t = np.clip(t, 0.0, 1.0)
    phi = 2.0 * np.pi * rng.random()
    alpha = np.sqrt(t) * np.exp(1j * phi)
    beta = np.sqrt(1.0 - t)
    return alpha * v1 + beta * v2


def _get_haar_rho_dual(psi: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Dual frame operator D = 2*(3|psi><psi| - I) for a Haar pure state."""
    rho = np.outer(psi, psi.conj())
    return 2.0 * (3.0 * rho - np.eye(2, dtype=np.complex128))


def _sample_random_intervention_sequence(
    k: int, rng: np.random.Generator
) -> tuple[list[tuple[NDArray[np.complex128], NDArray[np.complex128]]], list[NDArray[np.complex128]]]:
    """Sample k pairs of (psi_meas, psi_prep) and their 4x4 dual operators."""
    psi_pairs = []
    dual_ops = []
    for _ in range(k):
        pm = _sample_haar_pure_state(rng)
        pp = _sample_haar_pure_state(rng)
        psi_pairs.append((pm, pp))
        d_q = _get_haar_rho_dual(pp)
        p_mat = np.outer(pm, pm.conj())
        d_pt = 2.0 * (3.0 * p_mat.T - np.eye(2, dtype=np.complex128))
        dual_ops.append(np.kron(d_q, d_pt).T)
    return psi_pairs, dual_ops


def _continuous_dual_step(
    psi_meas: NDArray[np.complex128], psi_prep: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Compute the 4×4 dual operator for one continuous intervention step."""
    d_q = _get_haar_rho_dual(psi_prep)
    p_mat = np.outer(psi_meas, psi_meas.conj())
    d_pt = 2.0 * (3.0 * p_mat.T - np.eye(2, dtype=np.complex128))
    return np.kron(d_q, d_pt).T


def _logsumexp(x: NDArray[np.float64]) -> float:
    """Numerically stable log(sum(exp(x)))."""
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return -np.inf
    return float(x_max + np.log(np.sum(np.exp(x - x_max))))


def _normalize_log_weights(log_weights: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
    """Return normalized linear weights and log(sum w)."""
    log_w_sum = _logsumexp(log_weights)
    if not np.isfinite(log_w_sum):
        n = log_weights.shape[0]
        return np.full(n, 1.0 / n, dtype=np.float64), -np.inf
    w_norm = np.exp(log_weights - log_w_sum)
    return w_norm.astype(np.float64), log_w_sum
