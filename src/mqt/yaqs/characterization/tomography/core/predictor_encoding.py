# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Real-vector features for neural predictors on single-qubit states and Choi maps.

Maps 2x2 density matrices to 8 real floats (row-major Re/Im per entry) and 4x4 Choi
matrices to 32 reals, plus normalization consistent with
:class:`~mqt.yaqs.characterization.tomography.exact.combs.DenseComb`.
Related batch metrics use :func:`mqt.yaqs.characterization.tomography.core.metrics.trace_distance`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

CHOI_FLAT_DIM = 32  # 4x4 complex -> 16 complex -> 32 reals (Re/Im, row-major)


def pack_rho8(rho: np.ndarray) -> np.ndarray:
    """Unweighted 2x2 density matrix as 8 floats (Re/Im, row-major)."""
    r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
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


def unpack_rho8(y: np.ndarray) -> np.ndarray:
    """Convert 8 reals back to a Hermitian 2x2 density matrix."""
    t = np.asarray(y, dtype=np.float64).reshape(8)
    rho = np.array(
        [
            [t[0] + 1j * t[1], t[2] + 1j * t[3]],
            [t[4] + 1j * t[5], t[6] + 1j * t[7]],
        ],
        dtype=np.complex128,
    )
    return 0.5 * (rho + rho.conj().T)


def normalize_rho_like_densecomb(rho: np.ndarray) -> np.ndarray:
    """Match DenseComb.predict convention: Hermitize + trace/PSD projection."""
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 1e-12:
        rho = rho / tr

    w, V = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    rho = (V * w) @ V.conj().T

    tr2 = np.trace(rho)
    if abs(tr2) > 1e-15:
        rho = rho / tr2
    return rho


def normalize_rho_from_backend_output(rho_final: Any) -> np.ndarray:
    """DenseComb-like normalization for backend outputs."""
    rho_h = np.asarray(rho_final, dtype=np.complex128).reshape(2, 2)
    rho_h = 0.5 * (rho_h + rho_h.conj().T)
    return normalize_rho_like_densecomb(rho_h)


def flatten_choi4_to_real32(j: np.ndarray) -> np.ndarray:
    """Vectorize 4x4 complex Choi matrix to 32 floats (Re/Im, row-major)."""
    m = np.asarray(j, dtype=np.complex128).reshape(4, 4)
    flat = m.reshape(-1)
    interleaved = np.stack([flat.real, flat.imag], axis=-1).astype(np.float32)
    return interleaved.reshape(-1)


def build_choi_feature_table(choi_matrices: list[np.ndarray]) -> np.ndarray:
    """Return array of shape (16, 32): one feature row per fixed-basis index."""
    rows = [flatten_choi4_to_real32(c) for c in choi_matrices]
    return np.stack(rows, axis=0)


def concat_choi_features(alphas: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Concatenate ``table[alpha_t]`` for each step; table is (16, 32)."""
    a = np.asarray(alphas, dtype=np.int64).reshape(-1)
    parts = [table[int(ai)] for ai in a]
    return np.concatenate(parts, axis=0).astype(np.float32)


def random_density_matrix(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-inspired random physical 2x2 density matrix (Ginibre / normalize)."""
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = a @ a.conj().T
    tr = float(np.trace(rho).real)
    rho = rho / max(tr, 1e-15)
    return 0.5 * (rho + rho.conj().T)


def state_prep_map_from_rho(rho_in: np.ndarray) -> Any:
    """CP map ``sigma -> Tr(sigma) * rho_in`` (trace-preserving replacement preparation)."""
    rho0 = 0.5 * (rho_in + rho_in.conj().T)
    tr = np.trace(rho0)
    if abs(tr) > 1e-12:
        rho0 = rho0 / tr
    return lambda rho: np.trace(rho) * rho0


def random_pure_state(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random single-qubit pure state vector |psi> (shape (2,), complex128)."""
    v = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    return (v / n).astype(np.complex128)


def random_rank1_projector(rng: np.random.Generator) -> np.ndarray:
    """Sample rank-1 projector |psi><psi| as a 2x2 complex matrix."""
    psi = random_pure_state(rng)
    return np.outer(psi, psi.conj()).astype(np.complex128)


def sample_random_intervention(
    rng: np.random.Generator,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """Sample one continuous CP intervention map and its Choi matrix.

    Returns:
        emap: callable map ``emap(rho) = Tr(E @ rho) * rho_prep``
        rho_prep: 2x2 rank-1 density matrix
        E: 2x2 rank-1 measurement effect
        J: 4x4 Choi matrix ``kron(rho_prep, E.T)``
    """
    rho_prep = random_rank1_projector(rng)
    E = random_rank1_projector(rng)

    def emap(rho: np.ndarray, rho_prep: np.ndarray = rho_prep, E: np.ndarray = E) -> np.ndarray:
        r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
        return np.trace(E @ r) * rho_prep
    # Attach components for callers that need a backend-equivalent representation.
    setattr(emap, "rho_prep", rho_prep)
    setattr(emap, "effect", E)

    J = np.kron(rho_prep, E.T).astype(np.complex128)
    return emap, rho_prep, E, J


def sample_random_intervention_sequence(
    k: int,
    rng: np.random.Generator,
) -> tuple[list[Any], np.ndarray]:
    """Sample k fresh continuous interventions and return maps + per-step Choi features.

    Returns:
        maps: length-k list of callables ``rho -> Tr(E_t rho) * rho_prep_t``
        choi_features: shape ``(k, 32)``, each row from :func:`flatten_choi4_to_real32`
    """
    maps: list[Any] = []
    rows: list[np.ndarray] = []
    for _ in range(int(k)):
        emap, _rho_prep, _E, J = sample_random_intervention(rng)
        maps.append(emap)
        rows.append(flatten_choi4_to_real32(J))
    return maps, np.stack(rows, axis=0).astype(np.float32)
