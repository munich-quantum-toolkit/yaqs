# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Encoding utilities shared by process-tensor tomography and surrogates.

This includes:
- fixed-basis Choi feature encodings (used by tomography basis code and surrogate utilities)
- single-qubit density matrix encodings (rho8) and normalization helpers
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _flatten_choi4_to_real32(j: np.ndarray) -> np.ndarray:
    """Flatten a 4x4 Choi matrix into 32 real features.

    Args:
        j: Complex 4x4 Choi matrix.

    Returns:
        A float32 vector of shape ``(32,)`` with interleaved real/imag parts (row-major).
    """
    m = np.asarray(j, dtype=np.complex128).reshape(4, 4)
    flat = m.reshape(-1)
    interleaved = np.stack([flat.real, flat.imag], axis=-1).astype(np.float32)
    return interleaved.reshape(-1)


def build_choi_feature_table(choi_matrices: list[np.ndarray]) -> np.ndarray:
    """Build a feature table for a fixed 16-letter Choi basis.

    Args:
        choi_matrices: List of 16 complex 4x4 Choi matrices.

    Returns:
        Float32 array of shape ``(16, 32)`` with one feature row per basis index.
    """
    rows = [_flatten_choi4_to_real32(c) for c in choi_matrices]
    return np.stack(rows, axis=0)


def _normalize_density_like_densecomb(rho: np.ndarray) -> np.ndarray:
    """Project a 2x2 matrix onto a physical density matrix.

    Args:
        rho: Complex 2x2 matrix (not necessarily physical).

    Returns:
        A Hermitian, PSD, trace-1 2x2 density matrix.
    """
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 1e-12:
        rho /= tr

    w, V = np.linalg.eigh(rho)
    w = np.clip(w, 0.0, None)
    rho = (V * w) @ V.conj().T

    tr2 = np.trace(rho)
    if abs(tr2) > 1e-15:
        rho /= tr2
    return rho


def pack_rho8(rho: np.ndarray) -> np.ndarray:
    """Pack a 2x2 density matrix into 8 floats (rho8 encoding).

    Args:
        rho: Complex 2x2 matrix.

    Returns:
        Float32 vector of shape ``(8,)`` with interleaved real/imag parts (row-major).
    """
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
    """Unpack a rho8 vector back into a Hermitian 2x2 matrix.

    Args:
        y: Float vector of shape ``(8,)``.

    Returns:
        Hermitian complex 2x2 matrix (not normalized / projected).
    """
    t = np.asarray(y, dtype=np.float64).reshape(8)
    rho = np.array(
        [
            [t[0] + 1j * t[1], t[2] + 1j * t[3]],
            [t[4] + 1j * t[5], t[6] + 1j * t[7]],
        ],
        dtype=np.complex128,
    )
    return 0.5 * (rho + rho.conj().T)


def normalize_rho_from_backend_output(rho_final: Any) -> np.ndarray:
    """Normalize a backend 2x2 output into a physical density matrix.

    This applies hermitization and trace normalization, then uses a conservative fast-path check
    to skip PSD projection for already-near-physical outputs; otherwise it projects onto the PSD cone.

    Args:
        rho_final: Backend output convertible to a 2x2 complex array.

    Returns:
        Hermitian, PSD, trace-1 2x2 density matrix.
    """
    rho_h = np.asarray(rho_final, dtype=np.complex128).reshape(2, 2)
    rho_h = 0.5 * (rho_h + rho_h.conj().T)
    tr = np.trace(rho_h)
    if abs(tr) > 1e-12:
        rho_h /= tr
    else:
        return np.zeros((2, 2), dtype=np.complex128)

    # Fast path: for near-physical outputs, avoid full PSD projection (eigh) and only do a cheap check.
    # This is conservative: any small negativity falls back to the projection path.
    eps = 1e-12
    w = np.linalg.eigvalsh(rho_h).real
    if float(w.min()) >= -eps:
        tr2 = np.trace(rho_h)
        if abs(tr2) > 1e-15:
            rho_h /= tr2
        return rho_h

    return _normalize_density_like_densecomb(rho_h)
