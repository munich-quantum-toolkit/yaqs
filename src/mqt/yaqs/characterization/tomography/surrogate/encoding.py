# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Surrogate-only encodings and normalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def _normalize_density_like_densecomb(rho: np.ndarray) -> np.ndarray:
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


def normalize_rho_from_backend_output(rho_final: Any) -> np.ndarray:
    """Normalize a raw backend 2x2 output into a physical density matrix."""
    rho_h = np.asarray(rho_final, dtype=np.complex128).reshape(2, 2)
    rho_h = 0.5 * (rho_h + rho_h.conj().T)
    return _normalize_density_like_densecomb(rho_h)

