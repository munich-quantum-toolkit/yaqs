"""Tomography metrics and dense-comb utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rel_fro_error(A: NDArray[np.complex128], B: NDArray[np.complex128]) -> float:
    """Relative Frobenius error between two matrices."""
    num = np.linalg.norm(A - B, "fro")
    den = np.linalg.norm(B, "fro")
    return float(num / max(den, 1e-15))


def trace_distance(rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
    """Trace distance between two density matrices."""
    X = rho - sigma
    X = 0.5 * (X + X.conj().T)
    evals = np.linalg.eigvalsh(X)
    return float(0.5 * np.sum(np.abs(evals)))

