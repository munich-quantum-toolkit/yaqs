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


def canonicalize_upsilon(
    U: NDArray[np.complex128],
    *,
    hermitize: bool = True,
    psd_project: bool = False,
    normalize_trace: bool = True,
    psd_tol: float = 1e-12,
) -> NDArray[np.complex128]:
    """Canonicalize a comb Choi operator so that comparisons are meaningful.

    Steps:
    - Hermitize
    - Optional PSD projection
    - Optional trace normalization
    """
    U = U.copy()

    if hermitize:
        U = 0.5 * (U + U.conj().T)

    if psd_project:
        w, V = np.linalg.eigh(U)
        w = np.clip(w, 0.0, None)
        U = (V * w) @ V.conj().T

    if normalize_trace:
        tr = np.trace(U)
        if abs(tr) > 1e-15:
            U = U / tr

    return U


def reduced_upsilon(
    U: NDArray[np.complex128],
    k: int,
    keep_last_m: int = 1,
) -> NDArray[np.complex128]:
    """Reduce Υ by tracing out all but the last m past legs.

    Υ has dimension (2·4^k, 2·4^k) with index structure
    (output ⊗ past_0 ⊗ … ⊗ past_{k-1}).  This function traces out
    past_0 … past_{k-m-1} and returns the reduced operator of shape
    (2·4^m, 2·4^m).  For m=1 the result is always 8×8, independent of k,
    making it a constant-size metric for k-scaling experiments.
    """
    if keep_last_m > k:
        raise ValueError(f"keep_last_m={keep_last_m} > k={k}")
    if keep_last_m <= 0:
        raise ValueError(f"keep_last_m must be >= 1, got {keep_last_m}")

    dim_expected = 2 * 4**k
    if U.shape != (dim_expected, dim_expected):
        raise ValueError(
            f"Expected U with shape ({dim_expected},{dim_expected}) for k={k}, got {U.shape}"
        )

    dim_m = 2 * (4**keep_last_m)
    dim_traced = 4 ** (k - keep_last_m)

    if dim_traced == 1:
        # Nothing to trace: keep_last_m == k
        return U.reshape(dim_m, dim_m)

    U6 = U.reshape(2, dim_traced, 4**keep_last_m, 2, dim_traced, 4**keep_last_m)
    reduced = np.einsum("iabjac->ibjc", U6)  # (2, 4^m, 2, 4^m)

    return reduced.reshape(dim_m, dim_m)


def _partial_trace_dense(r: NDArray[np.complex128], dims_: list[int], keep: list[int]) -> NDArray[np.complex128]:
    """Compute partial trace of a dense operator."""
    keep = sorted(keep)
    n = len(dims_)
    if any(i < 0 or i >= n for i in keep):
        raise ValueError("keep indices out of range")

    reshaped = r.reshape(*(dims_ + dims_))  # (i0..in-1, j0..jn-1)
    trace_out = [i for i in range(n) if i not in keep]
    perm = keep + trace_out
    reshaped = reshaped.transpose(*(perm + [i + n for i in perm]))

    dim_keep = int(np.prod([dims_[i] for i in keep])) if keep else 1
    dim_out = int(np.prod([dims_[i] for i in trace_out])) if trace_out else 1
    reshaped = reshaped.reshape(dim_keep, dim_out, dim_keep, dim_out)

    return np.einsum("a b c b -> a c", reshaped)


def _entropy_dense(r: NDArray[np.complex128], base: int = 2) -> float:
    """Compute von Neumann entropy of a dense density matrix."""
    log_base = np.log(base)
    rH = 0.5 * (r + r.conj().T)
    tr = np.trace(rH)
    if abs(tr) < 1e-15:
        return 0.0
    rH = rH / tr
    evals = np.linalg.eigvalsh(rH).real
    evals = np.clip(evals, 0.0, 1.0)
    nz = evals[evals > 1e-15]
    if nz.size == 0:
        return 0.0
    return float(-(nz * (np.log(nz) / log_base)).sum())


def comb_qmi_from_upsilon_dense(
    Upsilon: NDArray[np.complex128],
    base: int = 2,
    past: str = "all",
    check_psd: bool = False,
    assume_canonical: bool = False,
) -> float:
    """Quantum mutual information I(F : P) from a dense comb Choi operator Υ."""
    if assume_canonical:
        rho = Upsilon
    else:
        U = 0.5 * (Upsilon + Upsilon.conj().T)
        if check_psd:
            lam_min = float(np.linalg.eigvalsh(U).min().real)
            if lam_min < -1e-9:
                raise ValueError(f"Reconstructed Υ not PSD (min eigenvalue {lam_min:.3e}).")
        tr = np.trace(U)
        rho = U / tr if abs(tr) > 1e-15 else U

    size = rho.shape[0]
    k_steps = int(np.round(np.log2(size / 2) / 2))
    dims = [2] + [4] * k_steps

    if past == "all":
        keep_P = list(range(1, k_steps + 1))
    elif past == "last":
        keep_P = [k_steps]
    elif past == "first":
        keep_P = [1]
    else:
        raise ValueError(f"Unknown past='{past}'.")

    rho_F = _partial_trace_dense(rho, dims, keep=[0])
    rho_P = _partial_trace_dense(rho, dims, keep=keep_P)

    return _entropy_dense(rho_P, base) + _entropy_dense(rho_F, base) - _entropy_dense(rho, base)


def comb_cmi_from_upsilon_dense(
    Upsilon: NDArray[np.complex128],
    base: int = 2,
    check_psd: bool = False,
    assume_canonical: bool = False,
) -> float:
    """Conditional Mutual Information I(F : P_{<k} | P_k) from dense Upsilon."""
    if assume_canonical:
        rho = Upsilon
    else:
        U = 0.5 * (Upsilon + Upsilon.conj().T)
        tr = np.trace(U)
        rho = U / tr if abs(tr) > 1e-15 else U

    size = rho.shape[0]
    k_steps = int(np.round(np.log2(size / 2) / 2))
    dims = [2] + [4] * k_steps

    if k_steps < 2:
        return 0.0

    idx_F = [0]
    idx_Pk = [k_steps]
    idx_P_less = list(range(1, k_steps))

    rho_FPk = _partial_trace_dense(rho, dims, keep=idx_F + idx_Pk)
    rho_P = _partial_trace_dense(rho, dims, keep=idx_P_less + idx_Pk)
    rho_Pk = _partial_trace_dense(rho, dims, keep=idx_Pk)

    return (
        _entropy_dense(rho_FPk, base)
        + _entropy_dense(rho_P, base)
        - _entropy_dense(rho_Pk, base)
        - _entropy_dense(rho, base)
    )


def comb_cmi_conditional_from_upsilon_dense(
    Upsilon: NDArray[np.complex128],
    base: int = 2,
    A: str = "first",
    B: str = "final",
    C: str = "last",
    normalize: bool = True,
    check_psd: bool = True,
) -> float:
    """Conditional mutual information I(A:B|C) from dense Υ.

    Subsystem names: "final" (F, dim 2), "first" (step 1), "last" (step k).
    Default I(first : final | last).
    """
    U = 0.5 * (Upsilon + Upsilon.conj().T)
    if check_psd:
        w, V = np.linalg.eigh(U)
        w = np.clip(w, 0.0, None)
        U = (V * w) @ V.conj().T
    if normalize:
        tr = np.trace(U)
        if abs(tr) < 1e-15:
            return 0.0
        rho = U / tr
    else:
        rho = U

    size = rho.shape[0]
    k_steps = int(np.round(np.log2(size / 2) / 2))
    if k_steps < 2:
        return 0.0
    dims = [2] + [4] * k_steps
    idx_final, idx_first, idx_last = 0, 1, k_steps

    def _idx(which: str) -> int:
        if which == "final":
            return idx_final
        if which == "first":
            return idx_first
        if which == "last":
            return idx_last
        raise ValueError(f"Unknown subsystem '{which}'. Use 'final', 'first', or 'last'.")

    iA, iB, iC = _idx(A), _idx(B), _idx(C)
    if len({iA, iB, iC}) != 3:
        raise ValueError("A, B, C must refer to three distinct subsystems.")

    rho_AC = _partial_trace_dense(rho, dims, keep=[iA, iC])
    rho_BC = _partial_trace_dense(rho, dims, keep=[iB, iC])
    rho_C = _partial_trace_dense(rho, dims, keep=[iC])
    return (
        _entropy_dense(rho_AC, base)
        + _entropy_dense(rho_BC, base)
        - _entropy_dense(rho_C, base)
        - _entropy_dense(rho, base)
    )

