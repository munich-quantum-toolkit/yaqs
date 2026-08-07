# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Independent dense Schmidt-support projectors and localized actions.

Site ordering
-------------
Dense statevectors use C-order flatten of ``psi.reshape([d] * N)`` with
**site 0 as the slowest axis** (first axis / MSB). Gate and projector actions
follow the same convention. This is independent of YAQS ``MPS.to_vec()``
(LSB) ordering used only in the production-code checks.

Kronecker convention
--------------------
With bipartition matrix ``X`` from that flatten, reshape-and-contract applies
``P_L @ X @ P_R`` for ``P_R = V V†``. Dense embeddings therefore use
``kron(P_L, P_R.T)`` (and the analogous three-factor form) so both sides of
every locality identity share one convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (
    ABS_ACTION_FLOOR,
    CHI,
    GENERATOR_SEED,
    SIGMA_DISC_REL,
    SIGMA_MIN_REL,
    D,
    N,
    bond_profile,
)


@dataclass(frozen=True)
class SchmidtData:
    """Schmidt bases and spectra for one dense MPS statevector."""

    psi: np.ndarray
    n: int
    profile: list[int]
    lefts: dict[int, np.ndarray]
    rights: dict[int, np.ndarray]
    spectra: dict[int, np.ndarray]
    p_left: dict[int, np.ndarray]
    p_right: dict[int, np.ndarray]
    sigma_min_retained: float
    sigma_max_discarded: float


class FixtureError(RuntimeError):
    """Raised when an MPS fixture fails rank or conditioning checks."""


def make_generic_generator(seed: int = GENERATOR_SEED) -> np.ndarray:
    """Unit-spectral-norm traceless Hermitian 4×4 generator."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    h = (a + a.conj().T) / 2.0
    h -= np.trace(h) * np.eye(4, dtype=np.complex128) / 4.0
    spectral = float(np.linalg.norm(h, 2))
    if spectral <= 0.0:
        msg = "Generic generator has vanishing spectral norm"
        raise FixtureError(msg)
    return (h / spectral).astype(np.complex128)


def random_exact_rank_state(
    seed: int,
    *,
    n: int = N,
    chi: int = CHI,
    d: int = D,
) -> np.ndarray:
    """Contract a random complex MPS with exact bond profile and normalize."""
    rng = np.random.default_rng(seed)
    profile = bond_profile(n, chi)
    tensors: list[np.ndarray] = []
    for site in range(n):
        shape = (profile[site], d, profile[site + 1])
        tensors.append(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    psi = tensors[0]
    for tensor in tensors[1:]:
        psi = np.tensordot(psi, tensor, axes=([-1], [0]))
    vec = psi.reshape(-1).astype(np.complex128)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        msg = f"Vanishing MPS norm for seed={seed}"
        raise FixtureError(msg)
    return vec / norm


def compute_schmidt(psi: np.ndarray, *, n: int = N, chi: int = CHI) -> SchmidtData:
    """Dense SVDs at every cut; verify exact ranks and conditioning."""
    profile = bond_profile(n, chi)
    lefts: dict[int, np.ndarray] = {}
    rights: dict[int, np.ndarray] = {}
    spectra: dict[int, np.ndarray] = {}
    p_left: dict[int, np.ndarray] = {0: np.ones((1, 1), dtype=np.complex128)}
    p_right: dict[int, np.ndarray] = {n: np.ones((1, 1), dtype=np.complex128)}

    sigma_min_retained = np.inf
    sigma_max_discarded = 0.0
    sigma_max_retained = 0.0

    for cut in range(1, n):
        matrix = psi.reshape(2**cut, 2 ** (n - cut))
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = profile[cut]
        numerical_rank = int(np.sum(s > 1e-12 * s[0])) if s.size and s[0] > 0 else 0
        if numerical_rank != rank:
            msg = (
                f"Schmidt rank mismatch at cut {cut}: "
                f"numerical={numerical_rank}, intended={rank}, s[:{rank + 1}]={s[: rank + 1]}"
            )
            raise FixtureError(msg)
        if len(s) < rank or s[rank - 1] <= 0.0:
            msg = f"Insufficient singular values at cut {cut}"
            raise FixtureError(msg)

        lefts[cut] = u[:, :rank]
        rights[cut] = vh[:rank, :].conj().T
        spectra[cut] = s
        p_left[cut] = lefts[cut] @ lefts[cut].conj().T
        p_right[cut] = rights[cut] @ rights[cut].conj().T

        sigma_max_retained = max(sigma_max_retained, float(s[0]))
        sigma_min_retained = min(sigma_min_retained, float(s[rank - 1]))
        discarded = float(s[rank]) if len(s) > rank else 0.0
        sigma_max_discarded = max(sigma_max_discarded, discarded)

    if sigma_max_retained <= 0.0:
        msg = "All retained singular values vanished"
        raise FixtureError(msg)
    if sigma_min_retained / sigma_max_retained < SIGMA_MIN_REL:
        msg = (
            f"Retained Schmidt floor too small: "
            f"{sigma_min_retained / sigma_max_retained:.3e} < {SIGMA_MIN_REL:.0e}"
        )
        raise FixtureError(msg)
    if sigma_max_discarded / sigma_max_retained > SIGMA_DISC_REL:
        msg = (
            f"Discarded Schmidt mass too large: "
            f"{sigma_max_discarded / sigma_max_retained:.3e} > {SIGMA_DISC_REL:.0e}"
        )
        raise FixtureError(msg)

    return SchmidtData(
        psi=psi,
        n=n,
        profile=profile,
        lefts=lefts,
        rights=rights,
        spectra=spectra,
        p_left=p_left,
        p_right=p_right,
        sigma_min_retained=float(sigma_min_retained),
        sigma_max_discarded=float(sigma_max_discarded),
    )


def apply_two_site_op(
    psi: np.ndarray,
    op4: np.ndarray,
    q0: int,
    q1: int,
    *,
    n: int | None = None,
) -> np.ndarray:
    """Apply a 4×4 operator on axes ``(q0, q1)`` without embedding a full matrix."""
    if q0 == q1:
        msg = "Two-site operator requires distinct sites"
        raise ValueError(msg)
    length = int(np.log2(psi.size)) if n is None else n
    tensor = psi.reshape([2] * length)
    op = np.asarray(op4, dtype=np.complex128).reshape(2, 2, 2, 2)
    tmp = np.tensordot(op, tensor, axes=([2, 3], [q0, q1]))
    remaining = [i for i in range(length) if i not in {q0, q1}]
    dest = [0] * length
    dest[q0] = 0
    dest[q1] = 1
    for offset, site in enumerate(remaining):
        dest[site] = 2 + offset
    return np.transpose(tmp, dest).reshape(-1)


def pauli_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def apply_xx(psi: np.ndarray, q0: int, q1: int, *, n: int) -> np.ndarray:
    """Apply ``X_q0 X_q1`` to a dense statevector."""
    x = pauli_x()
    tensor = psi.reshape([2] * n)
    tensor = np.tensordot(x, tensor, axes=([1], [q0]))
    tensor = np.moveaxis(tensor, 0, q0)
    tensor = np.tensordot(x, tensor, axes=([1], [q1]))
    tensor = np.moveaxis(tensor, 0, q1)
    return tensor.reshape(-1)


def _kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(a, np.kron(b, c))


def dense_s(sch: SchmidtData, k: int) -> np.ndarray:
    """Dense matrix for ``S_k = P_L[k] ⊗ I_2 ⊗ P_R[k+1]``.

    Under C-order vectorization with Schmidt row coefficients from ``vh``,
    the right factor enters as ``P_R.T`` so that the matrix action matches
    ``P_L @ X @ P_R`` used by the reshape-and-contract routines.
    """
    return _kron3(
        sch.p_left[k],
        np.eye(2, dtype=np.complex128),
        sch.p_right[k + 1].T,
    )


def dense_b(sch: SchmidtData, cut: int) -> np.ndarray:
    """Dense matrix for ``B_c = P_L[c] ⊗ P_R[c]`` (right factor as ``P_R.T``)."""
    return np.kron(sch.p_left[cut], sch.p_right[cut].T)


def dense_k(sch: SchmidtData, k: int) -> np.ndarray:
    """Dense matrix for ``K_k = P_L[k] ⊗ I_4 ⊗ P_R[k+2]`` (right factor as ``P_R.T``)."""
    return _kron3(
        sch.p_left[k],
        np.eye(4, dtype=np.complex128),
        sch.p_right[k + 2].T,
    )


def build_p1_full(sch: SchmidtData) -> np.ndarray:
    """Full-chain fixed-rank projector ``P^{[1]}``."""
    dim = 2**sch.n
    p = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(sch.n):
        p += dense_s(sch, k)
    for cut in range(1, sch.n):
        p -= dense_b(sch, cut)
    return p


def build_p2_full(sch: SchmidtData) -> np.ndarray:
    """Full-chain two-site projector ``P^{[2]}``."""
    dim = 2**sch.n
    p = np.zeros((dim, dim), dtype=np.complex128)
    for k in range(sch.n - 1):
        p += dense_k(sch, k)
    for k in range(1, sch.n - 1):
        p -= dense_s(sch, k)
    return p


def projector_diagnostics(p: np.ndarray) -> dict[str, float]:
    """Relative Hermiticity and idempotence residuals."""
    fro = float(np.linalg.norm(p, "fro"))
    if fro <= 0.0:
        return {"hermitian_rel": np.inf, "idempotent_rel": np.inf, "frobenius": 0.0}
    herm = float(np.linalg.norm(p - p.conj().T, "fro")) / fro
    idem = float(np.linalg.norm(p @ p - p, "fro")) / fro
    return {"hermitian_rel": herm, "idempotent_rel": idem, "frobenius": fro}


def apply_s_contract(x: np.ndarray, k: int, sch: SchmidtData) -> np.ndarray:
    """Reshape-and-contract action of ``S_k`` (no full matrix)."""
    n = sch.n
    t = x.reshape(2**k, 2 * 2 ** (n - k - 1))
    if k > 0:
        u = sch.lefts[k]
        t = u @ (u.conj().T @ t)
    t = t.reshape(2**k * 2, 2 ** (n - k - 1))
    if k < n - 1:
        v = sch.rights[k + 1]
        # Matches ``P_L @ X @ P_R`` with ``P_R = V V†`` (validate_locality convention).
        t = (t @ v) @ v.conj().T
    return t.reshape(-1)


def apply_b_contract(x: np.ndarray, cut: int, sch: SchmidtData) -> np.ndarray:
    """Reshape-and-contract action of ``B_c``."""
    n = sch.n
    m = x.reshape(2**cut, 2 ** (n - cut))
    u = sch.lefts[cut]
    v = sch.rights[cut]
    m = u @ (u.conj().T @ m)
    m = (m @ v) @ v.conj().T
    return m.reshape(-1)


def apply_k_contract(x: np.ndarray, k: int, sch: SchmidtData) -> np.ndarray:
    """Reshape-and-contract action of ``K_k``."""
    n = sch.n
    t = x.reshape(2**k, 4 * 2 ** (n - k - 2))
    if k > 0:
        u = sch.lefts[k]
        t = u @ (u.conj().T @ t)
    t = t.reshape(2**k * 4, 2 ** (n - k - 2))
    if k < n - 2:
        v = sch.rights[k + 2]
        t = (t @ v) @ v.conj().T
    return t.reshape(-1)


def localized_p1_action(x: np.ndarray, q0: int, q1: int, sch: SchmidtData) -> np.ndarray:
    """Fixed-rank windowed action on ``X = H|ψ⟩``."""
    out = np.zeros_like(x)
    for k in range(q0, q1 + 1):
        out += apply_s_contract(x, k, sch)
    for cut in range(q0 + 1, q1 + 1):
        out -= apply_b_contract(x, cut, sch)
    return out


def localized_p2_action(x: np.ndarray, q0: int, q1: int, sch: SchmidtData) -> np.ndarray:
    """Two-site enlarged-window action on ``X = H|ψ⟩``."""
    n = sch.n
    k_lo = max(0, q0 - 1)
    k_hi = min(q1, n - 2)
    s_lo = max(1, q0)
    s_hi = min(q1, n - 2)
    out = np.zeros_like(x)
    for k in range(k_lo, k_hi + 1):
        out += apply_k_contract(x, k, sch)
    for k in range(s_lo, s_hi + 1):
        out -= apply_s_contract(x, k, sch)
    return out


def fixed_rank_window(q0: int, q1: int) -> tuple[int, int]:
    return q0, q1


def two_site_window(q0: int, q1: int, *, n: int = N) -> tuple[int, int]:
    return max(0, q0 - 1), min(n - 1, q1 + 1)


def relative_residual(full: np.ndarray, windowed: np.ndarray) -> tuple[float, float, float]:
    """Return ``(abs_res, rel_res, full_norm)``."""
    abs_res = float(np.linalg.norm(full - windowed))
    full_norm = float(np.linalg.norm(full))
    rel = abs_res / full_norm if full_norm > 0.0 else np.inf
    return abs_res, rel, full_norm


def assert_nonvacuous_action(full_action: np.ndarray, x: np.ndarray) -> None:
    """Reject fixtures whose full projected action is nearly zero."""
    x_norm = float(np.linalg.norm(x))
    action_norm = float(np.linalg.norm(full_action))
    if x_norm <= 0.0 or action_norm < ABS_ACTION_FLOOR * x_norm:
        msg = (
            f"Vacuous projected action: ||P X||={action_norm:.3e}, "
            f"||X||={x_norm:.3e}, floor={ABS_ACTION_FLOOR:.0e}||X||"
        )
        raise FixtureError(msg)


def infidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized state infidelity ``1 - |⟨a|b⟩|² / (‖a‖²‖b‖²)``."""
    aa = float(np.real(np.vdot(a, a)))
    bb = float(np.real(np.vdot(b, b)))
    overlap = abs(np.vdot(a, b)) ** 2
    return float(1.0 - overlap / (aa * bb))
