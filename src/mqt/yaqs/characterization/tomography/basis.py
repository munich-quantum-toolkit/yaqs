# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Choi basis and dual frame utilities for process tomography."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _haar_unitary_2x2(rng: np.random.Generator) -> NDArray[np.complex128]:
    z = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    q, r = np.linalg.qr(z)
    diag = np.diag(r)
    phases = diag / np.where(np.abs(diag) > 0.0, np.abs(diag), 1.0)
    return (q * phases.conj()).astype(np.complex128)


def get_basis_states(
    *,
    basis: Literal["standard", "random_low_overlap", "tetrahedral", "random_unitary"] = "tetrahedral",
    seed: int | None = None,
) -> list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]]:
    """Return the 4 single-qubit basis states used to build the 16 CP-map basis.

    - standard: |0>, |1>, |+>, |i+>
    - random_low_overlap: pick 4 low-overlap states from a pool of 10 Haar-random
      unitaries applied to |0>, using a simple greedy max-overlap minimization.
    - tetrahedral: deterministic 4-state tetrahedral (SIC-like) frame on the Bloch sphere.
    """
    if basis == "standard":
        psi_0 = np.array([1, 0], dtype=np.complex128)
        psi_1 = np.array([0, 1], dtype=np.complex128)
        psi_plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
        psi_i_plus = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
        states = [("zeros", psi_0), ("ones", psi_1), ("x+", psi_plus), ("y+", psi_i_plus)]
        return [(name, psi, np.outer(psi, psi.conj())) for name, psi in states]

    if basis == "tetrahedral":
        # 4 Bloch vectors at the vertices of a tetrahedron (pairwise dot = -1/3).
        rs = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=np.float64,
        ) / np.sqrt(3.0)

        sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        sy = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
        sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
        I = np.eye(2, dtype=np.complex128)

        states = []
        for i, r in enumerate(rs):
            rho = 0.5 * (I + r[0] * sx + r[1] * sy + r[2] * sz)
            evals, evecs = np.linalg.eigh(rho)
            psi = evecs[:, int(np.argmax(evals.real))].astype(np.complex128)
            psi = psi / np.linalg.norm(psi)
            states.append((f"tet{i}", psi))
        return [(name, psi, np.outer(psi, psi.conj())) for name, psi in states]

    # Backwards-compat alias: random_unitary -> random_low_overlap
    if basis == "random_unitary":
        basis = "random_low_overlap"

    rng = np.random.default_rng(seed)
    psi0 = np.array([1, 0], dtype=np.complex128)
    pool: list[NDArray[np.complex128]] = []
    for _ in range(10):
        u = _haar_unitary_2x2(rng)
        v = (u @ psi0).astype(np.complex128)
        v = v / np.linalg.norm(v)
        pool.append(v)

    # Greedy selection: iteratively pick the state that minimizes max overlap with selected.
    selected: list[NDArray[np.complex128]] = []
    selected.append(pool[0])
    remaining = pool[1:]
    while len(selected) < 4:
        best_idx = 0
        best_score = float("inf")
        for i, cand in enumerate(remaining):
            overlaps = [abs(np.vdot(cand, s)) ** 2 for s in selected]
            score = max(overlaps) if overlaps else 0.0
            if score < best_score:
                best_score = score
                best_idx = i
        selected.append(remaining.pop(best_idx))

    states = [(f"rlo{i}", psi) for i, psi in enumerate(selected)]
    return [(name, psi, np.outer(psi, psi.conj())) for name, psi in states]


def get_choi_basis(
    *,
    basis: Literal["standard", "random_low_overlap", "tetrahedral", "random_unitary"] = "standard",
    seed: int | None = None,
) -> tuple[list[NDArray[np.complex128]], list[tuple[int, int]]]:
    r"""Generate the 16 basis CP-map Choi matrices from the 4 basis states.

    A basis CP map is A_{p,m}(rho) = Tr(E_m rho) rho_p.
    Its Choi matrix is B_{p,m} = rho_p \otimes E_m^T.
    """
    basis_set = get_basis_states(basis=basis, seed=seed)
    choi_matrices, indices = [], []
    for p, (_, _, rho_p) in enumerate(basis_set):
        for m, (_, _, e_m) in enumerate(basis_set):
            choi_matrices.append(np.kron(rho_p, e_m.T))
            indices.append((p, m))
    return choi_matrices, indices


def gram_offdiag_metrics(basis_matrices: list[NDArray[np.complex128]]) -> tuple[float, float]:
    """Return (max_offdiag, mean_offdiag) for the real Gram matrix overlaps."""
    b = [m.reshape(-1) for m in basis_matrices]
    n = len(b)
    g = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            g[i, j] = float(np.real(np.vdot(b[i], b[j])))
    off = g.copy()
    np.fill_diagonal(off, 0.0)
    return float(np.max(np.abs(off))), float(np.mean(np.abs(off)))


def gram_condition_number(basis_matrices: list[NDArray[np.complex128]]) -> float:
    """Condition number of the frame matrix built from vec(basis_matrices)."""
    frame_matrix = np.column_stack([m.reshape(-1) for m in basis_matrices])
    s = np.linalg.svd(frame_matrix, compute_uv=False)
    if s[-1] <= 0.0:
        return float("inf")
    return float(s[0] / s[-1])


def dual_norm_metrics(dual_basis: list[NDArray[np.complex128]]) -> dict[str, float]:
    norms = [float(np.linalg.norm(d, "fro")) for d in dual_basis]
    return {
        "mean_dual_norm": float(np.mean(norms)),
        "max_dual_norm": float(np.max(norms)),
    }


def _finalize_sequence_averages(
    acc: dict[tuple[int, ...], list[Any]],
    weight_scale: float,
) -> tuple[list[tuple[int, ...]], list[NDArray[np.complex128]], list[float]]:
    """Consolidated logic for result collection, normalization, and weight assignment."""
    final_seqs = []
    final_outputs = []
    final_weights = []

    for seq, (rho_weighted_sum, weight_sum, count) in acc.items():
        if weight_sum > 1e-30:
            rho_avg = (rho_weighted_sum / count) / (weight_sum / count)
        else:
            rho_avg = np.zeros((2, 2), dtype=np.complex128)
        final_seqs.append(seq)
        final_outputs.append(rho_avg)
        final_weights.append(weight_sum / weight_scale)

    return final_seqs, final_outputs, final_weights


def calculate_dual_choi_basis(
    basis_matrices: list[NDArray[np.complex128]],
) -> list[NDArray[np.complex128]]:
    """Calculate the dual frame for the given Choi basis matrices."""
    frame_matrix = np.column_stack([m.reshape(-1) for m in basis_matrices])
    dual_frame = np.linalg.pinv(frame_matrix).conj().T
    dim = basis_matrices[0].shape[0]
    return [dual_frame[:, k].reshape(dim, dim) for k in range(dual_frame.shape[1])]
