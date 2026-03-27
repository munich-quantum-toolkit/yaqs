# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Choi basis and dual frame utilities for process tomography."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from .predictor_encoding import build_choi_feature_table

TomographyBasis = Literal["standard", "tetrahedral", "random"]


def get_basis_states(
    *,
    basis: TomographyBasis = "tetrahedral",
    seed: int | None = None,
) -> list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]]:
    """Return the 4 single-qubit basis states used to build the 16 CP-map basis.

    - standard: |0>, |1>, |+>, |i+>
    - tetrahedral: deterministic 4-state tetrahedral (SIC-like) frame on the Bloch sphere.
    - random: four independent Haar-random pure states (seeded).
    """
    if basis == "random":
        rng = np.random.default_rng(seed)
        states: list[tuple[str, NDArray[np.complex128]]] = []
        for i in range(4):
            z = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            psi = (z / np.linalg.norm(z)).astype(np.complex128)
            states.append((f"rand{i}", psi))
        return [(name, psi, np.outer(psi, psi.conj())) for name, psi in states]

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

    raise TypeError(f"Unknown basis {basis!r}")


def get_choi_basis(
    *,
    basis: TomographyBasis = "standard",
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


def intervention_from_alpha(
    alpha: int,
    basis_set: list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]],
    choi_pm_pairs: list[tuple[int, int]],
) -> Any:
    """Discrete basis map ``E_alpha(rho) = Tr(E_m rho) * rho_p`` for comb prediction / surrogates."""
    p, m = choi_pm_pairs[int(alpha)]
    rho_p = np.asarray(basis_set[p][2], dtype=np.complex128)
    e_m = np.asarray(basis_set[m][2], dtype=np.complex128)

    def emap(rho: np.ndarray) -> np.ndarray:
        r = np.asarray(rho, dtype=np.complex128).reshape(2, 2)
        coeff = np.trace(e_m @ r)
        return coeff * rho_p

    return emap


def build_basis_for_fixed_alphabet(
    *,
    basis: TomographyBasis | str,
    basis_seed: int | None = None,
) -> tuple[
    list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]],
    list[NDArray[np.complex128]],
    list[tuple[int, int]],
    np.ndarray,
]:
    """Bundle used by NN predictors: basis states, Choi list, index pairs, flat Choi rows (16, 32)."""
    basis_t = cast(TomographyBasis, basis)
    seed_for_basis = int(basis_seed) if basis_seed is not None else None
    basis_set = get_basis_states(basis=basis_t, seed=seed_for_basis if basis == "random" else None)
    choi_matrices, choi_pm_pairs = get_choi_basis(basis=basis_t, seed=seed_for_basis if basis == "random" else None)
    choi_feat_table = build_choi_feature_table(choi_matrices)
    return basis_set, choi_matrices, choi_pm_pairs, choi_feat_table


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
