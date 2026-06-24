# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Discrete Choi basis and dual frame utilities for tomography estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from ..core.encoding import build_choi_feature_table

TomographyBasis = Literal["standard", "tetrahedral", "random"]


def _finalize_sequence_averages(
    acc: dict[tuple[int, ...], list[Any]],
    weight_scale: float,
) -> tuple[list[tuple[int, ...]], list[NDArray[np.complex128]], list[float]]:
    """Finalize per-sequence weighted averages.

    Args:
        acc: Mapping from sequences to accumulator tuples ``[rho_weighted_sum, weight_sum, count]``.
        weight_scale: Scale factor applied to weights (e.g., number of trajectories).

    Returns:
        Tuple ``(sequences, outputs, weights)`` where outputs are averaged 2x2 density matrices.
    """
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


def get_basis_states(
    *,
    basis: TomographyBasis = "tetrahedral",
    seed: int | None = None,
) -> list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]]:
    """Return the 4 single-qubit basis states used for the 16-map CP basis.

    Args:
        basis: Basis choice.
        seed: Optional seed used when ``basis="random"``.

    Returns:
        List of 4 tuples ``(name, psi, rho)`` where ``psi`` is a ket and ``rho = |psi><psi|``.

    Raises:
        TypeError: If ``basis`` is not recognized.
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
            psi /= np.linalg.norm(psi)
            states.append((f"tet{i}", psi))
        return [(name, psi, np.outer(psi, psi.conj())) for name, psi in states]

    msg = f"Unknown basis {basis!r}"
    raise TypeError(msg)


def get_choi_basis(
    *,
    basis: TomographyBasis = "standard",
    seed: int | None = None,
) -> tuple[list[NDArray[np.complex128]], list[tuple[int, int]]]:
    """Generate the 16 CP-map Choi basis matrices.

    Args:
        basis: Basis choice for the underlying 4 states.
        seed: Optional seed used when ``basis="random"``.

    Returns:
        Tuple ``(choi_matrices, indices)`` where:
        - ``choi_matrices`` is a list of 16 complex 4x4 Choi matrices.
        - ``indices`` gives the corresponding ``(prep_index, meas_index)`` pairs.
    """
    basis_set = get_basis_states(basis=basis, seed=seed)
    choi_matrices, indices = [], []
    for p, (_, _, rho_p) in enumerate(basis_set):
        for m, (_, _, e_m) in enumerate(basis_set):
            choi_matrices.append(np.kron(rho_p, e_m.T))
            indices.append((p, m))
    return choi_matrices, indices


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
    """Build the discrete basis bundle for tomography and surrogate feature encoding.

    Args:
        basis: Basis name (``"standard"``, ``"tetrahedral"``, ``"random"``).
        basis_seed: Optional seed used when ``basis="random"``.

    Returns:
        Tuple ``(basis_set, choi_mats, choi_idx, choi_features)`` where ``choi_features`` has shape
        ``(16, 32)``.
    """
    basis_t = cast("TomographyBasis", basis)
    seed_for_basis = int(basis_seed) if basis_seed is not None else None
    basis_set = get_basis_states(basis=basis_t, seed=seed_for_basis if basis == "random" else None)
    choi_matrices, choi_pm_pairs = get_choi_basis(basis=basis_t, seed=seed_for_basis if basis == "random" else None)
    choi_feat_table = build_choi_feature_table(choi_matrices)
    return basis_set, choi_matrices, choi_pm_pairs, choi_feat_table


def calculate_dual_choi_basis(
    basis_matrices: list[NDArray[np.complex128]],
) -> list[NDArray[np.complex128]]:
    """Compute the dual frame for a Choi basis.

    Args:
        basis_matrices: List of basis Choi matrices.

    Returns:
        List of dual-frame matrices with the same shapes as ``basis_matrices``.
    """
    frame_matrix = np.column_stack([m.reshape(-1) for m in basis_matrices])
    dual_frame = np.linalg.pinv(frame_matrix).conj().T
    dim = basis_matrices[0].shape[0]
    return [dual_frame[:, k].reshape(dim, dim) for k in range(dual_frame.shape[1])]
