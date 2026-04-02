# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Tomography estimate container (raw reconstruction data)."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TomographyEstimate:
    """Raw tomography estimate / reconstruction container."""

    def __init__(
        self,
        tensor: NDArray[np.complex128] | None,
        weights: NDArray[np.float64] | None,
        timesteps: list[float],
        choi_duals: list[NDArray[np.complex128]] | None = None,
        choi_indices: list[tuple[int, int]] | None = None,
        choi_basis: list[NDArray[np.complex128]] | None = None,
        dense_choi: NDArray[np.complex128] | None = None,
    ) -> None:
        self.tensor = tensor
        self.weights = weights
        self.timesteps = timesteps
        self.choi_duals = choi_duals
        self.choi_indices = choi_indices
        self.choi_basis = choi_basis
        self.dense_choi = dense_choi

    @classmethod
    def from_dense_choi(
        cls,
        dense_choi: NDArray[np.complex128],
        timesteps: list[float],
    ) -> "TomographyEstimate":
        return cls(
            tensor=None,
            weights=None,
            timesteps=timesteps,
            dense_choi=dense_choi,
        )

    # ── Comb views ──────────────────────────────────────────────────────────
    def to_dense_comb(self):
        from ..exact.combs import DenseComb

        U = self.reconstruct_comb_choi(check=True)
        return DenseComb(U, self.timesteps)

    def to_mpo_comb(
        self,
        *,
        d: int = 2,
        max_bond_dim: int | None = None,
        cutoff: float = 1e-12,
    ):
        from mqt.yaqs.core.data_structures.networks import MPO
        from ..exact.combs import MPOComb

        U = self.reconstruct_comb_choi(check=True)
        mpo = MPO.from_matrix(U, d=d, max_bond=max_bond_dim, cutoff=cutoff)
        return MPOComb(mpo, self.timesteps)

    def to_linear_map_matrix(self) -> NDArray[np.complex128]:
        num_inputs = np.prod(self.tensor.shape[1:])
        return self.tensor.reshape(4, num_inputs)

    def reconstruct_comb_choi(
        self,
        check: bool = True,
        atol: float = 1e-8,
        return_convention: bool = False,
    ) -> NDArray[np.complex128] | tuple[NDArray[np.complex128], str]:
        if (self.choi_basis is None or len(self.choi_basis) != 16) and self.dense_choi is not None:
            if return_convention:
                return self.dense_choi, "id"
            return self.dense_choi

        if self.choi_basis is None or len(self.choi_basis) != 16:
            raise ValueError("Need `self.choi_basis` of length 16 to reconstruct Υ.")
        if self.choi_duals is None or len(self.choi_duals) != 16:
            raise ValueError("Need `self.choi_duals` of length 16 to reconstruct Υ.")
        if self.tensor.shape[0] != 4:
            raise ValueError(f"Expected tensor[0] dim 4 (vec of 2x2 output), got {self.tensor.shape[0]}.")

        k = self.tensor.ndim - 1
        if k == 0:
            rho = self.tensor.reshape(2, 2)
            if return_convention:
                return rho, "id"
            return rho

        def v2rho(v: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return v.reshape(2, 2)

        def _T_id(x):  # noqa: ANN001
            return x

        def _T_T(x):  # noqa: ANN001
            return x.T

        def _T_conj(x):  # noqa: ANN001
            return x.conj()

        def _T_dag(x):  # noqa: ANN001
            return x.conj().T

        candidates = {
            "id": _T_id,
            "T": _T_T,
            "conj": _T_conj,
            "dag": _T_dag,
        }

        def build_upsilon(dual_transform):  # noqa: ANN001
            dim = (2 * (4**k))
            U = np.zeros((dim, dim), dtype=np.complex128)
            for alphas in itertools.product(range(16), repeat=k):
                w = self.weights[alphas]
                rho_out = v2rho(self.tensor[(slice(None), *alphas)])
                past = dual_transform(self.choi_duals[alphas[0]])
                for a in alphas[1:]:
                    past = np.kron(past, dual_transform(self.choi_duals[a]))
                U += np.kron(w * rho_out, past)
            return U

        def predict_from_upsilon(U: NDArray[np.complex128], alphas: tuple[int, ...]) -> NDArray[np.complex128]:
            past = self.choi_basis[alphas[0]]
            for a in alphas[1:]:
                past = np.kron(past, self.choi_basis[a])
            dim_p = 4**k
            U4 = U.reshape(2, dim_p, 2, dim_p)
            ins = past.T.reshape(dim_p, dim_p)
            rho = np.einsum("s p q r, r p -> s q", U4, ins)
            return rho

        seqs = list(itertools.product(range(16), repeat=k))
        test_seqs = seqs if (check and len(seqs) <= 256) else (seqs[:64] if check else [])

        best_name = None
        best_err = np.inf
        best_U = None

        for name, dual_T in candidates.items():
            U = build_upsilon(dual_T)
            if check:
                err = 0.0
                n_used = 0
                for alphas in test_seqs:
                    w = float(self.weights[alphas])
                    if w <= 1e-30:
                        continue
                    rho_true = w * self.tensor[(slice(None), *alphas)].reshape(2, 2)
                    rho_pred = predict_from_upsilon(U, alphas)
                    err += np.linalg.norm(rho_true - rho_pred)
                    n_used += 1
                err /= max(1, n_used)
            else:
                err = 0.0
            if err < best_err:
                best_err, best_name, best_U = err, name, U

        assert best_U is not None
        if check and best_err > atol:
            raise ValueError(
                f"Could not find a consistent Υ reconstruction (best convention={best_name}, "
                f"mean test error={best_err:.3e} > atol={atol})."
            )
        if return_convention:
            return best_U, cast(str, best_name)
        return best_U

