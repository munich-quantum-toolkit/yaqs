# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tomography estimator utilities.

This module provides the low-level representation of raw tomography estimates
(`TomographyEstimate`) together with helper functions to work with dense comb
Choi operators (Υ). Higher-level comb wrappers live in `comb.py`.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPO

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def _cptp_to_choi(emap: Callable[[NDArray[np.complex128]], NDArray[np.complex128]]) -> NDArray[np.complex128]:
    """Convert a CPTP map callable to its Choi matrix J(E).

    J(E) = sum_{i,j} E(|i><j|) ⊗ |i><j| (order matches predict_from_dense_upsilon).
    """
    j_choi = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            e_in = np.zeros((2, 2), dtype=complex)
            e_in[i, j] = 1.0
            # E(|i><j|) ⊗ |i><j|
            j_choi += np.kron(emap(e_in), e_in)
    return j_choi


def predict_from_dense_upsilon(
    U: NDArray[np.complex128],
    interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
) -> NDArray[np.complex128]:
    """Predict final state from a dense Choi operator (Upsilon) by contraction.

    Upsilon index order: output ⊗ past_0 ⊗ past_1 ⊗ ... ⊗ past_{k-1}.
    """
    k_steps = len(interventions)
    past_list = []
    for emap in interventions:
        j_choi = _cptp_to_choi(emap)
        past_list.append(j_choi)

    # Contract
    # For k=1: rho = Tr_past[ U (I ⊗ J.T) ]
    # Kronecker order of past: J1 ⊗ J2 ⊗ ... ⊗ Jk
    past_total = past_list[0]
    for p in past_list[1:]:
        past_total = np.kron(past_total, p)

    dim_p = 4**k_steps
    U4 = U.reshape(2, dim_p, 2, dim_p)
    ins = past_total.T.reshape(dim_p, dim_p)
    return np.einsum("s p q r, r p -> s q", U4, ins)


def _vec_to_rho(vec4: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Convert a 4-element vector to a 2x2 density matrix.

    Args:
        vec4: A 4-element vector.

    Returns:
        A 2x2 density matrix.
    """
    assert len(vec4) == 4, "Vector must have 4 elements"
    rho = vec4.reshape(2, 2)
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 1e-13: # Changed from 1e-15 to 1e-13
        rho /= tr
    return rho




class TomographyEstimate:
    """Raw tomography estimate / reconstruction container.

    This holds the coefficients of a tomography run in a fixed frame together
    with the associated basis/dual information. From this data the underlying
    comb Choi operator Υ can be reconstructed in dense form.

    Attributes:
        tensor (NDArray): The raw tensor data of shape (4, N, ..., N).
        weights (NDArray): The probabilities of each sequence of shape (N, ..., N).
        timesteps (list[float]): The time points where interventions/measurements occurred.
    """

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
        """Initialize the tomography estimate container.

        Args:
            tensor: The raw tensor data (coefficients in a basis).
            weights: The probabilities of each sequence in that basis.
            timesteps: The time points where interventions/measurements occurred.
            choi_duals: List of dual matrices used for tensor contraction.
            choi_indices: List mapping tensor index alpha to (prep_idx, meas_idx).
            choi_basis: Optional list of original basis matrices.
            dense_choi: Optional dense matrix representation of the comb Choi operator.
        """
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
        """Build a TomographyEstimate from its dense Choi matrix representation."""
        return cls(
            tensor=None,
            weights=None,
            timesteps=timesteps,
            dense_choi=dense_choi,
        )

    # ── Comb views ──────────────────────────────────────────────────────────
    def to_dense_comb(self):
        """Return a DenseComb view of the underlying comb Choi operator Υ.

        This uses the same reconstruction as ``reconstruct_comb_choi`` and does
        not change any estimator semantics.
        """
        from .comb import DenseComb

        U = self.reconstruct_comb_choi(check=True)
        return DenseComb(U, self.timesteps)

    def to_mpo_comb(self):
        """Return an MPOComb view (API hook).

        The current implementation does not track an MPO directly on the
        tomography estimate, so generic reconstruction of an MPO comb from the
        raw tensor is intentionally not implemented here.
        """
        from .comb import MPOComb  # noqa: F401  (for future use)

        msg = "MPO comb reconstruction from `TomographyEstimate` is not implemented."
        raise NotImplementedError(msg)

    def to_linear_map_matrix(self) -> NDArray[np.complex128]:
        """Convert to matrix view (final output vs all inputs).

        Returns matrix of shape (4, N^k) where:
        - 4: final output (vectorized 2x2 density matrix)
        - N: number of frame states per input slot
        - k: number of steps

        For Pauli frame (N=6), k=2 gives shape (4, 36).

        Returns:
            NDArray[np.complex128]: Matrix of shape (4, N^k).
        """
        # Shape is [4, N, N, ...] -> reshape to [4, N^k]
        num_inputs = np.prod(self.tensor.shape[1:])  # Product of all input dimensions
        return self.tensor.reshape(4, num_inputs)

    def reconstruct_comb_choi(
        self,
        check: bool = True,
        atol: float = 1e-8,
        return_convention: bool = False,
    ) -> NDArray[np.complex128] | tuple[NDArray[np.complex128], str]:
        """Reconstruct the *actual* comb Choi operator Υ from tomography data.

        Returns:
            Υ as a dense matrix of shape (2*4^k, 2*4^k), i.e. on H_F ⊗ (⊗_{t=1}^k H_{choi,t},
            where dim(H_F)=2 and each dim(H_{choi,t})=4 (a two-qubit Choi leg flattened as a Hilbert space).

        Notes:
            - Uses the stored primal basis `choi_basis` and duals `choi_duals`.
            - Automatically selects the correct transpose/conjugation convention by validating that
              contracting Υ with the primal basis reproduces the stored outputs.
        """
        # If this estimate was constructed directly from a dense comb (no basis/dual
        # information), simply return the stored matrix.
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
            # Single output state only; comb is just that state on the final leg.
            rho = self.tensor.reshape(2, 2)
            if return_convention:
                return rho, "id"
            return rho

        # Helper: vec <-> mat for final output
        def v2rho(v: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return v.reshape(2, 2)

        # Candidate transforms applied to the duals in the reconstruction
        # (we'll choose the one that makes the forward prediction match stored outputs)
        def _T_id(x): return x
        def _T_T(x): return x.T
        def _T_conj(x): return x.conj()
        def _T_dag(x): return x.conj().T

        candidates = {
            "id": _T_id,
            "T": _T_T,
            "conj": _T_conj,
            "dag": _T_dag,
        }

        # Build Υ given a transform on duals
        def build_upsilon(dual_transform) -> NDArray[np.complex128]:
            # Υ lives on H_F (dim 2) ⊗ ⊗_{t=1}^k H_choi,t (dim 4 each)
            dim = (2 * (4**k))
            U = np.zeros((dim, dim), dtype=np.complex128)

            # Enumerate all alpha tuples (16^k)
            for alphas in itertools.product(range(16), repeat=k):
                # self.tensor is normalized, but Upsilon requires weighted contributions J = sum w*rho*D
                w = self.weights[alphas]
                rho_out = v2rho(self.tensor[(slice(None), *alphas)])

                # Past operator on ⊗_t H_choi,t, each factor is 4x4
                past = dual_transform(self.choi_duals[alphas[0]])
                for a in alphas[1:]:
                    past = np.kron(past, dual_transform(self.choi_duals[a]))

                # Expansion in dual basis.
                U += np.kron(w * rho_out, past)

            return U

        # Forward contraction: given Υ, recover rho_out(alpha) by contracting with primal basis
        # We test: rho_pred(alpha) = Tr_past[ (I ⊗ (⊗_t B_alpha_t^†)) Υ ]    (or without dag depending on convention)
        # Rather than overcomplicate, we test multiple simple variants consistent with our dual choice.
        def predict_from_upsilon(U: NDArray[np.complex128], alphas: tuple[int, ...]) -> NDArray[np.complex128]:
            past = self.choi_basis[alphas[0]]
            for a in alphas[1:]:
                past = np.kron(past, self.choi_basis[a])

            # reshape Υ as blocks: (2 x 4^k) by (2 x 4^k)
            # Take partial trace over past with an operator insertion.
            # Implement: rho = Tr_past[ Υ (I ⊗ past^T) ]  -- this matches common Choi/link conventions.
            dim_p = 4**k
            U4 = U.reshape(2, dim_p, 2, dim_p)  # indices: s, p, s', p'

            # choose insertion = past^T (this is the only remaining convention; dual_transform handles the rest)
            ins = past.T.reshape(dim_p, dim_p)

            # rho_{s,s'} = Σ_{p,p'} U_{s,p,s',p'} * ins_{p',p}
            rho = np.einsum("s p q r, r p -> s q", U4, ins)
            return rho

        # Choose the best candidate by checking a small sample of alphas (or all, for k small)
        seqs = list(itertools.product(range(16), repeat=k))
        test_seqs = seqs if (check and len(seqs) <= 256) else (seqs[:64] if check else [])

        best_name = None
        best_err = np.inf
        best_U = None

        for name, dual_T in candidates.items():
            U = build_upsilon(dual_T)

            if check:
                err = 0.0
                for alphas in test_seqs:
                    w = self.weights[alphas]
                    rho_true = w * self.tensor[(slice(None), *alphas)].reshape(2, 2)
                    rho_pred = predict_from_upsilon(U, alphas)
                    err += np.linalg.norm(rho_true - rho_pred)
                err /= max(1, len(test_seqs))
            else:
                err = 0.0

            if err < best_err:
                best_err, best_name, best_U = err, name, U

        assert best_U is not None

        if check and best_err > atol:
            raise ValueError(
                f"Could not find a consistent Υ reconstruction (best convention={best_name}, "
                f"mean test error={best_err:.3e} > atol={atol}). "
                "This usually means a convention mismatch in how basis/duals are defined or stored."
            )

        if return_convention:
            return best_U, best_name
        return best_U



def rank1_upsilon_mpo_term(
    rho_final: NDArray[np.complex128],
    dual_ops: list[NDArray[np.complex128]],
    weight: float = 1.0,
) -> MPO:
    """Build a rank-1 MPO term representing a single sample's contribution to Upsilon.

    The comb Choi operator lives on sites [F, P1, P2, ..., Pk].
    Site 0 (F) is the final state density matrix (2x2).
    Sites 1..k are the dual operators (4x4) corresponding to interventions.

    Args:
        rho_final: (2x2) density matrix of the final state.
        dual_ops: list of k (4x4) dual matrices for the time steps.
        weight: Scalar importance weight for this sample.

    Returns:
        MPO with bond dimension 1 representing `weight * (rho_final ⊗ D1 ⊗ D2 ⊗ ... ⊗ Dk)`.
    """
    k = len(dual_ops)
    length = k + 1
    phys_dims = [2] + [4] * k

    # Build local tensors with dummy bond dims of 1
    # YAQS MPO tensor order: (phys_out, phys_in, chi_left, chi_right)
    tensors = []
    t0 = (weight * rho_final).reshape(2, 2, 1, 1)
    tensors.append(t0)

    for D in dual_ops:
        tD = D.reshape(4, 4, 1, 1)
        tensors.append(tD)

    m = MPO()
    # `custom` sets tensors, length, physical_dimension, checks bounds
    m.custom(tensors, transpose=False)
    m.physical_dimension = phys_dims

    return m

