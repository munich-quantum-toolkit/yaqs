# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Restricted Process Tensor representation."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from qiskit.quantum_info import DensityMatrix, entropy

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


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
    if abs(tr) > 1e-15:
        rho /= tr
    return rho


class ProcessTensor:
    """Class to represent a Process Tensor.

    A Process Tensor generalizes the concept of a quantum channel to multiple time steps,
    capturing non-Markovian memory effects.

    Attributes:
        tensor (NDArray): The raw tensor data of shape (4, N, ..., N).
        weights (NDArray): The probabilities of each sequence of shape (N, ..., N).
        timesteps (list[float]): The time points where interventions/measurements occurred.
    """

    def __init__(
        self,
        tensor: NDArray[np.complex128],
        weights: NDArray[np.float64],
        timesteps: list[float],
        choi_duals: list[NDArray[np.complex128]],
        choi_indices: list[tuple[int, int]],
        choi_basis: list[NDArray[np.complex128]] | None = None,
    ) -> None:
        """Initialize the ProcessTensor.

        Args:
            tensor: The raw tensor data.
            weights: The probabilities of each sequence.
            timesteps: The time points where interventions/measurements occurred.
            choi_duals: List of 16 dual matrices (4x4) used for tensor contraction.
            choi_indices: List mapping tensor index alpha to (prep_idx, meas_idx).
            choi_basis: Optional list of 16 original basis matrices (4x4).
        """
        self.tensor = tensor
        self.weights = weights
        self.timesteps = timesteps
        self.choi_duals = choi_duals
        self.choi_indices = choi_indices
        self.choi_basis = choi_basis

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

    def predict_final_state(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Predict final state using dual-frame contraction for arbitrary interventions.

        Args:
            interventions: A list of callables representing CPTP maps for each intervention step.
                           The first callable is the initial state preparation at t=0.

        Returns:
            Predicted final density matrix (2x2)

        Raises:
            ValueError: If the number of interventions does not match num_steps.
        """
        k_steps = len(self.timesteps)
        if len(interventions) != k_steps:
            msg = f"Expected {k_steps} interventions (including t=0 prep), got {len(interventions)}."
            raise ValueError(msg)

        # Precompute the Choi matrices and their projection onto the dual basis.
        # For a CP map E(\\rho), its Choi matrix is J(E) = sum_{i,j} E(|i><j|) \\otimes |i><j|^T
        # which in our basis choice maps directly to \\rho_p \\otimes E_m^T.

        c_maps = []
        for emap in interventions:
            j_choi = np.zeros((4, 4), dtype=complex)
            for i in range(2):
                for j in range(2):
                    e_in = np.zeros((2, 2), dtype=complex)
                    e_in[i, j] = 1.0
                    rho_out = emap(e_in)
                    j_choi += np.kron(rho_out, e_in)

            # Project onto duals: c_a = Tr(D_a^dag J)
            c_a = np.array([np.trace(d.conj().T @ j_choi) for d in self.choi_duals])
            c_maps.append(c_a)

        # Tensor contraction
        # self.tensor has shape (4, 16, 16, ..., 16).
        # We want to contract the k_steps indices (dimensions 1 to k_steps) with the c_maps coefficients.
        result_tensor = self.tensor
        for step in reversed(range(k_steps)):
            # Multiply and sum out the last axis (axis -1) with c_maps[step]
            result_tensor = np.tensordot(result_tensor, c_maps[step], axes=([-1], [0]))

        return result_tensor.reshape(2, 2)

    def quantum_mutual_information(self, base: int = 2) -> float:
        """Calculate the Quantum Mutual Information between the input sequence and the output state.

        I(A:B) = S(B) - sum_a p(a) S(B|a)
        where A is the sequence distribution, B is the output state.

        Args:
            base: Logarithm base for entropy calculation (default 2 for bits).

        Returns:
            float: The Quantum Mutual Information.

        Raises:
            ValueError: If the output dimension is not 4.
        """
        out_dim = self.tensor.shape[0]
        if out_dim != 4:
            msg = f"Expected output dimension 4, got {out_dim}."
            raise ValueError(msg)

        steps_k = self.tensor.ndim - 1
        if steps_k == 0:
            return 0.0

        states_n = self.tensor.shape[1]
        seqs = list(itertools.product(range(states_n), repeat=steps_k))

        # Build ensemble and average output
        rhos = {}
        rho_avg = np.zeros((2, 2), dtype=np.complex128)

        # Normalize weights if they are provided
        total_weight = np.sum(self.weights)
        norm_weights = self.weights / total_weight if total_weight > 0 else np.ones_like(self.weights) / len(seqs)

        for seq in seqs:
            vec = self.tensor[(slice(None), *seq)]
            rho = _vec_to_rho(vec)
            rhos[seq] = rho
            rho_avg += norm_weights[seq] * rho

        # Compute Entropies
        entropy_b = entropy(DensityMatrix(rho_avg), base=base) if np.trace(rho_avg) > 1e-12 else 0.0

        entropy_b_given_a = 0.0
        for seq in seqs:
            if norm_weights[seq] > 1e-12 and np.trace(rhos[seq]) > 1e-12:
                entropy_b_given_a += norm_weights[seq] * entropy(DensityMatrix(rhos[seq]), base=base)

        return entropy_b - entropy_b_given_a

    def comb_quantum_mutual_information(
        self,
        base: int = 2,
        past: str = "all",
        normalize: bool = True,
    ) -> float:
        """Compute a *comb-level* (basis-independent) quantum mutual information from the full process tensor.

        This treats the reconstructed Choi operator of the k-step process tensor as a (normalized) quantum state
        on the tensor product of its legs, and computes

            I(P:F) = S(ρ_P) + S(ρ_F) - S(ρ_PF),

        where F is the final output leg, and P is a chosen subset of the remaining legs ("the past").

        IMPORTANT:
        - This is *not* the same as the cq/Holevo quantity computed by `quantum_mutual_information`.
        - It requires that the full comb Choi operator can be reconstructed from `self.tensor` + `self.choi_duals`
          (i.e., your tomography basis/dual is consistent).

        Assumptions / conventions (match your current codebase):
        - Each time step index α ∈ {0..15} corresponds to a 4x4 Choi-basis element for that intervention slot.
        - `self.tensor` stores coefficients such that contracting with duals predicts outputs (as in `predict_final_state`).
        - The final output leg is a qubit operator in a 4-dimensional vectorization, which we interpret as a 4-dim Hilbert leg
          for the purpose of QMI (i.e., we work in the Liouville/Choi space consistently).

        Args:
            base: Logarithm base for von Neumann entropies (2 => bits).
            past: Which "past" registers to include:
                  - "all": all k intervention legs (default, typical memory cut).
                  - "last": only the most recent intervention leg.
                  - "first": only the earliest intervention leg.
            normalize: If True, normalize the reconstructed comb Choi operator to trace 1 before entropies.

        Returns:
            The quantum mutual information I(P:F) in units of log(base).

        Raises:
            ValueError: on inconsistent shapes / missing duals.
        """
        # ----------------------------
        # 0) Basic checks
        # ----------------------------
        if self.tensor.ndim < 2:
            return 0.0

        out_dim = self.tensor.shape[0]
        if out_dim != 4:
            raise ValueError(f"Expected output dimension 4 (single-qubit operator space), got {out_dim}.")

        k_steps = self.tensor.ndim - 1  # number of intervention legs
        if k_steps == 0:
            return 0.0

        # Expect 16 basis elements per intervention leg
        if any(self.tensor.shape[i] != 16 for i in range(1, self.tensor.ndim)):
            raise ValueError(
                "Expected each intervention axis to have dimension 16 (single-qubit Choi basis). "
                f"Got tensor shape {self.tensor.shape}."
            )

        if self.choi_duals is None or len(self.choi_duals) != 16:
            raise ValueError("Expected `self.choi_duals` to be a list of 16 dual 4x4 matrices.")

        # ----------------------------
        # 1) Reconstruct the full comb Choi operator as a dense matrix
        # ----------------------------
        # Your tensor stores coefficients over the dual frame.
        # If predict_final_state does:  vec_out = Σ_{α1..αk} T[:,α1..αk] Π_t Tr(D_{αt}^† J_t)
        # then the natural reconstruction of the underlying operator uses the *primal* basis B_α such that
        # Tr(D_α^† B_β) = δ_{αβ}.
        #
        # If you did not store `choi_basis` explicitly, you can often take the primal to be the *dual-dual*
        # only if your frame is orthonormal; otherwise you need the actual primal basis.
        #
        # Here we use:
        #   - if choi_basis is provided: use it (recommended).
        #   - else: assume the duals are actually the primal basis too (orthonormal case).
        primal_basis = self.choi_basis if self.choi_basis is not None else self.choi_duals
        if len(primal_basis) != 16:
            raise ValueError("Expected `choi_basis` (if provided) to have length 16.")

        # We interpret the comb Choi operator to live on:
        #   H = H_F ⊗ H_1 ⊗ ... ⊗ H_k
        # where each H_t is 4-dim (single-qubit Choi/Liouville leg), and H_F is 4-dim (final output leg).
        #
        # So the full matrix dimension is: D = 4 * 4^k = 4^(k+1).
        D = 4 ** (k_steps + 1)
        Upsilon = np.zeros((D, D), dtype=np.complex128)

        # Efficient-ish reconstruction:
        # We expand as
        #   Upsilon = Σ_{μ,α1..αk} c_{μ,α1..αk} (F_μ ⊗ B_{α1} ⊗ ... ⊗ B_{αk})
        # But we only have μ as a length-4 vectorization of a 2x2 operator.
        # We need a 4x4 operator basis {F_μ} on the final output Liouville leg.
        #
        # We'll use the canonical matrix units in Liouville space:
        #   E_{ij} for i,j=0..3, but we only have 4 coefficients not 16.
        # Therefore: your `tensor[μ,...]` is not a general 4x4 operator on H_F; it's a *vector* in a 4-dim space.
        # The consistent way is to treat the final leg as a *ket* in Liouville space, i.e., Upsilon is a state vector
        # on H_F tensored with operators on the past legs.
        #
        # To make QMI well-defined as a density operator, we embed that vector as a rank-1 operator on H_F:
        #   |v><v| on H_F.
        #
        # Practically: form a pure-state density on H_F from the coefficient vector, and tensor with the past operator.
        #
        # This is the mildest consistent choice given your stored format. If you later store the full 16 coefficients
        # for the final leg, replace this with a true operator-basis expansion.

        # Helper: build |v><v| on the 4-dim final Liouville leg
        def _ketbra_from_vec(v: NDArray[np.complex128]) -> NDArray[np.complex128]:
            v = v.reshape(4)
            nrm = np.vdot(v, v)
            if abs(nrm) < 1e-15:
                return np.zeros((4, 4), dtype=np.complex128)
            return np.outer(v, v.conj()) / nrm

        # Enumerate all α-tuples and accumulate kron products.
        # Warning: this is exponential in k. Only feasible for small k (which matches PT tomography use).
        for alphas in itertools.product(range(16), repeat=k_steps):
            v = self.tensor[(slice(None), *alphas)]  # length-4 vector on final leg
            F_op = _ketbra_from_vec(v)               # 4x4 operator on final leg (embedding)

            # Past operator = ⊗_t B_{αt}
            past_op = primal_basis[alphas[0]]
            for a in alphas[1:]:
                past_op = np.kron(past_op, primal_basis[a])

            Upsilon += np.kron(F_op, past_op)

        # Normalize to trace 1 if requested
        if normalize:
            tr = np.trace(Upsilon)
            if abs(tr) > 1e-15:
                Upsilon = Upsilon / tr

        # ----------------------------
        # 2) Choose partition P vs F and compute entropies
        # ----------------------------
        # We treat the total Hilbert space as:
        #   H_total = H_F (dim 4) ⊗ H_1 (dim 4) ⊗ ... ⊗ H_k (dim 4)
        dims = [4] + [4] * k_steps  # [F, step1, ..., stepk]

        # Pick which past subsystems to keep
        if past == "all":
            keep_past = list(range(1, k_steps + 1))  # keep all steps
        elif past == "last":
            keep_past = [k_steps]                    # last step
        elif past == "first":
            keep_past = [1]                          # first step
        else:
            raise ValueError(f"Unknown past='{past}'. Use 'all', 'last', or 'first'.")

        # Helper: partial trace over selected subsystems for a state on ⊗ dims.
        def _partial_trace(rho: NDArray[np.complex128], dims_: list[int], keep: list[int]) -> NDArray[np.complex128]:
            """Partial trace keeping subsystems in `keep` (indices into dims_)."""
            keep = sorted(keep)
            n = len(dims_)
            if any(i < 0 or i >= n for i in keep):
                raise ValueError("keep indices out of range")

            # reshape to 2n indices: (i0..in-1, j0..jn-1)
            reshaped = rho.reshape(*dims_, *dims_)
            # trace out subsystems not in keep
            trace_out = [i for i in range(n) if i not in keep]

            # Move kept systems to front for both bra/ket
            perm = keep + trace_out
            reshaped = reshaped.transpose(*(perm + [i + n for i in perm]))

            # Now trace over the tail systems
            dim_keep = int(np.prod([dims_[i] for i in keep])) if keep else 1
            dim_out = int(np.prod([dims_[i] for i in trace_out])) if trace_out else 1

            reshaped = reshaped.reshape(dim_keep, dim_out, dim_keep, dim_out)
            # trace over dim_out
            return np.einsum("a b c b -> a c", reshaped)

        # Reduced states
        rho_PF = Upsilon
        rho_F = _partial_trace(rho_PF, dims, keep=[0])                # keep final only
        rho_P = _partial_trace(rho_PF, dims, keep=keep_past)          # keep chosen past

        # Entropies (guard tiny traces)
        def _S(r: NDArray[np.complex128]) -> float:
            tr = np.trace(r)
            if abs(tr) < 1e-12:
                return 0.0
            # ensure Hermitian numeric stability
            rH = 0.5 * (r + r.conj().T)
            # renormalize in case partial trace drifted
            rH = rH / np.trace(rH)
            return float(entropy(DensityMatrix(rH), base=base))

        return _S(rho_P) + _S(rho_F) - _S(rho_PF)