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

    def reconstruct_comb_choi(
        self,
        check: bool = True,
        atol: float = 1e-8,
    ) -> NDArray[np.complex128]:
        """Reconstruct the *actual* comb Choi operator Υ from tomography data.

        Returns:
            Υ as a dense matrix of shape (2*4^k, 2*4^k), i.e. on H_F ⊗ (⊗_{t=1}^k H_{choi,t},
            where dim(H_F)=2 and each dim(H_{choi,t})=4 (a two-qubit Choi leg flattened as a Hilbert space).

        Notes:
            - Uses the stored primal basis `choi_basis` and duals `choi_duals`.
            - Automatically selects the correct transpose/conjugation convention by validating that
              contracting Υ with the primal basis reproduces the stored outputs.
        """
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
                rho_out = v2rho(self.tensor[(slice(None), *alphas)])

                # Past operator on ⊗_t H_choi,t, each factor is 4x4
                past = dual_transform(self.choi_duals[alphas[0]])
                for a in alphas[1:]:
                    past = np.kron(past, dual_transform(self.choi_duals[a]))

                # Υ = Σ ρ_out ⊗ past
                U += np.kron(rho_out, past)

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
                    rho_true = self.tensor[(slice(None), *alphas)].reshape(2, 2)
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
                "This usually means a convention mismatch in how basis/duals are defined or how outputs are stored."
            )

        return best_U

    def comb_qmi_from_upsilon(
        self,
        base: int = 2,
        past: str = "all",
        normalize: bool = True,
        check_psd: bool = True,
    ) -> float:
        """Quantum mutual information from the reconstructed comb Choi operator Υ.

        We reconstruct the genuine comb Choi operator Υ (dense), normalize it to a state ρ=Υ/Tr(Υ),
        then compute I(P:F)=S(ρ_P)+S(ρ_F)-S(ρ).

        Partition convention:
            - F is the final system output leg (dim 2)
            - Each step contributes one Choi-leg Hilbert space of dim 4
              (because your basis elements are 4x4 Choi matrices).

        Args:
            base: log base for entropy (2 => bits).
            past: "all", "last", "first" (which Choi legs are included in P).
            normalize: normalize Υ to trace 1 before entropies.
            check_psd: if True, sanity check Υ is Hermitian PSD (up to numerical tolerance).

        Returns:
            I(P:F) as float.
        """
        # 1) reconstruct true comb Choi tensor Υ
        Upsilon = self.reconstruct_comb_choi(check=True)  # uses your dual/primal consistency

        # 2) Hermitize + (optional) PSD check
        Upsilon = 0.5 * (Upsilon + Upsilon.conj().T)

        if check_psd:
            # allow tiny negatives from numerics
            lam_min = float(np.linalg.eigvalsh(Upsilon).min().real)
            if lam_min < -1e-9:
                raise ValueError(f"Reconstructed Υ not PSD (min eigenvalue {lam_min:.3e}).")

        # 3) normalize to density operator
        if normalize:
            tr = np.trace(Upsilon)
            if abs(tr) < 1e-15:
                return 0.0
            rho = Upsilon / tr
        else:
            rho = Upsilon

        # 4) subsystem dims: [F, step1, ..., stepk] = [2, 4, 4, ..., 4]
        k_steps = self.tensor.ndim - 1  # number of intervention slots (same k)
        dims = [2] + [4] * k_steps

        # choose which past legs to keep
        if past == "all":
            keep_P = list(range(1, k_steps + 1))
        elif past == "last":
            keep_P = [k_steps]
        elif past == "first":
            keep_P = [1]
        else:
            raise ValueError(f"Unknown past='{past}'. Use 'all', 'last', or 'first'.")

        # partial trace helper
        def _partial_trace(r: NDArray[np.complex128], dims_: list[int], keep: list[int]) -> NDArray[np.complex128]:
            keep = sorted(keep)
            n = len(dims_)
            if any(i < 0 or i >= n for i in keep):
                raise ValueError("keep indices out of range")

            reshaped = r.reshape(*dims_, *dims_)  # (i0..in-1, j0..jn-1)
            trace_out = [i for i in range(n) if i not in keep]
            perm = keep + trace_out
            reshaped = reshaped.transpose(*(perm + [i + n for i in perm]))

            dim_keep = int(np.prod([dims_[i] for i in keep])) if keep else 1
            dim_out = int(np.prod([dims_[i] for i in trace_out])) if trace_out else 1
            reshaped = reshaped.reshape(dim_keep, dim_out, dim_keep, dim_out)

            return np.einsum("a b c b -> a c", reshaped)

        rho_F = _partial_trace(rho, dims, keep=[0])
        rho_P = _partial_trace(rho, dims, keep=keep_P)

        # von Neumann entropy (eigenvalues)
        log_base = np.log(base)

        def _S(r: NDArray[np.complex128]) -> float:
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

        return _S(rho_P) + _S(rho_F) - _S(rho)

    def quantum_mutual_information(
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
            return np.einsum("abcb ->ac", reshaped)

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

    def conditional_mutual_information_from_upsilon(
    self,
    base: int = 2,
    A: str = "first",
    B: str = "final",
    C: str = "last",
    normalize: bool = True,
    check_psd: bool = True,
) -> float:
    """Conditional mutual information I(A:B|C) from the reconstructed comb Choi operator Υ.

    We reconstruct the genuine comb Choi operator Υ (dense), normalize it to a state ρ=Υ/Tr(Υ),
    then compute

        I(A:B|C) = S(AC) + S(BC) - S(C) - S(ABC).

    Default choice corresponds to the “memory beyond last operation” metric:
        I(first_slot : final | last_slot).

    Subsystem conventions:
        - The total Hilbert space is ordered as [F, step1, step2, ..., stepk]
        - dims = [2, 4, 4, ..., 4]
          where dim(F)=2 and each step leg is dim 4 (because each intervention basis element is a 4x4 Choi matrix).

    Args:
        base: Log base for entropy (2 => bits).
        A: Which past register is A: {"first","last"} (extendable).
        B: Which register is B: currently only {"final"}.
        C: Which past register is C: {"first","last"} (must be different from A for meaningful use).
        normalize: If True, normalize Υ to trace 1 before entropies.
        check_psd: If True, sanity check that Υ is Hermitian PSD up to tolerance.

    Returns:
        Conditional mutual information I(A:B|C).

    Raises:
        ValueError: If k<2 or invalid subsystem selections.
    """
    # Need at least two intervention legs to define a nontrivial conditioning.
    k_steps = self.tensor.ndim - 1
    if k_steps < 2:
        raise ValueError("Conditional MI requires at least k>=2 intervention legs.")

    # 1) Reconstruct Υ and form density operator ρ
    Upsilon = self.reconstruct_comb_choi(check=True)
    Upsilon = 0.5 * (Upsilon + Upsilon.conj().T)

    if check_psd:
        lam_min = float(np.linalg.eigvalsh(Upsilon).min().real)
        if lam_min < -1e-9:
            raise ValueError(f"Reconstructed Υ not PSD (min eigenvalue {lam_min:.3e}).")

    if normalize:
        tr = np.trace(Upsilon)
        if abs(tr) < 1e-15:
            return 0.0
        rho = Upsilon / tr
    else:
        rho = Upsilon

    # 2) Define subsystem indices for partial traces.
    # Order: [F, step1, ..., stepk]
    dims = [2] + [4] * k_steps
    idx_final = 0
    idx_first = 1
    idx_last = k_steps  # because steps run 1..k

    def _idx(which: str) -> int:
        if which == "final":
            return idx_final
        if which == "first":
            return idx_first
        if which == "last":
            return idx_last
        raise ValueError(f"Unknown subsystem spec '{which}'. Use 'final', 'first', or 'last'.")

    iA = _idx(A)
    iB = _idx(B)
    iC = _idx(C)

    if len({iA, iB, iC}) != 3:
        raise ValueError("A, B, C must refer to three distinct subsystems.")

    # 3) Partial trace helper
    def _partial_trace(r: NDArray[np.complex128], dims_: list[int], keep: list[int]) -> NDArray[np.complex128]:
        keep = sorted(keep)
        n = len(dims_)
        if any(i < 0 or i >= n for i in keep):
            raise ValueError("keep indices out of range")

        reshaped = r.reshape(*dims_, *dims_)
        trace_out = [i for i in range(n) if i not in keep]
        perm = keep + trace_out
        reshaped = reshaped.transpose(*(perm + [i + n for i in perm]))

        dim_keep = int(np.prod([dims_[i] for i in keep])) if keep else 1
        dim_out = int(np.prod([dims_[i] for i in trace_out])) if trace_out else 1
        reshaped = reshaped.reshape(dim_keep, dim_out, dim_keep, dim_out)
        return np.einsum("a b c b -> a c", reshaped)

    # 4) Entropy helper (eigenvalues, stable, no qiskit dependency needed here)
    log_base = np.log(base)

    def _S(r: NDArray[np.complex128]) -> float:
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

    # 5) Compute CMI
    rho_ABC = rho
    rho_AC = _partial_trace(rho_ABC, dims, keep=[iA, iC])
    rho_BC = _partial_trace(rho_ABC, dims, keep=[iB, iC])
    rho_C = _partial_trace(rho_ABC, dims, keep=[iC])

    return _S(rho_AC) + _S(rho_BC) - _S(rho_C) - _S(rho_ABC)