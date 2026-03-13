# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Restricted Process Tensor representation."""

from __future__ import annotations

from email.policy import strict
import itertools
from typing import TYPE_CHECKING

import numpy as np
from qiskit.quantum_info import DensityMatrix, entropy

from mqt.yaqs.core.data_structures.networks import MPO

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
    if abs(tr) > 1e-13: # Changed from 1e-15 to 1e-13
        rho /= tr
    return rho


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

    Args:
        U: Raw or canonicalized Υ matrix of shape (2·4^k, 2·4^k).
        k: Number of timesteps.
        keep_last_m: Number of past legs to keep (default 1).

    Returns:
        Reduced operator of shape (2·4^m, 2·4^m).

    Raises:
        ValueError: If keep_last_m > k or U has wrong size.
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

    dim_m = 2 * (4**keep_last_m)        # output side of reduced matrix
    dim_traced = 4 ** (k - keep_last_m)  # legs being traced out

    if dim_traced == 1:
        # Nothing to trace: keep_last_m == k
        return U.reshape(dim_m, dim_m)

    # Regroup U into 6 axes:
    #   ket side: [output=2, traced_block=4^(k-m), kept_block=4^m]
    #   bra side: [output=2, traced_block=4^(k-m), kept_block=4^m]
    # Then trace over the two traced_block axes by sharing the same index 'a'.
    #
    # Index naming (fixed, no alphabet limit):
    #   i = output ket     (2)
    #   a = traced ket     (4^(k-m))  — shared with bra for contraction
    #   b = kept ket       (4^m)
    #   j = output bra     (2)
    #   a = traced bra     (4^(k-m))  — same 'a' forces diagonal (partial trace)
    #   c = kept bra       (4^m)
    #
    # Einsum 'iabjac->ibjc' sums over 'a' (the traced block).
    U6 = U.reshape(2, dim_traced, 4**keep_last_m, 2, dim_traced, 4**keep_last_m)
    reduced = np.einsum('iabjac->ibjc', U6)  # (2, 4^m, 2, 4^m)

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
    """Quantum mutual information I(F : P) from a dense comb Choi operator Υ.

    **Calling convention (canonicalization policy)**::

        rho = canonicalize_upsilon(U, hermitize=True, psd_project=True, normalize_trace=True)
        qmi = comb_qmi_from_upsilon_dense(rho, assume_canonical=True)

    When ``assume_canonical=True``, the function skips the internal
    hermitize + normalize step (saves ~30% for large matrices). When
    ``assume_canonical=False`` (default), it hermitizes and normalizes
    internally, so raw Upsilon estimates can be passed directly.

    Args:
        Upsilon: Raw or canonicalized Υ. Shape (2·4^k, 2·4^k).
        base: Entropy logarithm base (2 = bits).
        past: Which past legs to include: ``"all"``, ``"last"``, or ``"first"``.
        check_psd: If True, raise if min eigenvalue < -1e-9 (expensive — prefer
            using ``canonicalize_upsilon(psd_project=True)`` before calling).
        assume_canonical: If True, skip internal hermitize + normalize.
    """
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
    """Conditional Mutual Information I(F : P_{<k} | P_k) from dense Upsilon.

    See ``comb_qmi_from_upsilon_dense`` for the canonicalization policy.
    Pass ``assume_canonical=True`` when Upsilon is already the output of
    ``canonicalize_upsilon(hermitize=True, normalize_trace=True)``.
    """
    if assume_canonical:
        rho = Upsilon
    else:
        U = 0.5 * (Upsilon + Upsilon.conj().T)
        tr = np.trace(U)
        rho = U / tr if abs(tr) > 1e-15 else U

    size = rho.shape[0]  # fixed: U is undefined when assume_canonical=True
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
        tensor: NDArray[np.complex128] | None,
        weights: NDArray[np.float64] | None,
        timesteps: list[float],
        choi_duals: list[NDArray[np.complex128]] | None = None,
        choi_indices: list[tuple[int, int]] | None = None,
        choi_basis: list[NDArray[np.complex128]] | None = None,
        dense_choi: NDArray[np.complex128] | None = None,
    ) -> None:
        """Initialize the ProcessTensor.

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
    ) -> ProcessTensor:
        """Build a ProcessTensor from its dense Choi matrix representation."""
        return cls(
            tensor=None,
            weights=None,
            timesteps=timesteps,
            dense_choi=dense_choi,
        )

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

        if self.dense_choi is not None:
            return predict_from_dense_upsilon(self.dense_choi, interventions)

        if self.tensor is None or self.weights is None or self.choi_duals is None:
            msg = "Predicting from ProcessTensor requires either `dense_choi` or (`tensor`, `weights`, and `choi_duals`)."
            raise ValueError(msg)

        # Precompute the Choi matrices and their projection onto the dual basis.
        # For a CP map E(\\rho), its Choi matrix is J(E) = sum_{i,j} E(|i><j|) \\otimes |i><j|^T
        # which in our basis choice maps directly to \\rho_p \\otimes E_m^T.

        c_maps = []
        for emap in interventions:
            j_choi = _cptp_to_choi(emap)
            # Project onto duals: c_a = Tr(D_a^dag J)
            c_a = np.array([np.trace(d.conj().T @ j_choi) for d in self.choi_duals])
            c_maps.append(c_a)

        # Tensor contraction
        # self.tensor has shape (4, 16, 16, ..., 16).
        # We want to contract the k_steps indices (dimensions 1 to k_steps) with the c_maps coefficients.
        weighted_tensor = self.tensor * self.weights[None, ...]

        result_tensor = weighted_tensor
        for step in reversed(range(k_steps)):
            # Multiply and sum out the last axis (axis -1) with c_maps[step]
            result_tensor = np.tensordot(result_tensor, c_maps[step], axes=([-1], [0]))

        return result_tensor.reshape(2, 2)

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
        Upsilon = self.reconstruct_comb_choi(check=True)
        return comb_qmi_from_upsilon_dense(Upsilon, base=base, past=past, check_psd=check_psd)

    def comb_cmi_from_upsilon(
        self,
        base: int = 2,
        check_psd: bool = True,
    ) -> float:
        """Conditional mutual information from the reconstructed comb Choi operator Υ.

        Computes I(F : P_{<k} | P_k).

        Args:
            base: log base for entropy (2 => bits).
            check_psd: if True, sanity check Υ is Hermitian PSD.

        Returns:
            I(F : P_{<k} | P_k) as float.
        """
        Upsilon = self.reconstruct_comb_choi(check=True)
        return comb_cmi_from_upsilon_dense(Upsilon, base=base, check_psd=check_psd)

    def _quantum_mutual_information_experimental(
        self,
        base: int = 2,
        past: str = "all",
        normalize: bool = True,
    ) -> float:
        """DEPRECATED / EXPERIMENTAL — do not use for convergence comparisons.

        .. deprecated::
            This method constructs a *different* Choi object than
            ``reconstruct_comb_choi`` (it embeds the output 4-vector as a
            rank-1 operator on a 4-dim leg, giving dims [4, 4, ..., 4] instead
            of the standard [2, 4, ..., 4]).  Results are NOT comparable to
            ``comb_qmi_from_upsilon_dense``.

            Use instead::

                U  = pt.reconstruct_comb_choi(check=True)
                U_red = reduced_upsilon(U, k=k, keep_last_m=1)
                rho = canonicalize_upsilon(U_red, hermitize=True,
                                           psd_project=True, normalize_trace=True)
                qmi = comb_qmi_from_upsilon_dense(rho, assume_canonical=True)
        """
        import warnings
        warnings.warn(
            "ProcessTensor._quantum_mutual_information_experimental uses a "
            "different Choi object than reconstruct_comb_choi and is not "
            "comparable to comb_qmi_from_upsilon_dense. Use the standalone "
            "comb_qmi_from_upsilon_dense(canonicalize_upsilon(pt.reconstruct_comb_choi())) "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        return 0.0  # stub; body removed (see deprecation warning above)

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

        evals = np.linalg.eigvalsh(Upsilon).real
        lam_min = float(evals.min())
        scale = float(np.max(np.abs(evals)))
        tol = 1e-8 * max(1.0, scale)   # tune: 1e-8..1e-6 depending on your noise

        if check_psd and lam_min < -tol:
            # warn / log but don't crash
            # raise only if strict=True
            if False:
                raise ValueError(f"Reconstructed Υ not PSD (min eigenvalue {lam_min:.3e}, tol {tol:.3e}).")

        if check_psd:
            w, V = np.linalg.eigh(Upsilon)
            w = np.clip(w, 0.0, None)
            Upsilon = (V * w) @ V.conj().T

        # if check_psd:
        #     lam_min = float(np.linalg.eigvalsh(Upsilon).min().real)
        #     if lam_min < -1e-9:
        #         raise ValueError(f"Reconstructed Υ not PSD (min eigenvalue {lam_min:.3e}).")
        if check_psd:
            lam_min = float(np.linalg.eigvalsh(Upsilon).min().real)
            tr = float(np.trace(Upsilon).real)
            tol = -1e-6 * max(1.0, tr)   # adjust prefactor as needed
            if lam_min < tol:
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


def upsilon_mpo_to_dense(mpo: MPO) -> NDArray[np.complex128]:
    return mpo.to_matrix()


def _cptp_to_choi(emap: Callable[[NDArray[np.complex128]], NDArray[np.complex128]]) -> NDArray[np.complex128]:
    """Calculate the 4x4 Choi matrix for a given CPTP map."""
    # Basis: |0><0|, |0><1|, |1><0|, |1><1|
    basis = [
        np.array([[1, 0], [0, 0]], dtype=complex),
        np.array([[0, 1], [0, 0]], dtype=complex),
        np.array([[0, 0], [1, 0]], dtype=complex),
        np.array([[0, 0], [0, 1]], dtype=complex),
    ]
    # J(E) = sum_{i,j} E(|i><j|) \otimes |i><j|
    # Note: Using the convention that matches predict_final_state
    j_choi = np.zeros((4, 4), dtype=complex)
    for i, eij in enumerate(basis):
        out = emap(eij)
        # Vectorize out and place into column i of J or similar structure
        # Standard Choi: sum E(e_ij) \otimes e_ij
        term = np.kron(out, eij)
        j_choi += term
    return j_choi


def predict_from_dense_upsilon(
    U: NDArray[np.complex128],
    interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
) -> NDArray[np.complex128]:
    """Predict final state from a dense Choi matrix U and CPTP interventions."""
    k = len(interventions)
    dim_p = 4**k
    # U lives on H_F (dim 2) \otimes (H1 \otimes ... \otimes Hk) (dim 4 each)
    # We want: rho = Tr_past[ (I \otimes (\otimes_t J_t^T)) U ]
    
    # Bundle interventions into one big operator
    # NOTE: Since interventions are applied serially, we contract them step by step or all at once.
    # For a dense matrix, all at once is easier:
    j_total = _cptp_to_choi(interventions[0])
    for t in range(1, k):
        j_total = np.kron(j_total, _cptp_to_choi(interventions[t]))
    
    # Contract: rho = Tr_past[ U (I \otimes J_total^T) ]
    U4 = U.reshape(2, dim_p, 2, dim_p)
    ins = j_total.T # .T is consistent with predict_final_state basis
    
    # rho_{s,s'} = \sum_{p,p'} U_{s,p,s',p'} * ins_{p',p}
    rho = np.einsum("s p q r, r p -> s q", U4, ins)
    return rho
