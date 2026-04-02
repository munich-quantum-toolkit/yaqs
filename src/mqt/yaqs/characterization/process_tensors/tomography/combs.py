"""Dense and MPO comb (process-tensor) wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from mqt.yaqs.core.data_structures.networks import MPO

if TYPE_CHECKING:
    from collections.abc import Callable
    from numpy.typing import NDArray


def _cptp_to_choi(emap: Callable[[NDArray[np.complex128]], NDArray[np.complex128]]) -> NDArray[np.complex128]:
    """Convert a CPTP map callable to its Choi matrix J(E).

    J(E) = sum_{i,j} E(|i><j|) ⊗ |i><j| (order matches predict contraction).
    """
    j_choi = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            e_in = np.zeros((2, 2), dtype=complex)
            e_in[i, j] = 1.0
            j_choi += np.kron(emap(e_in), e_in)
    return j_choi


def _partial_trace_dense(r: NDArray[np.complex128], dims: list[int], keep: list[int]) -> NDArray[np.complex128]:
    """Partial trace of a dense operator keeping subsystems in keep."""
    keep = sorted(keep)
    n = len(dims)
    if any(i < 0 or i >= n for i in keep):
        raise ValueError("keep indices out of range")
    reshaped = r.reshape(*(dims + dims))
    trace_out = [i for i in range(n) if i not in keep]
    perm = keep + trace_out
    reshaped = reshaped.transpose(*(perm + [i + n for i in perm]))
    dim_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    dim_out = int(np.prod([dims[i] for i in trace_out])) if trace_out else 1
    reshaped = reshaped.reshape(dim_keep, dim_out, dim_keep, dim_out)
    return np.einsum("a b c b -> a c", reshaped)


def _entropy_dense(r: NDArray[np.complex128], base: int = 2) -> float:
    """Von Neumann entropy of a dense density matrix."""
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


class DenseComb:
    """Wrapper around a dense comb Choi operator Υ."""

    def __init__(self, upsilon: NDArray[np.complex128], timesteps: list[float]) -> None:
        self.upsilon = upsilon
        self.timesteps = timesteps

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the underlying dense comb matrix Υ."""
        return self.upsilon

    # NOTE: previously there was a `DenseComb.fit(...)` entry point here.
    # The library now exposes only the exhaustive `construct(...) -> SequenceData -> to_*_comb()` path.

    def _k_steps(self) -> int:
        """Number of intervention steps from Υ shape (2·4^k, 2·4^k)."""
        size = self.upsilon.shape[0]
        return int(np.round(np.log2(size / 2) / 2))

    def canonicalize(
        self,
        *,
        hermitize: bool = True,
        psd_project: bool = False,
        normalize_trace: bool = True,
        psd_tol: float = 1e-12,
    ) -> DenseComb:
        """Return a new DenseComb with canonicalized Υ (hermitize, optional PSD, optional normalize)."""
        U = self.upsilon.copy()
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
        return DenseComb(U, self.timesteps)

    def reduced(self, keep_last_m: int = 1) -> DenseComb:
        """Return a DenseComb with Υ reduced to the last keep_last_m past legs."""
        k = self._k_steps()
        if keep_last_m > k:
            raise ValueError(f"keep_last_m={keep_last_m} > k={k}")
        if keep_last_m <= 0:
            raise ValueError(f"keep_last_m must be >= 1, got {keep_last_m}")
        dim_m = 2 * (4**keep_last_m)
        dim_traced = 4 ** (k - keep_last_m)
        if dim_traced == 1:
            U_red = self.upsilon.reshape(dim_m, dim_m)
        else:
            U6 = self.upsilon.reshape(
                2, dim_traced, 4**keep_last_m, 2, dim_traced, 4**keep_last_m
            )
            U_red = np.einsum("iabjac->ibjc", U6).reshape(dim_m, dim_m)
        t_red = self.timesteps[-keep_last_m:] if len(self.timesteps) >= keep_last_m else self.timesteps
        return DenseComb(U_red, t_red)

    def _predict_raw(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Raw contraction to a 2x2 matrix (not guaranteed physical)."""
        k_steps = len(interventions)
        past_list = [_cptp_to_choi(emap) for emap in interventions]
        past_total = past_list[0]
        for p in past_list[1:]:
            past_total = np.kron(past_total, p)
        dim_p = 4**k_steps
        U4 = self.upsilon.reshape(2, dim_p, 2, dim_p)
        ins = past_total.T.reshape(dim_p, dim_p)
        return np.einsum("s p q r, r p -> s q", U4, ins)

    def predict(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Predict final output state given a list of CPTP interventions.

        The raw contraction is projected to a physical density matrix (Hermitian, PSD,
        trace-1) via Hermitization, trace normalization, and PSD projection.
        """
        rho = self._predict_raw(interventions)

        # Hermitize
        rho = 0.5 * (rho + rho.conj().T)

        # Normalize trace (if non-negligible)
        tr = np.trace(rho)
        if abs(tr) > 1e-12:
            rho = rho / tr

        # PSD projection
        w, V = np.linalg.eigh(rho)
        w = np.clip(w, 0.0, None)
        rho = (V * w) @ V.conj().T
        tr2 = np.trace(rho)
        if abs(tr2) > 1e-15:
            rho = rho / tr2
        return rho

    def qmi(
        self,
        base: int = 2,
        past: str = "all",
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Quantum mutual information I(F:P) from this comb."""
        if assume_canonical:
            rho = self.upsilon
        else:
            U = 0.5 * (self.upsilon + self.upsilon.conj().T)
            if check_psd:
                lam_min = float(np.linalg.eigvalsh(U).min().real)
                if lam_min < -1e-9:
                    raise ValueError(f"Υ not PSD (min eigenvalue {lam_min:.3e}).")
            tr = np.trace(U)
            rho = U / tr if abs(tr) > 1e-15 else U

        k_steps = self._k_steps()
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

    def cmi(
        self,
        base: int = 2,
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Conditional mutual information I(F:P_{<k} | P_k) from this comb."""
        if assume_canonical:
            rho = self.upsilon
        else:
            U = 0.5 * (self.upsilon + self.upsilon.conj().T)
            tr = np.trace(U)
            rho = U / tr if abs(tr) > 1e-15 else U

        k_steps = self._k_steps()
        if k_steps < 2:
            return 0.0
        dims = [2] + [4] * k_steps
        rho_FPk = _partial_trace_dense(rho, dims, keep=[0, k_steps])
        rho_P = _partial_trace_dense(rho, dims, keep=list(range(1, k_steps)) + [k_steps])
        rho_Pk = _partial_trace_dense(rho, dims, keep=[k_steps])
        return (
            _entropy_dense(rho_FPk, base)
            + _entropy_dense(rho_P, base)
            - _entropy_dense(rho_Pk, base)
            - _entropy_dense(rho, base)
        )

    def cmi_conditional(
        self,
        *,
        A: str = "first",
        B: str = "final",
        C: str = "last",
        base: int = 2,
        normalize: bool = True,
        check_psd: bool = True,
    ) -> float:
        """I(A:B|C) with subsystems 'final', 'first', 'last'."""
        U = 0.5 * (self.upsilon + self.upsilon.conj().T)
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

        k_steps = self._k_steps()
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
            raise ValueError(f"Unknown subsystem '{which}'.")

        iA, iB, iC = _idx(A), _idx(B), _idx(C)
        if len({iA, iB, iC}) != 3:
            raise ValueError("A, B, C must be three distinct subsystems.")

        rho_AC = _partial_trace_dense(rho, dims, keep=[iA, iC])
        rho_BC = _partial_trace_dense(rho, dims, keep=[iB, iC])
        rho_C = _partial_trace_dense(rho, dims, keep=[iC])
        return (
            _entropy_dense(rho_AC, base)
            + _entropy_dense(rho_BC, base)
            - _entropy_dense(rho_C, base)
            - _entropy_dense(rho, base)
        )


class MPOComb(MPO):
    """Wrapper around an MPO representation of a comb Choi operator Υ."""

    def __init__(self, upsilon_mpo: MPO, timesteps: list[float]) -> None:
        """Initialize MPOComb from an existing MPO and associated timesteps."""
        # Copy underlying MPO tensors/state into this subclass
        super().__init__()
        self.tensors = [t.copy() for t in upsilon_mpo.tensors]
        self.length = upsilon_mpo.length
        self.physical_dimension = upsilon_mpo.physical_dimension
        self.timesteps = timesteps

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the dense matrix representation of Υ."""
        return super().to_matrix()

    def to_dense(self) -> DenseComb:
        """Convert the MPO comb to a DenseComb."""
        return DenseComb(self.to_matrix(), self.timesteps)

    def predict(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Predict final state given a list of CPTP interventions.

        This uses only MPO-local operations:

        1. Build local Choi operators J(E_t) for each intervention.
        2. Apply J(E_t)^T on the corresponding past sites of the comb MPO.
        3. Trace out all past sites on the physical level.
        4. Read out the remaining 2x2 final state from the single-site MPO.
        """
        if not interventions:
            msg = "interventions list must be non-empty."
            raise ValueError(msg)

        k_steps = len(interventions)
        if self.length != k_steps + 1:
            msg = (
                f"MPOComb length {self.length} inconsistent with number of "
                f"interventions {k_steps} (expected length = k + 1)."
            )
            raise ValueError(msg)

        # Work on a copy so the original MPOComb remains unchanged.
        work = MPO()
        work.length = self.length
        work.physical_dimension = self.physical_dimension
        work.tensors = [t.copy() for t in self.tensors]

        # Apply local Choi operators (with transpose as in DenseComb.predict) on past sites.
        for t, emap in enumerate(interventions):
            j_choi = _cptp_to_choi(emap)  # 4x4
            work.apply_local_operator(site=t + 1, op=j_choi.T, left_action=True)

        # Trace out all past sites, keep only the final site (index 0).
        reduced = work.partial_trace_sites([0])

        # The remaining MPO encodes a single 2x2 matrix on the final leg.
        rho = reduced.to_matrix()

        # Match DenseComb.predict: Hermitian, PSD, trace-1.
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho)
        if abs(tr) > 1e-12:
            rho = rho / tr
        w, V = np.linalg.eigh(rho)
        w = np.clip(w, 0.0, None)
        rho = (V * w) @ V.conj().T
        tr2 = np.trace(rho)
        if abs(tr2) > 1e-15:
            rho = rho / tr2
        return rho

    def qmi(
        self,
        base: int = 2,
        past: str = "all",
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Quantum mutual information I(F:P) from this MPO comb.

        This delegates to the dense implementation via ``DenseComb``.
        """
        return self.to_dense().qmi(
            base=base,
            past=past,
            check_psd=check_psd,
            assume_canonical=assume_canonical,
        )

    def cmi(
        self,
        base: int = 2,
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Conditional mutual information I(F:P_{<k} | P_k) from this MPO comb.

        This delegates to the dense implementation via ``DenseComb``.
        """
        return self.to_dense().cmi(
            base=base,
            check_psd=check_psd,
            assume_canonical=assume_canonical,
        )

    def cmi_conditional(
        self,
        *,
        A: str = "first",
        B: str = "final",
        C: str = "last",
        base: int = 2,
        normalize: bool = True,
        check_psd: bool = True,
    ) -> float:
        """I(A:B|C) with subsystems 'final', 'first', 'last' for this MPO comb.

        This delegates to the dense implementation via ``DenseComb``.
        """
        return self.to_dense().cmi_conditional(
            A=A,
            B=B,
            C=C,
            base=base,
            normalize=normalize,
            check_psd=check_psd,
        )


