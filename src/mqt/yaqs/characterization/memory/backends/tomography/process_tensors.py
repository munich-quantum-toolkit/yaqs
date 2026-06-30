# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dense and MPO process-tensor wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from mqt.yaqs.core.data_structures.mpo import MPO

from ...operational_memory.grid import assemble_probe_sequence
from ...shared.encoding import encode_rho_pauli
from ..surrogates.utils import InterventionMap

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ...operational_memory.samples import ProbeSet

_RHO0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
_Z0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)

AnyInput = dict[str, Any] | tuple[Any, Any]


def convert_probe_step(step: AnyInput) -> InterventionMap | np.ndarray:
    """Normalize a probe-grid step to an intervention map or unitary matrix.

    Args:
        step: Structured dict step or measure/prepare ket pair.

    Returns:
        :class:`InterventionMap` or ``2 x 2`` unitary array.

    Raises:
        ValueError: If the step type is unsupported.
    """
    if isinstance(step, dict):
        step_type = str(step.get("type", "")).lower()
        if step_type == "unitary":
            return np.asarray(step["U"], dtype=np.complex128).reshape(2, 2)
        if step_type == "cut_measurement":
            psi_meas = np.asarray(step["psi_meas"], dtype=np.complex128).reshape(2)
            if "psi_reset" in step:
                psi_reset = np.asarray(step["psi_reset"], dtype=np.complex128).reshape(2)
            else:
                psi_reset = psi_meas
            return InterventionMap(
                rho_prep=np.outer(psi_reset, psi_reset.conj()),
                effect=np.outer(psi_meas, psi_meas.conj()),
            )
        if step_type == "cut_preparation":
            psi_prep = np.asarray(step["psi_prep"], dtype=np.complex128).reshape(2)
            return InterventionMap(
                rho_prep=np.outer(psi_prep, psi_prep.conj()),
                effect=np.eye(2, dtype=np.complex128),
            )
        msg = f"Unsupported probe step type: {step_type!r}"
        raise ValueError(msg)
    psi_meas, psi_prep = step
    psi_m = np.asarray(psi_meas, dtype=np.complex128).reshape(2)
    psi_p = np.asarray(psi_prep, dtype=np.complex128).reshape(2)
    return InterventionMap(
        rho_prep=np.outer(psi_p, psi_p.conj()),
        effect=np.outer(psi_m, psi_m.conj()),
    )


def convert_probe_callable(
    step: AnyInput,
) -> Callable[[NDArray[np.complex128]], NDArray[np.complex128]]:
    """Convert a probe-grid step to a CPTP map callable for :meth:`DenseProcessTensor.predict`.

    Args:
        step: Structured dict step or measure/prepare ket pair.

    Returns:
        Callable implementing the single-qubit map for ``step``.
    """
    inter = convert_probe_step(step)
    if isinstance(inter, np.ndarray):
        u_mat = cast("NDArray[np.complex128]", np.asarray(inter, dtype=np.complex128).reshape(2, 2))

        def unitary_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return u_mat @ rho @ u_mat.conj().T

        return unitary_map
    return inter


def evaluate_dense_probes(process_tensor: DenseProcessTensor, probe_set: ProbeSet) -> np.ndarray:
    """Evaluate split-cut probe Pauli responses on a dense process tensor.

    Args:
        process_tensor: Dense reference process-tensor backend.
        probe_set: Sampled split-cut probes.

    Returns:
        Array of shape ``(n_pasts, n_futures, 4)`` with Pauli tomography coefficients.
    """
    n_p = len(probe_set.past_pairs)
    n_f = len(probe_set.future_pairs)
    pauli = np.empty((n_p, n_f, 4), dtype=np.float32)
    for i in range(n_p):
        for j in range(n_f):
            steps = assemble_probe_sequence(probe_set, i, j)
            interventions = [convert_probe_callable(s) for s in steps]
            pauli[i, j] = encode_rho_pauli(process_tensor.predict(interventions))
    return pauli


def encode_cptp_choi(emap: Callable[[NDArray[np.complex128]], NDArray[np.complex128]]) -> NDArray[np.complex128]:
    """Convert a CPTP map callable into its Choi matrix.

    Args:
        emap: Callable implementing a single-qubit map ``rho -> emap(rho)``.

    Returns:
        4x4 Choi matrix for ``emap`` using the convention that matches the `predict` contraction.
    """
    j_choi = np.zeros((4, 4), dtype=complex)
    for i in range(2):
        for j in range(2):
            e_in = np.zeros((2, 2), dtype=complex)
            e_in[i, j] = 1.0
            j_choi += np.kron(emap(e_in), e_in)
    return j_choi


def trace_partial_dense(r: NDArray[np.complex128], dims: list[int], keep: list[int]) -> NDArray[np.complex128]:
    """Compute a partial trace of a dense operator.

    Args:
        r: Dense operator on the tensor product space.
        dims: Dimensions of each subsystem.
        keep: Indices of subsystems to keep.

    Returns:
        Reduced operator after tracing out subsystems not in ``keep``.

    Raises:
        ValueError: If ``keep`` contains out-of-range indices.
    """
    keep = sorted(keep)
    n = len(dims)
    if any(i < 0 or i >= n for i in keep):
        msg = "keep indices out of range"
        raise ValueError(msg)
    reshaped = r.reshape(*(dims + dims))
    trace_out = [i for i in range(n) if i not in keep]
    perm = keep + trace_out
    reshaped = reshaped.transpose(*(perm + [i + n for i in perm]))
    dim_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    dim_out = int(np.prod([dims[i] for i in trace_out])) if trace_out else 1
    reshaped = reshaped.reshape(dim_keep, dim_out, dim_keep, dim_out)
    return np.einsum("a b c b -> a c", reshaped)


def compute_entropy_dense(r: NDArray[np.complex128], base: int = 2) -> float:
    """Compute von Neumann entropy of a (possibly unnormalized) density matrix.

    Args:
        r: Density matrix.
        base: Logarithm base.

    Returns:
        Von Neumann entropy in the given base.

    Raises:
        ValueError: If ``base`` is not greater than 1.
    """
    if base <= 1:
        msg = f"entropy base must be > 1, got {base!r}."
        raise ValueError(msg)
    log_base = np.log(base)
    rho_herm = 0.5 * (r + r.conj().T)
    tr = np.trace(rho_herm)
    if abs(tr) < 1e-15:
        return 0.0
    rho_herm /= tr
    evals = np.linalg.eigvalsh(rho_herm).real
    evals = np.clip(evals, 0.0, 1.0)
    nz = evals[evals > 1e-15]
    if nz.size == 0:
        return 0.0
    return float(-(nz * (np.log(nz) / log_base)).sum())


class DenseProcessTensor:
    """Wrapper around a dense process-tensor Choi operator Upsilon."""

    def __init__(self, upsilon: NDArray[np.complex128], timesteps: list[float]) -> None:
        """Create a dense process-tensor wrapper.

        Args:
            upsilon: Dense process-tensor matrix.
            timesteps: Per-step evolution durations.
        """
        self.upsilon = upsilon
        self.timesteps = timesteps

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the underlying dense process-tensor matrix.

        Returns:
            Dense process-tensor matrix.
        """
        return self.upsilon

    # NOTE: previously there was a `DenseProcessTensor.fit(...)` entry point here.
    # The library now exposes only the exhaustive
    # `build_process_tensor(...) -> SequenceData -> to_*_process_tensor()` path.

    def _num_interventions(self) -> int:
        """Infer number of intervention steps from the process-tensor matrix shape.

        Returns:
            Number of steps ``num_interventions`` such that the shape is
            ``(2*4**num_interventions, 2*4**num_interventions)``.
        """
        size = self.upsilon.shape[0]
        return int(np.round(np.log2(size / 2) / 2))

    def canonicalize(
        self,
        *,
        hermitize: bool = True,
        psd_project: bool = False,
        normalize_trace: bool = True,
        _psd_tol: float = 1e-12,
    ) -> DenseProcessTensor:
        """Return a canonicalized process-tensor matrix.

        Args:
            hermitize: If ``True``, symmetrize to enforce Hermiticity.
            psd_project: If ``True``, project eigenvalues onto the PSD cone.
            normalize_trace: If ``True``, normalize by the trace when nonzero.
            _psd_tol: PSD tolerance (currently unused; kept for compatibility).

        Returns:
            New `DenseProcessTensor` with canonicalized matrix.
        """
        comb_mat = self.upsilon.copy()
        if hermitize:
            comb_mat = 0.5 * (comb_mat + comb_mat.conj().T)
        if psd_project:
            w, eig_vecs = np.linalg.eigh(comb_mat)
            w = np.clip(w, 0.0, None)
            comb_mat = (eig_vecs * w) @ eig_vecs.conj().T
        if normalize_trace:
            tr = np.trace(comb_mat)
            if abs(tr) > 1e-15:
                comb_mat /= tr
        return DenseProcessTensor(comb_mat, self.timesteps)

    def reduced(self, keep_last_m: int = 1) -> DenseProcessTensor:
        """Reduce the process tensor by tracing out early past legs.

        Args:
            keep_last_m: Number of most-recent past legs to keep.

        Returns:
            Reduced process tensor as a new `DenseProcessTensor`.

        Raises:
            ValueError: If ``keep_last_m`` is out of range.
        """
        k = self._num_interventions()
        if keep_last_m > k:
            msg = f"keep_last_m={keep_last_m} > num_interventions={k}"
            raise ValueError(msg)
        if keep_last_m <= 0:
            msg = f"keep_last_m must be >= 1, got {keep_last_m}"
            raise ValueError(msg)
        dim_m = 2 * (4**keep_last_m)
        dim_traced = 4 ** (k - keep_last_m)
        if dim_traced == 1:
            comb_reduced = self.upsilon.reshape(dim_m, dim_m)
        else:
            comb_6d = self.upsilon.reshape(2, dim_traced, 4**keep_last_m, 2, dim_traced, 4**keep_last_m)
            comb_reduced = np.einsum("iabjac->ibjc", comb_6d).reshape(dim_m, dim_m)
        t_red = self.timesteps[-keep_last_m:] if len(self.timesteps) >= keep_last_m else self.timesteps
        return DenseProcessTensor(comb_reduced, t_red)

    def _predict_raw(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Contract the process tensor with interventions without physicalization.

        Args:
            interventions: List of CPTP maps, one per step.

        Returns:
            Raw 2x2 complex matrix from the process-tensor contraction (not guaranteed physical).
        """
        k_steps = len(interventions)
        if k_steps == 0:
            return np.asarray(self.upsilon, dtype=np.complex128).reshape(2, 2).copy()
        past_list = [encode_cptp_choi(emap) for emap in interventions]
        past_total = past_list[0]
        for p in past_list[1:]:
            past_total = np.kron(past_total, p)
        dim_p = 4**k_steps
        comb_4d = self.upsilon.reshape(2, dim_p, 2, dim_p)
        ins = past_total.T.reshape(dim_p, dim_p)
        return np.einsum("s p q r, r p -> s q", comb_4d, ins)

    def predict(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Predict the final reduced state for a sequence of interventions.

        Args:
            interventions: List of CPTP maps, one per step.

        Returns:
            Physicalized 2x2 density matrix (Hermitian, PSD, trace-1).

        Raises:
            ValueError: If the number of interventions does not match the process-tensor length.
        """
        num_steps = self._num_interventions()
        if len(interventions) != num_steps:
            msg = (
                f"DenseProcessTensor expects {num_steps} interventions for "
                f"num_interventions={num_steps}, got {len(interventions)}."
            )
            raise ValueError(msg)
        rho = self._predict_raw(interventions)

        # Hermitize
        rho = 0.5 * (rho + rho.conj().T)

        # Normalize trace (if non-negligible)
        tr = np.trace(rho)
        if abs(tr) > 1e-12:
            rho /= tr

        # PSD projection
        w, eig_vecs = np.linalg.eigh(rho)
        w = np.clip(w, 0.0, None)
        rho = (eig_vecs * w) @ eig_vecs.conj().T
        tr2 = np.trace(rho)
        if abs(tr2) > 1e-15:
            rho /= tr2
        return rho

    def _num_interventions_for_probe(self) -> int:
        return self._num_interventions()

    def evaluate_probes(self, probe_set: ProbeSet) -> np.ndarray:
        """Evaluate split-cut probe Pauli responses.

        Returns:
            Array of shape ``(n_pasts, n_futures, 4)``.
        """
        return evaluate_dense_probes(self, probe_set)

    def qmi(
        self,
        base: int = 2,
        past: str = "all",
        *,
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Compute quantum mutual information between final and past subsystems.

        Args:
            base: Log base for entropy.
            past: Which past legs to include: ``"all"``, ``"first"``, or ``"last"``.
            check_psd: If ``True``, validate PSD before normalizing.
            assume_canonical: If ``True``, treat ``upsilon`` as already canonicalized.

        Returns:
            Quantum mutual information.

        Raises:
            ValueError: If ``past`` is invalid or PSD check fails.
        """
        if assume_canonical:
            rho = self.upsilon
        else:
            comb_mat = 0.5 * (self.upsilon + self.upsilon.conj().T)
            if check_psd:
                lam_min = float(np.linalg.eigvalsh(comb_mat).min().real)
                if lam_min < -1e-9:
                    msg = f"Upsilon not PSD (min eigenvalue {lam_min:.3e})."
                    raise ValueError(msg)
            tr = np.trace(comb_mat)
            rho = comb_mat / tr if abs(tr) > 1e-15 else comb_mat

        k_steps = self._num_interventions()
        if k_steps == 0:
            if past not in {"all", "first", "last"}:
                msg = f"Unknown past='{past}'."
                raise ValueError(msg)
            return 0.0

        dims = [2] + [4] * k_steps
        if past == "all":
            keep_past = list(range(1, k_steps + 1))
        elif past == "last":
            keep_past = [k_steps]
        elif past == "first":
            keep_past = [1]
        else:
            msg = f"Unknown past='{past}'."
            raise ValueError(msg)

        rho_final_sub = trace_partial_dense(rho, dims, keep=[0])
        rho_past_sub = trace_partial_dense(rho, dims, keep=keep_past)
        return (
            compute_entropy_dense(rho_past_sub, base)
            + compute_entropy_dense(rho_final_sub, base)
            - compute_entropy_dense(rho, base)
        )

    def cmi(
        self,
        base: int = 2,
        *,
        _check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Compute conditional mutual information I(F:P_{<k} | P_k).

        Args:
            base: Log base for entropy.
            _check_psd: If ``True``, validate PSD before normalizing (currently ignored).
            assume_canonical: If ``True``, treat ``upsilon`` as already canonicalized.

        Returns:
            Conditional mutual information. Returns 0.0 for ``k<2``.
        """
        if assume_canonical:
            rho = self.upsilon
        else:
            comb_mat = 0.5 * (self.upsilon + self.upsilon.conj().T)
            tr = np.trace(comb_mat)
            rho = comb_mat / tr if abs(tr) > 1e-15 else comb_mat

        k_steps = self._num_interventions()
        if k_steps < 2:
            return 0.0
        dims = [2] + [4] * k_steps
        rho_final_past_k = trace_partial_dense(rho, dims, keep=[0, k_steps])
        rho_past_sub = trace_partial_dense(rho, dims, keep=[*list(range(1, k_steps)), k_steps])
        rho_past_k = trace_partial_dense(rho, dims, keep=[k_steps])
        return (
            compute_entropy_dense(rho_final_past_k, base)
            + compute_entropy_dense(rho_past_sub, base)
            - compute_entropy_dense(rho_past_k, base)
            - compute_entropy_dense(rho, base)
        )

    def cmi_conditional(
        self,
        *,
        a_label: str = "first",
        b_label: str = "final",
        c_label: str = "last",
        base: int = 2,
        normalize: bool = True,
        check_psd: bool = True,
    ) -> float:
        """Compute I(A:B|C) for selected subsystem labels.

        Args:
            a_label: Label for subsystem A: ``"first"``, ``"last"``, or ``"final"``.
            b_label: Label for subsystem B: ``"first"``, ``"last"``, or ``"final"``.
            c_label: Label for subsystem C: ``"first"``, ``"last"``, or ``"final"``.
            base: Log base for entropy.
            normalize: Whether to normalize the process-tensor matrix by trace.
            check_psd: Whether to project onto PSD before computing entropies.

        Returns:
            Conditional mutual information.

        Raises:
            ValueError: If a subsystem label is invalid.
        """
        comb_mat = 0.5 * (self.upsilon + self.upsilon.conj().T)
        if check_psd:
            w, eig_vecs = np.linalg.eigh(comb_mat)
            w = np.clip(w, 0.0, None)
            comb_mat = (eig_vecs * w) @ eig_vecs.conj().T
        if normalize:
            tr = np.trace(comb_mat)
            if abs(tr) < 1e-15:
                return 0.0
            rho = comb_mat / tr
        else:
            rho = comb_mat

        k_steps = self._num_interventions()
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
            msg = f"Unknown subsystem '{which}'."
            raise ValueError(msg)

        idx_a, idx_b, idx_c = _idx(a_label), _idx(b_label), _idx(c_label)
        if len({idx_a, idx_b, idx_c}) != 3:
            msg = "A, B, C must be three distinct subsystems."
            raise ValueError(msg)

        rho_ac = trace_partial_dense(rho, dims, keep=[idx_a, idx_c])
        rho_bc = trace_partial_dense(rho, dims, keep=[idx_b, idx_c])
        rho_c = trace_partial_dense(rho, dims, keep=[idx_c])
        return (
            compute_entropy_dense(rho_ac, base)
            + compute_entropy_dense(rho_bc, base)
            - compute_entropy_dense(rho_c, base)
            - compute_entropy_dense(rho, base)
        )


class MPOProcessTensor(MPO):
    """Wrapper around an MPO representation of a process-tensor Choi operator Upsilon."""

    def __init__(self, upsilon_mpo: MPO, timesteps: list[float]) -> None:
        """Create an MPO process-tensor wrapper.

        Args:
            upsilon_mpo: MPO representation of the process-tensor matrix.
            timesteps: Per-step evolution durations.
        """
        # Copy underlying MPO tensors/state into this subclass
        super().__init__()
        self.tensors = [t.copy() for t in upsilon_mpo.tensors]
        self.length = upsilon_mpo.length
        self.physical_dimension = upsilon_mpo.physical_dimension
        self.timesteps = timesteps

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the dense matrix representation.

        Returns:
            Dense process-tensor matrix.
        """
        return super().to_matrix()

    def to_dense(self) -> DenseProcessTensor:
        """Convert this MPO process tensor to a dense process tensor.

        Returns:
            Dense process-tensor wrapper.
        """
        return DenseProcessTensor(self.to_matrix(), self.timesteps)

    def _num_interventions_for_probe(self) -> int:
        return int(self.length) - 1

    def evaluate_probes(self, probe_set: ProbeSet) -> np.ndarray:
        """Evaluate split-cut probe Pauli responses.

        Returns:
            Array of shape ``(n_pasts, n_futures, 4)`` with Pauli tomography coefficients.
        """
        return self.to_dense().evaluate_probes(probe_set)

    def predict(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Predict the final reduced state for a sequence of interventions.

        Args:
            interventions: List of CPTP maps, one per past leg.

        Returns:
            Physicalized 2x2 density matrix (Hermitian, PSD, trace-1).

        Raises:
            ValueError: If the interventions list is empty or length mismatches the process tensor.
        """
        if not interventions:
            if self.length == 1:
                reduced = self.partial_trace_sites([0])
                rho = reduced.to_matrix()
                rho = 0.5 * (rho + rho.conj().T)
                tr = np.trace(rho)
                if abs(tr) > 1e-12:
                    rho /= tr
                else:
                    rho = np.eye(2, dtype=np.complex128) / 2.0
                w, eig_vecs = np.linalg.eigh(rho)
                w = np.clip(w, 0.0, None)
                rho = (eig_vecs * w) @ eig_vecs.conj().T
                tr = np.trace(rho)
                if abs(tr) > 1e-12:
                    rho /= tr
                return rho.astype(np.complex128, copy=False)

            msg = "interventions list must be non-empty."
            raise ValueError(msg)

        k_steps = len(interventions)
        if self.length != k_steps + 1:
            msg = (
                f"MPOProcessTensor length {self.length} inconsistent with number of "
                f"interventions {k_steps} (expected length = k + 1)."
            )
            raise ValueError(msg)

        # Work on a copy so the original MPOProcessTensor remains unchanged.
        work = MPO()
        work.length = self.length
        work.physical_dimension = self.physical_dimension
        work.tensors = [t.copy() for t in self.tensors]

        # Apply local Choi operators (with transpose as in DenseProcessTensor.predict) on past sites.
        for t, emap in enumerate(interventions):
            j_choi = encode_cptp_choi(emap)  # 4x4
            work.apply_local_operator(site=t + 1, op=j_choi.T, left_action=True)

        # Trace out all past sites, keep only the final site (index 0).
        reduced = work.partial_trace_sites([0])

        # The remaining MPO encodes a single 2x2 matrix on the final leg.
        rho = reduced.to_matrix()

        # Match DenseProcessTensor.predict: Hermitian, PSD, trace-1.
        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho)
        if abs(tr) > 1e-12:
            rho /= tr
        w, eig_vecs = np.linalg.eigh(rho)
        w = np.clip(w, 0.0, None)
        rho = (eig_vecs * w) @ eig_vecs.conj().T
        tr2 = np.trace(rho)
        if abs(tr2) > 1e-15:
            rho /= tr2
        return rho

    def qmi(
        self,
        base: int = 2,
        past: str = "all",
        *,
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Compute quantum mutual information between final and past subsystems.

        Args:
            base: Log base for entropy.
            past: Which past legs to include: ``"all"``, ``"first"``, or ``"last"``.
            check_psd: Passed through to the dense implementation.
            assume_canonical: Passed through to the dense implementation.

        Returns:
            Quantum mutual information.
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
        *,
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Compute conditional mutual information I(F:P_{<k} | P_k).

        Args:
            base: Log base for entropy.
            check_psd: Passed through to the dense implementation.
            assume_canonical: Passed through to the dense implementation.

        Returns:
            Conditional mutual information.
        """
        return self.to_dense().cmi(
            base=base,
            _check_psd=check_psd,
            assume_canonical=assume_canonical,
        )

    def cmi_conditional(
        self,
        *,
        a_label: str = "first",
        b_label: str = "final",
        c_label: str = "last",
        base: int = 2,
        normalize: bool = True,
        check_psd: bool = True,
    ) -> float:
        """Compute I(A:B|C) for selected subsystem labels.

        Args:
            a_label: Label for subsystem A: ``"first"``, ``"last"``, or ``"final"``.
            b_label: Label for subsystem B: ``"first"``, ``"last"``, or ``"final"``.
            c_label: Label for subsystem C: ``"first"``, ``"last"``, or ``"final"``.
            base: Log base for entropy.
            normalize: Passed through to the dense implementation.
            check_psd: Passed through to the dense implementation.

        Returns:
            Conditional mutual information.
        """
        return self.to_dense().cmi_conditional(
            a_label=a_label,
            b_label=b_label,
            c_label=c_label,
            base=base,
            normalize=normalize,
            check_psd=check_psd,
        )
