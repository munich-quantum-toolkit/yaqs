# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dense and MPO process-tensor wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from mqt.yaqs.core.data_structures.mpo import MPO

from ...operational_memory.grid import assemble_probe_sequence
from ...shared.encoding import DEFAULT_INITIAL_RHO0, encode_rho_pauli
from ...shared.intervention_steps import AnyInterventionStep, build_intervention_operator
from ...shared.probabilities import PROBABILITY_ATOL

_HERMITICITY_ATOL = 1e-10
_PSD_ATOL = 1e-10
_TRACE_ATOL = 1e-12
_ZERO_BRANCH_NORM_ATOL = 64.0 * np.finfo(np.float64).eps

if TYPE_CHECKING:
    from collections.abc import Callable

    import scipy.sparse
    from numpy.typing import NDArray

    from ...operational_memory.samples import ProbeSet


class SupportsPredict(Protocol):
    """Process-tensor backends that map intervention sequences to a final state."""

    def predict(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Predict the final reduced state for a sequence of interventions."""
        ...


def _validate_hermitian_matrix(
    matrix: NDArray[np.complex128],
    *,
    name: str,
) -> NDArray[np.complex128]:
    """Validate and remove roundoff-scale anti-Hermitian residuals.

    Args:
        matrix: Matrix to validate.
        name: Name used in error messages.

    Returns:
        Hermitian average of ``matrix``.

    Raises:
        ValueError: If the matrix is malformed, non-finite, or not Hermitian.
    """
    value = np.asarray(matrix, dtype=np.complex128)
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[0] != value.shape[1]:
        msg = f"{name} must be a nonempty square rank-2 matrix, got shape {value.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(value)):
        msg = f"{name} must contain only finite values."
        raise ValueError(msg)
    scale = float(np.linalg.norm(value))
    residual = float(np.linalg.norm(value - value.conj().T))
    if residual > _HERMITICITY_ATOL * scale:
        msg = f"{name} must be Hermitian; residual norm is {residual:.3e}."
        raise ValueError(msg)
    return 0.5 * (value + value.conj().T)


def _validate_psd(
    matrix: NDArray[np.complex128],
    *,
    name: str,
) -> NDArray[np.float64]:
    """Validate positive semidefiniteness up to roundoff.

    Args:
        matrix: Hermitian matrix to validate.
        name: Name used in error messages.

    Returns:
        Eigenvalues of ``matrix``.

    Raises:
        ValueError: If an eigenvalue is negative beyond numerical tolerance.
    """
    eigenvalues = np.linalg.eigvalsh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    minimum = float(eigenvalues[0])
    if minimum < -_PSD_ATOL * scale:
        msg = f"{name} must be positive semidefinite; minimum eigenvalue is {minimum:.3e}."
        raise ValueError(msg)
    return eigenvalues.real


def _normalize_conditional_branch(
    rho_branch: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], float]:
    """Validate a subnormalized qubit branch and return its conditional state.

    Args:
        rho_branch: Subnormalized final-system outcome branch.

    Returns:
        Pair with the normalized conditional state and branch probability. An
        impossible branch returns the zero matrix and probability zero.

    Raises:
        ValueError: If the branch is malformed, non-Hermitian, has an invalid
            probability, has near-zero trace but nonzero norm, or is not positive
            semidefinite.
    """
    rho = np.asarray(rho_branch, dtype=np.complex128)
    if rho.shape != (2, 2):
        msg = f"Process-tensor branch must have shape (2, 2), got {rho.shape}."
        raise ValueError(msg)
    if not np.all(np.isfinite(rho)):
        msg = "Process-tensor branch must contain only finite values."
        raise ValueError(msg)
    if np.linalg.norm(rho) <= _ZERO_BRANCH_NORM_ATOL:
        return np.zeros((2, 2), dtype=np.complex128), 0.0

    rho = _validate_hermitian_matrix(rho, name="Process-tensor branch")
    weight = float(np.trace(rho).real)
    if weight < 0.0 or weight > 1.0 + PROBABILITY_ATOL:
        msg = (
            f"Process-tensor branch trace must be a probability in [0, 1], got {weight}. "
            "Direct-MPO compression or tomography error can make a reconstructed process tensor "
            "nonphysical. For a noiseless process, remove the experimental finite cap by setting "
            "max_bond_dim=None, or use return_type='dense'; for sampled tomography, improve the reconstruction."
        )
        raise ValueError(msg)
    if weight <= 0.0:
        msg = "Process-tensor branch has near-zero trace but a nonzero subnormalized output."
        raise ValueError(msg)

    normalized = rho / weight
    weight = float(np.clip(weight, 0.0, 1.0))
    eigenvalues = _validate_psd(normalized, name="Process-tensor conditional state")
    if np.any(eigenvalues < 0.0):
        eigenvalues, eigenvectors = np.linalg.eigh(normalized)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        normalized = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
        normalized /= np.trace(normalized)
    return normalized, weight


def _normalize_prediction(rho_branch: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Validate and normalize a process-tensor prediction.

    A zero-probability branch remains the zero matrix because its normalized
    conditional state is undefined.

    Args:
        rho_branch: Subnormalized final-system outcome branch.

    Returns:
        Validated final matrix, normalized for a nonzero branch.

    """
    rho, _ = _normalize_conditional_branch(rho_branch)
    return rho


def validate_initial_rho(
    rho0: NDArray[np.complex128],
    reference: NDArray[np.complex128],
    *,
    atol: float = 1e-8,
) -> None:
    """Raise if ``rho0`` does not match the process-tensor reference initial state.

    Args:
        rho0: User-supplied initial reduced state at the cut.
        reference: Reference site-0 state stored on the process tensor.
        atol: Absolute tolerance for element-wise comparison.

    Raises:
        ValueError: If the matrices differ beyond ``atol``.
    """
    got = np.asarray(rho0, dtype=np.complex128).reshape(2, 2)
    ref = np.asarray(reference, dtype=np.complex128).reshape(2, 2)
    if not np.allclose(got, ref, atol=atol):
        msg = "rho0 does not match the process-tensor reference initial state."
        raise ValueError(msg)


def convert_probe_callable(
    step: AnyInterventionStep,
) -> Callable[[NDArray[np.complex128]], NDArray[np.complex128]]:
    """Convert a probe-grid step to a CP map callable for :meth:`~SupportsPredict.predict`.

    Args:
        step: Structured dict step or measure/prepare ket pair.

    Returns:
        Callable implementing the single-qubit map for ``step``.
    """
    inter = build_intervention_operator(step)
    if isinstance(inter, np.ndarray):
        u_mat = cast("NDArray[np.complex128]", np.asarray(inter, dtype=np.complex128).reshape(2, 2))

        def unitary_map(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return u_mat @ rho @ u_mat.conj().T

        return unitary_map
    return inter


def evaluate_probes(process_tensor: SupportsPredict, probe_set: ProbeSet) -> np.ndarray:
    """Evaluate split-cut probe Pauli responses via process-tensor :meth:`predict`.

    Shared by dense and MPO process tensors for operational-memory V-matrix assembly.
    Does not densify MPO tensors.

    Args:
        process_tensor: Backend implementing :meth:`~SupportsPredict.predict`.
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


def _evaluate_probes_with_weights(
    contract_subnormalized_branch: Callable[
        [list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]]], NDArray[np.complex128]
    ],
    probe_set: ProbeSet,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate normalized final system responses and joint retained-outcome probabilities.

    Each subnormalized branch is checked for Hermiticity, a probability-valued
    trace, and positive semidefiniteness before normalization.

    Args:
        contract_subnormalized_branch: Bound method that contracts a subnormalized outcome branch.
        probe_set: Sampled split-cut probes.

    Returns:
        Tuple ``(pauli_ixyz_ij, weights_ij)``. The Pauli array has shape
        ``(n_pasts, n_futures, 4)`` and contains normalized final system responses.
        The weights are joint retained-outcome probabilities from the corresponding
        subnormalized outcome branches.

    """
    n_p = len(probe_set.past_pairs)
    n_f = len(probe_set.future_pairs)
    pauli = np.empty((n_p, n_f, 4), dtype=np.float64)
    weights = np.empty((n_p, n_f), dtype=np.float64)
    for i in range(n_p):
        for j in range(n_f):
            steps = assemble_probe_sequence(probe_set, i, j)
            interventions = [convert_probe_callable(step) for step in steps]
            normalized, weight = _normalize_conditional_branch(contract_subnormalized_branch(interventions))
            if weight <= 0.0:
                normalized = np.eye(2, dtype=np.complex128) / 2.0
            weights[i, j] = weight
            pauli[i, j] = encode_rho_pauli(normalized)
    return pauli, weights


def encode_map_choi(emap: Callable[[NDArray[np.complex128]], NDArray[np.complex128]]) -> NDArray[np.complex128]:
    """Convert a single-qubit map callable into its Choi matrix.

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


def _normalize_entropy_state(
    matrix: NDArray[np.complex128],
    base: int,
    *,
    name: str,
) -> tuple[NDArray[np.complex128], float]:
    """Normalize a positive matrix and compute its von Neumann entropy.

    Args:
        matrix: Positive semidefinite matrix.
        base: Logarithm base.
        name: Name used in error messages.

    Returns:
        Trace-normalized matrix and its entropy.

    Raises:
        ValueError: If ``base`` is not finite and greater than 1, or if ``matrix``
            does not define a nonzero positive semidefinite matrix.
    """
    if not np.isfinite(base) or base <= 1:
        msg = f"entropy base must be > 1, got {base!r}."
        raise ValueError(msg)
    rho_herm = _validate_hermitian_matrix(matrix, name=name)
    trace = float(np.trace(rho_herm).real)
    if not np.isfinite(trace) or trace <= _TRACE_ATOL:
        msg = f"{name} must have trace greater than {_TRACE_ATOL:.0e}, got {trace:.3e}."
        raise ValueError(msg)
    rho_herm /= trace
    evals = _validate_psd(rho_herm, name=name)
    evals = np.clip(evals, 0.0, None)
    evals /= float(evals.sum())
    nz = evals[evals > 0.0]
    entropy = float(-(nz * (np.log(nz) / np.log(base))).sum())
    upper_bound = float(np.log(rho_herm.shape[0]) / np.log(base))
    return rho_herm, float(np.clip(entropy, 0.0, upper_bound))


def compute_entropy_dense(r: NDArray[np.complex128], base: int = 2) -> float:
    """Compute von Neumann entropy of a (possibly unnormalized) density matrix.

    Args:
        r: Density matrix.
        base: Logarithm base.

    Returns:
        Von Neumann entropy in the given base.
    """
    _, entropy = _normalize_entropy_state(r, base, name="Entropy input")
    return entropy


def _validate_cut(cut: int, num_interventions: int) -> None:
    if cut < 1 or cut > num_interventions:
        msg = f"cut must satisfy 1 <= cut <= num_interventions ({num_interventions}), got {cut}."
        raise ValueError(msg)


def _unfuse_slot_index(fused: int, *, out_first: bool = True) -> tuple[int, int]:
    """Split a fused 4-index Choi leg into ``(output, input)`` qubit indices.

    ``encode_map_choi`` uses ``kron(output, input)`` so ``f = 2 * out + in`` by default.

    Returns:
        Tuple ``(output_index, input_index)`` each in ``{0, 1}``.
    """
    if out_first:
        return fused // 2, fused % 2
    return fused % 2, fused // 2


def _upsilon_to_unfused_operator(
    upsilon: NDArray[np.complex128],
    num_interventions: int,
    *,
    out_first: bool = True,
) -> NDArray[np.complex128]:
    """Reshape a process-tensor Choi operator into explicit ket/bra qubit axes.

    Subsystem order in ``upsilon`` is ``[final(2), slot_1(4), …, slot_k(4)]`` with
    ``slot_t = output_t ⊗ input_t`` and ``f = 2 * output + input`` when ``out_first=True``.

    Returns:
        Tensor with axes ``final_ket/bra`` then per-slot ``out/in`` ket/bra pairs.

    Raises:
        ValueError: If ``upsilon`` shape is inconsistent with ``num_interventions``.
    """
    k = num_interventions
    expected = 2 * (4**k)
    ups = np.asarray(upsilon, dtype=np.complex128)
    if ups.shape != (expected, expected):
        msg = f"Expected upsilon shape ({expected}, {expected}) for k={k}, got {ups.shape}."
        raise ValueError(msg)
    dims = [2] + [4] * k
    mat = ups.reshape(*dims, *dims)
    out = np.zeros([2, 2] + [2, 2, 2, 2] * k, dtype=np.complex128)
    for idx in np.ndindex(*dims, *dims):
        sub_k = idx[: k + 1]
        sub_b = idx[k + 1 :]
        coords: list[int] = [sub_k[0], sub_b[0]]
        for t in range(k):
            ok, ik = _unfuse_slot_index(sub_k[t + 1], out_first=out_first)
            ob, ib = _unfuse_slot_index(sub_b[t + 1], out_first=out_first)
            coords.extend([ok, ik, ob, ib])
        out[tuple(coords)] = mat[idx]
    return out


def _block_axis_indices(num_interventions: int) -> list[list[int]]:
    """Return unfused tensor axis indices for causal blocks ``B_0 … B_k``.

    Axis numbering matches :func:`_upsilon_to_unfused_operator`:

    - ``final_ket=0``, ``final_bra=1``
    - slot ``t`` (0-based): ``out_ket=2+4t``, ``in_ket=3+4t``, ``out_bra=4+4t``, ``in_bra=5+4t``

    Args:
        num_interventions: Number of intervention slots ``k``.

    Returns:
        List of ``k + 1`` blocks of axis indices.
    """
    k = num_interventions
    blocks: list[list[int]] = [[3, 5]]
    blocks.extend([2 + 4 * t, 3 + 4 * (t + 1), 4 + 4 * t, 5 + 4 * (t + 1)] for t in range(k - 1))
    blocks.append([2 + 4 * (k - 1), 0, 4 + 4 * (k - 1), 1])
    return blocks


def compute_temporal_entropy(
    upsilon: NDArray[np.complex128],
    num_interventions: int,
    cut: int,
    *,
    rtol: float = 1e-12,
    weight_tol: float = 1e-30,
) -> dict[str, NDArray[np.float64] | float | int]:
    r"""Compute temporal entanglement of the process tensor at a causal cut.

    Partitions causal blocks ``B_0, \ldots, B_k`` at cut ``c`` as::

        LEFT  = B_0, …, B_{c-1}
        RIGHT = B_c, …, B_k

    and computes the operator-Schmidt spectrum of the unfused Choi operator without
    partial tracing or trace normalization. The result is temporal entanglement
    :math:`S_{PT}(c)`, distinct from operational response entropy :math:`S_V(c)`.

    Args:
        upsilon: Dense process-tensor Choi matrix.
        num_interventions: Intervention count ``k``.
        cut: Causal cut index ``c`` matching the response protocol.
        rtol: Relative threshold ``s_i > rtol * s_0`` for resolved Schmidt rank.
        weight_tol: Absolute floor on ``sum(s**2)``; below this raises ``ValueError``.

    Returns:
        Dictionary with keys ``entropy`` (:math:`S_{PT}`), ``effective_rank``,
        ``schmidt_rank``, ``singular_values``, and ``weights``.

    Raises:
        ValueError: If ``cut`` is invalid or the squared-Schmidt weight sum is below ``weight_tol``.
    """
    _validate_cut(cut, num_interventions)
    op = _upsilon_to_unfused_operator(upsilon, num_interventions)
    blocks = _block_axis_indices(num_interventions)
    left_axes = [i for b in blocks[:cut] for i in b]
    right_axes = [i for b in blocks[cut:] for i in b]
    perm = left_axes + right_axes
    tensor_perm = np.transpose(op, perm)
    dim_left = int(np.prod([tensor_perm.shape[i] for i in range(len(left_axes))], dtype=np.int64))
    dim_right = int(
        np.prod([tensor_perm.shape[i] for i in range(len(left_axes), len(left_axes) + len(right_axes))], dtype=np.int64)
    )
    mat = tensor_perm.reshape(dim_left, dim_right)
    singular_values = np.linalg.svd(mat, compute_uv=False).astype(np.float64)
    total_weight = float(np.sum(singular_values**2))
    if total_weight < weight_tol:
        msg = f"Operator-Schmidt weight sum {total_weight:.3e} below tolerance {weight_tol:.3e}."
        raise ValueError(msg)
    weights = singular_values**2 / total_weight
    nz = weights > weight_tol
    entropy = float(-np.sum(weights[nz] * np.log(weights[nz]))) if np.any(nz) else 0.0
    if singular_values.size and singular_values[0] > 0.0:
        resolved = singular_values > rtol * singular_values[0]
    else:
        resolved = singular_values > 0.0
    schmidt_rank = int(np.sum(resolved))
    effective_rank = float(np.exp(entropy)) if entropy > 0.0 else 1.0
    return {
        "entropy": entropy,
        "effective_rank": effective_rank,
        "schmidt_rank": schmidt_rank,
        "singular_values": singular_values,
        "weights": weights,
    }


class DenseProcessTensor:
    """Wrapper around a dense process-tensor Choi operator Upsilon."""

    def __init__(
        self,
        upsilon: NDArray[np.complex128],
        timesteps: list[float],
        *,
        initial_rho: NDArray[np.complex128] | None = None,
    ) -> None:
        r"""Create a dense process-tensor wrapper.

        Args:
            upsilon: Dense process-tensor matrix.
            timesteps: Evolution schedule. A tensor with ``k > 0`` intervention
                legs requires ``k + 1`` durations. A zero-leg tensor requires an
                empty schedule.
            initial_rho: Site-0 reference state after ``U_0`` (defaults to ``|0\\rangle\\langle 0|``).

        Raises:
            ValueError: If ``upsilon`` is not a finite square rank-2 matrix with
                dimension ``2 * 4**k``, or if the schedule length does not match
                the inferred intervention count.
        """
        matrix = np.asarray(upsilon, dtype=np.complex128)
        if matrix.ndim != 2:
            msg = f"upsilon must be a rank-2 matrix, got shape {matrix.shape}."
            raise ValueError(msg)
        if matrix.shape[0] != matrix.shape[1]:
            msg = f"upsilon must be square, got shape {matrix.shape}."
            raise ValueError(msg)
        if not np.all(np.isfinite(matrix)):
            msg = "upsilon must contain only finite values."
            raise ValueError(msg)

        dimension = matrix.shape[0]
        slot_dimension = dimension // 2 if dimension % 2 == 0 else 0
        num_interventions = 0
        while slot_dimension > 1 and slot_dimension % 4 == 0:
            slot_dimension //= 4
            num_interventions += 1
        if slot_dimension != 1:
            msg = f"upsilon dimension must equal 2 * 4**k for a nonnegative intervention count k, got {dimension}."
            raise ValueError(msg)

        expected_timesteps = 0 if num_interventions == 0 else num_interventions + 1
        if len(timesteps) != expected_timesteps:
            msg = (
                f"A DenseProcessTensor with {num_interventions} intervention legs requires "
                f"{expected_timesteps} timesteps, got {len(timesteps)}."
            )
            raise ValueError(msg)

        self.upsilon = matrix
        self.timesteps = list(timesteps)
        self._num_intervention_steps = num_interventions
        self.initial_rho = (
            DEFAULT_INITIAL_RHO0.copy()
            if initial_rho is None
            else np.asarray(initial_rho, dtype=np.complex128).reshape(2, 2)
        )

    def check_initial_rho(
        self,
        rho0: NDArray[np.complex128],
        *,
        atol: float = 1e-8,
    ) -> None:
        """Validate ``rho0`` against :attr:`initial_rho`.

        Args:
            rho0: User-supplied initial reduced state at the cut.
            atol: Absolute tolerance for element-wise comparison.
        """
        validate_initial_rho(rho0, self.initial_rho, atol=atol)

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the underlying dense process-tensor matrix.

        Returns:
            Dense process-tensor matrix.
        """
        return self.upsilon

    def _num_interventions(self) -> int:
        """Infer number of intervention steps from the process-tensor matrix shape.

        Returns:
            Number of steps ``num_interventions`` such that the shape is
            ``(2*4**num_interventions, 2*4**num_interventions)``.
        """
        return self._num_intervention_steps

    def compute_temporal_entropy(
        self,
        cut: int,
        *,
        rtol: float = 1e-12,
        weight_tol: float = 1e-30,
    ) -> dict[str, NDArray[np.float64] | float | int]:
        """Compute temporal entanglement :math:`S_{PT}(c)` at ``cut``.

        Args:
            cut: Causal cut index ``c`` matching the response protocol.
            rtol: Relative Schmidt threshold for ``schmidt_rank``.
            weight_tol: Absolute floor on ``sum(s**2)``.

        Returns:
            Result dictionary from :func:`compute_temporal_entropy`.
        """
        return compute_temporal_entropy(
            self.upsilon,
            self._num_interventions(),
            cut,
            rtol=rtol,
            weight_tol=weight_tol,
        )

    def _contract_subnormalized_branch(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Contract a subnormalized outcome branch without validation or normalization.

        Args:
            interventions: List of CP intervention maps, one per step.

        Returns:
            Subnormalized final-system ``2 x 2`` matrix from the process-tensor contraction.

        Raises:
            ValueError: If the number of interventions does not match the process-tensor length.
        """
        k_steps = len(interventions)
        num_steps = self._num_interventions()
        if k_steps != num_steps:
            msg = (
                f"DenseProcessTensor expects {num_steps} interventions for "
                f"num_interventions={num_steps}, got {k_steps}."
            )
            raise ValueError(msg)
        if k_steps == 0:
            return np.asarray(self.upsilon, dtype=np.complex128).reshape(2, 2).copy()
        past_list = [encode_map_choi(emap) for emap in interventions]
        past_total = past_list[0]
        for p in past_list[1:]:
            past_total = np.kron(past_total, p)
        dim_p = 4**k_steps
        upsilon_4d = self.upsilon.reshape(2, dim_p, 2, dim_p)
        ins = past_total.T.reshape(dim_p, dim_p)
        return np.einsum("s p q r, r p -> s q", upsilon_4d, ins)

    def predict(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Predict the final reduced state for a sequence of interventions.

        Args:
            interventions: List of CP intervention maps, one per step.

        Returns:
            Validated 2x2 matrix, trace-normalized for a nonzero branch.

        """
        return _normalize_prediction(self._contract_subnormalized_branch(interventions))

    def _num_interventions_for_probe(self) -> int:
        return self._num_interventions()

    def evaluate_probes(self, probe_set: ProbeSet) -> np.ndarray:
        """Evaluate split-cut probe Pauli responses for V-matrix assembly.

        Args:
            probe_set: Sampled split-cut probes.

        Returns:
            Array of shape ``(n_pasts, n_futures, 4)``.
        """
        return evaluate_probes(self, probe_set)

    def evaluate_probes_with_weights(self, probe_set: ProbeSet) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate normalized final system responses with joint retained-outcome probabilities.

        Args:
            probe_set: Sampled split-cut probes.

        Returns:
            Tuple ``(pauli_ixyz_ij, weights_ij)`` derived from subnormalized outcome branches.
        """
        return _evaluate_probes_with_weights(self._contract_subnormalized_branch, probe_set)

    def qmi(
        self,
        base: int = 2,
        past: str = "all",
    ) -> float:
        """Compute quantum mutual information between final and past subsystems.

        Args:
            base: Log base for entropy.
            past: Which past legs to include: ``"all"``, ``"first"``, or ``"last"``.

        Returns:
            Quantum mutual information.

        Raises:
            ValueError: If ``past`` is invalid or the process tensor is nonphysical.
        """
        if past not in {"all", "first", "last"}:
            msg = f"Unknown past='{past}'."
            raise ValueError(msg)
        rho, entropy_total = _normalize_entropy_state(self.upsilon, base, name="Upsilon")

        k_steps = self._num_interventions()
        if k_steps == 0:
            return 0.0

        dims = [2] + [4] * k_steps
        if past == "all":
            keep_past = list(range(1, k_steps + 1))
        elif past == "last":
            keep_past = [k_steps]
        else:
            keep_past = [1]

        rho_final_sub = trace_partial_dense(rho, dims, keep=[0])
        rho_past_sub = trace_partial_dense(rho, dims, keep=keep_past)
        entropy_joint = (
            entropy_total
            if len(keep_past) == k_steps
            else compute_entropy_dense(trace_partial_dense(rho, dims, keep=[0, *keep_past]), base)
        )
        return compute_entropy_dense(rho_past_sub, base) + compute_entropy_dense(rho_final_sub, base) - entropy_joint

    def cmi(
        self,
        base: int = 2,
    ) -> float:
        """Compute conditional mutual information I(F:P_{<k} | P_k).

        Args:
            base: Log base for entropy.

        Returns:
            Conditional mutual information. Returns 0.0 for ``k<2``.

        """
        rho, entropy_total = _normalize_entropy_state(self.upsilon, base, name="Upsilon")

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
            - entropy_total
        )


class MPOProcessTensor(MPO):
    """Wrapper around an MPO representation of a process-tensor Choi operator Upsilon."""

    def __init__(
        self,
        upsilon_mpo: MPO,
        timesteps: list[float],
        *,
        initial_rho: NDArray[np.complex128] | None = None,
    ) -> None:
        r"""Create an MPO process-tensor wrapper.

        Args:
            upsilon_mpo: MPO representation of the process-tensor matrix.
            timesteps: Evolution schedule. A tensor with ``k > 0`` intervention
                legs requires ``k + 1`` durations. A zero-leg tensor requires an
                empty schedule.
            initial_rho: Site-0 reference state after ``U_0`` (defaults to ``|0\\rangle\\langle 0|``).

        Raises:
            ValueError: If the MPO tensors do not have the process-tensor site
                dimensions and consistent bonds, contain non-finite values, or
                disagree with the stored length or schedule.
        """
        tensors = list(upsilon_mpo.tensors)
        if not tensors:
            msg = "MPOProcessTensor requires at least one tensor."
            raise ValueError(msg)
        if upsilon_mpo.length != len(tensors):
            msg = f"MPO length metadata is {upsilon_mpo.length}, but the MPO contains {len(tensors)} tensors."
            raise ValueError(msg)

        previous_right_bond: int | None = None
        for site, tensor in enumerate(tensors):
            if tensor.ndim != 4:
                msg = f"MPO process-tensor site {site} must be rank 4, got shape {tensor.shape}."
                raise ValueError(msg)
            expected_physical_dimension = 2 if site == 0 else 4
            if tensor.shape[:2] != (expected_physical_dimension, expected_physical_dimension):
                msg = (
                    f"MPO process-tensor site {site} must have physical dimensions "
                    f"({expected_physical_dimension}, {expected_physical_dimension}), got {tensor.shape[:2]}."
                )
                raise ValueError(msg)
            left_bond, right_bond = tensor.shape[2:]
            if left_bond <= 0 or right_bond <= 0:
                msg = f"MPO process-tensor site {site} must have positive bond dimensions, got {tensor.shape[2:]}."
                raise ValueError(msg)
            if not np.all(np.isfinite(tensor)):
                msg = f"MPO process-tensor site {site} must contain only finite values."
                raise ValueError(msg)
            if previous_right_bond is not None and left_bond != previous_right_bond:
                msg = (
                    f"MPO process-tensor bond mismatch before site {site}: "
                    f"expected left bond {previous_right_bond}, got {left_bond}."
                )
                raise ValueError(msg)
            previous_right_bond = right_bond
        if tensors[0].shape[2] != 1 or tensors[-1].shape[3] != 1:
            msg = "MPO process tensor must have unit left and right boundary bonds."
            raise ValueError(msg)

        num_interventions = len(tensors) - 1
        expected_timesteps = 0 if num_interventions == 0 else num_interventions + 1
        if len(timesteps) != expected_timesteps:
            msg = (
                f"An MPOProcessTensor with {num_interventions} intervention legs requires "
                f"{expected_timesteps} timesteps, got {len(timesteps)}."
            )
            raise ValueError(msg)

        super().__init__()
        self.tensors = [tensor.copy() for tensor in tensors]
        self.length = len(tensors)
        self.physical_dimension = 2
        self.timesteps = list(timesteps)
        self.initial_rho = (
            DEFAULT_INITIAL_RHO0.copy()
            if initial_rho is None
            else np.asarray(initial_rho, dtype=np.complex128).reshape(2, 2)
        )

    def check_initial_rho(
        self,
        rho0: NDArray[np.complex128],
        *,
        atol: float = 1e-8,
    ) -> None:
        """Validate ``rho0`` against :attr:`initial_rho`.

        Args:
            rho0: User-supplied initial reduced state at the cut.
            atol: Absolute tolerance for element-wise comparison.
        """
        validate_initial_rho(rho0, self.initial_rho, atol=atol)

    def to_matrix(self) -> NDArray[np.complex128]:
        """Return the dense matrix in process-tensor causal-leg order.

        Process tensors store the final output leg first, followed by past
        intervention slots. This causal order is distinct from the spatial
        site-0-LSB order used by :class:`~mqt.yaqs.MPO`.

        Returns:
            Dense process-tensor matrix.
        """
        return self._to_matrix_site0_msb()

    def to_sparse_matrix(self) -> scipy.sparse.csr_matrix:
        """Return the sparse matrix in process-tensor causal-leg order.

        Process tensors store the final output leg first, followed by past
        intervention slots. This method matches :meth:`to_matrix`, rather than
        the spatial site-0-LSB order used by :class:`~mqt.yaqs.MPO`.

        Returns:
            Sparse process-tensor matrix in CSR format.
        """
        return self.reflected().to_sparse_matrix()

    def to_dense(self) -> DenseProcessTensor:
        """Convert this MPO process tensor to a dense process tensor.

        Returns:
            Dense process-tensor wrapper.
        """
        return DenseProcessTensor(self.to_matrix(), self.timesteps, initial_rho=self.initial_rho.copy())

    def _num_interventions_for_probe(self) -> int:
        return int(self.length) - 1

    def compute_temporal_entropy(
        self,
        cut: int,
        *,
        rtol: float = 1e-12,
        weight_tol: float = 1e-30,
    ) -> dict[str, NDArray[np.float64] | float | int]:
        """Compute temporal entanglement :math:`S_{PT}(c)` at ``cut``.

        This method densifies the complete MPO before the operator-Schmidt
        decomposition. For ``k`` intervention legs, the dense complex matrix
        uses ``64 * 16**k`` bytes before decomposition workspace. Restrict this
        method to about five intervention legs on a typical workstation. At six
        legs, the matrix alone uses 1 GiB.

        Args:
            cut: Causal cut index ``c`` matching the response protocol.
            rtol: Relative Schmidt threshold for ``schmidt_rank``.
            weight_tol: Absolute floor on ``sum(s**2)``.

        Returns:
            Result dictionary from :func:`compute_temporal_entropy`.
        """
        return self.to_dense().compute_temporal_entropy(cut, rtol=rtol, weight_tol=weight_tol)

    def evaluate_probes(self, probe_set: ProbeSet) -> np.ndarray:
        """Evaluate split-cut probe Pauli responses for V-matrix assembly.

        Uses native MPO :meth:`predict` (does not densify the process tensor).

        Args:
            probe_set: Sampled split-cut probes.

        Returns:
            Array of shape ``(n_pasts, n_futures, 4)`` with Pauli tomography coefficients.
        """
        return evaluate_probes(self, probe_set)

    def evaluate_probes_with_weights(self, probe_set: ProbeSet) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate normalized final system responses with joint retained-outcome probabilities.

        Uses native MPO contractions and does not densify the process tensor.

        Args:
            probe_set: Sampled split-cut probes.

        Returns:
            Tuple ``(pauli_ixyz_ij, weights_ij)`` derived from subnormalized outcome branches.
        """
        return _evaluate_probes_with_weights(self._contract_subnormalized_branch, probe_set)

    def _contract_subnormalized_branch(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Contract a subnormalized outcome branch without validation or normalization.

        Args:
            interventions: List of intervention maps, one per process-tensor leg.

        Returns:
            Subnormalized final ``2 x 2`` matrix.

        Raises:
            ValueError: If the interventions list is empty or its length mismatches the process tensor.
        """
        if not interventions:
            if self.length == 1:
                return self.partial_trace_sites([0]).to_matrix()
            msg = "interventions list must be non-empty."
            raise ValueError(msg)

        k_steps = len(interventions)
        if self.length != k_steps + 1:
            msg = (
                f"MPOProcessTensor length {self.length} inconsistent with number of "
                f"interventions {k_steps} (expected length = k + 1)."
            )
            raise ValueError(msg)

        work = MPO()
        work.length = self.length
        work.physical_dimension = self.physical_dimension
        work.tensors = [tensor.copy() for tensor in self.tensors]
        for t, emap in enumerate(interventions):
            j_choi = encode_map_choi(emap)
            work.apply_local_operator(site=t + 1, op=j_choi.T, left_action=True)
        return work.partial_trace_sites([0]).to_matrix()

    def predict(
        self,
        interventions: list[Callable[[NDArray[np.complex128]], NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        """Predict the final reduced state for a sequence of interventions.

        Args:
            interventions: List of CP intervention maps, one per past leg.

        Returns:
            Validated 2x2 matrix, trace-normalized for a nonzero branch.

        """
        return _normalize_prediction(self._contract_subnormalized_branch(interventions))

    def qmi(
        self,
        base: int = 2,
        past: str = "all",
    ) -> float:
        """Compute quantum mutual information between final and past subsystems.

        Args:
            base: Log base for entropy.
            past: Which past legs to include: ``"all"``, ``"first"``, or ``"last"``.

        Returns:
            Quantum mutual information.
        """
        return self.to_dense().qmi(base=base, past=past)

    def cmi(
        self,
        base: int = 2,
    ) -> float:
        """Compute conditional mutual information I(F:P_{<k} | P_k).

        Args:
            base: Log base for entropy.

        Returns:
            Conditional mutual information.
        """
        return self.to_dense().cmi(base=base)
