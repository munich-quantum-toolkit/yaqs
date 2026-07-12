# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Dense and MPO process-tensor wrappers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import numpy as np

from mqt.yaqs.core.data_structures.mpo import MPO

from ...operational_memory.grid import assemble_probe_sequence
from ...shared.encoding import DEFAULT_INITIAL_RHO0, encode_rho_pauli
from ...shared.intervention_steps import AnyInterventionStep, build_intervention_operator

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ...operational_memory.samples import ProbeSet


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
    """Convert a probe-grid step to a CPTP map callable for :meth:`DenseProcessTensor.predict`.

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


def _canonicalize_upsilon(
    upsilon: NDArray[np.complex128],
    *,
    check_psd: bool = False,
    assume_canonical: bool = False,
) -> NDArray[np.complex128]:
    """Hermitize and trace-normalize a process-tensor Choi operator.

    Returns:
        Trace-normalized Hermitian Choi operator.

    Raises:
        ValueError: If ``check_psd`` is ``True`` and the operator is not PSD.
    """
    if assume_canonical:
        return upsilon
    upsilon_mat = 0.5 * (upsilon + upsilon.conj().T)
    if check_psd:
        lam_min = float(np.linalg.eigvalsh(upsilon_mat).min().real)
        if lam_min < -1e-9:
            msg = f"Upsilon not PSD (min eigenvalue {lam_min:.3e})."
            raise ValueError(msg)
    tr = np.trace(upsilon_mat)
    return upsilon_mat / tr if abs(tr) > 1e-15 else upsilon_mat


def _validate_cut(cut: int, num_interventions: int) -> None:
    if cut < 1 or cut > num_interventions:
        msg = f"cut must satisfy 1 <= cut <= num_interventions ({num_interventions}), got {cut}."
        raise ValueError(msg)


def _unfuse_slot_index(fused: int, *, out_first: bool = True) -> tuple[int, int]:
    """Split a fused 4-index Choi leg into ``(output, input)`` qubit indices.

    ``encode_cptp_choi`` uses ``kron(output, input)`` so ``f = 2 * out + in`` by default.

    Returns:
        Tuple ``(output_index, input_index)`` each in ``{0, 1}``.
    """
    if out_first:
        return fused // 2, fused % 2
    return fused % 2, fused // 2


def _fuse_slot_index(out: int, inp: int, *, out_first: bool = True) -> int:
    if out_first:
        return 2 * out + inp
    return 2 * inp + out


def upsilon_to_unfused_operator(
    upsilon: NDArray[np.complex128],
    num_interventions: int,
    *,
    out_first: bool = True,
) -> NDArray[np.complex128]:
    """Reshape a process-tensor Choi operator into explicit ket/bra qubit axes.

    Subsystem order in ``upsilon`` is ``[final(2), slot_1(4), …, slot_k(4)]`` with
    ``slot_t = output_t ⊗ input_t`` and ``f = 2 * output + input`` when ``out_first=True``.

    Returns:
        Tensor with axes, in order::

            final_ket, final_bra,
            slot1_out_ket, slot1_in_ket, slot1_out_bra, slot1_in_bra,
            …,
            slotk_out_ket, slotk_in_ket, slotk_out_bra, slotk_in_bra.

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


def refold_unfused_to_upsilon(
    op: NDArray[np.complex128],
    num_interventions: int,
    *,
    out_first: bool = True,
) -> NDArray[np.complex128]:
    """Inverse of :func:`upsilon_to_unfused_operator`.

    Returns:
        Dense process-tensor Choi matrix with fused intervention legs.
    """
    k = num_interventions
    dims = [2] + [4] * k
    mat = np.zeros((2 * 4**k, 2 * 4**k), dtype=np.complex128)
    view = mat.reshape(*dims, *dims)
    n = k + 1
    for idx in np.ndindex(*dims, *dims):
        sub_k, sub_b = idx[:n], idx[n:]
        coords: list[int] = [sub_k[0], sub_b[0]]
        for t in range(k):
            ok, ik = _unfuse_slot_index(sub_k[t + 1], out_first=out_first)
            ob, ib = _unfuse_slot_index(sub_b[t + 1], out_first=out_first)
            coords.extend([ok, ik, ob, ib])
        view[idx] = op[tuple(coords)]
    return mat


def causal_block_axis_labels(num_interventions: int) -> list[list[str]]:
    """Return semantic labels for causal blocks ``B_0 … B_k``.

    Args:
        num_interventions: Number of intervention slots ``k``.

    Returns:
        List of ``k + 1`` blocks, each a list of axis-name strings.
    """
    k = num_interventions
    blocks: list[list[str]] = [["slot1_in_ket", "slot1_in_bra"]]
    blocks.extend(
        [
            f"slot{t + 1}_out_ket",
            f"slot{t + 2}_in_ket",
            f"slot{t + 1}_out_bra",
            f"slot{t + 2}_in_bra",
        ]
        for t in range(k - 1)
    )
    blocks.append([
        f"slot{k}_out_ket",
        "final_ket",
        f"slot{k}_out_bra",
        "final_bra",
    ])
    return blocks


def causal_block_axis_indices(num_interventions: int) -> list[list[int]]:
    """Return unfused tensor axis indices for causal blocks ``B_0 … B_k``.

    Axis numbering matches :func:`upsilon_to_unfused_operator`:

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


def causal_block_operator_entropy(
    upsilon: NDArray[np.complex128],
    num_interventions: int,
    cut: int,
    *,
    rtol: float = 1e-12,
    weight_tol: float = 1e-30,
) -> dict[str, NDArray[np.complex128] | float | int | list[int] | list[list[int]] | list[list[str]]]:
    """Operator-space entropy across causally regrouped channel blocks.

    Partitions causal blocks ``B_0, …, B_k`` at cut ``c`` as::

        LEFT  = B_0, …, B_{c-1}
        RIGHT = B_c, …, B_k

    and computes the operator-Schmidt spectrum of the unfused Choi operator without
    partial tracing or trace normalization.

    Args:
        upsilon: Dense process-tensor Choi matrix.
        num_interventions: Intervention count ``k``.
        cut: Causal cut index ``c`` matching the response protocol.
        rtol: Relative threshold ``s_i > rtol * s_0`` for resolved Schmidt rank.
        weight_tol: Absolute floor on ``sum(s**2)``; below this raises ``ValueError``.

    Returns:
        Dictionary with keys ``entropy`` (:math:`S_{PT}^{cb}`), ``effective_rank``,
        ``schmidt_rank``, ``singular_values``, ``weights``, ``left_axes``, ``right_axes``,
        ``blocks``, ``block_labels``.

    Raises:
        ValueError: If ``cut`` is invalid or the squared-Schmidt weight sum is below ``weight_tol``.
    """
    _validate_cut(cut, num_interventions)
    op = upsilon_to_unfused_operator(upsilon, num_interventions)
    blocks = causal_block_axis_indices(num_interventions)
    labels = causal_block_axis_labels(num_interventions)
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
        "left_axes": left_axes,
        "right_axes": right_axes,
        "blocks": blocks,
        "block_labels": labels,
    }


def _is_ket_unfused_axis(axis: int) -> bool:
    """Return whether an unfused process-tensor axis carries a ket index."""
    if axis == 0:
        return True
    if axis == 1:
        return False
    return (axis - 2) % 4 < 2


def _von_neumann_entropy_natural(rho: NDArray[np.complex128], *, eig_tol: float = 1e-15) -> float:
    r"""Von Neumann entropy in natural units for a density matrix.

    Returns:
        Entropy :math:`S(\rho)` using natural logarithms.
    """
    herm = 0.5 * (rho + rho.conj().T)
    tr = float(np.trace(herm).real)
    if tr <= eig_tol:
        return 0.0
    herm /= tr
    evals = np.linalg.eigvalsh(herm).real
    evals = np.clip(evals, 0.0, None)
    nz = evals[evals > eig_tol]
    if nz.size == 0:
        return 0.0
    return float(-np.sum(nz * np.log(nz)))


def causal_block_mutual_information(
    upsilon: NDArray[np.complex128],
    num_interventions: int,
    cut: int,
    *,
    eig_tol: float = 1e-15,
    psd_tol: float = 1e-10,
) -> dict[str, NDArray[np.complex128] | float | int | list[int] | list[list[int]] | list[list[str]]]:
    r"""Past-future mutual information across causally regrouped channel blocks.

    Forms the normalized Choi state :math:`\rho_{PF}` from the unfused operator with
    causal blocks

        P = B_0 \cdots B_{c-1}, \quad F = B_c \cdots B_k,

    and returns

        I_{PT}(c) = S(\rho_P) + S(\rho_F) - S(\rho_{PF}).

    Args:
        upsilon: Dense process-tensor Choi matrix.
        num_interventions: Intervention count ``k``.
        cut: Causal cut index ``c``.
        eig_tol: Discard eigenvalues below this floor in entropy sums.
        psd_tol: Allowed negative eigenvalue floor after Hermitization.

    Returns:
        Dictionary with ``mutual_information``, ``entropy_p``, ``entropy_f``,
        ``entropy_pf``, ``trace``, ``min_eigenvalue``, ``hermiticity_error``,
        ``blocks``, ``block_labels``, ``p_axes``, ``f_axes``.

    Raises:
        ValueError: If ``cut`` is invalid or the trace is non-positive.
    """
    _validate_cut(cut, num_interventions)
    op = upsilon_to_unfused_operator(upsilon, num_interventions)
    blocks = causal_block_axis_indices(num_interventions)
    labels = causal_block_axis_labels(num_interventions)
    p_axes = [i for b in blocks[:cut] for i in b]
    f_axes = [i for b in blocks[cut:] for i in b]
    p_ket = [a for a in p_axes if _is_ket_unfused_axis(a)]
    p_bra = [a for a in p_axes if not _is_ket_unfused_axis(a)]
    f_ket = [a for a in f_axes if _is_ket_unfused_axis(a)]
    f_bra = [a for a in f_axes if not _is_ket_unfused_axis(a)]
    perm = p_ket + f_ket + p_bra + f_bra
    tensor = np.transpose(op, perm)
    n_p = len(p_ket)
    n_f = len(f_ket)
    n_ket = n_p + n_f
    dim = 1 << n_ket
    dims = [2] * n_ket
    rho_pf = tensor.reshape(dim, dim).astype(np.complex128)
    herm_err = float(np.linalg.norm(rho_pf - rho_pf.conj().T, ord="fro"))
    rho_pf = 0.5 * (rho_pf + rho_pf.conj().T)
    trace = float(np.trace(rho_pf).real)
    if trace <= eig_tol:
        msg = f"Regrouped Choi state trace {trace:.3e} is not positive."
        raise ValueError(msg)
    min_eval = float(np.min(np.linalg.eigvalsh(rho_pf).real))
    if min_eval < -psd_tol:
        msg = f"Regrouped Choi state min eigenvalue {min_eval:.3e} below PSD tolerance {psd_tol:.3e}."
        raise ValueError(msg)
    rho_pf /= trace
    keep_p = list(range(n_p))
    keep_f = list(range(n_p, n_ket))
    rho_p = trace_partial_dense(rho_pf, dims, keep=keep_p)
    rho_f = trace_partial_dense(rho_pf, dims, keep=keep_f)
    s_p = _von_neumann_entropy_natural(rho_p, eig_tol=eig_tol)
    s_f = _von_neumann_entropy_natural(rho_f, eig_tol=eig_tol)
    s_pf = _von_neumann_entropy_natural(rho_pf, eig_tol=eig_tol)
    mutual = float(s_p + s_f - s_pf)
    return {
        "mutual_information": mutual,
        "entropy_p": s_p,
        "entropy_f": s_f,
        "entropy_pf": s_pf,
        "trace": trace,
        "min_eigenvalue": min_eval,
        "hermiticity_error": herm_err,
        "blocks": blocks,
        "block_labels": labels,
        "p_axes": p_axes,
        "f_axes": f_axes,
    }


def _temporal_bond_dimension_dense(
    rho: NDArray[np.complex128],
    num_interventions: int,
    cut: int,
    *,
    tol: float = 1e-12,
) -> int:
    """Schmidt rank of the Choi state across MPO bond index ``cut``.

    Returns:
        Schmidt rank at the cut, at least 1.
    """
    if cut <= 1:
        return 1
    dims = [2] + [4] * num_interventions
    dim_l = int(np.prod(dims[:cut], dtype=np.int64))
    dim_r = int(np.prod(dims[cut:], dtype=np.int64))
    mat = np.asarray(rho, dtype=np.complex128).reshape(dim_l, dim_r)
    singular_values = np.linalg.svd(mat, compute_uv=False)
    rank = int(np.sum(singular_values > tol))
    return max(1, rank)


def cut_entanglement_entropy_from_upsilon(
    upsilon: NDArray[np.complex128],
    cut: int,
    num_interventions: int,
    *,
    base: float = math.e,
    check_psd: bool = False,
    assume_canonical: bool = False,
) -> float:
    """Von Neumann entropy of past intervention legs ``{1, ..., cut-1}`` in the Choi state.

    Returns:
        Cut entanglement entropy ``S_PT(c)``; ``0.0`` when ``cut <= 1``.
    """
    _validate_cut(cut, num_interventions)
    if cut <= 1:
        return 0.0
    rho = _canonicalize_upsilon(upsilon, check_psd=check_psd, assume_canonical=assume_canonical)
    dims = [2] + [4] * num_interventions
    keep_past = list(range(1, cut))
    rho_past = trace_partial_dense(rho, dims, keep=keep_past)
    if math.isclose(float(base), math.e, rel_tol=0.0, abs_tol=0.0):
        rho_herm = 0.5 * (rho_past + rho_past.conj().T)
        tr = np.trace(rho_herm)
        if abs(tr) < 1e-15:
            return 0.0
        rho_herm /= tr
        evals = np.linalg.eigvalsh(rho_herm).real
        evals = np.clip(evals, 0.0, 1.0)
        nz = evals[evals > 1e-15]
        if nz.size == 0:
            return 0.0
        return float(-np.sum(nz * np.log(nz)))
    return compute_entropy_dense(rho_past, base=int(base) if base == int(base) else 2)


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
            timesteps: Per-step evolution durations.
            initial_rho: Site-0 reference state after ``U_0`` (defaults to ``|0\\rangle\\langle 0|``).
        """
        self.upsilon = upsilon
        self.timesteps = timesteps
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
        size = self.upsilon.shape[0]
        return int(np.round(np.log2(size / 2) / 2))

    def temporal_bond_dimension(self, cut: int, *, tol: float = 1e-12) -> int:
        """Return the PT-MPO bond dimension across temporal cut ``cut``.

        Args:
            cut: Causal cut index ``c`` (bond between MPO sites ``c-1`` and ``c``).
            tol: Schmidt threshold for counting bond dimensions.

        Returns:
            Bond dimension ``chi(c)``, with ``chi(1)=1``.
        """
        k = self._num_interventions()
        _validate_cut(cut, k)
        rho = _canonicalize_upsilon(self.upsilon)
        return _temporal_bond_dimension_dense(rho, k, cut, tol=tol)

    def cut_entanglement_entropy(
        self,
        cut: int,
        *,
        base: float = math.e,
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Von Neumann entropy of past intervention legs ``{1, ..., cut-1}``.

        Args:
            cut: Causal cut index ``c``.
            base: Logarithm base (default natural log).
            check_psd: If ``True``, validate PSD before normalizing.
            assume_canonical: If ``True``, treat ``upsilon`` as already canonicalized.

        Returns:
            Cut entanglement entropy ``S_PT(c)``.
        """
        return cut_entanglement_entropy_from_upsilon(
            self.upsilon,
            cut,
            self._num_interventions(),
            base=base,
            check_psd=check_psd,
            assume_canonical=assume_canonical,
        )

    def causal_block_operator_entropy(
        self,
        cut: int,
        *,
        rtol: float = 1e-12,
        weight_tol: float = 1e-30,
    ) -> dict[str, NDArray[np.complex128] | float | int | list[int] | list[list[int]] | list[list[str]]]:
        """Operator-space entropy across causally regrouped channel blocks.

        Args:
            cut: Causal cut index ``c`` matching the response protocol.
            rtol: Relative Schmidt threshold for ``schmidt_rank``.
            weight_tol: Absolute floor on ``sum(s**2)``.

        Returns:
            Result dictionary from :func:`causal_block_operator_entropy`.
        """
        return causal_block_operator_entropy(
            self.upsilon,
            self._num_interventions(),
            cut,
            rtol=rtol,
            weight_tol=weight_tol,
        )

    def causal_block_mutual_information(
        self,
        cut: int,
        *,
        eig_tol: float = 1e-15,
        psd_tol: float = 1e-10,
    ) -> dict[str, NDArray[np.complex128] | float | int | list[int] | list[list[int]] | list[list[str]]]:
        """Past-future mutual information across causally regrouped channel blocks.

        Args:
            cut: Causal cut index ``c``.
            eig_tol: Eigenvalue floor for entropies.
            psd_tol: Allowed negative eigenvalue floor after Hermitization.

        Returns:
            Result dictionary from :func:`causal_block_mutual_information`.
        """
        return causal_block_mutual_information(
            self.upsilon,
            self._num_interventions(),
            cut,
            eig_tol=eig_tol,
            psd_tol=psd_tol,
        )

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
        upsilon_4d = self.upsilon.reshape(2, dim_p, 2, dim_p)
        ins = past_total.T.reshape(dim_p, dim_p)
        return np.einsum("s p q r, r p -> s q", upsilon_4d, ins)

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
            upsilon_mat = 0.5 * (self.upsilon + self.upsilon.conj().T)
            if check_psd:
                lam_min = float(np.linalg.eigvalsh(upsilon_mat).min().real)
                if lam_min < -1e-9:
                    msg = f"Upsilon not PSD (min eigenvalue {lam_min:.3e})."
                    raise ValueError(msg)
            tr = np.trace(upsilon_mat)
            rho = upsilon_mat / tr if abs(tr) > 1e-15 else upsilon_mat

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
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Compute conditional mutual information I(F:P_{<k} | P_k).

        Args:
            base: Log base for entropy.
            check_psd: If ``True``, validate PSD before normalizing.
            assume_canonical: If ``True``, treat ``upsilon`` as already canonicalized.

        Returns:
            Conditional mutual information. Returns 0.0 for ``k<2``.

        Raises:
            ValueError: If PSD check fails.
        """
        if assume_canonical:
            rho = self.upsilon
        else:
            upsilon_mat = 0.5 * (self.upsilon + self.upsilon.conj().T)
            if check_psd:
                lam_min = float(np.linalg.eigvalsh(upsilon_mat).min().real)
                if lam_min < -1e-9:
                    msg = f"Upsilon not PSD (min eigenvalue {lam_min:.3e})."
                    raise ValueError(msg)
            tr = np.trace(upsilon_mat)
            rho = upsilon_mat / tr if abs(tr) > 1e-15 else upsilon_mat

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
            timesteps: Per-step evolution durations.
            initial_rho: Site-0 reference state after ``U_0`` (defaults to ``|0\\rangle\\langle 0|``).
        """
        # Copy underlying MPO tensors/state into this subclass
        super().__init__()
        self.tensors = [t.copy() for t in upsilon_mpo.tensors]
        self.length = upsilon_mpo.length
        self.physical_dimension = upsilon_mpo.physical_dimension
        self.timesteps = timesteps
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
        return DenseProcessTensor(self.to_matrix(), self.timesteps, initial_rho=self.initial_rho.copy())

    def _num_interventions_for_probe(self) -> int:
        return int(self.length) - 1

    def temporal_bond_dimension(self, cut: int, *, tol: float = 1e-12) -> int:
        """Return the PT-MPO bond dimension across temporal cut ``cut``.

        Args:
            cut: Causal cut index ``c`` (bond between MPO sites ``c-1`` and ``c``).
            tol: Unused for MPO tensors; bond dimension is read directly.

        Returns:
            Bond dimension ``chi(c)``, with ``chi(1)=1``.
        """
        _ = tol
        k = self._num_interventions_for_probe()
        _validate_cut(cut, k)
        if cut <= 1:
            return 1
        return int(self.tensors[cut - 1].shape[3])

    def cut_entanglement_entropy(
        self,
        cut: int,
        *,
        base: float = math.e,
        check_psd: bool = False,
        assume_canonical: bool = False,
    ) -> float:
        """Von Neumann entropy of past intervention legs ``{1, ..., cut-1}``.

        Args:
            cut: Causal cut index ``c``.
            base: Logarithm base (default natural log).
            check_psd: If ``True``, validate PSD before normalizing.
            assume_canonical: If ``True``, treat the reduced operator as already canonicalized.

        Returns:
            Cut entanglement entropy ``S_PT(c)``.

        Raises:
            ValueError: If ``cut`` is outside ``[1, k]``.
        """
        k = self._num_interventions_for_probe()
        _validate_cut(cut, k)
        if cut <= 1:
            return 0.0
        keep_past = list(range(1, cut))
        rho_past = self.partial_trace_sites(keep_past).to_matrix()
        if assume_canonical:
            rho = rho_past
        else:
            rho = 0.5 * (rho_past + rho_past.conj().T)
            if check_psd:
                lam_min = float(np.linalg.eigvalsh(rho).min().real)
                if lam_min < -1e-9:
                    msg = f"Reduced past state not PSD (min eigenvalue {lam_min:.3e})."
                    raise ValueError(msg)
        if math.isclose(float(base), math.e, rel_tol=0.0, abs_tol=0.0):
            tr = np.trace(rho)
            if abs(tr) < 1e-15:
                return 0.0
            rho /= tr
            evals = np.linalg.eigvalsh(rho).real
            evals = np.clip(evals, 0.0, 1.0)
            nz = evals[evals > 1e-15]
            if nz.size == 0:
                return 0.0
            return float(-np.sum(nz * np.log(nz)))
        return compute_entropy_dense(rho, base=int(base) if base == int(base) else 2)

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
            check_psd=check_psd,
            assume_canonical=assume_canonical,
        )
