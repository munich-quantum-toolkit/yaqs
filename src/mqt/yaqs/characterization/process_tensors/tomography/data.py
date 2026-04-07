# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Exhaustive discrete-basis process-tensor data + reconstruction helpers.

The main product of :func:`~mqt.yaqs.characterization.process_tensors.tomography.constructor.construct_process_tensor`
is :class:`SequenceData`. It can be converted to dense or MPO comb representations via
:meth:`SequenceData.to_dense_comb` and :meth:`SequenceData.to_mpo_comb`.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPO

from .combs import DenseComb, MPOComb

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _rank1_mpo_term(
    rho_final: "NDArray[np.complex128]",
    dual_ops: list["NDArray[np.complex128]"],
    weight: float = 1.0,
) -> MPO:
    """One rank-1 MPO term contributing to Υ."""
    num_steps = len(dual_ops)
    phys_dims = [2] + [4] * num_steps
    tensors: list[np.ndarray] = [(weight * rho_final).reshape(2, 2, 1, 1)]
    tensors.extend(D.reshape(4, 4, 1, 1) for D in dual_ops)

    mpo = MPO()
    mpo.custom(tensors, transpose=False)
    mpo.physical_dimension = phys_dims
    return mpo


def _accumulate_rank1(
    terms: Iterable[MPO],
    num_steps: int,
    dims: tuple[int, int] = (2, 2),
    compress_every: int = 100,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 4,
) -> MPO:
    """Accumulate rank-1 MPO terms with periodic SVD compression."""
    pending: list[MPO] = []
    running: MPO | None = None

    def _flush() -> None:
        nonlocal running, pending
        if not pending:
            return
        chunk = MPO.mpo_sum(pending)
        pending.clear()
        running = chunk if running is None else running + chunk
        running.compress(tol=tol, max_bond_dim=max_bond_dim, n_sweeps=n_sweeps)

    for term in terms:
        pending.append(term)
        if len(pending) >= compress_every:
            _flush()
    _flush()
    if running is None:
        return _rank1_mpo_term(
            np.zeros(dims, dtype=np.complex128), [np.eye(4, dtype=np.complex128)] * num_steps, weight=0.0
        )
    return running


def _pack_outputs(data: "SequenceData") -> tuple["NDArray[np.complex128]", "NDArray[np.float64]"]:
    """Pack list-of-sequences data into dense tensors indexed by alpha tuples."""
    num_steps = len(data.timesteps)
    out_vecs = np.zeros([4] + [16] * num_steps, dtype=np.complex128)
    seq_weights = np.zeros([16] * num_steps, dtype=np.float64)
    for i, alpha in enumerate(data.sequences):
        out_vecs[(slice(None), *alpha)] = np.asarray(data.outputs[i], dtype=np.complex128).reshape(-1)
        seq_weights[alpha] = float(data.weights[i])
    return out_vecs, seq_weights


def _iter_rank1_terms(data: "SequenceData") -> Iterable[MPO]:
    """Yield rank-1 MPO terms for Υ accumulation."""
    for i, alpha in enumerate(data.sequences):
        rho_out = data.outputs[i]
        w = float(data.weights[i])
        dual_ops = [data.choi_duals[a].T for a in alpha]
        yield _rank1_mpo_term(rho_out, dual_ops, weight=w)


def _reconstruct_upsilon(
    *,
    out_vecs: "NDArray[np.complex128]",
    seq_weights: "NDArray[np.float64]",
    dual_ops: list["NDArray[np.complex128]"],
    basis_ops: list["NDArray[np.complex128]"],
    check: bool,
    atol: float,
) -> "NDArray[np.complex128]":
    """Reconstruct dense Υ from packed tensor/weights (fixed dual convention).

    Convention matches the MPO build path: each dual frame operator is transposed
    (``dual_ops[a].T``) before Kronecker accumulation.
    """
    if len(basis_ops) != 16:
        raise ValueError("Need choi_basis of length 16 to reconstruct Υ.")
    if len(dual_ops) != 16:
        raise ValueError("Need choi_duals of length 16 to reconstruct Υ.")
    if out_vecs.shape[0] != 4:
        raise ValueError(f"Expected out_vecs[0] dim 4 (vec of 2x2 output), got {out_vecs.shape[0]}.")

    num_steps = out_vecs.ndim - 1
    if num_steps == 0:
        return out_vecs.reshape(2, 2)

    dim_past = 4**num_steps
    dim_total = 2 * dim_past

    upsilon = np.zeros((dim_total, dim_total), dtype=np.complex128)
    for alpha in np.ndindex(*([16] * num_steps)):
        w = float(seq_weights[alpha])
        if w <= 1e-30:
            continue
        rho_out = out_vecs[(slice(None), *alpha)].reshape(2, 2)
        past = dual_ops[alpha[0]].T
        for a in alpha[1:]:
            past = np.kron(past, dual_ops[a].T)
        upsilon += np.kron(w * rho_out, past)

    if not check:
        return upsilon

    U4 = upsilon.reshape(2, dim_past, 2, dim_past)
    err_sum = 0.0
    n_used = 0
    max_checks = 64 if dim_past > 256 else 256
    for alpha in np.ndindex(*([16] * num_steps)):
        if n_used >= max_checks:
            break
        w = float(seq_weights[alpha])
        if w <= 1e-30:
            continue
        rho_true = w * out_vecs[(slice(None), *alpha)].reshape(2, 2)
        past = basis_ops[alpha[0]]
        for a in alpha[1:]:
            past = np.kron(past, basis_ops[a])
        ins = past.T.reshape(dim_past, dim_past)
        rho_pred = np.einsum("s p q r, r p -> s q", U4, ins)
        err_sum += float(np.linalg.norm(rho_true - rho_pred))
        n_used += 1

    mean_err = err_sum / max(1, n_used)
    if mean_err > atol:
        raise ValueError(f"Υ reconstruction self-check failed (mean_err={mean_err:.3e} > atol={atol}).")

    return upsilon


@dataclass
class SequenceData:
    """Discrete tomography data: one row per **sequence** (Choi index tuple of length ``k``)."""

    sequences: list[tuple[int, ...]]
    outputs: list[np.ndarray]  # (2, 2) density matrices
    weights: list[float]
    choi_basis: list[np.ndarray]
    choi_indices: list[tuple[int, int]]
    choi_duals: list[np.ndarray]
    timesteps: list[float]

    def to_dense_comb(self, *, check: bool = True, atol: float = 1e-8) -> DenseComb:
        """Reconstruct dense comb Υ."""
        out_vecs, seq_weights = _pack_outputs(self)
        upsilon = _reconstruct_upsilon(
            out_vecs=out_vecs,
            seq_weights=seq_weights,
            dual_ops=self.choi_duals,
            basis_ops=self.choi_basis,
            check=check,
            atol=atol,
        )
        return DenseComb(upsilon, list(self.timesteps))

    def to_mpo_comb(
        self,
        *,
        compress_every: int = 100,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 2,
    ) -> MPOComb:
        """Build an MPO comb Υ via rank-1 accumulation."""
        num_steps = len(self.timesteps)
        mpo = _accumulate_rank1(
            _iter_rank1_terms(self),
            num_steps=num_steps,
            dims=(2, 2),
            compress_every=compress_every,
            tol=tol,
            max_bond_dim=max_bond_dim,
            n_sweeps=n_sweeps,
        )
        return MPOComb(mpo, self.timesteps)
