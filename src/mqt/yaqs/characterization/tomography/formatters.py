# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Format tomography data into TomographyEstimate or MPO (Upsilon)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPO

from .estimator import TomographyEstimate
from .sampling import SamplingData, SequenceData

if TYPE_CHECKING:
    from numpy.typing import NDArray


def rank1_upsilon_mpo_term(
    rho_final: NDArray[np.complex128],
    dual_ops: list[NDArray[np.complex128]],
    weight: float = 1.0,
) -> MPO:
    """Build a rank-1 MPO term representing a single sample's contribution to Upsilon.

    The comb Choi operator lives on sites [F, P1, P2, ..., Pk].
    Site 0 (F) is the final state density matrix (2x2).
    Sites 1..k are the dual operators (4x4) corresponding to interventions.
    """
    k = len(dual_ops)
    phys_dims = [2] + [4] * k
    tensors = []
    t0 = (weight * rho_final).reshape(2, 2, 1, 1)
    tensors.append(t0)
    for D in dual_ops:
        tD = D.reshape(4, 4, 1, 1)
        tensors.append(tD)
    m = MPO()
    m.custom(tensors, transpose=False)
    m.physical_dimension = phys_dims
    return m


def _accumulate_rank1_terms(
    terms: Iterable[MPO],
    k: int,
    dims: tuple[int, int] = (2, 2),
    compress_every: int = 100,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 4,
) -> MPO:
    """Accumulate an iterable of rank-1 MPO terms with periodic SVD compression."""
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
        return rank1_upsilon_mpo_term(
            np.zeros(dims, dtype=np.complex128), [np.eye(4, dtype=np.complex128)] * k, weight=0.0
        )
    return running


def _to_dense(data: SequenceData | SamplingData) -> TomographyEstimate:
    """Format estimation results into a dense tomography estimate."""
    if isinstance(data, SamplingData):
        k = len(data.timesteps)
        dim = 2 * (4**k)
        ups = np.zeros((dim, dim), dtype=np.complex128)
        for rho, duals, w in zip(data.outputs, data.dual_ops, data.weights):
            if abs(w) < 1e-30 or np.linalg.norm(rho) < 1e-30:
                continue
            if len(duals) != k:
                msg = f"Nonzero sample has incomplete dual history: expected {k}, got {len(duals)}."
                raise ValueError(msg)
            past = duals[0]
            for d_step in duals[1:]:
                past = np.kron(past, d_step)
            ups += w * np.kron(rho, past)
        return TomographyEstimate.from_dense_choi(ups, data.timesteps)

    k = len(data.timesteps)
    tensor_shape = [4] + [16] * k
    dense_data = np.zeros(tensor_shape, dtype=np.complex128)
    dense_weights = np.zeros([16] * k, dtype=np.float64)
    for i, seq in enumerate(data.sequences):
        rho = data.outputs[i]
        w = data.weights[i]
        dense_data[(slice(None), *seq)] = rho.reshape(-1)
        dense_weights[seq] = w
    return TomographyEstimate(
        tensor=dense_data,
        weights=dense_weights,
        timesteps=data.timesteps,
        choi_duals=data.choi_duals,
        choi_indices=data.choi_indices,
        choi_basis=data.choi_basis,
    )


def _to_mpo(
    data: SequenceData | SamplingData,
    compress_every: int = 100,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 2,
) -> MPO:
    """Format estimation results into an MPO (Upsilon) representation."""
    k = len(data.timesteps)
    if isinstance(data, SamplingData):
        def _sampling_terms():
            for i in range(len(data.outputs)):
                rho = data.outputs[i]
                w = data.weights[i]
                if abs(w) < 1e-30 or np.linalg.norm(rho) < 1e-30:
                    continue
                dual_ops = data.dual_ops[i]
                if len(dual_ops) != k:
                    msg = f"Nonzero sample has incomplete dual history: expected {k}, got {len(dual_ops)}."
                    raise ValueError(msg)
                yield rank1_upsilon_mpo_term(rho, dual_ops, weight=w)

        return _accumulate_rank1_terms(
            _sampling_terms(), k=k, dims=(2, 2),
            compress_every=compress_every, tol=tol, max_bond_dim=max_bond_dim, n_sweeps=n_sweeps
        )

    def _sequence_terms():
        for i, seq in enumerate(data.sequences):
            rho = data.outputs[i]
            w = data.weights[i]
            dual_ops = [data.choi_duals[a].T for a in seq]
            yield rank1_upsilon_mpo_term(rho, dual_ops, weight=w)

    return _accumulate_rank1_terms(
        _sequence_terms(), k=k, dims=(2, 2),
        compress_every=compress_every, tol=tol, max_bond_dim=max_bond_dim, n_sweeps=n_sweeps
    )
