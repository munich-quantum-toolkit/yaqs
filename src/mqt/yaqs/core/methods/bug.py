# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Implements the Basis-Update and Galerkin Method (BUG) for MPS.

Refer to Ceruti et al. (2023) doi:10.1137/22M1473790 for details of the method
for TTN. Select BUG with ``EvolutionMode.BUG``. Each physical step uses
center-augmented alternating endpoints, one SVD compression after the full
``dt`` composition, and renormalization after compression.
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from .decompositions import left_qr, right_qr
from .tdvp.primitives import update_left_environment, update_right_environment, update_site

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..data_structures.mpo import MPO
    from ..data_structures.mps import MPS
    from ..data_structures.simulation_parameters import AnalogSimParams, DigitalSimParams


def prepare_canonical_site_tensors(
    state: MPS, mpo: MPO
) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
    """Build coefficient-bearing centers and left MPO environments on a copied list.

    Performs one left-to-right QR preparation without mutating the physical MPS.

    Args:
        state: The MPS.
        mpo: The MPO.

    Returns:
        canon_tensors: The list of the canonical site tensors.
        left_blocks: The list of the left environments.
    """
    canon_tensors = copy(state.tensors)
    left_end_dimension = state.tensors[0].shape[1]
    left_blocks: list[NDArray[np.complex128]] = [
        np.eye(left_end_dimension, dtype=np.complex128).reshape(left_end_dimension, 1, left_end_dimension)
    ]
    for i, old_local_tensor in enumerate(canon_tensors[1:], start=1):
        left_tensor = canon_tensors[i - 1]
        left_q, left_r = right_qr(left_tensor)
        local_tensor = np.tensordot(left_r, old_local_tensor, axes=(1, 1)).transpose(1, 0, 2)
        canon_tensors[i] = np.asarray(local_tensor, dtype=np.complex128)
        left_blocks.append(update_left_environment(left_q, left_q, mpo.tensors[i - 1], left_blocks[i - 1]))
    return canon_tensors, left_blocks


def build_trial_basis(
    *,
    old_q: NDArray[np.complex128],
    working_center: NDArray[np.complex128],
    predictor: NDArray[np.complex128],
    deeper_overlap: NDArray[np.complex128],
    is_endpoint: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Build a center-augmented right-isometric trial basis and block-overlap factor.

    Returns:
        ``(new_q, new_overlap)`` with MPS leg order on ``new_q``.
    """
    old_basis_current = np.asarray(np.tensordot(old_q, deeper_overlap, axes=(2, 0)), dtype=np.complex128)
    retained = old_q if is_endpoint else working_center
    stacked = np.concatenate((retained, predictor), axis=1)
    new_q, _ = left_qr(stacked)
    new_overlap = np.asarray(
        np.tensordot(old_basis_current, new_q.conj(), axes=([0, 2], [0, 2])),
        dtype=np.complex128,
    )
    return new_q, new_overlap


def _local_update(
    state: MPS,
    mpo: MPO,
    left_blocks: list[NDArray[np.complex128]],
    right_block: NDArray[np.complex128],
    canon_center_tensors: list[NDArray[np.complex128]],
    site: int,
    right_m_block: NDArray[np.complex128],
    *,
    dt: float,
    krylov_tol: float,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Evolve one non-root site, form a trial basis, and transport the left center.

    Returns:
        The site overlap matrix and the updated right MPO environment.
    """
    working_center = canon_center_tensors[site]
    predictor = update_site(
        left_blocks[site],
        right_block,
        mpo.tensors[site],
        working_center,
        dt,
        krylov_tol=krylov_tol,
    )
    new_q, basis_change_m = build_trial_basis(
        old_q=state.tensors[site],
        working_center=working_center,
        predictor=predictor,
        deeper_overlap=right_m_block,
        is_endpoint=(site == state.length - 1),
    )
    state.tensors[site] = new_q
    canon_center_tensors[site - 1] = np.asarray(
        np.tensordot(canon_center_tensors[site - 1], basis_change_m, axes=(2, 0)),
        dtype=np.complex128,
    )
    return basis_change_m, update_right_environment(new_q, new_q, mpo.tensors[site], right_block)


def bug_sweep(
    state: MPS,
    mpo: MPO,
    *,
    dt: float,
    krylov_tol: float,
) -> None:
    """Apply one uncompressed left-root endpoint BUG sweep in place.

    Requires a known orthogonality center at site ``0``. The caller is responsible
    for supplying an endpoint-canonical MPS in that gauge (typically via
    :meth:`~mqt.yaqs.core.data_structures.mps.MPS.set_canonical_form` /
    :meth:`~mqt.yaqs.core.data_structures.mps.MPS.set_center`); this routine checks
    the tracked center metadata but does not re-validate full right-canonicality
    of every tensor. Performs no compression or normalization.

    Args:
        state: MPS to evolve in place.
        mpo: Time-independent Hermitian Hamiltonian as an MPO.
        dt: Sweep duration.
        krylov_tol: Krylov tolerance for local exponentials.

    Raises:
        ValueError: If lengths differ or the center is not at site ``0``.
    """
    if mpo.length != state.length:
        msg = "MPS and Hamiltonian must have the same number of sites"
        raise ValueError(msg)

    state.assert_center(0, context="bug_sweep")

    canon_center_tensors, left_envs = prepare_canonical_site_tensors(state, mpo)
    right_end_dimension = state.tensors[-1].shape[2]
    right_block = np.eye(right_end_dimension, dtype=np.complex128).reshape(right_end_dimension, 1, right_end_dimension)
    right_m_block = np.eye(right_end_dimension, dtype=np.complex128)

    for site in range(mpo.length - 1, 0, -1):
        right_m_block, right_block = _local_update(
            state,
            mpo,
            left_envs,
            right_block,
            canon_center_tensors,
            site,
            right_m_block,
            dt=dt,
            krylov_tol=krylov_tol,
        )

    state.tensors[0] = update_site(
        left_envs[0],
        right_block,
        mpo.tensors[0],
        canon_center_tensors[0],
        dt,
        krylov_tol=krylov_tol,
    )
    state.set_center(0)


def _postprocess_bug_state(
    state: MPS,
    sim_params: AnalogSimParams | DigitalSimParams,
    *,
    normalize: bool = True,
) -> None:
    """Apply SVD compression and optional renormalization after a full BUG step.

    Args:
        state: MPS to compress (and optionally normalize) in place.
        sim_params: Simulation parameters supplying ``svd_threshold``,
            ``max_bond_dim``, and ``trunc_mode`` for compression.
        normalize: If ``True`` (default), rescale the tracked center after
            compression. If ``False``, leave the post-compression norm unchanged
            for auxiliary correlator states.
    """
    state.compress(
        sim_params.svd_threshold,
        max_bond_dim=sim_params.max_bond_dim,
        trunc_mode=sim_params.trunc_mode,
    )
    if normalize:
        state.normalize_center()


def bug(
    state: MPS,
    mpo: MPO,
    sim_params: AnalogSimParams | DigitalSimParams,
    *,
    normalize: bool = True,
) -> None:
    """Perform one BUG physical step according to ``sim_params``.

    Uses center-augmented alternating endpoints (two half-sweeps of ``dt / 2``),
    then one SVD compression and optional renormalization.

    Args:
        state: The MPS to evolve in place.
        mpo: Hamiltonian represented as an MPO.
        sim_params: Simulation parameters (``dt``, truncation, Krylov tolerance).
        normalize: If ``True`` (default), renormalize after compression. Pass
            ``False`` for auxiliary correlator states so non-unitary probe
            amplitudes are preserved.

    Raises:
        ValueError: If lengths differ or the gauge contract fails.
    """
    if mpo.length != state.length:
        msg = "MPS and Hamiltonian must have the same number of sites"
        raise ValueError(msg)

    half_dt = sim_params.dt / 2.0
    state.assert_center(0, context="bug")
    bug_sweep(state, mpo, dt=half_dt, krylov_tol=sim_params.krylov_tol)

    flipped = False
    try:
        state.flip_network()
        flipped = True
        state.set_canonical_form(0, decomposition="QR")
        state.set_center(0)
        bug_sweep(state, mpo.reflected(), dt=half_dt, krylov_tol=sim_params.krylov_tol)
    finally:
        if flipped:
            state.flip_network()
            state.set_canonical_form(0, decomposition="QR")
            state.set_center(0)

    _postprocess_bug_state(state, sim_params, normalize=normalize)
