# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Implements the Basis-Update and Galerkin Method (BUG) for MPS.

Refer to Ceruti et al. (2023) doi:10.1137/22M1473790 for details of the method
for TTN. The MPS endpoint formulation implemented here follows the YAQS
single-left-endpoint, center-augmented canonical BUG sweep with optional
paper-facing configuration via :class:`~mqt.yaqs.core.data_structures.simulation_parameters.BUGConfig`.
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..data_structures.simulation_parameters import BUGConfig
from .decompositions import left_qr, right_qr
from .tdvp.primitives import update_left_environment, update_right_environment, update_site

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..data_structures.mpo import MPO
    from ..data_structures.mps import MPS
    from ..data_structures.simulation_parameters import AnalogSimParams, DigitalSimParams

BUGBasisMode = Literal["center", "explicit_old_basis", "fixed_profile"]


def prepare_canonical_site_tensors(
    state: MPS, mpo: MPO
) -> tuple[list[NDArray[np.complex128]], list[NDArray[np.complex128]]]:
    """Build coefficient-bearing centers and left MPO environments on a copied list.

    Performs one left-to-right QR preparation without mutating the physical MPS.
    Assumes non-root sites are right-isometric when the orthogonality center is at
    site ``0``.

    Args:
        state: The MPS.
        mpo: The MPO.

    Returns:
        canon_tensors: The list of the canonical site tensors.
        left_blocks: The list of the left environments.

    """
    # This will merely do a shallow copy of the MPS.
    canon_tensors = copy(state.tensors)
    left_end_dimension = state.tensors[0].shape[1]
    left_blocks: list[NDArray[np.complex128]] = [
        np.eye(left_end_dimension, dtype=np.complex128).reshape(left_end_dimension, 1, left_end_dimension)
    ]
    for i, old_local_tensor in enumerate(canon_tensors[1:], start=1):
        left_tensor = canon_tensors[i - 1]
        left_q, left_r = right_qr(left_tensor)
        # Legs of right_r: (new, old_right)
        local_tensor = np.tensordot(left_r, old_local_tensor, axes=(1, 1))
        # Leg order of local_tensor: (left, phys, right)
        local_tensor = local_tensor.transpose(1, 0, 2)
        # Correct leg order: (phys, left, right) and orth center
        canon_tensors[i] = np.asarray(local_tensor, dtype=np.complex128)
        new_env = update_left_environment(left_q, left_q, mpo.tensors[i - 1], left_blocks[i - 1])
        left_blocks.append(new_env)
    return canon_tensors, left_blocks


def choose_stack_tensor(
    site: int, canon_center_tensors: list[NDArray[np.complex128]], state: MPS
) -> NDArray[np.complex128]:
    """Return the retained tensor for center-augmented stacking.

    If the site is the last one and thus the leaf site, we need to choose the
    MPS tensor, when the MPS was in left-canonical form. Otherwise, we choose
    the MPS tensor, when the local site was the orthogonality center.

    Args:
        site: The site to be updated.
        canon_center_tensors: The canonical site tensors.
        state: The MPS.

    Returns:
        NDArray[np.complex128]: The tensor to be stacked.

    """
    if site == state.length - 1:  # ruff:ignore[if-else-block-instead-of-if-exp]
        # This is the only leaf case.
        old_stack_tensor = state.tensors[site]
    else:
        old_stack_tensor = canon_center_tensors[site]
    return old_stack_tensor


def find_new_q(
    old_stack_tensor: NDArray[np.complex128], updated_tensor: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Finds the new Q tensor after the update with enlarged left virtual leg.

    Args:
        old_stack_tensor: The tensor to be stacked with the updated tensor.
        updated_tensor: The tensor after the update.

    Returns:
        new_q: The new Q tensor with MPS leg order (phys, left, right).

    """
    stacked_tensor = np.concatenate((old_stack_tensor, updated_tensor), axis=1)
    new_q, _ = left_qr(stacked_tensor)
    return new_q


def build_trial_basis(
    *,
    old_q: NDArray[np.complex128],
    working_center: NDArray[np.complex128],
    predictor: NDArray[np.complex128],
    deeper_overlap: NDArray[np.complex128],
    is_endpoint: bool,
    basis_mode: BUGBasisMode,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Build the new right-isometric trial basis and the block-overlap factor.

    Args:
        old_q: Old right-isometric site tensor ``(phys, left, right)``.
        working_center: Coefficient-bearing center at this site before the local
            predictor evolution.
        predictor: Locally evolved working tensor used as the update direction.
        deeper_overlap: Block-overlap matrix from the deeper (right) bond,
            shape ``(old_right, new_right)``.
        is_endpoint: Whether this site is the rightmost non-root endpoint.
        basis_mode: Trial-basis construction mode.

    Returns:
        Tuple ``(new_q, new_overlap)`` where ``new_q`` is right-isometric with
        MPS leg order ``(phys, left, right)`` and ``new_overlap`` has shape
        ``(old_left, new_left)``.

    Raises:
        ValueError: If ``basis_mode`` is not recognized.
    """
    old_basis_current = np.asarray(np.tensordot(old_q, deeper_overlap, axes=(2, 0)), dtype=np.complex128)

    if basis_mode == "center":
        retained = old_q if is_endpoint else working_center
        stacked = np.concatenate((retained, predictor), axis=1)
    elif basis_mode == "explicit_old_basis":
        stacked = np.concatenate((old_basis_current, predictor), axis=1)
    elif basis_mode == "fixed_profile":
        stacked = predictor
    else:
        msg = f"Unknown BUG basis mode: {basis_mode!r}"
        raise ValueError(msg)

    new_q, _ = left_qr(stacked)
    new_overlap = np.asarray(
        np.tensordot(old_basis_current, new_q.conj(), axes=([0, 2], [0, 2])),
        dtype=np.complex128,
    )
    return new_q, new_overlap


def build_basis_change_tensor(
    old_q: NDArray[np.complex128], new_q: NDArray[np.complex128], old_m: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Build a new basis change tensor M.

    Args:
        old_q: The old tensor of the site, when the MPS was in left-canonical
            form. The leg order is (phys, left, right).
        new_q: The extended local base tensor after the update. Same leg order
            as an MPS tensor. The leg order is (phys, left, right).
        old_m: The basis change matrix of the site to the right. The leg order
            is (old,new).

    Returns:
        new_m: The basis change tensor M. The leg order is (old,new).

    """
    new_m = np.tensordot(old_q, old_m, axes=(2, 0))
    return np.asarray(np.tensordot(new_m, new_q.conj(), axes=([0, 2], [0, 2])), dtype=np.complex128)


def local_update(
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
    basis_mode: BUGBasisMode = "center",
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Single-site BUG non-root update.

    Evolves the working center once, builds a trial basis, transports the
    coefficient-bearing center at ``site - 1``, and updates the right MPO
    environment. Does not absorb the L factor from ``left_qr``.

    Args:
        state: The MPS.
        mpo: The MPO.
        left_blocks: The left environments.
        right_block: The right environment.
        canon_center_tensors: The canonical site tensors.
        site: The site to be updated.
        right_m_block: The basis update matrix of the site to the right.
        dt: Sweep duration for the local predictor exponential.
        krylov_tol: Krylov tolerance for the local exponential.
        basis_mode: Trial-basis construction mode.

    Returns:
        basis_change_m: The basis update matrix of this site.
        new_right_block: The right environment of this site.
    """
    working_center = canon_center_tensors[site]
    updated_tensor = update_site(
        left_blocks[site],
        right_block,
        mpo.tensors[site],
        working_center,
        dt,
        krylov_tol=krylov_tol,
    )
    old_q = state.tensors[site]
    new_q, basis_change_m = build_trial_basis(
        old_q=old_q,
        working_center=working_center,
        predictor=updated_tensor,
        deeper_overlap=right_m_block,
        is_endpoint=(site == state.length - 1),
        basis_mode=basis_mode,
    )
    state.tensors[site] = new_q
    canon_center_tensors[site - 1] = np.asarray(
        np.tensordot(canon_center_tensors[site - 1], basis_change_m, axes=(2, 0)),
        dtype=np.complex128,
    )
    new_right_block = update_right_environment(new_q, new_q, mpo.tensors[site], right_block)
    return basis_change_m, new_right_block


def bug_sweep(
    state: MPS,
    mpo: MPO,
    *,
    dt: float,
    krylov_tol: float,
    basis_mode: BUGBasisMode = "center",
) -> None:
    """Apply one uncompressed left-root endpoint BUG sweep in place.

    Preconditions:
        - ``state.length == mpo.length``
        - known orthogonality center at site ``0``
        - non-root tensors are right-isometric

    The output is the uncompressed state
    ``[updated root, new right-isometric site 1, ..., new right-isometric site L-1]``
    with the orthogonality center tracked at site ``0``. No rank cap, SVD
    compression, or normalization is performed.

    Args:
        state: MPS to evolve in place.
        mpo: Time-independent Hermitian Hamiltonian as an MPO.
        dt: Sweep duration (may be a half-step for alternating schedules).
        krylov_tol: Krylov tolerance for local exponentials.
        basis_mode: Trial-basis construction mode.

    Raises:
        ValueError: If lengths differ, the center is unknown, or the center is
            not at site ``0``.
    """
    num_sites = mpo.length
    if num_sites != state.length:
        msg = "MPS and Hamiltonian must have the same number of sites"
        raise ValueError(msg)

    state.assert_center(0, context="bug_sweep")

    canon_center_tensors, left_envs = prepare_canonical_site_tensors(state, mpo)
    right_end_dimension = state.tensors[-1].shape[2]
    right_block = np.eye(right_end_dimension, dtype=np.complex128).reshape(right_end_dimension, 1, right_end_dimension)
    right_m_block = np.eye(right_end_dimension, dtype=np.complex128)
    # Sweep from right to left over non-root sites.
    for site in range(num_sites - 1, 0, -1):
        right_m_block, right_block = local_update(
            state,
            mpo,
            left_envs,
            right_block,
            canon_center_tensors,
            site,
            right_m_block,
            dt=dt,
            krylov_tol=krylov_tol,
            basis_mode=basis_mode,
        )
    # Root solve: evolve the transported tensor at site 0 exactly once.
    updated_tensor = update_site(
        left_envs[0],
        right_block,
        mpo.tensors[0],
        canon_center_tensors[0],
        dt,
        krylov_tol=krylov_tol,
    )
    state.tensors[0] = updated_tensor
    state.set_center(0)


def _resolve_bug_config(sim_params: AnalogSimParams | DigitalSimParams) -> BUGConfig:
    """Return the BUG configuration, defaulting when digital params omit it."""
    bug_config = getattr(sim_params, "bug_config", None)
    if bug_config is None:
        return BUGConfig()
    return bug_config  # type: ignore[no-any-return]


def _postprocess_bug_state(state: MPS, sim_params: AnalogSimParams | DigitalSimParams, config: BUGConfig) -> None:
    """Apply configured SVD compression and optional normalization."""
    state.compress(
        sim_params.svd_threshold,
        max_bond_dim=sim_params.max_bond_dim,
        trunc_mode=sim_params.trunc_mode,
    )
    if config.normalize_after_compression:
        state.normalize()
    state.set_center(0)


def _gauge_canonicalize_to_zero(state: MPS) -> None:
    """Gauge-only rerooting to site 0 without SVD truncation."""
    state.set_canonical_form(0, decomposition="QR")
    state.set_center(0)


def _bond_profile(state: MPS) -> list[int]:
    """Return internal bond dimensions ``[chi_1, ..., chi_{L-1}]``."""
    return [int(state.tensors[i].shape[2]) for i in range(state.length - 1)]


def _run_single_endpoint(
    state: MPS,
    mpo: MPO,
    sim_params: AnalogSimParams | DigitalSimParams,
    config: BUGConfig,
    *,
    dt: float,
) -> None:
    """Run one left-root endpoint sweep and optional after-sweep postprocessing."""
    bug_sweep(
        state,
        mpo,
        dt=dt,
        krylov_tol=sim_params.krylov_tol,
        basis_mode=config.basis_mode,
    )
    if config.compression == "after_sweep":
        _postprocess_bug_state(state, sim_params, config)


def _run_alternating_endpoints(
    state: MPS,
    mpo: MPO,
    sim_params: AnalogSimParams | DigitalSimParams,
    config: BUGConfig,
) -> None:
    """Apply two reflected half-sweeps of duration ``dt / 2``."""
    half_dt = sim_params.dt / 2.0
    state.assert_center(0, context="bug")
    _run_single_endpoint(state, mpo, sim_params, config, dt=half_dt)

    flipped = False
    try:
        state.flip_network()
        flipped = True
        _gauge_canonicalize_to_zero(state)
        _run_single_endpoint(state, mpo.reflected(), sim_params, config, dt=half_dt)
    finally:
        if flipped:
            state.flip_network()
            _gauge_canonicalize_to_zero(state)

    if config.compression == "after_step":
        _postprocess_bug_state(state, sim_params, config)
    elif config.compression == "none":
        state.set_center(0)


def bug(state: MPS, mpo: MPO, sim_params: AnalogSimParams | DigitalSimParams) -> None:
    """Orchestrate the configured BUG schedule, compression, and normalization.

    Default configuration preserves the historical single left-endpoint,
    center-augmented sweep of duration ``sim_params.dt`` followed by one
    compression and no explicit normalization.

    Args:
        state: The initial state represented as an MPS.
        mpo: Hamiltonian represented as an MPO.
        sim_params: Simulation parameters containing time step ``dt``, SVD
            threshold, and (for analog params) ``bug_config``.

    Raises:
        ValueError: If the state and Hamiltonian have different numbers of
            sites, the input gauge contract is violated, or the schedule is
            unsupported.
        RuntimeError: If ``fixed_profile`` mode enlarges the entry bond profile
            when compression is disabled.
    """
    num_sites = mpo.length
    if num_sites != state.length:
        msg = "MPS and Hamiltonian must have the same number of sites"
        raise ValueError(msg)

    config = _resolve_bug_config(sim_params)
    entry_profile = _bond_profile(state) if config.basis_mode == "fixed_profile" else None

    if config.schedule == "single_endpoint":
        state.assert_center(0, context="bug")
        _run_single_endpoint(state, mpo, sim_params, config, dt=sim_params.dt)
        if config.compression == "after_step":
            _postprocess_bug_state(state, sim_params, config)
        elif config.compression == "none":
            state.set_center(0)
    elif config.schedule == "alternating_endpoints":
        _run_alternating_endpoints(state, mpo, sim_params, config)
    else:
        msg = f"Unknown BUG schedule: {config.schedule!r}"
        raise ValueError(msg)

    if entry_profile is not None and config.compression == "none":
        exit_profile = _bond_profile(state)
        if any(out > inp for out, inp in zip(exit_profile, entry_profile, strict=True)):
            msg = (
                f"fixed_profile BUG enlarged the bond profile: entry={entry_profile}, exit={exit_profile}. "
                "Fixed-profile mode must not enlarge bonds during the uncompressed sweep."
            )
            raise RuntimeError(msg)
