# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Digital long-range gate TDVP helpers.

Core :func:`mqt.yaqs.core.methods.tdvp.tdvp` with ``mode="dynamic"`` is the analog
integrator. This module provides :func:`gate_tdvp` for long-range two-qubit gates
with explicit retained-bond support during a forked dynamic TDVP sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
import opt_einsum as oe

from ..core import linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.mpo import MPO
    from ..core.data_structures.mps import MPS
    from ..core.data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams
    from ..core.methods.decompositions import SvdDistribution, TruncMode


def _support_enabled(sim_params: StrongSimParams | WeakSimParams) -> bool:
    """Return whether long-range TDVP should retain at least two support dimensions."""
    cap = sim_params.max_bond_dim
    return cap is None or cap >= 2


def _min_bond_dim(sim_params: AnalogSimParams | StrongSimParams | WeakSimParams) -> int:
    """Minimum bond dimension on gate-crossed bonds.

    Returns:
        ``min(2, max_bond_dim)`` when capped, otherwise ``2``.
    """
    cap = sim_params.max_bond_dim
    if cap is None:
        return 2
    return min(2, cap)


def _entanglement_threshold(sim_params: AnalogSimParams | StrongSimParams | WeakSimParams) -> float:
    """Second-Schmidt ratio floor for entanglement on crossed bonds.

    Returns:
        Entanglement detection threshold derived from ``svd_threshold``.
    """
    threshold = sim_params.svd_threshold
    return max(0.01, np.sqrt(threshold), 100.0 * threshold)


def _clamp_bond_dim(dim: int, sim_params: AnalogSimParams | StrongSimParams | WeakSimParams) -> int:
    """Clamp a bond dimension to the gate-support window.

    Returns:
        Bond dimension clipped to the gate-support range.
    """
    clamped = max(dim, _min_bond_dim(sim_params))
    cap = sim_params.max_bond_dim
    if cap is not None:
        clamped = min(clamped, cap)
    return clamped


def protected_bonds_for_two_site_gate(
    site0: int,
    site1: int,
    window_left: int,
    window_right: int,
) -> frozenset[int]:
    """Return window-local bond indices crossed by a two-site gate.

    Args:
        site0: First acted-on site (global index).
        site1: Second acted-on site (global index).
        window_left: Leftmost site included in the local TDVP window.
        window_right: Rightmost site included in the local TDVP window.

    Returns:
        Window-local indices of bonds between ``site0`` and ``site1``.
    """
    left_site = min(site0, site1)
    right_site = max(site0, site1)
    return frozenset(
        bond - window_left for bond in range(left_site, right_site) if window_left <= bond <= window_right - 1
    )


def retention_crossed_bonds(crossed_bonds: frozenset[int], num_sites: int) -> frozenset[int]:
    """Return crossed bonds that receive active support during gate TDVP.

    Args:
        crossed_bonds: Window-local bonds crossed by the gate.
        num_sites: Number of sites in the local TDVP window.

    Returns:
        Anchor-half subset of ``crossed_bonds`` monitored during sweeps.
    """
    midpoint = (num_sites - 1) // 2
    return frozenset(bond for bond in crossed_bonds if bond < midpoint)


def _pad_canonical(
    site_tensor: NDArray[np.complex128],
    bond_tensor: NDArray[np.complex128],
    target_dim: int,
    *,
    outgoing: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    if outgoing:
        chi = int(site_tensor.shape[2])
        if chi >= target_dim:
            return site_tensor, bond_tensor
        phys_l, chi_l, _ = site_tensor.shape
        _, chi_r = bond_tensor.shape
        new_site = np.zeros((phys_l, chi_l, target_dim), dtype=site_tensor.dtype)
        new_site[:, :, :chi] = site_tensor
        new_bond = np.zeros((target_dim, chi_r), dtype=bond_tensor.dtype)
        new_bond[:chi, :] = bond_tensor
        return new_site, new_bond
    chi = int(site_tensor.shape[1])
    if chi >= target_dim:
        return site_tensor, bond_tensor
    phys, _, chi_out = site_tensor.shape
    _, chi_in_right = bond_tensor.shape
    new_site = np.zeros((phys, target_dim, chi_out), dtype=site_tensor.dtype)
    new_site[:, :chi, :] = site_tensor
    new_bond = np.zeros((target_dim, chi_in_right), dtype=bond_tensor.dtype)
    new_bond[:chi, :] = bond_tensor
    return new_site, new_bond


def _second_schmidt_ratio(state: MPS, bond_index: int) -> float:
    left = state.tensors[bond_index]
    right = state.tensors[bond_index + 1]
    theta = np.tensordot(left, right, axes=(2, 1))
    mat = theta.reshape(left.shape[0] * left.shape[1], right.shape[0] * right.shape[2])
    _u, s_vec, _v = linalg.svd(mat, full_matrices=False)
    if len(s_vec) < 2 or float(s_vec[0]) <= 0.0:
        return 0.0
    return float(s_vec[1] / s_vec[0])


def _merged_second_schmidt_ratio(
    merged: NDArray[np.complex128],
    physical_dimensions: list[int],
) -> float:
    d_left, d_right = physical_dimensions
    tensor_reshaped = merged.reshape(d_left, d_right, merged.shape[1], merged.shape[2])
    tensor_transposed = tensor_reshaped.transpose((0, 2, 1, 3))
    shape_transposed = tensor_transposed.shape
    theta_mat = tensor_transposed.reshape(
        shape_transposed[0] * shape_transposed[1],
        shape_transposed[2] * shape_transposed[3],
    )
    _u, s_vec, _v = linalg.svd(theta_mat, full_matrices=False)
    if len(s_vec) < 2 or float(s_vec[0]) <= 0.0:
        return 0.0
    return float(s_vec[1] / s_vec[0])


def _reseed_support(
    state: MPS,
    bond_index: int,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> None:
    target = _min_bond_dim(sim_params)
    if target < 2:
        return
    eps = max(np.sqrt(sim_params.svd_threshold), np.finfo(float).eps)
    left = state.tensors[bond_index]
    right = state.tensors[bond_index + 1]
    phys_l, chi_l, chi = left.shape
    phys_r, _, chi_r = right.shape
    chi_copy = min(chi, target)
    new_left = np.zeros((phys_l, chi_l, target), dtype=left.dtype)
    new_right = np.zeros((phys_r, target, chi_r), dtype=right.dtype)
    new_left[:, :, :chi_copy] = left[:, :, :chi_copy]
    new_right[:, :chi_copy, :] = right[:, :chi_copy, :]
    if phys_l >= 2 and target >= 2:
        new_left[0, :, 1] = eps
        new_left[1, :, 1] = -eps
        new_right[0, 1, :] = 1.0
        new_right[1, 1, :] = 1.0
    state.tensors[bond_index] = new_left
    state.tensors[bond_index + 1] = new_right


@dataclass
class _GateSupportTracker:
    """Per-substep Schmidt monitoring state for retained gate bonds."""

    merged_peak: dict[int, float] = field(default_factory=dict)
    last_second: dict[int, float] = field(default_factory=dict)


def _canon_retained_ltr(
    tensor: NDArray[np.complex128],
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    tensor_shape = tensor.shape
    reshaped = tensor.reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
    chi_r = tensor_shape[2]
    eff_min = _min_bond_dim(sim_params)
    cap = sim_params.max_bond_dim
    entanglement_threshold = _entanglement_threshold(sim_params)
    if chi_r >= eff_min >= 2:
        u_mat, s_vec, vh_mat = linalg.svd(reshaped, full_matrices=False)
        if len(s_vec) >= 2 and float(s_vec[1] / max(float(s_vec[0]), 1e-30)) >= entanglement_threshold:
            keep = min(chi_r, max(eff_min, u_mat.shape[1]))
            if cap is not None:
                keep = min(keep, cap)
            site_tensor = u_mat[:, :keep].reshape((tensor_shape[0], tensor_shape[1], keep))
            return site_tensor, vh_mat[:keep, :]
    site_tensor, bond_tensor = np.linalg.qr(reshaped)
    chi_out = int(site_tensor.shape[1])
    site_tensor = site_tensor.reshape((tensor_shape[0], tensor_shape[1], chi_out))
    if eff_min >= 2 and cap is not None:
        target = _clamp_bond_dim(chi_out, sim_params)
        site_tensor, bond_tensor = _pad_canonical(site_tensor, bond_tensor, target, outgoing=True)
        chi_out = target
    if cap is not None and chi_out > cap:
        site_tensor = site_tensor[:, :, :cap]
        bond_tensor = bond_tensor[:cap, :]
    return site_tensor, bond_tensor


def _canon_retained_rtl(
    tensor: NDArray[np.complex128],
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    tensor_t = tensor.transpose((0, 2, 1))
    tensor_shape = tensor_t.shape
    reshaped = tensor_t.reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
    chi_r = tensor_shape[2]
    eff_min = _min_bond_dim(sim_params)
    cap = sim_params.max_bond_dim
    entanglement_threshold = _entanglement_threshold(sim_params)
    if chi_r >= eff_min >= 2:
        u_mat, s_vec, vh_mat = linalg.svd(reshaped, full_matrices=False)
        if len(s_vec) >= 2 and float(s_vec[1] / max(float(s_vec[0]), 1e-30)) >= entanglement_threshold:
            keep = min(chi_r, max(eff_min, u_mat.shape[1]))
            if cap is not None:
                keep = min(keep, cap)
            site_tensor = u_mat[:, :keep].reshape((tensor_shape[0], tensor_shape[1], keep)).transpose((0, 2, 1))
            bond_tensor = vh_mat[:keep, :].transpose()
            return site_tensor, bond_tensor.transpose()
    site_tensor, bond_tensor = np.linalg.qr(reshaped)
    site_tensor = site_tensor.reshape((tensor_shape[0], tensor_shape[1], site_tensor.shape[1])).transpose((
        0,
        2,
        1,
    ))
    if eff_min >= 2 and cap is not None:
        target = _clamp_bond_dim(int(site_tensor.shape[1]), sim_params)
        site_tensor, bond_tensor = _pad_canonical(site_tensor, bond_tensor, target, outgoing=False)
    if cap is not None and int(site_tensor.shape[1]) > cap:
        site_tensor = site_tensor[:, :cap, :]
        bond_tensor = bond_tensor[:cap, :]
    return site_tensor, bond_tensor.transpose()


def _split_gate_two_site(
    merged: NDArray[np.complex128],
    sim_params: StrongSimParams | WeakSimParams,
    physical_dimensions: list[int],
    svd_distribution: str,
    *,
    bond_index: int,
    retained_bonds: frozenset[int],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Split a merged two-site tensor with gate-support truncation policy.

    Returns:
        Left and right MPS site tensors after split and truncation.

    Raises:
        ValueError: If ``physical_dimensions`` is invalid, ``trunc_mode`` is
            unrecognized, or ``svd_distribution`` is invalid.
    """
    threshold = sim_params.svd_threshold
    trunc_mode = cast("TruncMode", sim_params.trunc_mode)
    svd_dist = cast("SvdDistribution", svd_distribution)
    cap = sim_params.max_bond_dim
    max_bond_dim = cap
    min_keep = 2 if cap is None else min(2, cap)
    if bond_index in retained_bonds:
        min_keep = max(min_keep, _min_bond_dim(sim_params))
        threshold = 0.0

    if len(physical_dimensions) != 2:
        msg = f"physical_dimensions must have exactly 2 elements (d_left, d_right); got {len(physical_dimensions)}."
        raise ValueError(msg)
    d_left = physical_dimensions[0]
    d_right = physical_dimensions[1]
    if merged.shape[0] != d_left * d_right:
        msg = "The first dimension of the tensor must be a combination of the given physical dimensions."
        raise ValueError(msg)

    tensor_reshaped = merged.reshape(d_left, d_right, merged.shape[1], merged.shape[2])
    tensor_transposed = tensor_reshaped.transpose((0, 2, 1, 3))
    shape_transposed = tensor_transposed.shape

    theta_mat = tensor_transposed.reshape(
        shape_transposed[0] * shape_transposed[1],
        shape_transposed[2] * shape_transposed[3],
    )
    u_mat, s_vec, v_mat = linalg.svd(theta_mat, full_matrices=False)

    if trunc_mode == "discarded_weight":
        keep = linalg.truncate(
            s_vec,
            mode="discarded_weight",
            threshold=threshold,
            max_bond_dim=max_bond_dim,
            min_keep=min_keep,
        )
    elif trunc_mode == "relative":
        keep = linalg.truncate(
            s_vec,
            mode="relative",
            threshold=threshold,
            max_bond_dim=max_bond_dim,
            min_keep=min_keep,
        )
    else:
        msg = f"Unknown truncation mode: {trunc_mode!r}"
        raise ValueError(msg)

    left_tensor = u_mat[:, :keep]
    s_vec = s_vec[:keep]
    right_tensor = v_mat[:keep, :]

    left_tensor = left_tensor.reshape((shape_transposed[0], shape_transposed[1], keep))
    right_tensor = right_tensor.reshape((keep, shape_transposed[2], shape_transposed[3]))

    if svd_dist == "left":
        left_tensor *= s_vec
    elif svd_dist == "right":
        right_tensor *= s_vec[:, None, None]
    elif svd_dist == "sqrt":
        sqrt_sigma = np.sqrt(s_vec)
        left_tensor *= sqrt_sigma
        right_tensor *= sqrt_sigma[:, None, None]
    else:
        msg = "svd_distribution parameter must be left, right, or sqrt."
        raise ValueError(msg)

    right_tensor = right_tensor.transpose((1, 0, 2))
    return left_tensor, right_tensor


def _after_retained_split(
    state: MPS,
    bond_index: int,
    merged: NDArray[np.complex128],
    physical_dimensions: list[int],
    sim_params: StrongSimParams | WeakSimParams,
    tracker: _GateSupportTracker,
) -> None:
    min_dim = _min_bond_dim(sim_params)
    threshold = sim_params.svd_threshold
    propagation = _entanglement_threshold(sim_params)
    if min_dim >= 2:
        state._ensure_internal_bond_dims(  # noqa: SLF001
            (bond_index,), min_dim, max_dim=sim_params.max_bond_dim
        )
    pre_ratio = _merged_second_schmidt_ratio(merged, physical_dimensions)
    tracker.merged_peak[bond_index] = max(tracker.merged_peak.get(bond_index, 0.0), pre_ratio)
    post_ratio = _second_schmidt_ratio(state, bond_index)
    if min_dim >= 2 and pre_ratio >= propagation and post_ratio < threshold:
        _reseed_support(state, bond_index, sim_params)
        post_ratio = _second_schmidt_ratio(state, bond_index)
    tracker.last_second[bond_index] = post_ratio


def _after_gate_substep(
    state: MPS,
    retained_bonds: frozenset[int],
    sim_params: StrongSimParams | WeakSimParams,
    tracker: _GateSupportTracker,
) -> None:
    min_dim = _min_bond_dim(sim_params)
    if min_dim < 2:
        return
    threshold = sim_params.svd_threshold
    propagation = _entanglement_threshold(sim_params)
    ratios = {bond: _second_schmidt_ratio(state, bond) for bond in retained_bonds}
    bonds_to_repad = [bond for bond in retained_bonds if state.tensors[bond].shape[2] < min_dim]
    if bonds_to_repad:
        state._ensure_internal_bond_dims(  # noqa: SLF001
            tuple(bonds_to_repad), min_dim, max_dim=sim_params.max_bond_dim
        )
        for bond in bonds_to_repad:
            ratios[bond] = _second_schmidt_ratio(state, bond)
    entangled = any(ratio >= propagation for ratio in ratios.values())
    entangled_cutoff = propagation if sim_params.max_bond_dim == 2 else threshold
    reseeded = False
    for bond in sorted(retained_bonds):
        ratio = ratios[bond]
        collapsed = tracker.last_second.get(bond, 0.0) >= propagation and ratio < threshold
        merged_relative = tracker.merged_peak.get(bond, 0.0) / max(ratio, 1e-30)
        if (
            collapsed
            or (entangled and ratio < entangled_cutoff)
            or (merged_relative >= propagation and ratio < threshold)
        ):
            _reseed_support(state, bond, sim_params)
            ratio = _second_schmidt_ratio(state, bond)
            reseeded = True
        tracker.last_second[bond] = ratio
    if reseeded:
        state.normalize()


def _gate_dynamic_tdvp_sweep(
    state: MPS,
    operator: MPO,
    sim_params: StrongSimParams | WeakSimParams,
    retained_bonds: frozenset[int],
    *,
    step_scale: float = 1.0,
    sweep_plan: list[float] | None = None,
) -> None:
    from ..core.methods.decompositions import merge_two_site  # noqa: PLC0415
    from ..core.methods.tdvp import (  # noqa: PLC0415
        initialize_right_environments,
        merge_mpo_tensors,
        update_bond,
        update_left_environment,
        update_right_environment,
        update_site,
    )
    from ..core.methods.tdvp_utils import (  # noqa: PLC0415
        _bond_dim_at_or_above_cap,
        _bond_dims_mismatched,
        _canonicalize_site_ltr,
        _canonicalize_site_rtl,
        _contract_bond_target_dim,
        _enforce_global_bond_cap,
        _prepare_substep_evolution_dt,
        _resize_bond,
        _sync_bond_dim,
    )

    if sweep_plan is not None:
        for plan_step_scale in sweep_plan:
            _gate_dynamic_tdvp_sweep(
                state,
                operator,
                sim_params,
                retained_bonds,
                step_scale=plan_step_scale,
            )
        return

    tracker = _GateSupportTracker()
    _enforce_global_bond_cap(state, sim_params)

    num_sites = operator.length

    right_blocks = initialize_right_environments(state, operator)
    left_blocks = [np.empty((0, 0, 0), dtype=np.complex128) for _ in range(num_sites)]
    chi0 = state.tensors[0].shape[1]
    mpo_dim = operator.tensors[0].shape[2]
    eye = np.zeros((chi0, mpo_dim, chi0), dtype=np.complex128)
    for i in range(chi0):
        eye[i, :, i] = 1
    left_blocks[0] = eye

    substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, step_scale)

    # ----- LEFT-TO-RIGHT DYNAMIC SWEEP -----
    for i in range(num_sites):
        bond_dim = state.tensors[i].shape[2]
        if _bond_dim_at_or_above_cap(bond_dim, sim_params.max_bond_dim):
            state.tensors[i] = update_site(
                left_blocks[i],
                right_blocks[i],
                operator.tensors[i],
                state.tensors[i],
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            if i != num_sites - 1:
                if i in retained_bonds:
                    site_tensor, bond_tensor = _canon_retained_ltr(state.tensors[i], sim_params)
                else:
                    site_tensor, bond_tensor = _canonicalize_site_ltr(state.tensors[i], sim_params)
                state.tensors[i] = site_tensor
                left_blocks[i + 1] = update_left_environment(
                    state.tensors[i], state.tensors[i], operator.tensors[i], left_blocks[i]
                )
                bond_tensor = update_bond(
                    left_blocks[i + 1],
                    right_blocks[i],
                    bond_tensor,
                    -0.5 * substep_evolution_dt,
                    krylov_tol=sim_params.krylov_tol,
                )
                if sim_params.max_bond_dim is not None:
                    if _bond_dims_mismatched(state, i):
                        _sync_bond_dim(state, i, _contract_bond_target_dim(state, i, sim_params))
                        state.normalize()
                    bond_tensor = _resize_bond(
                        bond_tensor,
                        lead=int(state.tensors[i].shape[2]),
                        trail=int(state.tensors[i + 1].shape[1]),
                    )
                state.tensors[i + 1] = oe.contract(state.tensors[i + 1], (0, 3, 2), bond_tensor, (1, 3), (0, 1, 2))
                if sim_params.max_bond_dim is not None and _bond_dims_mismatched(state, i):
                    _sync_bond_dim(state, i, _contract_bond_target_dim(state, i, sim_params))
                    state.normalize()
        elif i == num_sites - 1:
            continue
        elif i == num_sites - 2:
            merged_tensor = merge_two_site(state.tensors[i], state.tensors[i + 1])
            merged_mpo = merge_mpo_tensors(operator.tensors[i], operator.tensors[i + 1])
            merged_tensor = update_site(
                left_blocks[i],
                right_blocks[i + 1],
                merged_mpo,
                merged_tensor,
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            phys_dims = [state.physical_dimensions[i], state.physical_dimensions[i + 1]]
            state.tensors[i], state.tensors[i + 1] = _split_gate_two_site(
                merged_tensor,
                sim_params,
                phys_dims,
                "right",
                bond_index=i,
                retained_bonds=retained_bonds,
            )
            if i in retained_bonds:
                _after_retained_split(state, i, merged_tensor, phys_dims, sim_params, tracker)
            right_blocks[i] = update_right_environment(
                state.tensors[i + 1], state.tensors[i + 1], operator.tensors[i + 1], right_blocks[i + 1]
            )
            left_blocks[i + 1] = update_left_environment(
                state.tensors[i], state.tensors[i], operator.tensors[i], left_blocks[i]
            )

        else:
            merged_tensor = merge_two_site(state.tensors[i], state.tensors[i + 1])
            merged_mpo = merge_mpo_tensors(operator.tensors[i], operator.tensors[i + 1])
            merged_tensor = update_site(
                left_blocks[i],
                right_blocks[i + 1],
                merged_mpo,
                merged_tensor,
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            phys_dims = [state.physical_dimensions[i], state.physical_dimensions[i + 1]]
            state.tensors[i], state.tensors[i + 1] = _split_gate_two_site(
                merged_tensor,
                sim_params,
                phys_dims,
                "right",
                bond_index=i,
                retained_bonds=retained_bonds,
            )
            if i in retained_bonds:
                _after_retained_split(state, i, merged_tensor, phys_dims, sim_params, tracker)
            left_blocks[i + 1] = update_left_environment(
                state.tensors[i], state.tensors[i], operator.tensors[i], left_blocks[i]
            )
            state.tensors[i + 1] = update_site(
                left_blocks[i + 1],
                right_blocks[i + 1],
                operator.tensors[i + 1],
                state.tensors[i + 1],
                -0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
    substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, step_scale)

    # ----- RIGHT-TO-LEFT DYNAMIC SWEEP -----
    for i in reversed(range(num_sites)):
        bond_dim = state.tensors[i].shape[1]
        if _bond_dim_at_or_above_cap(bond_dim, sim_params.max_bond_dim):
            state.tensors[i] = update_site(
                left_blocks[i],
                right_blocks[i],
                operator.tensors[i],
                state.tensors[i],
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            if i != 0:
                if (i - 1) in retained_bonds:
                    site_tensor, bond_tensor = _canon_retained_rtl(state.tensors[i], sim_params)
                else:
                    site_tensor, bond_tensor = _canonicalize_site_rtl(state.tensors[i], sim_params)
                state.tensors[i] = site_tensor
                right_blocks[i - 1] = update_right_environment(
                    state.tensors[i], state.tensors[i], operator.tensors[i], right_blocks[i]
                )
                bond_tensor = bond_tensor.transpose()
                bond_tensor = update_bond(
                    left_blocks[i],
                    right_blocks[i - 1],
                    bond_tensor,
                    -0.5 * substep_evolution_dt,
                    krylov_tol=sim_params.krylov_tol,
                )
                if sim_params.max_bond_dim is not None:
                    if _bond_dims_mismatched(state, i - 1):
                        _sync_bond_dim(state, i - 1, _contract_bond_target_dim(state, i - 1, sim_params))
                        state.normalize()
                    bond_tensor = _resize_bond(
                        bond_tensor,
                        lead=int(state.tensors[i - 1].shape[2]),
                        trail=int(state.tensors[i].shape[1]),
                    )
                state.tensors[i - 1] = oe.contract(state.tensors[i - 1], (0, 1, 3), bond_tensor, (3, 2), (0, 1, 2))
                if sim_params.max_bond_dim is not None and _bond_dims_mismatched(state, i - 1):
                    _sync_bond_dim(state, i - 1, _contract_bond_target_dim(state, i - 1, sim_params))
                    state.normalize()
        elif i == 0:
            continue
        else:
            merged_tensor = merge_two_site(state.tensors[i - 1], state.tensors[i])
            merged_mpo = merge_mpo_tensors(operator.tensors[i - 1], operator.tensors[i])
            merged_tensor = update_site(
                left_blocks[i - 1],
                right_blocks[i],
                merged_mpo,
                merged_tensor,
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            phys_dims = [state.physical_dimensions[i - 1], state.physical_dimensions[i]]
            state.tensors[i - 1], state.tensors[i] = _split_gate_two_site(
                merged_tensor,
                sim_params,
                phys_dims,
                "left",
                bond_index=i - 1,
                retained_bonds=retained_bonds,
            )
            if (i - 1) in retained_bonds:
                _after_retained_split(state, i - 1, merged_tensor, phys_dims, sim_params, tracker)
            right_blocks[i - 1] = update_right_environment(
                state.tensors[i], state.tensors[i], operator.tensors[i], right_blocks[i]
            )
            if i != 1:
                state.tensors[i - 1] = update_site(
                    left_blocks[i - 1],
                    right_blocks[i - 1],
                    operator.tensors[i - 1],
                    state.tensors[i - 1],
                    -0.5 * substep_evolution_dt,
                    krylov_tol=sim_params.krylov_tol,
                )

    _after_gate_substep(state, retained_bonds, sim_params, tracker)
    if sim_params.max_bond_dim is not None:
        state.normalize()


def prepare_retained_bonds(
    state: MPS,
    site0: int,
    site1: int,
    window: tuple[int, int],
    sim_params: StrongSimParams | WeakSimParams,
) -> frozenset[int] | None:
    """Build retained bond indices for a long-range gate, pre-padding support dims.

    Returns:
        ``None`` when bond support is disabled under the current χ budget.
    """
    if not _support_enabled(sim_params):
        return None
    crossed = protected_bonds_for_two_site_gate(site0, site1, window[0], window[1])
    bonds = retention_crossed_bonds(crossed, state.length)
    min_dim = _min_bond_dim(sim_params)
    state._ensure_internal_bond_dims(  # noqa: SLF001
        tuple(bonds), min_dim, max_dim=sim_params.max_bond_dim
    )
    return bonds


def gate_tdvp(
    state: MPS,
    operator: MPO,
    sim_params: StrongSimParams | WeakSimParams,
    *,
    retained_bonds: frozenset[int] | None = None,
) -> None:
    """Evolve a window MPS under a gate generator with optional retained-bond support.

    Entry point for digital long-range gates. When ``retained_bonds`` is set, a
    forked dynamic TDVP sweep enforces gate bond support; otherwise the core
    dynamic sweep is used (e.g. χ=1 degraded path).

    Raises:
        ValueError: If ``state`` and ``operator`` lengths mismatch.
    """
    from ..core.methods.tdvp import (  # noqa: PLC0415
        _local_dynamic_tdvp_sweep,
        _run_sweeps,
        _single_site_tdvp_sweep,
    )

    if operator.length != state.length:
        msg = "MPS and operator must have the same number of sites."
        raise ValueError(msg)
    if operator.length == 1:
        _run_sweeps(_single_site_tdvp_sweep, state, operator, sim_params)
        return
    if retained_bonds:
        _run_sweeps(_gate_dynamic_tdvp_sweep, state, operator, sim_params, retained_bonds)
    else:
        _run_sweeps(_local_dynamic_tdvp_sweep, state, operator, sim_params)
