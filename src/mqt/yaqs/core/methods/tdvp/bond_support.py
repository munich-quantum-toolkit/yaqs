# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Long-range digital gate bond support for dynamic TDVP.

Bond geometry, preparation helpers, and sweep hooks used by
:func:`mqt.yaqs.core.methods.tdvp.tdvp.tdvp` when ``support_bonds`` is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ... import linalg
from ..decompositions import split_two_site
from .sweep_utils import _renorm_if_digital, compute_min_keep

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...data_structures.mps import MPS
    from ...data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams
    from ..decompositions import SvdDistribution, TruncMode

__all__ = [
    "prepare_support_bonds",
    "protected_bonds_for_two_site_gate",
    "select_protected_seed_bonds",
]


# --- Public bond geometry and setup ---


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


def prepare_support_bonds(
    state: MPS,
    site0: int,
    site1: int,
    window: tuple[int, int],
    sim_params: StrongSimParams | WeakSimParams,
) -> frozenset[int] | None:
    """Build support bond indices for a long-range gate, pre-padding support dims.

    Gate-local protected-bond support retention preserves the minimal virtual
    support required by an active long-range entangling gate, subject to
    ``max_bond_dim``. All bonds crossed by the gate within the TDVP window are
    returned as the protected set; seed bonds (see
    :func:`select_protected_seed_bonds`) are pre-padded for null-space
    initialization. Returns ``None`` when ``max_bond_dim < 2``.

    Returns:
        Window-local bond indices receiving active support during sweeps, or
        ``None`` when bond support is disabled under the current χ budget.
    """
    if not _support_enabled(sim_params):
        return None
    protected_bonds = protected_bonds_for_two_site_gate(site0, site1, window[0], window[1])
    min_dim = compute_min_keep(sim_params)
    seed_bonds = select_protected_seed_bonds(protected_bonds)
    state._ensure_internal_bond_dims(tuple(seed_bonds), min_dim, max_dim=sim_params.max_bond_dim)
    return protected_bonds


def select_protected_seed_bonds(protected_bonds: frozenset[int]) -> frozenset[int]:
    """Select crossed bonds used to initialize protected virtual support.

    The selected bonds are always a subset of the active gate-crossed bonds.
    They are used only to initialize a nonzero virtual direction so dynamic
    TDVP can leave rank-deficient product manifolds during long-range
    entangling updates.

    The rule is deterministic and independent of system size or TDVP window
    size: sort the crossed-bond interval and seed the left half
    (``floor(n/2)`` bonds). For a single crossed bond, seed that bond. This
    provides a reproducible left-to-right support-initialization convention
    without hardcoded anchors.

    This helper does not override ``max_bond_dim``. If the effective bond cap
    is one, no rank-2 support should be created by the caller.

    Args:
        protected_bonds: Gate-crossed bonds within the TDVP window.

    Returns:
        Subset of ``protected_bonds`` chosen for support seeding.
    """
    if not protected_bonds:
        return frozenset()
    ordered = sorted(protected_bonds)
    split_at = len(ordered) // 2
    if split_at <= 0:
        return frozenset({ordered[0]})
    return frozenset(ordered[:split_at])


# --- Support policy ---


def _support_enabled(sim_params: StrongSimParams | WeakSimParams) -> bool:
    """Return whether long-range TDVP should retain at least two support dimensions."""
    cap = sim_params.max_bond_dim
    return cap is None or cap >= 2


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
    clamped = max(dim, compute_min_keep(sim_params))
    cap = sim_params.max_bond_dim
    if cap is not None:
        clamped = min(clamped, cap)
    return clamped


# --- Schmidt metrics and padding ---


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


# --- Sweep hooks ---


def _init_support_null_direction(
    state: MPS,
    bond_index: int,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> None:
    target = compute_min_keep(sim_params)
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


def _canon_support_site(
    tensor: NDArray[np.complex128],
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    *,
    ltr: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    eff_min = compute_min_keep(sim_params)
    cap = sim_params.max_bond_dim
    entanglement_threshold = _entanglement_threshold(sim_params)

    if ltr:
        tensor_shape = tensor.shape
        reshaped = tensor.reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        chi_r = tensor_shape[2]
    else:
        tensor_t = tensor.transpose((0, 2, 1))
        tensor_shape = tensor_t.shape
        reshaped = tensor_t.reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        chi_r = tensor_shape[2]

    if chi_r >= eff_min >= 2:
        u_mat, s_vec, vh_mat = linalg.svd(reshaped, full_matrices=False)
        if len(s_vec) >= 2 and float(s_vec[1] / max(float(s_vec[0]), 1e-30)) >= entanglement_threshold:
            keep = min(chi_r, max(eff_min, u_mat.shape[1]))
            if cap is not None:
                keep = min(keep, cap)
            if ltr:
                site_tensor = u_mat[:, :keep].reshape((tensor_shape[0], tensor_shape[1], keep))
                return site_tensor, vh_mat[:keep, :]
            site_tensor = u_mat[:, :keep].reshape((tensor_shape[0], tensor_shape[1], keep)).transpose((0, 2, 1))
            bond_tensor = vh_mat[:keep, :].transpose()
            return site_tensor, bond_tensor.transpose()

    site_tensor, bond_tensor = np.linalg.qr(reshaped)
    if ltr:
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


def _split_support_two_site(
    merged: NDArray[np.complex128],
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    physical_dimensions: list[int],
    svd_distribution: str,
    *,
    bond_index: int,
    seed_bonds: frozenset[int],
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Split a merged two-site tensor with bond-support truncation policy.

    Returns:
        Left and right MPS site tensors after split and truncation.
    """
    cap = sim_params.max_bond_dim
    min_keep = compute_min_keep(sim_params)
    threshold = 0.0 if bond_index in seed_bonds else sim_params.svd_threshold
    return split_two_site(
        merged,
        physical_dimensions,
        svd_distribution=cast("SvdDistribution", svd_distribution),
        trunc_mode=cast("TruncMode", sim_params.trunc_mode),
        threshold=threshold,
        max_bond_dim=cap,
        min_keep=min_keep,
    )


def _after_support_split(
    state: MPS,
    bond_index: int,
    merged: NDArray[np.complex128],
    physical_dimensions: list[int],
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    merged_peak: dict[int, float],
    last_second: dict[int, float],
    *,
    seed_bonds: frozenset[int],
) -> None:
    min_dim = compute_min_keep(sim_params)
    threshold = sim_params.svd_threshold
    propagation = _entanglement_threshold(sim_params)
    if min_dim >= 2:
        state._ensure_internal_bond_dims((bond_index,), min_dim, max_dim=sim_params.max_bond_dim)
    pre_ratio = _merged_second_schmidt_ratio(merged, physical_dimensions)
    merged_peak[bond_index] = max(merged_peak.get(bond_index, 0.0), pre_ratio)
    post_ratio = _second_schmidt_ratio(state, bond_index)
    if (
        bond_index in seed_bonds
        and min_dim >= 2
        and pre_ratio >= propagation
        and post_ratio < threshold
    ):
        _init_support_null_direction(state, bond_index, sim_params)
        post_ratio = _second_schmidt_ratio(state, bond_index)
    last_second[bond_index] = post_ratio


def _after_support_substep(
    state: MPS,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    merged_peak: dict[int, float],
    last_second: dict[int, float],
    *,
    seed_bonds: frozenset[int],
) -> None:
    min_dim = compute_min_keep(sim_params)
    if min_dim < 2:
        return
    threshold = sim_params.svd_threshold
    propagation = _entanglement_threshold(sim_params)
    ratios = {bond: _second_schmidt_ratio(state, bond) for bond in seed_bonds}
    bonds_to_repad = [bond for bond in seed_bonds if state.tensors[bond].shape[2] < min_dim]
    if bonds_to_repad:
        state._ensure_internal_bond_dims(tuple(bonds_to_repad), min_dim, max_dim=sim_params.max_bond_dim)
        for bond in bonds_to_repad:
            ratios[bond] = _second_schmidt_ratio(state, bond)
    entangled = any(ratio >= propagation for ratio in ratios.values())
    entangled_cutoff = propagation if sim_params.max_bond_dim == 2 else threshold
    initialized = False
    for bond in sorted(seed_bonds):
        ratio = ratios[bond]
        collapsed = last_second.get(bond, 0.0) >= propagation and ratio < threshold
        merged_relative = merged_peak.get(bond, 0.0) / max(ratio, 1e-30)
        if (
            collapsed
            or (entangled and ratio < entangled_cutoff)
            or (merged_relative >= propagation and ratio < threshold)
        ):
            _init_support_null_direction(state, bond, sim_params)
            ratio = _second_schmidt_ratio(state, bond)
            initialized = True
        last_second[bond] = ratio
    if initialized:
        _renorm_if_digital(state, sim_params)
