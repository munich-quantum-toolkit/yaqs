# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Digital long-range gate TDVP helpers.

Core :func:`mqt.yaqs.core.methods.tdvp.tdvp` with ``mode="dynamic"`` is the fixed
integrator protocol for analog and general circuit evolution. This module adds
bond-support hooks for long-range two-qubit gates only (via :func:`gate_tdvp`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core import linalg

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.mpo import MPO
    from ..core.data_structures.mps import MPS
    from ..core.data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams


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


class BondHooks:
    """Bond-support hooks for long-range gate TDVP sweeps."""

    def __init__(
        self,
        bonds: frozenset[int],
        sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    ) -> None:
        """Store retained bond indices and per-substep monitoring state."""
        self.bonds = bonds
        self._params = sim_params
        self._merged_peak: dict[int, float] = {}
        self._last_second: dict[int, float] = {}

    def split(self, bond: int, min_dim: int, threshold: float) -> tuple[int, float]:
        """Adjust truncation on a retained bond split.

        Returns:
            ``(min_dim, threshold)`` for ordinary bonds; boosted floor and zero
            threshold on retained bonds.
        """
        if bond in self.bonds:
            return max(min_dim, _min_bond_dim(self._params)), 0.0
        return min_dim, threshold

    def canon(
        self,
        tensor: NDArray[np.complex128],
        sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
        *,
        rtl: bool,
    ) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
        """Canonicalize a site on a retained bond during a dynamic sweep.

        Returns:
            Canonical site tensor and bond transfer matrix.
        """
        if rtl:
            return self._canon_rtl(tensor, sim_params)
        return self._canon_ltr(tensor, sim_params)

    def _canon_ltr(  # noqa: PLR6301
        self,
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

    def _canon_rtl(  # noqa: PLR6301
        self,
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

    def after_split(
        self,
        state: MPS,
        bond_index: int,
        merged: NDArray[np.complex128],
        physical_dimensions: list[int],
        sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    ) -> None:
        """Pad and re-seed a retained bond immediately after a two-site split."""
        min_dim = _min_bond_dim(sim_params)
        threshold = sim_params.svd_threshold
        propagation = _entanglement_threshold(sim_params)
        if min_dim >= 2:
            state._ensure_internal_bond_dims(  # noqa: SLF001
                (bond_index,), min_dim, max_dim=sim_params.max_bond_dim
            )
        pre_ratio = _merged_second_schmidt_ratio(merged, physical_dimensions)
        self._merged_peak[bond_index] = max(self._merged_peak.get(bond_index, 0.0), pre_ratio)
        post_ratio = _second_schmidt_ratio(state, bond_index)
        if min_dim >= 2 and pre_ratio >= propagation and post_ratio < threshold:
            _reseed_support(state, bond_index, sim_params)
            post_ratio = _second_schmidt_ratio(state, bond_index)
        self._last_second[bond_index] = post_ratio

    def after_substep(
        self,
        state: MPS,
        sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    ) -> None:
        """Re-pad and re-seed retained bonds after one dynamic TDVP substep."""
        min_dim = _min_bond_dim(sim_params)
        if min_dim < 2:
            return
        threshold = sim_params.svd_threshold
        propagation = _entanglement_threshold(sim_params)
        ratios = {bond: _second_schmidt_ratio(state, bond) for bond in self.bonds}
        bonds_to_repad = [bond for bond in self.bonds if state.tensors[bond].shape[2] < min_dim]
        if bonds_to_repad:
            state._ensure_internal_bond_dims(  # noqa: SLF001
                tuple(bonds_to_repad), min_dim, max_dim=sim_params.max_bond_dim
            )
            for bond in bonds_to_repad:
                ratios[bond] = _second_schmidt_ratio(state, bond)
        entangled = any(ratio >= propagation for ratio in ratios.values())
        entangled_cutoff = propagation if sim_params.max_bond_dim == 2 else threshold
        reseeded = False
        for bond in sorted(self.bonds):
            ratio = ratios[bond]
            collapsed = self._last_second.get(bond, 0.0) >= propagation and ratio < threshold
            merged_relative = self._merged_peak.get(bond, 0.0) / max(ratio, 1e-30)
            if (
                collapsed
                or (entangled and ratio < entangled_cutoff)
                or (merged_relative >= propagation and ratio < threshold)
            ):
                _reseed_support(state, bond, sim_params)
                ratio = _second_schmidt_ratio(state, bond)
                reseeded = True
            self._last_second[bond] = ratio
        if reseeded:
            state.normalize()


def make_hooks(
    state: MPS,
    site0: int,
    site1: int,
    window: tuple[int, int],
    sim_params: StrongSimParams | WeakSimParams,
) -> BondHooks | None:
    """Build bond hooks for a long-range gate, pre-padding retained bonds.

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
    return BondHooks(bonds, sim_params)


def gate_tdvp(
    state: MPS,
    operator: MPO,
    sim_params: StrongSimParams | WeakSimParams,
    *,
    hooks: BondHooks | None = None,
) -> None:
    """Evolve a window MPS under a gate generator with optional bond hooks.

    Unlike :func:`mqt.yaqs.core.methods.tdvp.tdvp` ``mode="dynamic"``, this entry
    point is for digital long-range gates and accepts :class:`BondHooks`.

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
    _run_sweeps(_local_dynamic_tdvp_sweep, state, operator, sim_params, hooks=hooks)
