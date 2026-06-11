# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""TDVP sweep helpers.

Substep timestep scaling, truncation-policy split adapter, and fixed-χ bond
bookkeeping used by :mod:`mqt.yaqs.core.methods.tdvp.integrators`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ...data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams
from ..decompositions import merge_two_site, split_two_site

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...data_structures.mps import MPS
    from ..decompositions import SvdDistribution, TruncMode


# --- Truncation policy ---


def get_min_keep(sim_params: AnalogSimParams | StrongSimParams | WeakSimParams) -> int:
    """Return the minimum bond dimension to retain during TDVP truncation.

    Args:
        sim_params: Simulation parameters supplying ``max_bond_dim``.

    Returns:
        ``min(2, max_bond_dim)`` when a cap is set, otherwise ``2``.

    """
    cap = sim_params.max_bond_dim
    if cap is None:
        return 2
    return min(2, cap)


def split_tdvp(
    merged: NDArray[np.complex128],
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    physical_dimensions: list[int],
    svd_distribution: str,
    *,
    dynamic: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Split a merged two-site tensor using TDVP simulation truncation policy.

    Thin adapter around :func:`mqt.yaqs.core.methods.decompositions.split_two_site`.
    When ``dynamic`` is True, no ``max_bond_dim`` cap is applied during truncation
    (bond growth is handled by the dynamic TDVP sweep and global χ enforcement).
    Otherwise the cap is ``sim_params.max_bond_dim``.

    Args:
        merged: Two-site tensor ``(d_left * d_right, D0, D2)``.
        sim_params: Simulation parameters with ``svd_threshold``, ``trunc_mode``,
            and ``max_bond_dim``.
        physical_dimensions: ``[d_left, d_right]`` physical dimensions.
        svd_distribution: How to absorb singular values (``"left"``, ``"right"``, ``"sqrt"``).
        dynamic: If True, pass ``max_bond_dim=None`` to truncation (dynamic TDVP path).

    Returns:
        Left and right MPS site tensors after split and truncation.

    """
    return split_two_site(
        merged,
        physical_dimensions,
        svd_distribution=cast("SvdDistribution", svd_distribution),
        trunc_mode=cast("TruncMode", sim_params.trunc_mode),
        threshold=sim_params.svd_threshold,
        max_bond_dim=None if dynamic else sim_params.max_bond_dim,
        min_keep=get_min_keep(sim_params),
    )


# --- Substep timing ---


def _scale_dt(
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    step_scale: float,
) -> float:
    """Return the TDVP evolution timestep for the current symmetric substep.

    Args:
        sim_params: Analog parameters use ``dt * step_scale``; digital gate
            parameters treat ``step_scale`` as the full substep time.
        step_scale: Fraction of one evolution step assigned to this substep.

    Returns:
        Effective local evolution time for site and bond updates.

    """
    if not isinstance(sim_params, (StrongSimParams, WeakSimParams)):
        return float(sim_params.dt) * step_scale
    return step_scale


# --- Fixed-χ bond bookkeeping ---


def _sync_bond_dim(
    state: MPS,
    bond_index: int,
    target_dim: int,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams | None = None,
) -> None:
    """Set both tensors on an internal bond to share dimension ``target_dim``.

    When truncation is required, the two-site block is merged and re-split with
    SVD rather than raw axis slicing so the bond is reduced in Schmidt space.

    Args:
        state: MPS updated in place.
        bond_index: Internal bond index ``0 <= b < length - 1``.
        target_dim: Bond dimension enforced on both adjacent virtual indices.
        sim_params: Optional truncation settings for SVD compression.

    """
    left = state.tensors[bond_index]
    right = state.tensors[bond_index + 1]
    chi_out = int(left.shape[2])
    chi_in = int(right.shape[1])
    if chi_out == target_dim and chi_in == target_dim:
        return
    if chi_out != chi_in:
        align_dim = max(chi_out, chi_in)
        state.ensure_internal_bond_dims((bond_index,), align_dim, max_dim=align_dim)
        left = state.tensors[bond_index]
        right = state.tensors[bond_index + 1]
        chi_out = int(left.shape[2])
        chi_in = int(right.shape[1])
        if chi_out == target_dim and chi_in == target_dim:
            return
    if chi_out > target_dim or chi_in > target_dim:
        trunc_mode = cast("TruncMode", sim_params.trunc_mode) if sim_params is not None else "relative"
        threshold = sim_params.svd_threshold if sim_params is not None else 0.0
        merged = merge_two_site(left, right)
        phys_dims = [int(left.shape[0]), int(right.shape[0])]
        left_new, right_new = split_two_site(
            merged,
            phys_dims,
            svd_distribution="sqrt",
            trunc_mode=trunc_mode,
            threshold=threshold,
            max_bond_dim=target_dim,
            min_keep=1,
        )
        state.tensors[bond_index] = left_new
        state.tensors[bond_index + 1] = right_new
        return
    state.ensure_internal_bond_dims((bond_index,), target_dim, max_dim=target_dim)


def _get_bond_dim(
    state: MPS,
    bond_index: int,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> int:
    """Return the shared bond dimension to use before a bond transfer contraction.

    Args:
        state: MPS whose bond shapes are read.
        bond_index: Internal bond index ``0 <= b < length - 1``.
        sim_params: Supplies optional ``max_bond_dim`` cap.

    Returns:
        Target bond dimension, at least ``1``.

    """
    chi_left = int(state.tensors[bond_index].shape[2])
    chi_right = int(state.tensors[bond_index + 1].shape[1])
    chi_target = max(chi_left, chi_right)
    cap = sim_params.max_bond_dim
    if cap is not None:
        chi_target = min(chi_target, cap)
    return max(chi_target, 1)


def uses_fixed_chi(sim_params: AnalogSimParams | StrongSimParams | WeakSimParams) -> bool:
    """Return whether fixed-χ digital renormalization policy applies.

    Args:
        sim_params: Simulation parameters inspected for ``max_bond_dim``.

    Returns:
        True when ``max_bond_dim`` is set on digital simulation parameters.
        Analog Hamiltonian evolution is excluded (per-sweep renorm breaks ensembles).

    """
    return sim_params.max_bond_dim is not None and not isinstance(sim_params, AnalogSimParams)


def _get_norm(state: MPS) -> float:
    """Return the L2 norm of the full MPS state vector.

    Args:
        state: MPS whose norm is measured via ``scalar_product``.

    Returns:
        Non-negative Euclidean norm of the represented state vector.

    """
    overlap = state.scalar_product(state)
    norm_sq = float(np.real(np.asarray(overlap, dtype=np.complex128).flat[0]))
    return float(np.sqrt(max(norm_sq, 0.0)))


def renorm_trunc(state: MPS, _sim_params: AnalogSimParams | StrongSimParams | WeakSimParams) -> None:
    """Renormalize after explicit bond truncation (call only when fixed-χ digital).

    Args:
        state: MPS normalized in place.
        _sim_params: Reserved for call-site symmetry with :func:`renorm_drift`.

    """
    state.normalize()


def renorm_drift(state: MPS, sim_params: AnalogSimParams | StrongSimParams | WeakSimParams) -> None:
    """Renormalize when global norm drift exceeds tolerance (call only when fixed-χ digital).

    Args:
        state: MPS normalized in place when drift exceeds tolerance.
        sim_params: Supplies ``svd_threshold`` used to derive the drift tolerance.

    """
    drift_tol = max(1e-10, float(np.sqrt(sim_params.svd_threshold)))
    norm = _get_norm(state)
    if abs(norm - 1.0) > drift_tol:
        state.normalize()


def _align_bond(
    state: MPS,
    bond_index: int,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> None:
    """Align a bond to the fixed-χ target and optionally renormalize (digital only).

    Args:
        state: MPS updated in place when bond dimensions disagree.
        bond_index: Internal bond index ``0 <= b < length - 1``.
        sim_params: Fixed-χ digital parameters; no-op when ``max_bond_dim`` is unset.

    """
    if sim_params.max_bond_dim is None:
        return
    left = state.tensors[bond_index]
    right = state.tensors[bond_index + 1]
    if int(left.shape[2]) == int(right.shape[1]):
        return
    _sync_bond_dim(state, bond_index, _get_bond_dim(state, bond_index, sim_params), sim_params)
    if uses_fixed_chi(sim_params):
        renorm_trunc(state, sim_params)


def _cap_bonds(
    state: MPS,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> None:
    """Truncate all internal bonds to ``max_bond_dim`` before a fixed-χ sweep.

    Args:
        state: MPS whose bonds are truncated and optionally renormalized in place.
        sim_params: Supplies ``max_bond_dim``; no-op when unset.

    """
    cap = sim_params.max_bond_dim
    if cap is None:
        return
    changed = False
    for bond in range(state.length - 1):
        chi_out = int(state.tensors[bond].shape[2])
        chi_in = int(state.tensors[bond + 1].shape[1])
        if chi_out > cap or chi_in > cap:
            _sync_bond_dim(state, bond, cap, sim_params)
            changed = True
    if changed and uses_fixed_chi(sim_params):
        renorm_trunc(state, sim_params)


# --- Bond transfer geometry ---


def _resize_bond(
    bond_tensor: NDArray[np.complex128],
    *,
    lead: int | None = None,
    trail: int | None = None,
) -> NDArray[np.complex128]:
    """Resize leading and/or trailing axes of a bond transfer matrix.

    Args:
        bond_tensor: Two-index bond transfer matrix.
        lead: Optional target size for axis ``0``; unchanged when ``None``.
        trail: Optional target size for axis ``1``; unchanged when ``None``.

    Returns:
        Resized bond tensor.

    """
    out = bond_tensor
    if lead is not None:
        current = int(out.shape[0])
        if current != lead:
            if current > lead:
                out = out[:lead, :]
            else:
                padded = np.zeros((lead, out.shape[1]), dtype=out.dtype)
                padded[:current, :] = out
                out = padded
    if trail is not None:
        current = int(out.shape[1])
        if current != trail:
            if current > trail:
                out = out[:, :trail]
            else:
                padded = np.zeros((out.shape[0], trail), dtype=out.dtype)
                padded[:, :current] = out
                out = padded
    return out
