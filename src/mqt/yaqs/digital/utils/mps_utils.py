# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MPS utilities for digital circuit simulation.

Applies operators to MPS states via generic local MPO--MPS contraction followed by
bond compression, including long-range two-qubit gates built as extended gate MPOs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ...core.data_structures.mps import MPS
from ...core.libraries.gate_library import extend_gate
from ...core.methods.decompositions import merge_two_site, split_two_site

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.data_structures.simulation_parameters import StrongSimParams, WeakSimParams
    from ...core.libraries.gate_library import BaseGate
    from ...core.methods.decompositions import TruncMode


def _identity_mpo_site(physical_dimension: int) -> NDArray[np.complex128]:
    """Single-site identity MPO tensor ``(phys_out, phys_in, 1, 1)``."""
    tensor = np.eye(physical_dimension, dtype=np.complex128)
    return np.expand_dims(np.expand_dims(tensor, axis=2), axis=3)


def contract_mpo_site_with_mps_site(
    mpo_tensor: NDArray[np.complex128],
    mps_tensor: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    """Contract one MPO site with one MPS site using library leg ordering.

    MPO layout: ``(phys_out, phys_in, chi_left, chi_right)``.
    MPS layout: ``(phys, chi_left, chi_right)``.

    Virtual bonds are fused with MPS indices first and MPO indices second on both sides.

    Args:
        mpo_tensor: Gate or operator MPO tensor at one site.
        mps_tensor: MPS site tensor.

    Returns:
        Updated MPS site tensor ``(phys_out, chi_left * mpo_left, chi_right * mpo_right)``.
    """
    operator = np.asarray(mpo_tensor, dtype=np.complex128)
    site = np.asarray(mps_tensor, dtype=np.complex128)
    theta = np.tensordot(operator, site, axes=([1], [0]))
    phys_out, mpo_left, mpo_right, mps_left, mps_right = theta.shape
    return np.asarray(
        theta.transpose(0, 3, 1, 4, 2).reshape(
            phys_out,
            mps_left * mpo_left,
            mps_right * mpo_right,
        ),
        dtype=np.complex128,
    )


def apply_mpo_to_mps(
    state: MPS,
    mpo_tensors: list[NDArray[np.complex128]],
) -> None:
    """Apply an MPO to an MPS by local site contraction (no compression).

    Args:
        state: MPS updated in place.
        mpo_tensors: One MPO tensor per site, same length as ``state``.

    Raises:
        ValueError: If ``len(mpo_tensors) != state.length``.
    """
    if len(mpo_tensors) != state.length:
        msg = f"MPO length {len(mpo_tensors)} does not match MPS length {state.length}."
        raise ValueError(msg)

    for site, operator in enumerate(mpo_tensors):
        state.tensors[site] = contract_mpo_site_with_mps_site(operator, state.tensors[site])


def _compress_mps(
    state: MPS,
    sim_params: StrongSimParams | WeakSimParams,
) -> None:
    """Compress an MPS after MPO application using two-site SVD sweeps."""
    if state.length == 1:
        return

    trunc_mode = cast("TruncMode", sim_params.trunc_mode)
    canonical = state.check_canonical_form()
    orth_center = canonical[0] if canonical and canonical[0] >= 0 else state.length - 1

    for site in range(orth_center):
        left_tensor = state.tensors[site]
        right_tensor = state.tensors[site + 1]
        merged = merge_two_site(left_tensor, right_tensor)
        left_new, right_new = split_two_site(
            merged,
            [left_tensor.shape[0], right_tensor.shape[0]],
            svd_distribution="right",
            trunc_mode=trunc_mode,
            threshold=sim_params.svd_threshold,
            max_bond_dim=sim_params.max_bond_dim,
            min_bond_dim=sim_params.min_bond_dim,
        )
        state.tensors[site] = left_new
        state.tensors[site + 1] = right_new

    state.flip_network()
    orth_flipped = state.length - 1 - orth_center
    for site in range(orth_flipped):
        left_tensor = state.tensors[site]
        right_tensor = state.tensors[site + 1]
        merged = merge_two_site(left_tensor, right_tensor)
        left_new, right_new = split_two_site(
            merged,
            [left_tensor.shape[0], right_tensor.shape[0]],
            svd_distribution="right",
            trunc_mode=trunc_mode,
            threshold=sim_params.svd_threshold,
            max_bond_dim=sim_params.max_bond_dim,
            min_bond_dim=sim_params.min_bond_dim,
        )
        state.tensors[site] = left_new
        state.tensors[site + 1] = right_new
    state.flip_network()


def _gate_tensor_in_mps_order(
    gate: BaseGate,
    left_site: int,
    right_site: int,
) -> NDArray[np.complex128]:
    """Return ``gate.tensor`` as ``U[out_l, out_r, in_l, in_r]`` on ascending MPS sites."""
    if gate.sites[0] == left_site and gate.sites[1] == right_site:
        return np.asarray(gate.tensor, dtype=np.complex128)
    if gate.sites[0] == right_site and gate.sites[1] == left_site:
        return np.asarray(np.transpose(gate.tensor, (1, 0, 3, 2)), dtype=np.complex128)
    msg = f"Gate sites {gate.sites!r} are not consistent with MPS sites ({left_site}, {right_site})."
    raise ValueError(msg)


def _extended_gate_mpo_on_chain(
    gate: BaseGate,
    chain_length: int,
) -> list[NDArray[np.complex128]]:
    """Embed an extended two-qubit gate MPO into a full-length chain with identity outside."""
    site0, site1 = gate.sites[0], gate.sites[1]
    first_site = min(site0, site1)
    last_site = max(site0, site1)
    ordered_tensor = _gate_tensor_in_mps_order(gate, first_site, last_site)
    support = extend_gate(ordered_tensor, [first_site, last_site])
    distance = last_site - first_site + 1
    if distance != len(support):
        msg = f"Expected {distance} gate MPO sites, got {len(support)}."
        raise ValueError(msg)

    phys_dim = support[0].shape[0]
    identity_site = _identity_mpo_site(phys_dim)
    tensors: list[NDArray[np.complex128]] = []
    for site in range(chain_length):
        if site < first_site or site > last_site:
            tensors.append(np.array(identity_site, copy=True))
        else:
            tensors.append(np.asarray(support[site - first_site], dtype=np.complex128))
    return tensors


def apply_long_range_gate(
    state: MPS,
    gate: BaseGate,
    sim_params: StrongSimParams | WeakSimParams,
) -> tuple[int, int]:
    """Apply a long-range two-qubit gate via generic MPO--MPS application.

    The extended gate MPO from :func:`~mqt.yaqs.core.libraries.gate_library.extend_gate`
    is embedded on the full chain (identity outside the support), contracted locally at
    every site, then the enlarged MPS is compressed with the usual truncation settings.

    Args:
        state: MPS updated in place.
        gate: Internal gate object with ``mpo_tensors`` populated.
        sim_params: Truncation settings for the compression sweep.

    Returns:
        ``(first_site, last_site)`` spanning the gate support in MPS order.

    Raises:
        AttributeError: If the gate has no ``mpo_tensors``.
        ValueError: If the gate MPO length does not match the support distance.
    """
    site0, site1 = gate.sites[0], gate.sites[1]
    first_site = min(site0, site1)
    last_site = max(site0, site1)

    mpo_tensors = _extended_gate_mpo_on_chain(gate, state.length)
    apply_mpo_to_mps(state, mpo_tensors)
    _compress_mps(state, sim_params)

    return first_site, last_site
