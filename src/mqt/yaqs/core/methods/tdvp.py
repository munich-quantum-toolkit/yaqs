# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""1TDVP + 2TDVP sweep orchestration and public entry point.

This module implements Time-Dependent Variational Principle (TDVP) integrators
for Matrix Product States (MPS). Low-level kernels (MPO environments, dense
effective operators, local updates, split/canon helpers) live in
:mod:`mqt.yaqs.core.methods.tdvp_primitives`. Long-range digital gate bond
support hooks live in :mod:`mqt.yaqs.core.methods.tdvp_bond_support`.

Two-site MPS merge/split with SVD truncation lives in
:mod:`mqt.yaqs.core.methods.decompositions`.

These methods are designed for simulating the dynamics of quantum many-body systems and are based on
techniques described in Haegeman et al., Phys. Rev. B 94, 165116 (2016).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import opt_einsum as oe

from ..data_structures.simulation_parameters import AnalogSimParams
from .decompositions import merge_two_site
from .tdvp_bond_support import (
    _after_support_split,
    _after_support_substep,
    _canon_support_site,
    _split_support_two_site,
)
from .tdvp_primitives import (
    _bond_dim_at_or_above_cap,
    _bond_dims_mismatched,
    _canonicalize_site_ltr,
    _canonicalize_site_rtl,
    _contract_bond_target_dim,
    _enforce_global_bond_cap,
    _prepare_substep_evolution_dt,
    _resize_bond,
    _split_two_site_tdvp,
    _sync_bond_dim,
    initialize_right_environments,
    merge_mpo_tensors,
    update_bond,
    update_left_environment,
    update_right_environment,
    update_site,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..data_structures.mpo import MPO
    from ..data_structures.mps import MPS
    from ..data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams


Mode = Literal["1site", "2site", "dynamic"]


# --- Sweep orchestration ---


def _build_tdvp_sweep_plan(
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> list[float]:
    """Build the sweep plan for one evolution step (gate or analog ``dt``).

    Each substep is symmetric (LTR then RTL) at ``step_time / tdvp_sweeps``.

    Returns:
        Ordered step-scale factors for one batched kernel call.
    """
    step_scale = 1.0 / sim_params.tdvp_sweeps
    return [step_scale for _ in range(sim_params.tdvp_sweeps)]


def _run_sweeps(
    evolve_once: Callable[..., None],
    state: MPS,
    operator: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    /,
    *args: object,
    **kwargs: object,
) -> None:
    """Run ``sim_params.tdvp_sweeps`` TDVP substeps per evolution step.

    Public TDVP entry points validate inputs and delegate here. Private ``_*_sweep``
    kernels perform one symmetric substep and accept ``step_scale``; they must not
    re-enter the public wrappers.

    Substep geometry (``_build_tdvp_sweep_plan``): ``tdvp_sweeps`` symmetric substeps
    (LTR then RTL each) at ``step_time / tdvp_sweeps`` for analog ``dt`` and digital gates.
    """
    evolve_once(
        state,
        operator,
        sim_params,
        *args,
        sweep_plan=_build_tdvp_sweep_plan(sim_params),
        **kwargs,
    )


# --- TDVP integrators ---


def _single_site_tdvp_sweep(
    state: MPS,
    operator: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    *,
    step_scale: float = 1.0,
    sweep_plan: list[float] | None = None,
) -> None:
    if sweep_plan is not None:
        for plan_step_scale in sweep_plan:
            _single_site_tdvp_sweep(
                state,
                operator,
                sim_params,
                step_scale=plan_step_scale,
            )
        return

    num_sites = operator.length

    right_blocks = initialize_right_environments(state, operator)

    left_blocks = [np.empty((0, 0, 0), dtype=np.complex128) for _ in range(num_sites)]
    left_virtual_dim = state.tensors[0].shape[1]
    mpo_left_dim = operator.tensors[0].shape[2]
    left_identity = np.zeros((left_virtual_dim, mpo_left_dim, left_virtual_dim), dtype=right_blocks[0].dtype)
    for i in range(left_virtual_dim):
        for a in range(mpo_left_dim):
            left_identity[i, a, i] = 1
    left_blocks[0] = left_identity

    substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, step_scale)

    # Left-to-right sweep: Update sites 0 to L-2.
    for i in range(num_sites - 1):
        state.tensors[i] = update_site(
            left_blocks[i],
            right_blocks[i],
            operator.tensors[i],
            state.tensors[i],
            0.5 * substep_evolution_dt,
            krylov_tol=sim_params.krylov_tol,
        )
        tensor_shape = state.tensors[i].shape
        reshaped_tensor = state.tensors[i].reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        site_tensor, bond_tensor = np.linalg.qr(reshaped_tensor)
        state.tensors[i] = site_tensor.reshape((tensor_shape[0], tensor_shape[1], site_tensor.shape[1]))
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
        state.tensors[i + 1] = oe.contract(state.tensors[i + 1], (0, 3, 2), bond_tensor, (1, 3), (0, 1, 2))

    last = num_sites - 1
    state.tensors[last] = update_site(
        left_blocks[last],
        right_blocks[last],
        operator.tensors[last],
        state.tensors[last],
        substep_evolution_dt,
        krylov_tol=sim_params.krylov_tol,
    )

    substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, step_scale)

    # Right-to-left sweep: Update sites 1 to L-1.
    for i in reversed(range(1, num_sites)):
        state.tensors[i] = state.tensors[i].transpose((0, 2, 1))
        tensor_shape = state.tensors[i].shape
        reshaped_tensor = state.tensors[i].reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        site_tensor, bond_tensor = np.linalg.qr(reshaped_tensor)
        state.tensors[i] = site_tensor.reshape((tensor_shape[0], tensor_shape[1], site_tensor.shape[1])).transpose((
            0,
            2,
            1,
        ))
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
        state.tensors[i - 1] = oe.contract(state.tensors[i - 1], (0, 1, 3), bond_tensor, (3, 2), (0, 1, 2))
        state.tensors[i - 1] = update_site(
            left_blocks[i - 1],
            right_blocks[i - 1],
            operator.tensors[i - 1],
            state.tensors[i - 1],
            0.5 * substep_evolution_dt,
            krylov_tol=sim_params.krylov_tol,
        )


def _two_site_tdvp_sweep(
    state: MPS,
    operator: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    *,
    step_scale: float = 1.0,
    sweep_plan: list[float] | None = None,
) -> None:
    num_sites = operator.length
    plan = sweep_plan if sweep_plan is not None else [step_scale]

    right_blocks = initialize_right_environments(state, operator)
    left_blocks = [np.empty((0, 0, 0), dtype=np.complex128) for _ in range(num_sites)]
    left_virtual_dim = state.tensors[0].shape[1]
    mpo_left_dim = operator.tensors[0].shape[2]
    left_identity = np.zeros((left_virtual_dim, mpo_left_dim, left_virtual_dim), dtype=right_blocks[0].dtype)
    for i in range(left_virtual_dim):
        for a in range(mpo_left_dim):
            left_identity[i, a, i] = 1
    left_blocks[0] = left_identity

    for plan_step_scale in plan:
        substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, plan_step_scale)

        for i in range(num_sites - 2):
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
            state.tensors[i], state.tensors[i + 1] = _split_two_site_tdvp(
                merged_tensor,
                sim_params,
                [state.physical_dimensions[i], state.physical_dimensions[i + 1]],
                "right",
                dynamic=False,
            )
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

        i = num_sites - 2
        merged_tensor = merge_two_site(state.tensors[i], state.tensors[i + 1])
        merged_mpo = merge_mpo_tensors(operator.tensors[i], operator.tensors[i + 1])
        merged_tensor = update_site(
            left_blocks[i],
            right_blocks[i + 1],
            merged_mpo,
            merged_tensor,
            substep_evolution_dt,
            krylov_tol=sim_params.krylov_tol,
        )
        state.tensors[i], state.tensors[i + 1] = _split_two_site_tdvp(
            merged_tensor,
            sim_params,
            [state.physical_dimensions[i], state.physical_dimensions[i + 1]],
            "left",
            dynamic=False,
        )

        right_blocks[i] = update_right_environment(
            state.tensors[i + 1], state.tensors[i + 1], operator.tensors[i + 1], right_blocks[i + 1]
        )

        substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, plan_step_scale)

        # For ``num_sites == 2`` the RTL loop is empty; the LTR final-bond update above
        # already applies the full substep time once. Do not duplicate it here.
        for i in reversed(range(num_sites - 2)):
            state.tensors[i + 1] = update_site(
                left_blocks[i + 1],
                right_blocks[i + 1],
                operator.tensors[i + 1],
                state.tensors[i + 1],
                -0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
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
            state.tensors[i], state.tensors[i + 1] = _split_two_site_tdvp(
                merged_tensor,
                sim_params,
                [state.physical_dimensions[i], state.physical_dimensions[i + 1]],
                "left",
                dynamic=False,
            )
            right_blocks[i] = update_right_environment(
                state.tensors[i + 1], state.tensors[i + 1], operator.tensors[i + 1], right_blocks[i + 1]
            )


def _dynamic_tdvp_sweep(
    state: MPS,
    operator: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    support_bonds: frozenset[int] | None = None,
    *,
    step_scale: float = 1.0,
    sweep_plan: list[float] | None = None,
) -> None:
    if sweep_plan is not None:
        for plan_step_scale in sweep_plan:
            _dynamic_tdvp_sweep(
                state,
                operator,
                sim_params,
                support_bonds,
                step_scale=plan_step_scale,
            )
        return

    _enforce_global_bond_cap(state, sim_params)

    num_sites = operator.length
    use_lock = support_bonds is None
    merged_peak: dict[int, float] = {}
    last_second: dict[int, float] = {}

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
    lock_final_site = False
    for i in range(num_sites):
        bond_dim = state.tensors[i].shape[2]
        if _bond_dim_at_or_above_cap(bond_dim, sim_params.max_bond_dim) or (use_lock and lock_final_site):
            state.tensors[i] = update_site(
                left_blocks[i],
                right_blocks[i],
                operator.tensors[i],
                state.tensors[i],
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            if i != num_sites - 1:
                if support_bonds is not None and i in support_bonds:
                    site_tensor, bond_tensor = _canon_support_site(state.tensors[i], sim_params, ltr=True)
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
            if use_lock and i == num_sites - 2:
                lock_final_site = True
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
            if support_bonds is not None:
                state.tensors[i], state.tensors[i + 1] = _split_support_two_site(
                    merged_tensor,
                    sim_params,
                    phys_dims,
                    "right",
                    bond_index=i,
                    support_bonds=support_bonds,
                )
                if i in support_bonds:
                    _after_support_split(state, i, merged_tensor, phys_dims, sim_params, merged_peak, last_second)
            else:
                state.tensors[i], state.tensors[i + 1] = _split_two_site_tdvp(
                    merged_tensor,
                    sim_params,
                    phys_dims,
                    "right",
                    dynamic=True,
                )
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
            if support_bonds is not None:
                state.tensors[i], state.tensors[i + 1] = _split_support_two_site(
                    merged_tensor,
                    sim_params,
                    phys_dims,
                    "right",
                    bond_index=i,
                    support_bonds=support_bonds,
                )
                if i in support_bonds:
                    _after_support_split(state, i, merged_tensor, phys_dims, sim_params, merged_peak, last_second)
            else:
                state.tensors[i], state.tensors[i + 1] = _split_two_site_tdvp(
                    merged_tensor,
                    sim_params,
                    phys_dims,
                    "right",
                    dynamic=True,
                )
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
    lock_final_site = False
    for i in reversed(range(num_sites)):
        bond_dim = state.tensors[i].shape[1]
        if _bond_dim_at_or_above_cap(bond_dim, sim_params.max_bond_dim) or (use_lock and lock_final_site):
            state.tensors[i] = update_site(
                left_blocks[i],
                right_blocks[i],
                operator.tensors[i],
                state.tensors[i],
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            if i != 0:
                if support_bonds is not None and (i - 1) in support_bonds:
                    site_tensor, bond_tensor = _canon_support_site(state.tensors[i], sim_params, ltr=False)
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
                if use_lock and i == 1:
                    lock_final_site = True
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
            bond_index = i - 1
            if support_bonds is not None:
                state.tensors[i - 1], state.tensors[i] = _split_support_two_site(
                    merged_tensor,
                    sim_params,
                    phys_dims,
                    "left",
                    bond_index=bond_index,
                    support_bonds=support_bonds,
                )
                if bond_index in support_bonds:
                    _after_support_split(
                        state, bond_index, merged_tensor, phys_dims, sim_params, merged_peak, last_second
                    )
            else:
                state.tensors[i - 1], state.tensors[i] = _split_two_site_tdvp(
                    merged_tensor,
                    sim_params,
                    phys_dims,
                    "left",
                    dynamic=True,
                )
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

    if support_bonds is not None:
        _after_support_substep(state, support_bonds, sim_params, merged_peak, last_second)
    if sim_params.max_bond_dim is not None:
        state.normalize()


# --- Public entry point ---


def tdvp(
    state: MPS,
    operator: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    mode: Mode = "dynamic",
    *,
    support_bonds: frozenset[int] | None = None,
) -> None:
    """Evolve an MPS under an MPO operator via TDVP.

    The operator may represent a Hamiltonian (analog simulation) or a gate
    generator MPO (digital simulation).

    Args:
        state: MPS state updated in place.
        operator: MPO defining the local generator at each site.
        sim_params: Simulation parameters including ``dt`` or gate time,
            ``tdvp_sweeps``, truncation, and Krylov settings.
        mode: TDVP integrator variant. ``"dynamic"`` (default) adaptively
            chooses single- or two-site updates per bond; ``"1site"`` and
            ``"2site"`` force 1TDVP or 2TDVP respectively.
        support_bonds: Optional window-local bond indices for long-range digital
            gate support during dynamic TDVP. Requires ``StrongSimParams`` or
            ``WeakSimParams``; analog callers should omit this.

    Raises:
        ValueError: If ``state`` and ``operator`` lengths mismatch, if
            ``mode="2site"`` with fewer than two sites, if ``support_bonds``
            is set with a non-dynamic mode, or if ``support_bonds`` is set with
            ``AnalogSimParams``.
    """
    if operator.length != state.length:
        msg = "MPS and operator must have the same number of sites."
        raise ValueError(msg)
    if mode == "2site" and operator.length < 2:
        msg = "Operator is too short for a two-site update (2TDVP)."
        raise ValueError(msg)
    if support_bonds is not None and mode != "dynamic":
        msg = "support_bonds is only supported with mode='dynamic'."
        raise ValueError(msg)
    if support_bonds is not None and isinstance(sim_params, AnalogSimParams):
        msg = "support_bonds is not supported with AnalogSimParams."
        raise ValueError(msg)

    if mode == "dynamic" and operator.length == 1:
        mode = "1site"

    if mode == "1site":
        _run_sweeps(_single_site_tdvp_sweep, state, operator, sim_params)
    elif mode == "2site":
        _run_sweeps(_two_site_tdvp_sweep, state, operator, sim_params)
    elif support_bonds:
        _run_sweeps(_dynamic_tdvp_sweep, state, operator, sim_params, support_bonds)
    else:
        _run_sweeps(_dynamic_tdvp_sweep, state, operator, sim_params)
