# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""1TDVP, 2TDVP, and dynamic TDVP sweep integrators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe

from ..decompositions import left_qr, merge_two_site, right_qr
from .primitives import (
    initialize_right_environments,
    merge_mpo_tensors,
    update_bond,
    update_left_environment,
    update_right_environment,
    update_site,
)
from .sweep_utils import (
    _enforce_global_bond_cap,
    _prepare_substep_dt,
    _renorm_if_digital,
    _resize_bond,
    _split_two_site_tdvp,
    _sync_fixed_chi_bond,
)

if TYPE_CHECKING:
    from ...data_structures.mpo import MPO
    from ...data_structures.mps import MPS
    from ...data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams


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

    substep_evolution_dt = _prepare_substep_dt(sim_params, step_scale)

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

    substep_evolution_dt = _prepare_substep_dt(sim_params, step_scale)

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
    renorm_after: bool = True,
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
        substep_evolution_dt = _prepare_substep_dt(sim_params, plan_step_scale)

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

        substep_evolution_dt = _prepare_substep_dt(sim_params, plan_step_scale)

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

        if renorm_after:
            _renorm_if_digital(state, sim_params)


def _dynamic_tdvp_sweep(
    state: MPS,
    operator: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    *,
    step_scale: float = 1.0,
    sweep_plan: list[float] | None = None,
    renorm_after: bool = True,
) -> None:
    if sweep_plan is not None:
        for plan_step_scale in sweep_plan:
            _dynamic_tdvp_sweep(
                state,
                operator,
                sim_params,
                step_scale=plan_step_scale,
                renorm_after=renorm_after,
            )
        return

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

    substep_evolution_dt = _prepare_substep_dt(sim_params, step_scale)

    # ----- LEFT-TO-RIGHT DYNAMIC SWEEP -----
    for i in range(num_sites):
        bond_dim = state.tensors[i].shape[2]
        cap = sim_params.max_bond_dim
        if cap is not None and bond_dim >= cap:
            state.tensors[i] = update_site(
                left_blocks[i],
                right_blocks[i],
                operator.tensors[i],
                state.tensors[i],
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            if i != num_sites - 1:
                site_tensor, bond_tensor = right_qr(state.tensors[i])
                if cap is not None and int(site_tensor.shape[2]) > cap:
                    site_tensor = site_tensor[:, :, :cap]
                    bond_tensor = bond_tensor[:cap, :]
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
                    _sync_fixed_chi_bond(state, i, sim_params)
                    bond_tensor = _resize_bond(
                        bond_tensor,
                        lead=int(state.tensors[i].shape[2]),
                        trail=int(state.tensors[i + 1].shape[1]),
                    )
                state.tensors[i + 1] = oe.contract(state.tensors[i + 1], (0, 3, 2), bond_tensor, (1, 3), (0, 1, 2))
                _sync_fixed_chi_bond(state, i, sim_params)
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
    substep_evolution_dt = _prepare_substep_dt(sim_params, step_scale)

    # ----- RIGHT-TO-LEFT DYNAMIC SWEEP -----
    for i in reversed(range(num_sites)):
        bond_dim = state.tensors[i].shape[1]
        cap = sim_params.max_bond_dim
        if cap is not None and bond_dim >= cap:
            state.tensors[i] = update_site(
                left_blocks[i],
                right_blocks[i],
                operator.tensors[i],
                state.tensors[i],
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            if i != 0:
                site_tensor, bond_tensor = left_qr(state.tensors[i])
                if cap is not None and int(site_tensor.shape[1]) > cap:
                    site_tensor = site_tensor[:, :cap, :]
                    bond_tensor = bond_tensor[:cap, :]
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
                    _sync_fixed_chi_bond(state, i - 1, sim_params)
                    bond_tensor = _resize_bond(
                        bond_tensor,
                        lead=int(state.tensors[i - 1].shape[2]),
                        trail=int(state.tensors[i].shape[1]),
                    )
                state.tensors[i - 1] = oe.contract(state.tensors[i - 1], (0, 1, 3), bond_tensor, (3, 2), (0, 1, 2))
                _sync_fixed_chi_bond(state, i - 1, sim_params)
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

    if renorm_after:
        _renorm_if_digital(state, sim_params)
