# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Variational MPO compression for the main-text benchmark."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from gate_runtime import SVD_THRESHOLD, TRUNC_MODE, _params

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.methods.decompositions import merge_two_site, split_two_site

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mps import MPS


@dataclass
class VariationalResult:
    """Outcome of variational MPO compression."""

    state: MPS
    runtime: float
    objective_initial: float
    objective_final: float
    sweeps: int
    converged: bool
    failed: bool = False
    failure_reason: str = ""
    objective_trace: list[float] = field(default_factory=list)


def _compression_objective(target: MPS, approx: MPS) -> float:
    inner = target.scalar_product(approx)
    tt = float(np.real(target.scalar_product(target)))
    aa = float(np.real(approx.scalar_product(approx)))
    return max(0.0, tt + aa - 2.0 * float(np.real(inner)))


def _apply_gate_mpo(state: MPS, gate, *, chi: int | None, compress: bool) -> None:
    params = _params(chi or 64, gate_mode="mpo", tdvp_sweeps=1) if compress else None
    mpo = MPO.from_gate(gate, state.length)
    if compress:
        assert params is not None
        mpo.multiply(state, sim_params=params, compress=True)
    else:
        mpo.multiply(state, compress=False)


def _bond_update_from_target(
    target: MPS,
    approx: MPS,
    *,
    bond: int,
    chi: int,
) -> tuple[MPS, float]:
    obj_before = _compression_objective(target, approx)
    trial = copy.deepcopy(approx)
    trial.set_canonical_form(bond, decomposition="SVD")
    target_local = copy.deepcopy(target)
    target_local.set_canonical_form(bond, decomposition="SVD")
    merged = merge_two_site(target_local.tensors[bond], target_local.tensors[bond + 1])
    dims = [trial.tensors[bond].shape[0], trial.tensors[bond + 1].shape[0]]
    new_left, new_right = split_two_site(
        merged,
        dims,
        svd_distribution="right",
        trunc_mode=TRUNC_MODE,
        threshold=SVD_THRESHOLD,
        max_bond_dim=chi,
    )
    trial.tensors[bond] = new_left
    trial.tensors[bond + 1] = new_right
    trial.set_center(bond + 1)
    try:
        obj_after = _compression_objective(target, trial)
    except ValueError:
        return approx, obj_before
    if obj_after <= obj_before + 1e-12:
        return trial, obj_after
    return approx, obj_before


def _sweep_variational(
    target: MPS,
    approx: MPS,
    *,
    chi: int,
    direction: str,
) -> tuple[MPS, float]:
    bonds = list(range(approx.length - 1))
    if direction == "rl":
        bonds.reverse()
    obj = _compression_objective(target, approx)
    state = approx
    for bond in bonds:
        state, obj = _bond_update_from_target(target, state, bond=bond, chi=chi)
    return state, obj


def variational_compress(
    target: MPS,
    initial: MPS,
    *,
    chi: int,
    max_sweeps: int = 12,
    rel_tol: float = 1e-10,
    abs_floor: float = 1e-14,
) -> VariationalResult:
    """Fit ``initial`` to ``target`` at bond dimension ``chi`` via alternating sweeps."""
    approx = copy.deepcopy(initial)
    obj0 = _compression_objective(target, approx)
    trace = [obj0]
    prev = obj0
    sweeps_done = 0
    converged = False

    if obj0 <= abs_floor:
        return VariationalResult(
            state=approx,
            runtime=0.0,
            objective_initial=obj0,
            objective_final=obj0,
            sweeps=0,
            converged=True,
            objective_trace=trace,
        )

    for sweep in range(max_sweeps):
        approx, obj = _sweep_variational(target, approx, chi=chi, direction="lr")
        approx, obj = _sweep_variational(target, approx, chi=chi, direction="rl")
        trace.append(obj)
        sweeps_done = sweep + 1
        if obj > prev + 1e-12:
            return VariationalResult(
                state=copy.deepcopy(initial),
                runtime=0.0,
                objective_initial=obj0,
                objective_final=obj,
                sweeps=sweeps_done,
                converged=False,
                failed=True,
                failure_reason=f"objective increased at sweep {sweeps_done}: {prev:.3e} -> {obj:.3e}",
                objective_trace=trace,
            )
        if prev - obj <= max(rel_tol * max(prev, abs_floor), abs_floor):
            converged = True
            break
        prev = obj

    obj_final = _compression_objective(target, approx)
    if obj_final > obj0 + 1e-12:
        return VariationalResult(
            state=copy.deepcopy(initial),
            runtime=0.0,
            objective_initial=obj0,
            objective_final=obj_final,
            sweeps=sweeps_done,
            converged=False,
            failed=True,
            failure_reason="final objective worse than zip-up initialization",
            objective_trace=trace,
        )
    return VariationalResult(
        state=approx,
        runtime=0.0,
        objective_initial=obj0,
        objective_final=obj_final,
        sweeps=sweeps_done,
        converged=converged or obj_final <= abs_floor,
        objective_trace=trace,
    )


def apply_variational_mpo_gate(
    initial_mps: MPS,
    node,
    *,
    chi: int,
    max_sweeps: int = 12,
    rel_tol: float = 1e-10,
    abs_floor: float = 1e-14,
) -> VariationalResult:
    """Apply a long-range gate with zip-up init and variational compression."""
    from mqt.yaqs.digital.digital_tjm import convert_dag_to_tensor_algorithm

    gate = convert_dag_to_tensor_algorithm(node)[0]
    t0 = time.perf_counter()
    target = copy.deepcopy(initial_mps)
    _apply_gate_mpo(target, gate, chi=None, compress=False)
    zip_init = copy.deepcopy(initial_mps)
    _apply_gate_mpo(zip_init, gate, chi=chi, compress=True)
    result = variational_compress(
        target,
        zip_init,
        chi=chi,
        max_sweeps=max_sweeps,
        rel_tol=rel_tol,
        abs_floor=abs_floor,
    )
    result.runtime = time.perf_counter() - t0
    return result
