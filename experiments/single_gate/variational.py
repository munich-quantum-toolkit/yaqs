# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Variational MPO compression for the main-text single-gate benchmark.

Multi-start, best-retained fitting of a χ-capped MPS to the uncapped MPO-applied
target. Local two-site updates project the target onto the approximant's virtual
spaces so external bond dimensions always match; dimension errors raise.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from gate_runtime import SVD_THRESHOLD, TRUNC_MODE, _params, normalized_state_fidelity

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.methods.decompositions import split_two_site

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mps import MPS as MPSType


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
    best_initializer: str = ""
    half_sweep_residuals: list[float] = field(default_factory=list)


def _apply_gate_mpo(state: MPSType, gate, *, chi: int | None, compress: bool) -> None:
    params = _params(chi or 64, gate_mode="mpo", tdvp_sweeps=1) if compress else None
    mpo = MPO.from_gate(gate, state.length)
    if compress:
        assert params is not None
        mpo.multiply(state, sim_params=params, compress=True)
    else:
        mpo.multiply(state, compress=False)


def tt_svd_from_vec(vec: np.ndarray, length: int, chi_max: int) -> MPS:
    """TT-SVD matching ``MPS.to_vec`` bit ordering (site ``i`` ↔ bit ``i``)."""
    psi = np.asarray(vec, dtype=np.complex128).reshape([2] * length)
    psi = np.transpose(psi, list(reversed(range(length))))
    tensors: list[np.ndarray] = []
    chi_l = 1
    rest: np.ndarray = psi
    for site in range(length - 1):
        rest = rest.reshape(chi_l * 2, -1)
        u, s, vh = np.linalg.svd(rest, full_matrices=False)
        keep = min(chi_max, int(s.shape[0]))
        u, s, vh = u[:, :keep], s[:keep], vh[:keep, :]
        core = u.reshape(chi_l, 2, keep).transpose(1, 0, 2)
        tensors.append(np.ascontiguousarray(core))
        chi_l = keep
        n_rem = length - site - 1
        rest = (np.diag(s) @ vh).reshape([chi_l] + [2] * n_rem)
    tensors.append(np.ascontiguousarray(rest.reshape(chi_l, 2, 1).transpose(1, 0, 2)))
    mps = MPS(length, tensors=tensors)
    mps.normalize(form="B", decomposition="SVD")
    return mps


def _normalize_mps(state: MPSType) -> None:
    norm = float(np.linalg.norm(state.to_vec()))
    if norm <= 0.0:
        msg = "Cannot normalize a zero MPS"
        raise ValueError(msg)
    state.tensors[0] = state.tensors[0] / norm


def _phase_align_to_target(target: MPSType, approx: MPSType) -> None:
    ov = target.scalar_product(approx)
    if abs(ov) <= 0.0:
        return
    approx.tensors[0] = approx.tensors[0] * (abs(ov) / ov)


def _euclidean_residual(target: MPSType, approx: MPSType) -> float:
    tt = float(np.real(target.scalar_product(target)))
    aa = float(np.real(approx.scalar_product(approx)))
    ov = float(np.real(target.scalar_product(approx)))
    return max(0.0, tt + aa - 2.0 * ov)


def _left_env(target: MPSType, approx: MPSType, bond: int) -> np.ndarray:
    """Overlap environment ``E[χa, χt]`` over sites ``0 … bond-1``."""
    env = np.ones((1, 1), dtype=np.complex128)
    for site in range(bond):
        a = approx.tensors[site]
        t = target.tensors[site]
        # E[al,tl] × conj(T[p,tl,tr]) × A[p,al,ar] → E'[ar,tr]
        env = np.einsum("ij,pjk,pil->lk", env, np.conj(t), a, optimize=True)
    return env


def _right_env(target: MPSType, approx: MPSType, bond: int) -> np.ndarray:
    """Overlap environment ``E[χt, χa]`` over sites ``bond+2 … L-1``."""
    env = np.ones((1, 1), dtype=np.complex128)
    for site in range(approx.length - 1, bond + 1, -1):
        a = approx.tensors[site]
        t = target.tensors[site]
        # E[tr,ar] × conj(T[p,tl,tr]) × A[p,al,ar] → E'[tl,al]
        env = np.einsum("ij,pki,plj->kl", env, np.conj(t), a, optimize=True)
    return env


def _bond_update_project(
    target: MPSType,
    approx: MPSType,
    *,
    bond: int,
    chi: int,
) -> tuple[MPSType, float]:
    """Project the target two-site block into approx's virtual spaces and truncate."""
    trial = copy.deepcopy(approx)
    trial.set_canonical_form(bond, decomposition="QR")
    target_c = copy.deepcopy(target)
    target_c.set_canonical_form(bond, decomposition="QR")

    left = _left_env(target_c, trial, bond)  # (χaL, χtL)
    right = _right_env(target_c, trial, bond)  # (χtR, χaR)
    t_l = target_c.tensors[bond]
    t_r = target_c.tensors[bond + 1]

    # θ[p,q,χaL,χaR] = left[χaL,χtL] t_l[p,χtL,g] t_r[q,g,χtR] right[χtR,χaR]
    theta = np.einsum("al,plg,qgr,rb->pqab", left, t_l, t_r, right, optimize=True)
    d0, d1, chi_l, chi_r = theta.shape
    expect_l = trial.tensors[bond].shape[1]
    expect_r = trial.tensors[bond + 1].shape[2]
    if chi_l != expect_l or chi_r != expect_r:
        msg = (
            f"Projected dims ({chi_l},{chi_r}) != approx virtual dims "
            f"({expect_l},{expect_r}) at bond {bond}"
        )
        raise ValueError(msg)

    merged = np.ascontiguousarray(theta.reshape(d0 * d1, chi_l, chi_r))
    new_l, new_r = split_two_site(
        merged,
        [d0, d1],
        svd_distribution="right",
        trunc_mode=TRUNC_MODE,
        threshold=SVD_THRESHOLD,
        max_bond_dim=chi,
    )
    if new_l.shape[1] != chi_l or new_r.shape[2] != chi_r:
        msg = (
            f"Truncation changed external dims at bond {bond}: "
            f"left {chi_l}->{new_l.shape[1]}, right {chi_r}->{new_r.shape[2]}"
        )
        raise ValueError(msg)
    if bond > 0 and new_l.shape[1] != trial.tensors[bond - 1].shape[2]:
        msg = f"Left neighbor bond mismatch after update at bond {bond}"
        raise ValueError(msg)
    if bond + 2 < trial.length and new_r.shape[2] != trial.tensors[bond + 2].shape[1]:
        msg = f"Right neighbor bond mismatch after update at bond {bond}"
        raise ValueError(msg)

    trial.tensors[bond] = new_l
    trial.tensors[bond + 1] = new_r
    trial.set_center(bond + 1)
    _normalize_mps(trial)
    _phase_align_to_target(target, trial)
    return trial, _euclidean_residual(target, trial)


def _sweep(
    target: MPSType,
    approx: MPSType,
    *,
    chi: int,
    direction: str,
    residual_tol: float,
) -> tuple[MPSType, float, list[float]]:
    bonds = list(range(approx.length - 1))
    if direction == "rl":
        bonds.reverse()
    state = approx
    obj = _euclidean_residual(target, state)
    half: list[float] = [obj]
    for bond in bonds:
        new_state, new_obj = _bond_update_project(target, state, bond=bond, chi=chi)
        if new_obj > obj + residual_tol:
            # Reject ascent; keep prior state. Do not swallow dimension errors.
            half.append(obj)
            continue
        state = new_state
        obj = new_obj
        half.append(obj)
    return state, obj, half


def variational_fit(
    target: MPSType,
    initial: MPSType,
    *,
    chi: int,
    max_sweeps: int = 8,
    rel_tol: float = 1e-10,
    abs_floor: float = 1e-14,
    residual_tol: float = 1e-12,
) -> VariationalResult:
    """Fit ``initial`` to ``target``; residual must be non-increasing within tolerance."""
    approx = copy.deepcopy(initial)
    _normalize_mps(approx)
    _phase_align_to_target(target, approx)
    obj0 = _euclidean_residual(target, approx)
    trace = [obj0]
    half_trace = [obj0]
    if obj0 <= abs_floor:
        return VariationalResult(
            state=approx,
            runtime=0.0,
            objective_initial=obj0,
            objective_final=obj0,
            sweeps=0,
            converged=True,
            objective_trace=trace,
            half_sweep_residuals=half_trace,
        )

    best_state = copy.deepcopy(approx)
    best_obj = obj0
    prev = obj0
    sweeps_done = 0
    converged = False

    for sweep in range(max_sweeps):
        approx, obj, ht = _sweep(target, approx, chi=chi, direction="lr", residual_tol=residual_tol)
        half_trace.extend(ht[1:])
        approx, obj, ht = _sweep(target, approx, chi=chi, direction="rl", residual_tol=residual_tol)
        half_trace.extend(ht[1:])
        # Relative slack avoids false positives when residuals are O(1e-6) and
        # floating-point noise exceeds a tiny absolute residual_tol.
        allowed = max(residual_tol, 1e-9 * max(prev, abs_floor))
        if obj > prev + allowed:
            msg = f"residual increased at full sweep {sweep + 1}: {prev:.3e} -> {obj:.3e}"
            raise RuntimeError(msg)
        trace.append(obj)
        sweeps_done = sweep + 1
        if obj < best_obj - residual_tol:
            best_obj = obj
            best_state = copy.deepcopy(approx)
        if prev - obj <= max(rel_tol * max(prev, abs_floor), abs_floor):
            converged = True
            break
        prev = obj

    if best_obj > obj0 + residual_tol:
        msg = f"best residual {best_obj:.3e} worse than initializer {obj0:.3e}"
        raise RuntimeError(msg)

    return VariationalResult(
        state=best_state,
        runtime=0.0,
        objective_initial=obj0,
        objective_final=best_obj,
        sweeps=sweeps_done,
        converged=converged or best_obj <= abs_floor,
        objective_trace=trace,
        half_sweep_residuals=half_trace,
    )


def apply_variational_mpo_gate(
    initial_mps: MPSType,
    node,
    *,
    chi: int,
    max_sweeps: int = 8,
    rel_tol: float = 1e-10,
    abs_floor: float = 1e-14,
    residual_tol: float = 1e-12,
    require_exact_when_chi_ge: int | None = 16,
) -> VariationalResult:
    """Multi-start variational MPO application (input / zip-up / default=zip-up).

    Dense TT-SVD is not a production initializer. TDVP init is available only via
    :func:`apply_variational_mpo_gate_tdvp_diagnostic`.

    Args:
        require_exact_when_chi_ge: If set, assert machine-precision agreement with the
            uncapped MPO target whenever ``chi`` is at least this value. Use ``None`` for
            circuit trajectories where the uncapped bond may exceed ``chi``.
    """
    from mqt.yaqs.digital.digital_tjm import convert_dag_to_tensor_algorithm

    gate = convert_dag_to_tensor_algorithm(node)[0]
    t0 = time.perf_counter()
    target = copy.deepcopy(initial_mps)
    _apply_gate_mpo(target, gate, chi=None, compress=False)
    _normalize_mps(target)

    zip_init = copy.deepcopy(initial_mps)
    _apply_gate_mpo(zip_init, gate, chi=chi, compress=True)

    # Scalable production initializers. "default" historically meant zip-up.
    starts = (
        ("input", copy.deepcopy(initial_mps)),
        ("zipup", zip_init),
        ("default", copy.deepcopy(zip_init)),
    )
    seen_vecs: list[np.ndarray] = []
    best: VariationalResult | None = None
    for name, cand in starts:
        v = cand.to_vec()
        if any(np.allclose(v, prev, atol=1e-14) for prev in seen_vecs):
            continue
        seen_vecs.append(v)
        fitted = variational_fit(
            target,
            cand,
            chi=chi,
            max_sweeps=max_sweeps,
            rel_tol=rel_tol,
            abs_floor=abs_floor,
            residual_tol=residual_tol,
        )
        fitted.best_initializer = name
        if best is None or fitted.objective_final < best.objective_final - 1e-15:
            best = fitted

    assert best is not None
    inf = normalized_state_fidelity(target.to_vec(), best.state.to_vec())["infidelity_normalized"]
    if require_exact_when_chi_ge is not None and chi >= require_exact_when_chi_ge and inf > 1e-10:
        msg = f"χ={chi} variational result not exact vs uncapped target: infidelity={inf:.3e}"
        raise RuntimeError(msg)
    best.runtime = time.perf_counter() - t0
    best.failure_reason = f"best_init={best.best_initializer}"
    return best


def apply_variational_mpo_gate_tdvp_diagnostic(
    initial_mps: MPSType,
    node,
    *,
    chi: int,
    tdvp_state: MPSType,
    max_sweeps: int = 8,
) -> VariationalResult:
    """TDVP-initialized fit for monotonicity diagnostics only (not publication)."""
    from mqt.yaqs.digital.digital_tjm import convert_dag_to_tensor_algorithm

    gate = convert_dag_to_tensor_algorithm(node)[0]
    target = copy.deepcopy(initial_mps)
    _apply_gate_mpo(target, gate, chi=None, compress=False)
    _normalize_mps(target)
    result = variational_fit(target, tdvp_state, chi=chi, max_sweeps=max_sweeps)
    result.best_initializer = "tdvp_diagnostic"
    return result
