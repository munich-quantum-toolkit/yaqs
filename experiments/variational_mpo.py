# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Variational MPS compression of an MPO-applied gate endpoint.

This module supports the publication accuracy control.  It deliberately lives
outside the production simulator API: the implementation is validated as an
endpoint comparison method, but is not optimized as a timing baseline.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.methods.decompositions import left_qr, split_two_site
from mqt.yaqs.digital.utils.dag_utils import convert_dag_to_tensor_algorithm

if TYPE_CHECKING:
    from qiskit.dagcircuit import DAGOpNode

    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.core.data_structures.simulation_parameters import DigitalSimParams
    from mqt.yaqs.core.libraries.gate_library import BaseGate
    from mqt.yaqs.core.methods.decompositions import SvdDistribution, TruncMode


@dataclass
class VariationalMPOResult:
    """Result and convergence diagnostics for one variational endpoint fit."""

    state: MPS
    objective_initial: float
    objective_final: float
    sweeps: int
    converged: bool
    objective_trace: list[float] = field(default_factory=list)
    update_trace: list[float] = field(default_factory=list)
    best_initializer: str = ""
    initializer_objectives: dict[str, float] = field(default_factory=dict)
    initializer_final_objectives: dict[str, float] = field(default_factory=dict)
    initializer_converged: dict[str, bool] = field(default_factory=dict)
    initializer_runtimes_s: dict[str, float] = field(default_factory=dict)
    rejected_nonimproving_updates: int = 0
    runtime_s: float = 0.0
    target_max_bond: int = 1
    target_parameter_count: int = 0
    fidelity_to_target: float = 0.0


def bond_profile(state: MPS) -> list[int]:
    """Return the MPS bond profile, including unit boundary bonds."""
    if not state.tensors:
        return []
    return [int(state.tensors[0].shape[1]), *(int(tensor.shape[2]) for tensor in state.tensors)]


def _norm_squared(state: MPS) -> float:
    value = complex(state.scalar_product(state))
    scale = max(1.0, abs(value.real))
    if not np.isfinite(value.real) or not np.isfinite(value.imag) or abs(value.imag) > 1e-10 * scale:
        msg = f"Invalid MPS norm squared {value!r}."
        raise ValueError(msg)
    norm_squared = float(value.real)
    if norm_squared <= 0.0:
        msg = f"MPS norm squared must be positive, got {norm_squared}."
        raise ValueError(msg)
    return norm_squared


def _scale_state(state: MPS, factor: complex) -> None:
    """Scale a state without invalidating a tracked canonical center."""
    site = state.orthogonality_center
    if site is None:
        site = 0
    state.tensors[site] = np.asarray(state.tensors[site], dtype=np.complex128) * factor


def _normalize(state: MPS) -> None:
    norm = float(np.sqrt(_norm_squared(state)))
    _scale_state(state, 1.0 / norm)


def _phase_align(target: MPS, approximate: MPS) -> None:
    overlap = complex(target.scalar_product(approximate))
    if abs(overlap) > 0.0:
        _scale_state(approximate, abs(overlap) / overlap)


def _squared_residual(target: MPS, approximate: MPS) -> float:
    target_norm = _norm_squared(target)
    approximate_norm = _norm_squared(approximate)
    overlap = float(np.real(target.scalar_product(approximate)))
    residual = target_norm + approximate_norm - 2.0 * overlap
    if residual < -1e-10:
        msg = f"Negative squared residual beyond roundoff: {residual}."
        raise ValueError(msg)
    return max(0.0, float(residual))


def normalized_mps_fidelity(target: MPS, approximate: MPS) -> float:
    """Return phase-insensitive normalized fidelity between two MPS."""
    numerator = abs(complex(target.scalar_product(approximate))) ** 2
    denominator = _norm_squared(target) * _norm_squared(approximate)
    fidelity = float(numerator / denominator)
    if fidelity < -1e-12 or fidelity > 1.0 + 1e-12:
        msg = f"Normalized fidelity {fidelity} lies outside [0, 1] beyond roundoff."
        raise ValueError(msg)
    return min(1.0, max(0.0, fidelity))


def _left_environment(target: MPS, approximate: MPS, bond: int) -> np.ndarray:
    """Contract the overlap environment over sites left of ``bond``."""
    environment = np.ones((1, 1), dtype=np.complex128)
    for site in range(bond):
        a_tensor = approximate.tensors[site]
        t_tensor = target.tensors[site]
        environment = np.einsum("ij,pjk,pil->lk", environment, np.conj(t_tensor), a_tensor, optimize=True)
    return environment


def _right_environment(target: MPS, approximate: MPS, bond: int) -> np.ndarray:
    """Contract the overlap environment over sites right of ``bond + 1``."""
    environment = np.ones((1, 1), dtype=np.complex128)
    for site in range(approximate.length - 1, bond + 1, -1):
        a_tensor = approximate.tensors[site]
        t_tensor = target.tensors[site]
        environment = np.einsum("ij,pki,plj->kl", environment, np.conj(t_tensor), a_tensor, optimize=True)
    return environment


def _projected_target_tensor(target: MPS, approximate: MPS, bond: int) -> np.ndarray:
    """Project the target ket into the approximation's exterior Schmidt bases."""
    left = _left_environment(target, approximate, bond)
    right = _right_environment(target, approximate, bond)
    target_left = target.tensors[bond]
    target_right = target.tensors[bond + 1]
    return np.einsum(
        "al,plg,qgr,rb->pqab",
        np.conj(left),
        target_left,
        target_right,
        np.conj(right),
        optimize=True,
    )


def _bond_update_reference(
    target: MPS,
    approximate: MPS,
    *,
    bond: int,
    compression_params: DigitalSimParams,
) -> tuple[MPS, float]:
    """Optimize one two-site block in the current external virtual spaces."""
    cap = compression_params.max_bond_dim
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        msg = f"A positive integer max_bond_dim is required, got {cap!r}."
        raise ValueError(msg)

    trial = copy.deepcopy(approximate)
    trial.set_canonical_form(bond, decomposition="QR")
    canonical_target = copy.deepcopy(target)
    canonical_target.set_canonical_form(bond, decomposition="QR")

    theta = _projected_target_tensor(canonical_target, trial, bond)

    d_left, d_right, chi_left, chi_right = theta.shape
    expected_left = trial.tensors[bond].shape[1]
    expected_right = trial.tensors[bond + 1].shape[2]
    if chi_left != expected_left or chi_right != expected_right:
        msg = (
            f"Projected external dimensions ({chi_left}, {chi_right}) do not match "
            f"the approximation ({expected_left}, {expected_right}) at bond {bond}."
        )
        raise ValueError(msg)

    merged = np.ascontiguousarray(theta.reshape(d_left * d_right, chi_left, chi_right))
    new_left, new_right = split_two_site(
        merged,
        [d_left, d_right],
        svd_distribution="right",
        trunc_mode=cast("TruncMode", compression_params.trunc_mode),
        threshold=float(compression_params.svd_threshold),
        max_bond_dim=cap,
        min_keep=1,
    )
    if new_left.shape[1] != chi_left or new_right.shape[2] != chi_right:
        msg = f"A variational update changed an external dimension at bond {bond}."
        raise ValueError(msg)
    if bond > 0 and new_left.shape[1] != trial.tensors[bond - 1].shape[2]:
        msg = f"Left-neighbor bond mismatch after update at bond {bond}."
        raise ValueError(msg)
    if bond + 2 < trial.length and new_right.shape[2] != trial.tensors[bond + 2].shape[1]:
        msg = f"Right-neighbor bond mismatch after update at bond {bond}."
        raise ValueError(msg)

    trial.tensors[bond] = new_left
    trial.tensors[bond + 1] = new_right
    trial.set_center(bond + 1)
    _normalize(trial)
    _phase_align(target, trial)
    return trial, _squared_residual(target, trial)


def _half_sweep_reference(
    target: MPS,
    approximate: MPS,
    *,
    compression_params: DigitalSimParams,
    direction: str,
    acceptance_abs_tol: float,
    acceptance_rel_tol: float,
) -> tuple[MPS, float, list[float], int]:
    bonds = list(range(approximate.length - 1))
    if direction == "right_to_left":
        bonds.reverse()
    elif direction != "left_to_right":
        msg = f"Unknown sweep direction {direction!r}."
        raise ValueError(msg)

    state = approximate
    objective = _squared_residual(target, state)
    trace = [objective]
    rejected = 0
    for bond in bonds:
        candidate, candidate_objective = _bond_update_reference(
            target,
            state,
            bond=bond,
            compression_params=compression_params,
        )
        if not np.isfinite(candidate_objective):
            msg = f"Nonfinite variational objective at bond {bond}."
            raise ValueError(msg)
        improvement_floor = max(
            acceptance_abs_tol,
            acceptance_rel_tol * max(objective, acceptance_abs_tol),
        )
        if objective - candidate_objective > improvement_floor:
            state = candidate
            objective = candidate_objective
        else:
            rejected += 1
        trace.append(objective)
    return state, objective, trace, rejected


def _variational_fit_reference(
    target: MPS,
    initial: MPS,
    *,
    compression_params: DigitalSimParams,
    max_sweeps: int = 8,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-14,
    acceptance_rel_tol: float = 1e-12,
) -> VariationalMPOResult:
    """Fit a capped MPS to ``target`` with alternating two-site sweeps."""
    cap = compression_params.max_bond_dim
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        msg = f"A positive integer max_bond_dim is required, got {cap!r}."
        raise ValueError(msg)
    if max_sweeps < 1:
        msg = f"max_sweeps must be positive, got {max_sweeps}."
        raise ValueError(msg)

    start_time = time.perf_counter()
    approximate = copy.deepcopy(initial)
    _normalize(approximate)
    _phase_align(target, approximate)
    if max(bond_profile(approximate), default=1) > cap:
        msg = "The variational initializer exceeds max_bond_dim."
        raise ValueError(msg)

    initial_objective = _squared_residual(target, approximate)
    objective = initial_objective
    objective_trace = [objective]
    update_trace = [objective]
    rejected = 0
    converged = objective <= abs_tol
    sweeps_done = 0

    for sweep in range(max_sweeps):
        if converged:
            break
        previous_state = copy.deepcopy(approximate)
        previous_objective = objective
        approximate, objective, trace, rejected_lr = _half_sweep_reference(
            target,
            approximate,
            compression_params=compression_params,
            direction="left_to_right",
            acceptance_abs_tol=abs_tol,
            acceptance_rel_tol=acceptance_rel_tol,
        )
        update_trace.extend(trace[1:])
        approximate, objective, trace, rejected_rl = _half_sweep_reference(
            target,
            approximate,
            compression_params=compression_params,
            direction="right_to_left",
            acceptance_abs_tol=abs_tol,
            acceptance_rel_tol=acceptance_rel_tol,
        )
        update_trace.extend(trace[1:])
        rejected += rejected_lr + rejected_rl
        if objective > previous_objective:
            approximate = previous_state
            objective = previous_objective
            rejected += 1
        objective_trace.append(objective)
        sweeps_done = sweep + 1
        improvement = previous_objective - objective
        tolerance = max(abs_tol, rel_tol * max(previous_objective, abs_tol))
        converged = objective <= abs_tol or improvement <= tolerance

    if objective > initial_objective:
        msg = f"Final objective {objective:.6e} exceeds initializer {initial_objective:.6e}."
        raise RuntimeError(msg)
    if max(bond_profile(approximate), default=1) > cap:
        msg = "The variational result exceeds max_bond_dim."
        raise RuntimeError(msg)

    return VariationalMPOResult(
        state=approximate,
        objective_initial=initial_objective,
        objective_final=objective,
        sweeps=sweeps_done,
        converged=converged,
        objective_trace=objective_trace,
        update_trace=update_trace,
        rejected_nonimproving_updates=rejected,
        runtime_s=time.perf_counter() - start_time,
    )


def _left_environment_step(
    environment: np.ndarray,
    target_tensor: np.ndarray,
    approximate_tensor: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "ij,pjk,pil->lk",
        environment,
        np.conj(target_tensor),
        approximate_tensor,
        optimize=True,
    )


def _right_environment_step(
    environment: np.ndarray,
    target_tensor: np.ndarray,
    approximate_tensor: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "ij,pki,plj->kl",
        environment,
        np.conj(target_tensor),
        approximate_tensor,
        optimize=True,
    )


def _projected_target_from_environments(
    target: MPS,
    *,
    bond: int,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    return np.einsum(
        "al,plg,qgr,rb->pqab",
        np.conj(left),
        target.tensors[bond],
        target.tensors[bond + 1],
        np.conj(right),
        optimize=True,
    )


def _split_projected_target(
    theta: np.ndarray,
    *,
    compression_params: DigitalSimParams,
    direction: str,
    target_norm: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split, normalize, and phase-align one projected two-site target."""
    cap = compression_params.max_bond_dim
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        msg = f"A positive integer max_bond_dim is required, got {cap!r}."
        raise ValueError(msg)
    d_left, d_right, chi_left, chi_right = theta.shape
    distribution = "right" if direction == "left_to_right" else "left"
    merged = np.ascontiguousarray(theta.reshape(d_left * d_right, chi_left, chi_right))
    new_left, new_right = split_two_site(
        merged,
        [d_left, d_right],
        svd_distribution=cast("SvdDistribution", distribution),
        trunc_mode=cast("TruncMode", compression_params.trunc_mode),
        threshold=float(compression_params.svd_threshold),
        max_bond_dim=cap,
        min_keep=1,
    )
    block = np.einsum("pag,qgb->pqab", new_left, new_right, optimize=True)
    norm_squared = float(np.real(np.vdot(block, block)))
    if not np.isfinite(norm_squared) or norm_squared <= 0.0:
        msg = f"Invalid projected two-site norm squared {norm_squared}."
        raise ValueError(msg)
    overlap = complex(np.vdot(theta, block))
    factor = 1.0 / np.sqrt(norm_squared)
    if abs(overlap) > 0.0:
        factor *= abs(overlap) / overlap
    if direction == "left_to_right":
        new_right *= factor
    else:
        new_left *= factor
    normalized_overlap = abs(overlap) / np.sqrt(norm_squared)
    objective = target_norm + 1.0 - 2.0 * normalized_overlap
    if objective < -1e-10:
        msg = f"Negative local squared residual beyond roundoff: {objective}."
        raise ValueError(msg)
    return new_left, new_right, max(0.0, float(objective))


def _shift_center_left_one(state: MPS, center: int) -> None:
    """Move a known center left by one site without flipping the full chain."""
    if center <= 0 or state.orthogonality_center != center:
        msg = f"Cannot shift center {state.orthogonality_center} left from site {center}."
        raise ValueError(msg)
    right_tensor, bond_tensor = left_qr(state.tensors[center])
    state.tensors[center] = right_tensor
    state.tensors[center - 1] = np.einsum(
        "pla,ab->plb",
        state.tensors[center - 1],
        bond_tensor,
        optimize=True,
    )
    state.set_center(center - 1)


def _cached_half_sweep(
    target: MPS,
    approximate: MPS,
    *,
    compression_params: DigitalSimParams,
    direction: str,
    target_norm: float,
    acceptance_abs_tol: float,
    acceptance_rel_tol: float,
) -> tuple[MPS, float, list[float], int]:
    """Perform one variational half-sweep with cached overlap environments."""
    state = approximate
    length = state.length
    objective = _squared_residual(target, state)
    trace = [objective]
    rejected = 0

    if direction == "left_to_right":
        if state.orthogonality_center != 0:
            state.set_canonical_form(0, decomposition="QR")
        right_environments: list[np.ndarray | None] = [None] * (length + 1)
        right_environments[length] = np.ones((1, 1), dtype=np.complex128)
        for site in range(length - 1, -1, -1):
            following = right_environments[site + 1]
            assert following is not None
            right_environments[site] = _right_environment_step(
                following,
                target.tensors[site],
                state.tensors[site],
            )
        left = np.ones((1, 1), dtype=np.complex128)
        bonds = range(length - 1)
    elif direction == "right_to_left":
        if state.orthogonality_center != length - 1:
            state.set_canonical_form(length - 1, decomposition="QR")
        left_environments: list[np.ndarray | None] = [None] * (length + 1)
        left_environments[0] = np.ones((1, 1), dtype=np.complex128)
        for site in range(length):
            preceding = left_environments[site]
            assert preceding is not None
            left_environments[site + 1] = _left_environment_step(
                preceding,
                target.tensors[site],
                state.tensors[site],
            )
        right = np.ones((1, 1), dtype=np.complex128)
        bonds = range(length - 2, -1, -1)
    else:
        msg = f"Unknown sweep direction {direction!r}."
        raise ValueError(msg)

    for bond in bonds:
        if direction == "left_to_right":
            cached_right = right_environments[bond + 2]
            assert cached_right is not None
            theta = _projected_target_from_environments(
                target,
                bond=bond,
                left=left,
                right=cached_right,
            )
        else:
            cached_left = left_environments[bond]
            assert cached_left is not None
            theta = _projected_target_from_environments(
                target,
                bond=bond,
                left=cached_left,
                right=right,
            )

        expected_left = state.tensors[bond].shape[1]
        expected_right = state.tensors[bond + 1].shape[2]
        if theta.shape[2:] != (expected_left, expected_right):
            msg = (
                f"Projected external dimensions {theta.shape[2:]} do not match "
                f"the approximation ({expected_left}, {expected_right}) at bond {bond}."
            )
            raise ValueError(msg)
        new_left, new_right, candidate_objective = _split_projected_target(
            theta,
            compression_params=compression_params,
            direction=direction,
            target_norm=target_norm,
        )
        improvement_floor = max(
            acceptance_abs_tol,
            acceptance_rel_tol * max(objective, acceptance_abs_tol),
        )
        if objective - candidate_objective > improvement_floor:
            state.tensors[bond] = new_left
            state.tensors[bond + 1] = new_right
            state.set_center(bond + 1 if direction == "left_to_right" else bond)
            objective = candidate_objective
        else:
            rejected += 1
            if direction == "left_to_right":
                state.shift_orthogonality_center_right(bond, decomposition="QR")
            else:
                _shift_center_left_one(state, bond + 1)
        trace.append(objective)

        if direction == "left_to_right":
            left = _left_environment_step(
                left,
                target.tensors[bond],
                state.tensors[bond],
            )
        else:
            right = _right_environment_step(
                right,
                target.tensors[bond + 1],
                state.tensors[bond + 1],
            )

    global_objective = _squared_residual(target, state)
    scale = max(1.0, objective, global_objective)
    if abs(global_objective - objective) > 1e-10 * scale:
        msg = f"Cached local objective {objective:.12e} disagrees with the full residual {global_objective:.12e}."
        raise RuntimeError(msg)
    return state, global_objective, trace, rejected


def variational_fit(
    target: MPS,
    initial: MPS,
    *,
    compression_params: DigitalSimParams,
    max_sweeps: int = 8,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-14,
    acceptance_rel_tol: float = 1e-12,
) -> VariationalMPOResult:
    """Fit a capped MPS using cached alternating two-site sweeps."""
    cap = compression_params.max_bond_dim
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        msg = f"A positive integer max_bond_dim is required, got {cap!r}."
        raise ValueError(msg)
    if max_sweeps < 1:
        msg = f"max_sweeps must be positive, got {max_sweeps}."
        raise ValueError(msg)

    start_time = time.perf_counter()
    target_work = copy.deepcopy(target)
    _normalize(target_work)
    target_work.set_canonical_form(0, decomposition="QR")
    target_norm = _norm_squared(target_work)
    approximate = copy.deepcopy(initial)
    _normalize(approximate)
    _phase_align(target_work, approximate)
    approximate.set_canonical_form(0, decomposition="QR")
    if max(bond_profile(approximate), default=1) > cap:
        msg = "The variational initializer exceeds max_bond_dim."
        raise ValueError(msg)

    initial_objective = _squared_residual(target_work, approximate)
    objective = initial_objective
    objective_trace = [objective]
    update_trace = [objective]
    rejected = 0
    converged = objective <= abs_tol
    sweeps_done = 0

    for sweep in range(max_sweeps):
        if converged:
            break
        previous_state = copy.deepcopy(approximate)
        previous_objective = objective
        approximate, _, trace, rejected_lr = _cached_half_sweep(
            target_work,
            approximate,
            compression_params=compression_params,
            direction="left_to_right",
            target_norm=target_norm,
            acceptance_abs_tol=abs_tol,
            acceptance_rel_tol=acceptance_rel_tol,
        )
        update_trace.extend(trace[1:])
        approximate, objective, trace, rejected_rl = _cached_half_sweep(
            target_work,
            approximate,
            compression_params=compression_params,
            direction="right_to_left",
            target_norm=target_norm,
            acceptance_abs_tol=abs_tol,
            acceptance_rel_tol=acceptance_rel_tol,
        )
        update_trace.extend(trace[1:])
        rejected += rejected_lr + rejected_rl
        if objective > previous_objective:
            approximate = previous_state
            objective = previous_objective
            rejected += 1
        objective_trace.append(objective)
        sweeps_done = sweep + 1
        improvement = previous_objective - objective
        tolerance = max(abs_tol, rel_tol * max(previous_objective, abs_tol))
        converged = objective <= abs_tol or improvement <= tolerance

    if objective > initial_objective:
        msg = f"Final objective {objective:.6e} exceeds initializer {initial_objective:.6e}."
        raise RuntimeError(msg)
    if max(bond_profile(approximate), default=1) > cap:
        msg = "The variational result exceeds max_bond_dim."
        raise RuntimeError(msg)

    return VariationalMPOResult(
        state=approximate,
        objective_initial=initial_objective,
        objective_final=objective,
        sweeps=sweeps_done,
        converged=converged,
        objective_trace=objective_trace,
        update_trace=update_trace,
        rejected_nonimproving_updates=rejected,
        runtime_s=time.perf_counter() - start_time,
    )


def apply_variational_mpo_gate(
    initial: MPS,
    gate: BaseGate,
    *,
    compression_params: DigitalSimParams,
    max_sweeps: int = 8,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-14,
    acceptance_rel_tol: float = 1e-12,
) -> VariationalMPOResult:
    """Variationally compress the exact MPO-applied endpoint.

    The input MPS and ordinary MPO contract-and-truncate result are independent
    initializers.  The lowest-residual converged fit is selected whenever one
    exists; if neither fit converges, the lower residual is returned with
    ``converged=False`` for diagnosis.  No fallback is performed.
    """
    start_time = time.perf_counter()
    cap = compression_params.max_bond_dim
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        msg = f"A positive integer max_bond_dim is required, got {cap!r}."
        raise ValueError(msg)

    gate_mpo = MPO.from_gate(gate, initial.length)
    target = copy.deepcopy(initial)
    gate_mpo.multiply(target, compress=False)
    _normalize(target)

    mpo_initializer = copy.deepcopy(initial)
    gate_mpo.multiply(mpo_initializer, sim_params=compression_params, compress=True)
    _normalize(mpo_initializer)
    _phase_align(target, mpo_initializer)

    starts = (
        ("mpo_contract_compress", mpo_initializer),
        ("input", copy.deepcopy(initial)),
    )
    fitted_results: dict[str, VariationalMPOResult] = {}
    for name, initializer in starts:
        fitted_results[name] = variational_fit(
            target,
            initializer,
            compression_params=compression_params,
            max_sweeps=max_sweeps,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
            acceptance_rel_tol=acceptance_rel_tol,
        )

    converged_names = [name for name, result in fitted_results.items() if result.converged]
    eligible_names = converged_names or list(fitted_results)
    best_name = min(
        eligible_names,
        key=lambda name: (
            fitted_results[name].objective_final,
            0 if name == "mpo_contract_compress" else 1,
        ),
    )
    best = fitted_results[best_name]
    best.best_initializer = best_name
    best.initializer_objectives = {name: result.objective_initial for name, result in fitted_results.items()}
    best.initializer_final_objectives = {name: result.objective_final for name, result in fitted_results.items()}
    best.initializer_converged = {name: result.converged for name, result in fitted_results.items()}
    best.initializer_runtimes_s = {name: result.runtime_s for name, result in fitted_results.items()}
    best.runtime_s = time.perf_counter() - start_time
    best.target_max_bond = max(bond_profile(target), default=1)
    best.target_parameter_count = int(sum(np.asarray(tensor).size for tensor in target.tensors))
    best.fidelity_to_target = normalized_mps_fidelity(target, best.state)
    if best.objective_final > best.initializer_objectives["mpo_contract_compress"] + 1e-13:
        msg = "Variational result is worse than its MPO contract-and-truncate initializer."
        raise RuntimeError(msg)
    return best


def apply_variational_mpo_node(
    initial: MPS,
    node: DAGOpNode,
    *,
    compression_params: DigitalSimParams,
    max_sweeps: int = 8,
    rel_tol: float = 1e-10,
    abs_tol: float = 1e-14,
    acceptance_rel_tol: float = 1e-12,
) -> VariationalMPOResult:
    """Convert one two-qubit DAG node and apply :func:`apply_variational_mpo_gate`."""
    gates = convert_dag_to_tensor_algorithm(node)
    if len(gates) != 1 or gates[0].interaction != 2:
        msg = "Variational MPO application requires exactly one two-qubit gate node."
        raise ValueError(msg)
    return apply_variational_mpo_gate(
        initial,
        gates[0],
        compression_params=compression_params,
        max_sweeps=max_sweeps,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        acceptance_rel_tol=acceptance_rel_tol,
    )
