# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Leg-by-leg process-tensor MPO construction without exhaustive ``16**k`` tomography."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.parallel_utils import ExecutionConfig, merge_execution_config, resolve_worker_ctx, run_indexed_jobs

from ...shared.encoding import normalize_backend_rho
from ...shared.intervention_steps import apply_intervention_to_backend
from ...shared.utils import (
    StochasticSolver,
    _evolve_backend_state,
    _initialize_backend_state,
    extract_site0_rho,
    make_mcwf_static_context,
    resolve_stochastic_solver,
)
from ..sequences.workers import _get_times_cached
from .basis import TomographyBasis, assemble_fixed_basis, compute_dual_choi_basis
from .constructor import _reference_initial_rho
from .data import _rank1_mpo_term, accumulate_rank1_terms
from .process_tensors import MPOProcessTensor, validate_initial_rho

if TYPE_CHECKING:
    from mqt.yaqs.analog.mcwf import MCWFContext
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

_N_CHOI = 16
BackendState = MPS | NDArray[np.complex128]


@dataclass
class _Branch:
    """One definite past intervention history and backend state before the next leg."""

    history: tuple[int, ...]
    psi: BackendState
    weight: float


def _clone_backend_state(state: BackendState) -> BackendState:
    """Return an independent copy of a dense or MPS backend state.

    Args:
        state: Dense vector (MCWF) or MPS (TJM).

    Returns:
        Deep-copied MPS or a contiguous dense copy.
    """
    if isinstance(state, MPS):
        return copy.deepcopy(state)
    return np.asarray(state, dtype=np.complex128).reshape(-1).copy()


def _compress_branches(
    branches: list[_Branch],
    *,
    max_bond_dim: int | None,
    tol: float,
) -> list[_Branch]:
    """Compress the branch ensemble to at most ``max_bond_dim`` states.

    Dense (MCWF) ensembles use a weighted SVD. MPS (TJM) ensembles keep the
    highest-weight branches without densifying.

    Args:
        branches: Current ensemble of weighted backend states.
        max_bond_dim: Optional cap on retained branches; ``None`` keeps all.
        tol: Singular-value floor when selecting kept dense modes.

    Returns:
        Compressed branch list (unchanged when already within the cap).
    """
    if max_bond_dim is None or len(branches) <= max_bond_dim:
        return branches
    if len(branches) == 1:
        return branches

    if isinstance(branches[0].psi, MPS):
        ordered = sorted(branches, key=lambda br: br.weight, reverse=True)
        return ordered[: int(max_bond_dim)]

    dim = int(np.asarray(branches[0].psi, dtype=np.complex128).reshape(-1).size)
    n = len(branches)
    mat = np.zeros((dim, n), dtype=np.complex128)
    for col, br in enumerate(branches):
        scale = float(np.sqrt(max(br.weight, 0.0)))
        mat[:, col] = scale * np.asarray(br.psi, dtype=np.complex128).reshape(-1)

    _u, singular_values, vh = np.linalg.svd(mat, full_matrices=False)
    keep = int(np.sum(singular_values > tol))
    keep = min(keep, int(max_bond_dim))
    keep = max(1, keep)

    out: list[_Branch] = []
    for row in range(keep):
        coeffs = vh[row, :]
        i_dom = int(np.argmax(np.abs(coeffs)))
        psi = mat @ coeffs.conj()
        norm = float(np.linalg.norm(psi))
        if norm <= 1e-15:
            psi = np.asarray(branches[i_dom].psi, dtype=np.complex128).reshape(-1).copy()
            norm = float(np.linalg.norm(psi))
        else:
            psi /= norm
        weight = float(singular_values[row] ** 2)
        out.append(_Branch(history=branches[i_dom].history, psi=psi, weight=weight))
    return out


def _prepare_step(
    operator: MPO,
    sim_params: AnalogSimParams,
    duration: float,
    *,
    solver: StochasticSolver,
) -> tuple[AnalogSimParams, MCWFContext | None]:
    """Build local sim params and MCWF context for one schedule slot.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        duration: Evolution duration for this slot.
        solver: Stochastic solver name.

    Returns:
        Tuple ``(step_params, static_ctx)``.
    """
    local_params = copy.copy(sim_params)
    local_params.get_state = True
    local_params.num_traj = 1
    static_ctx = make_mcwf_static_context(operator, local_params, noise_model=None) if solver == "MCWF" else None
    times_cache: dict[tuple[float, float], np.ndarray] = {}
    step_params = copy.copy(local_params)
    step_params.elapsed_time = float(duration)
    step_params.times = _get_times_cached(times_cache, dt=float(step_params.dt), duration=float(duration))
    return step_params, static_ctx


def _evolve_initial_state(
    operator: MPO,
    sim_params: AnalogSimParams,
    duration: float,
    *,
    solver: StochasticSolver,
) -> BackendState:
    """Evolve from ``|0...0>`` for one schedule slot.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        duration: Evolution duration for ``U_0``.
        solver: Stochastic solver name.

    Returns:
        Backend state after the initial evolution (dense for MCWF, MPS for TJM).
    """
    step_params, static_ctx = _prepare_step(operator, sim_params, duration, solver=solver)
    state = _initialize_backend_state(operator, solver)
    return _evolve_backend_state(
        state,
        operator,
        None,
        step_params,
        solver,
        traj_idx=0,
        static_ctx=static_ctx,
    )


def _branch_extension_worker(
    job_idx: int,
    job_payload: dict[str, Any] | None = None,
) -> tuple[tuple[int, ...], BackendState, float, NDArray[np.complex128]] | None:
    """Extend one branch by one Choi-basis intervention and post-evolution.

    Flat index layout is ``branch_index * 16 + choi_index``. Branch states keep their
    native backend type (dense for MCWF, MPS for TJM).

    Args:
        job_idx: Flat index over branches and Choi-basis elements.
        job_payload: Shared step payload; defaults to
            :data:`~mqt.yaqs.core.parallel_utils.WORKER_CTX`.

    Returns:
        ``(history, psi, weight, rho_out)`` for kept extensions, or ``None`` when the
        cumulative weight falls below the discard threshold.
    """
    ctx = resolve_worker_ctx(job_payload)
    br_idx, choi_idx = divmod(int(job_idx), _N_CHOI)
    br: _Branch = ctx["branches"][br_idx]
    prep_idx, meas_idx = ctx["choi_indices"][choi_idx]
    basis_set = ctx["basis_set"]
    meas_psi, prep_psi = basis_set[meas_idx][1], basis_set[prep_idx][1]

    state = _clone_backend_state(br.psi)
    state, step_prob = apply_intervention_to_backend(
        state,
        (meas_psi, prep_psi),
        solver=ctx["solver"],
        chain_length=int(ctx["chain_length"]),
    )
    weight = float(br.weight) * float(step_prob)
    if weight <= 1e-30:
        return None

    state = _evolve_backend_state(
        state,
        ctx["operator"],
        None,
        ctx["sim_params"],
        ctx["solver"],
        traj_idx=0,
        static_ctx=ctx["static_ctx"],
    )
    rho_out = normalize_backend_rho(extract_site0_rho(state))
    history = (*br.history, choi_idx)
    return history, state, weight, rho_out


def _apply_timestep(
    branches: list[_Branch],
    *,
    operator: MPO,
    sim_params: AnalogSimParams,
    duration: float,
    basis_set: list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]],
    choi_indices: list[tuple[int, int]],
    choi_duals: list[NDArray[np.complex128]],
    solver: StochasticSolver,
    execution: ExecutionConfig,
) -> tuple[list[_Branch], list[MPO]]:
    """Extend every branch by one local CPTP leg via indexed jobs.

    Args:
        branches: Ensemble before the intervention leg.
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        duration: Evolution duration after the intervention.
        basis_set: Discrete basis kets ``(label, ket, dual)``.
        choi_indices: Map from Choi index to ``(prep_idx, meas_idx)``.
        choi_duals: Dual Choi operators for rank-1 MPO assembly.
        solver: Stochastic solver name.
        execution: Parallel / serial job-dispatch configuration.

    Returns:
        Expanded branches and the corresponding rank-1 MPO terms.
    """
    step_params, static_ctx = _prepare_step(operator, sim_params, duration, solver=solver)
    n_jobs = len(branches) * _N_CHOI
    payload: dict[str, Any] = {
        "branches": branches,
        "operator": operator,
        "sim_params": step_params,
        "basis_set": basis_set,
        "choi_indices": choi_indices,
        "solver": solver,
        "static_ctx": static_ctx,
        "chain_length": int(operator.length),
    }
    job_results = run_indexed_jobs(
        _branch_extension_worker,
        payload=payload,
        n_jobs=n_jobs,
        config=execution,
        desc=f"MPO construction ({len(branches)} branches)",
    )

    expanded: list[_Branch] = []
    terms: list[MPO] = []
    for job_idx in range(n_jobs):
        out = job_results[job_idx]
        if out is None:
            continue
        history, state, weight, rho_out = out
        dual_ops = [choi_duals[idx].T for idx in history]
        terms.append(_rank1_mpo_term(rho_out, dual_ops, weight=weight))
        expanded.append(_Branch(history=history, psi=state, weight=weight))
    return expanded, terms


def build_process_tensor_direct(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    *,
    basis: TomographyBasis = "tetrahedral",
    basis_seed: int | None = None,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 2,
    compress_every: int = 16,
    solver: StochasticSolver | None = None,
    initial_rho: np.ndarray | None = None,
    initial_rho_atol: float = 1e-8,
    parallel: bool = True,
    _execution: ExecutionConfig | None = None,
) -> MPOProcessTensor:
    """Build a process-tensor MPO by leg-by-leg contraction.

    At each timestep only ``16 * chi`` local basis updates are simulated, where ``chi`` is the
    compressed branch count from the previous step. This avoids enumerating all ``16**k`` sequences.
    Construction is noiseless (site-0 interventions only). Within each intervention step, branch
    extensions are dispatched with :func:`~mqt.yaqs.core.parallel_utils.run_indexed_jobs`.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        timesteps: Process-tensor schedule of length ``num_interventions + 1``.
        basis: Discrete Choi basis name.
        basis_seed: Optional seed when ``basis="random"``.
        tol: MPO compression tolerance.
        max_bond_dim: Optional cap on the branch ensemble / MPO bond dimension. ``None`` (default)
            keeps all branches for an exact noiseless construction.
        n_sweeps: MPO compression sweeps after each step.
        compress_every: Rank-1 accumulation batch size before intermediate compression.
        solver: Stochastic solver (``"MCWF"`` or ``"TJM"``).
        initial_rho: Optional reference site-0 state after ``U_0``.
        initial_rho_atol: Tolerance for optional ``initial_rho`` validation.
        parallel: Whether to parallelize branch extensions within each intervention step.
        _execution: Optional internal execution configuration.

    Returns:
        MPO process-tensor wrapper.

    Raises:
        ValueError: If ``num_interventions`` is zero or the solver is unsupported.
    """
    if timesteps is None:
        dt = float(sim_params.dt)
        timesteps = [dt, dt]

    stochastic_solver = resolve_stochastic_solver(sim_params, solver=solver)
    if stochastic_solver not in {"MCWF", "TJM"}:
        msg = f"Direct construction requires solvers MCWF or TJM, got {stochastic_solver!r}."
        raise ValueError(msg)

    num_interventions = len(timesteps) - 1
    if num_interventions <= 0:
        msg = "Direct construction requires at least one intervention leg."
        raise ValueError(msg)

    basis_set, choi_basis, choi_indices, _choi_feat = assemble_fixed_basis(basis=basis, basis_seed=basis_seed)
    choi_duals = compute_dual_choi_basis(choi_basis)
    execution = merge_execution_config(_execution, parallel=parallel)

    ref_rho = _reference_initial_rho(
        operator,
        sim_params,
        timesteps,
        noise_model=None,
        solver=stochastic_solver,
        num_trajectories=1,
    )
    if initial_rho is not None:
        validate_initial_rho(np.asarray(initial_rho, dtype=np.complex128), ref_rho, atol=initial_rho_atol)

    psi0 = _evolve_initial_state(
        operator,
        sim_params,
        float(timesteps[0]),
        solver=stochastic_solver,
    )
    branches = [_Branch(history=(), psi=psi0, weight=1.0)]

    comb: MPO | None = None
    for step_idx in range(num_interventions):
        branches, terms = _apply_timestep(
            branches,
            operator=operator,
            sim_params=sim_params,
            duration=float(timesteps[step_idx + 1]),
            basis_set=basis_set,
            choi_indices=choi_indices,
            choi_duals=choi_duals,
            solver=stochastic_solver,
            execution=execution,
        )
        if not terms:
            msg = f"Direct construction produced no rank-1 terms at leg {step_idx + 1}."
            raise ValueError(msg)
        comb = accumulate_rank1_terms(
            terms,
            num_steps=step_idx + 1,
            tol=tol,
            max_bond_dim=max_bond_dim,
            n_sweeps=n_sweeps,
            compress_every=compress_every,
        )
        branches = _compress_branches(branches, max_bond_dim=max_bond_dim, tol=tol)

    if comb is None:
        comb = _rank1_mpo_term(ref_rho, [], weight=1.0)

    return MPOProcessTensor(comb, list(timesteps), initial_rho=ref_rho.copy())
