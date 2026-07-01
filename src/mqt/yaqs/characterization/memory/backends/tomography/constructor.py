# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Process-tensor tomography workflow: exhaustive discrete-basis simulation.

**Public** (see ``__all__`` in :mod:`mqt.yaqs.characterization.memory.backends.tomography`):
:func:`build_process_tensor` (this module,
:mod:`mqt.yaqs.characterization.memory.backends.tomography.constructor`).

:func:`build_process_tensor` is the high-level user entry point returning a process tensor directly
(:class:`~mqt.yaqs.characterization.memory.backends.tomography.process_tensors.DenseProcessTensor` or
:class:`~mqt.yaqs.characterization.memory.backends.tomography.process_tensors.MPOProcessTensor`).
The lower-level :func:`run_all_sequences` returns
:class:`~mqt.yaqs.characterization.memory.backends.tomography.data.SequenceData` covering all
``16**num_interventions`` Choi index sequences for ``num_interventions`` steps.

**Execution model** — Same pattern as :mod:`mqt.yaqs.simulator` and
:mod:`mqt.yaqs.characterization.memory.backends.surrogates.workflow`: build a picklable payload, optionally
install it as :data:`~mqt.yaqs.core.parallel_utils.WORKER_CTX`, then dispatch with
:func:`~mqt.yaqs.core.parallel_utils.run_indexed_jobs` (parallel or serial).

**Internals** — :func:`run_all_sequences` performs payload construction, aggregation, and
:func:`mqt.yaqs.characterization.memory.backends.tomography.basis._finalize_sequence_averages`.
"""

from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from mqt.yaqs.core.parallel_utils import (
    ExecutionConfig,
    merge_execution_config,
    run_indexed_jobs,
)

from ...shared.encoding import coerce_rho_matrix, normalize_backend_rho
from ...shared.utils import (
    StochasticSolver,
    _evolve_backend_state,
    _initialize_backend_state,
    extract_site0_rho,
    make_mcwf_static_context,
    resolve_stochastic_solver,
)
from ..sequences.workers import (
    _get_times_cached,
    _schedule_slots_for_sequence,
    _seq_final_worker,
    _validate_process_tensor_schedule_inputs,
)
from .basis import (
    _finalize_sequence_averages,
    assemble_fixed_basis,
    compute_dual_choi_basis,
)
from .data import SequenceData
from .process_tensors import validate_initial_rho

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.mps import MPS
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

    from .basis import TomographyBasis
    from .process_tensors import DenseProcessTensor, MPOProcessTensor


def _initial_psis_for_sequences(operator: MPO, solver: str, n_seq: int) -> list[np.ndarray | MPS]:
    """Build one fresh initial state per discrete-basis sequence.

    Args:
        operator: Hamiltonian MPO.
        solver: Backend solver name.
        n_seq: Number of parallel sequences.

    Returns:
        Initial backend states, one per sequence index.
    """
    psi0 = _initialize_backend_state(operator, solver)
    if isinstance(psi0, np.ndarray):
        template = np.asarray(psi0, dtype=np.complex128)
        return [template.copy() for _ in range(n_seq)]
    return [_initialize_backend_state(operator, solver) for _ in range(n_seq)]


def _reference_initial_rho(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    *,
    noise_model: NoiseModel | None,
    solver: StochasticSolver,
    num_trajectories: int,
) -> np.ndarray:
    r"""Return the site-0 reference state after ``U_0`` evolution from ``|0\\rangle^{\\otimes L}``.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        timesteps: Process-tensor schedule of length ``num_interventions + 1``.
        noise_model: Optional open-system noise model.
        solver: Stochastic solver (``"MCWF"`` or ``"TJM"``).
        num_trajectories: MCWF trajectories to average when ``noise_model`` is set.

    Returns:
        Normalized ``2 x 2`` reference density matrix at the cut.
    """
    num_interventions = len(timesteps) - 1
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True
    local_params.num_traj = 1

    mcwf_static_ctx = None
    if solver == "MCWF":
        mcwf_static_ctx = make_mcwf_static_context(operator, local_params, noise_model=noise_model)

    durs, ops, ctxs = _schedule_slots_for_sequence(
        sequence_idx=0,
        num_interventions=num_interventions,
        timesteps=timesteps,
        timesteps_rows=None,
        hamiltonian=operator,
        operators_list=None,
        mcwf_static_ctx=mcwf_static_ctx,
        mcwf_static_ctx_list=None,
    )

    n_traj = 1 if noise_model is None else int(num_trajectories)
    rho_acc = np.zeros((2, 2), dtype=np.complex128)
    times_cache: dict[tuple[float, float], np.ndarray] = {}
    duration = float(durs[0])

    for traj_idx in range(n_traj):
        state = _initialize_backend_state(operator, solver)
        step_params = copy.copy(local_params)
        step_params.elapsed_time = duration
        step_params.times = _get_times_cached(times_cache, dt=float(step_params.dt), duration=duration)
        state = _evolve_backend_state(
            state,
            ops[0],
            noise_model,
            step_params,
            solver,
            traj_idx=traj_idx,
            static_ctx=ctxs[0],
        )
        rho_acc += normalize_backend_rho(extract_site0_rho(state))

    return rho_acc / float(n_traj)


# ---------------------------------------------------------------------------
# Orchestration — build payload, run all ``16**num_interventions`` sequences, aggregate
# ---------------------------------------------------------------------------
def run_all_sequences(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float],
    *,
    parallel: bool = True,
    num_trajectories: int = 100,
    noise_model: NoiseModel | None = None,
    basis: TomographyBasis = "tetrahedral",
    basis_seed: int | None = None,
    solver: StochasticSolver | None = None,
    show_progress: bool = False,
    _execution: ExecutionConfig | None = None,
) -> SequenceData:
    """Run the backend for every one of the ``16**num_interventions`` discrete Choi index sequences.

    Prefer :func:`build_process_tensor` for the validated user entry; this routine assumes
    ``timesteps`` and solver compatibility are already correct.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        timesteps: Process-tensor schedule evolution durations (length ``num_interventions + 1``).
        parallel: Whether to parallelize over sequences.
        num_trajectories: MCWF trajectories per sequence (forced to 1 when noiseless).
        noise_model: Optional open-system noise model.
        basis: Tomography basis name.
        basis_seed: Optional seed when ``basis="random"``.
        solver: Stochastic solver (``"MCWF"`` or ``"TJM"``).
        show_progress: Whether to show a progress bar.
        _execution: Optional internal execution configuration.

    Returns:
        Exhaustive :class:`~mqt.yaqs.characterization.memory.backends.tomography.data.SequenceData`.

    Raises:
        ValueError: If ``num_interventions=0`` or the solver is unsupported.
    """
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True
    stochastic_solver = resolve_stochastic_solver(local_params, solver=solver)

    basis_set, choi_basis, choi_indices, _choi_feat = assemble_fixed_basis(basis=basis, basis_seed=basis_seed)
    choi_duals = compute_dual_choi_basis(choi_basis)

    num_interventions = len(timesteps) - 1
    if num_interventions <= 0:
        msg = "No sequences for num_interventions=0."
        raise ValueError(msg)
    if int(num_trajectories) != num_trajectories:
        msg = f"num_trajectories must be an integer, got {num_trajectories!r}."
        raise ValueError(msg)
    num_trajectories = int(num_trajectories)
    if num_trajectories < 0:
        msg = f"num_trajectories must be non-negative, got {num_trajectories}."
        raise ValueError(msg)
    if noise_model is not None and num_trajectories == 0:
        msg = "num_trajectories must be positive when noise_model is set."
        raise ValueError(msg)
    if noise_model is None:
        num_trajectories = 1

    initial_rho = _reference_initial_rho(
        operator,
        local_params,
        timesteps,
        noise_model=noise_model,
        solver=stochastic_solver,
        num_trajectories=num_trajectories,
    )

    def _enumerate_sequences(n_steps: int) -> list[tuple[int, ...]]:
        return list(itertools.product(range(16), repeat=n_steps))

    all_seqs = _enumerate_sequences(num_interventions)

    n_seq = len(all_seqs)
    samples_intervention_steps = [
        [(basis_set[choi_indices[a][1]][1], basis_set[choi_indices[a][0]][1]) for a in seq] for seq in all_seqs
    ]

    _validate_process_tensor_schedule_inputs(
        intervention_steps_list=samples_intervention_steps,
        timesteps=timesteps,
        timesteps_rows=None,
        operators_list=None,
        static_ctx_list=None,
    )

    mcwf_static_ctx = None
    if stochastic_solver == "MCWF":
        mcwf_static_ctx = make_mcwf_static_context(operator, local_params, noise_model=noise_model)
    elif stochastic_solver != "TJM":
        msg = f"Tomography does not support solver {stochastic_solver!r} (use MCWF or TJM)."
        raise ValueError(msg)

    total_jobs = n_seq * num_trajectories
    payload: dict[str, Any] = {
        "intervention_steps": samples_intervention_steps,
        "initial_psi": _initial_psis_for_sequences(operator, stochastic_solver, n_seq),
        "num_trajectories": num_trajectories,
        "operator": operator,
        "sim_params": local_params,
        "timesteps": timesteps,
        "timesteps_rows": None,
        "operators_list": None,
        "noise_model": noise_model,
        "mcwf_static_ctx": mcwf_static_ctx,
        "mcwf_static_ctx_list": None,
        "_times_cache": {},
        "solver": stochastic_solver,
    }

    aggregated_outputs = [np.zeros((2, 2), dtype=np.complex128) for _ in range(n_seq)]
    aggregated_weights = np.zeros(n_seq, dtype=np.float64)

    exec_cfg = merge_execution_config(_execution, parallel=parallel, show_progress=show_progress)
    job_results = run_indexed_jobs(
        _seq_final_worker,
        payload=payload,
        n_jobs=total_jobs,
        config=exec_cfg,
        desc=f"Simulating {n_seq} basis sequences",
    )
    for job_idx in range(total_jobs):
        s_idx, _traj_idx, rho_final, weight = job_results[job_idx]
        aggregated_outputs[s_idx] += rho_final * weight
        aggregated_weights[s_idx] += weight

    acc: dict[tuple[int, ...], list[Any]] = {}
    for i in range(n_seq):
        acc[all_seqs[i]] = [aggregated_outputs[i], aggregated_weights[i], num_trajectories]

    final_seqs, final_outputs, final_weights = _finalize_sequence_averages(acc, float(num_trajectories))

    return SequenceData(
        sequences=final_seqs,
        outputs=final_outputs,
        weights=final_weights,
        choi_basis=choi_basis,
        choi_indices=choi_indices,
        choi_duals=choi_duals,
        timesteps=timesteps,
        initial_rho=initial_rho,
    )


# ---------------------------------------------------------------------------
# Public entry — high-level façade (cf. surrogate ``workflow`` / ``simulator.run``)
# ---------------------------------------------------------------------------
def _construct_data(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    *,
    noise_model: NoiseModel | None = None,
    parallel: bool = True,
    num_trajectories: int = 100,
    basis: TomographyBasis = "tetrahedral",
    basis_seed: int | None = None,
    solver: StochasticSolver | None = None,
    show_progress: bool = False,
    _execution: ExecutionConfig | None = None,
) -> SequenceData:
    """Validate inputs and construct `SequenceData` via exhaustive simulation.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        timesteps: Optional process-tensor schedule (length ``num_interventions + 1``;
            defaults to ``[dt, dt]`` for one intervention leg).
        noise_model: Optional noise model.
        parallel: Whether to parallelize over sequences.
        num_trajectories: Number of MCWF trajectories per sequence (forced to 1 if noiseless).
        basis: Tomography basis name.
        basis_seed: Optional seed used when ``basis="random"``.
        solver: Stochastic solver name (``"MCWF"`` or ``"TJM"``).
        show_progress: Whether to show a progress bar during simulation.

    Returns:
        Exhaustive `SequenceData` containing all simulated sequences.

    Raises:
        ValueError: If ``solver`` is not MCWF or TJM.
    """
    if timesteps is None:
        dt = float(sim_params.dt)
        timesteps = [dt, dt]

    stochastic_solver = resolve_stochastic_solver(sim_params, solver=solver)
    valid_solvers = {"MCWF", "TJM"}
    if stochastic_solver not in valid_solvers:
        msg = f"Tomography requires solvers {valid_solvers}, got {stochastic_solver!r}."
        raise ValueError(msg)

    return run_all_sequences(
        operator,
        sim_params,
        timesteps,
        parallel=parallel,
        num_trajectories=num_trajectories,
        noise_model=noise_model,
        basis=basis,
        basis_seed=basis_seed,
        solver=stochastic_solver,
        show_progress=show_progress,
        _execution=_execution,
    )


def build_process_tensor(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    *,
    noise_model: NoiseModel | None = None,
    parallel: bool = True,
    num_trajectories: int = 100,
    basis: TomographyBasis = "tetrahedral",
    basis_seed: int | None = None,
    return_type: Literal["dense", "mpo"] = "dense",
    # Dense reconstruction
    check: bool = True,
    atol: float = 1e-8,
    # MPO reconstruction
    compress_every: int = 100,
    tol: float = 1e-12,
    max_bond_dim: int | None = None,
    n_sweeps: int = 2,
    solver: StochasticSolver | None = None,
    initial_rho: np.ndarray | None = None,
    initial_rho_atol: float = 1e-8,
    _execution: ExecutionConfig | None = None,
) -> DenseProcessTensor | MPOProcessTensor:
    """Construct a process tensor via exhaustive discrete-basis tomography.

    This simulates **every** ``16**num_interventions`` discrete basis sequence and returns a
    process tensor directly:

    - ``return_type="dense"``: reconstruct and return a :class:`DenseProcessTensor`.
    - ``return_type="mpo"``: build and return an :class:`MPOProcessTensor`.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        timesteps: Optional process-tensor schedule evolution durations (length
            ``num_interventions + 1``; defaults to ``[dt, dt]`` for one intervention leg).
        noise_model: Optional open-system noise model.
        parallel: Whether to parallelize over sequences.
        num_trajectories: MCWF trajectories per sequence (forced to 1 when noiseless).
        basis: Tomography basis name.
        basis_seed: Optional seed when ``basis="random"``.
        return_type: ``"dense"`` or ``"mpo"`` process-tensor representation.
        check: Run self-consistency check for dense reconstruction.
        atol: Absolute tolerance for the dense self-check.
        compress_every: MPO rank-1 accumulation compress interval.
        tol: MPO compression tolerance.
        max_bond_dim: Optional MPO bond-dimension cap.
        n_sweeps: MPO compression sweeps.
        solver: Stochastic solver (``"MCWF"`` or ``"TJM"``).
        initial_rho: Optional expected site-0 reference after ``U_0``; validated against the
            computed tomography reference when provided.
        initial_rho_atol: Tolerance for optional ``initial_rho`` validation.
        _execution: Optional internal execution configuration.

    Returns:
        Dense or MPO process-tensor wrapper depending on ``return_type``.

    Raises:
        ValueError: If ``return_type`` is not ``"dense"`` or ``"mpo"``.
    """
    data = _construct_data(
        operator,
        sim_params,
        timesteps,
        noise_model=noise_model,
        parallel=parallel,
        num_trajectories=num_trajectories,
        basis=basis,
        basis_seed=basis_seed,
        solver=solver,
        _execution=_execution,
    )

    if initial_rho is not None:
        validate_initial_rho(coerce_rho_matrix(initial_rho), data.initial_rho, atol=initial_rho_atol)

    if return_type == "dense":
        return data.to_dense_process_tensor(check=check, atol=atol)
    if return_type == "mpo":
        return data.to_mpo_process_tensor(
            compress_every=compress_every,
            tol=tol,
            max_bond_dim=max_bond_dim,
            n_sweeps=n_sweeps,
        )
    msg = f"Unknown return_type {return_type!r} (expected 'dense' or 'mpo')."
    raise ValueError(msg)
