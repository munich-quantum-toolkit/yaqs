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

:func:`build_process_tensor` is the high-level user entry point returning a comb directly
(:class:`~mqt.yaqs.characterization.memory.backends.tomography.combs.DenseComb` or
:class:`~mqt.yaqs.characterization.memory.backends.tomography.combs.MPOComb`).
The lower-level :func:`run_all_sequences` returns
:class:`~mqt.yaqs.characterization.memory.backends.tomography.data.SequenceData` covering all ``16^k``
Choi index sequences for ``k`` steps.

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
    resolve_worker_ctx,
    run_indexed_jobs,
    unpack_flat_job,
)

from ...shared.utils import (
    StochasticSolver,
    _evolve_backend_state,
    _initialize_backend_state,
    _reprepare_backend_state_forced,
    extract_site0_rho,
    make_mcwf_static_context,
    resolve_stochastic_solver,
)
from .basis import (
    _finalize_sequence_averages,
    assemble_fixed_basis,
    compute_dual_choi_basis,
)
from .data import SequenceData

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

    from .basis import TomographyBasis
    from .combs import DenseComb, MPOComb

# Re-export for worker docstrings referencing WORKER_CTX default.


# ---------------------------------------------------------------------------
# Parallel job payload (pickle-stable keys for WORKER_CTX)
# ---------------------------------------------------------------------------
# ``run_all_sequences`` passes this dict to
# :func:`~mqt.yaqs.core.parallel_utils.run_indexed_jobs` (initializer →
# :data:`~mqt.yaqs.core.parallel_utils.WORKER_CTX`) or directly to workers on the
# serial path. Workers use :func:`~mqt.yaqs.core.parallel_utils.resolve_worker_ctx`
# and :func:`~mqt.yaqs.core.parallel_utils.unpack_flat_job`.
#
#   psi_pairs           list[list[(meas, prep)]] — one inner list per sequence (length k)
#   num_trajectories    flat-index stride (1 when ``noise_model`` is None)
#   operator            Hamiltonian MPO
#   sim_params          :class:`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams`
#   timesteps           list[float] duration per step (length k)
#   noise_model         optional open-system noise
#   mcwf_static_ctx     from ``make_mcwf_static_context`` when solver is MCWF


# ---------------------------------------------------------------------------
# Pool workers — signature (job_idx, payload=None)
# ---------------------------------------------------------------------------
def _sequence_worker(
    job_idx: int,
    payload: dict[str, Any] | None = None,
) -> tuple[int, int, np.ndarray, float]:
    """Simulate one trajectory for one discrete-basis sequence.

    Args:
        job_idx: Flat index ``sequence_index * num_trajectories + trajectory_index``.
        payload: Optional worker payload (defaults to :data:`~mqt.yaqs.core.parallel_utils.WORKER_CTX`).

    Returns:
        Tuple ``(sequence_index, trajectory_index, rho_final, weight)`` where ``rho_final`` is the
        site-0 reduced density matrix and ``weight`` is the cumulative projection probability.
    """
    ctx = resolve_worker_ctx(payload)
    s_idx, traj_idx = unpack_flat_job(job_idx, int(ctx["num_trajectories"]))

    psi_pairs = ctx["psi_pairs"][s_idx]
    operator = ctx["operator"]
    sim_params = ctx["sim_params"]
    timesteps: list[float] = ctx["timesteps"]
    noise_model = ctx["noise_model"]

    if noise_model is None:
        assert int(ctx["num_trajectories"]) == 1, "num_trajectories must be 1 when noise_model is None."

    solver = resolve_stochastic_solver(sim_params, solver=ctx.get("solver"))
    current_state = _initialize_backend_state(operator, solver)

    weight = 1.0
    for step_i, (psi_meas, psi_prep) in enumerate(psi_pairs):
        current_state, step_prob = _reprepare_backend_state_forced(current_state, psi_meas, psi_prep, solver)
        weight *= float(step_prob)
        if weight < 1e-15:
            break

        duration = float(timesteps[step_i])
        step_params = copy.deepcopy(sim_params)
        step_params.elapsed_time = duration
        step_params.num_traj = 1
        step_params.get_state = True
        n_steps = max(1, int(np.ceil(duration / step_params.dt)))
        step_params.times = np.linspace(0, duration, n_steps + 1)

        current_state = _evolve_backend_state(
            current_state,
            operator,
            noise_model,
            step_params,
            solver,
            traj_idx=traj_idx,
            static_ctx=ctx.get("mcwf_static_ctx"),
        )

    rho_final = extract_site0_rho(current_state)
    return (s_idx, traj_idx, rho_final, float(weight))


# ---------------------------------------------------------------------------
# Orchestration — build payload, run all ``16^k`` sequences, aggregate
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
    """Run the backend for every one of the ``16^k`` discrete Choi index sequences.

    Prefer :func:`build_process_tensor` for the validated user entry; this routine assumes
    ``timesteps`` and solver compatibility are already correct.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        timesteps: Per-step evolution durations (length ``k``).
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
        ValueError: If ``k=0`` or the solver is unsupported.
    """
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True
    stochastic_solver = resolve_stochastic_solver(local_params, solver=solver)

    basis_set, choi_basis, choi_indices, _choi_feat = assemble_fixed_basis(basis=basis, basis_seed=basis_seed)
    choi_duals = compute_dual_choi_basis(choi_basis)

    k = len(timesteps)
    if k == 0:
        msg = "No sequences for k=0."
        raise ValueError(msg)

    def _enumerate_sequences(k_in: int) -> list[tuple[int, ...]]:
        return list(itertools.product(range(16), repeat=k_in))

    all_seqs = _enumerate_sequences(k)

    n_seq = len(all_seqs)
    samples_psi_pairs = [
        [(basis_set[choi_indices[a][1]][1], basis_set[choi_indices[a][0]][1]) for a in seq] for seq in all_seqs
    ]

    if noise_model is None:
        num_trajectories = 1

    mcwf_static_ctx = None
    if stochastic_solver == "MCWF":
        mcwf_static_ctx = make_mcwf_static_context(operator, local_params, noise_model=noise_model)
    elif stochastic_solver != "TJM":
        msg = f"Tomography does not support solver {stochastic_solver!r} (use MCWF or TJM)."
        raise ValueError(msg)

    total_jobs = n_seq * num_trajectories
    # Pickle-stable payload — schema documented above.
    payload = {
        "psi_pairs": samples_psi_pairs,
        "num_trajectories": num_trajectories,
        "operator": operator,
        "sim_params": local_params,
        "timesteps": timesteps,
        "noise_model": noise_model,
        "mcwf_static_ctx": mcwf_static_ctx,
        "solver": stochastic_solver,
    }

    aggregated_outputs = [np.zeros((2, 2), dtype=np.complex128) for _ in range(n_seq)]
    aggregated_weights = np.zeros(n_seq, dtype=np.float64)

    exec_cfg = merge_execution_config(_execution, parallel=parallel, show_progress=show_progress)
    job_results = run_indexed_jobs(
        _sequence_worker,
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
        timesteps: Optional per-step durations (defaults to ``[sim_params.elapsed_time]``).
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
        timesteps = [sim_params.elapsed_time]

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
    _execution: ExecutionConfig | None = None,
) -> DenseComb | MPOComb:
    """Construct a process tensor via exhaustive discrete-basis tomography.

    This simulates **every** ``16^k`` discrete basis sequence and returns a comb directly:

    - ``return_type="dense"``: reconstruct and return a :class:`DenseComb`.
    - ``return_type="mpo"``: build and return an :class:`MPOComb`.

    Args:
        operator: Hamiltonian MPO.
        sim_params: Analog simulation parameters.
        timesteps: Optional per-step durations (defaults to ``[sim_params.elapsed_time]``).
        noise_model: Optional open-system noise model.
        parallel: Whether to parallelize over sequences.
        num_trajectories: MCWF trajectories per sequence (forced to 1 when noiseless).
        basis: Tomography basis name.
        basis_seed: Optional seed when ``basis="random"``.
        return_type: ``"dense"`` or ``"mpo"`` comb representation.
        check: Run self-consistency check for dense reconstruction.
        atol: Absolute tolerance for the dense self-check.
        compress_every: MPO rank-1 accumulation compress interval.
        tol: MPO compression tolerance.
        max_bond_dim: Optional MPO bond-dimension cap.
        n_sweeps: MPO compression sweeps.
        solver: Stochastic solver (``"MCWF"`` or ``"TJM"``).
        _execution: Optional internal execution configuration.

    Returns:
        Dense or MPO comb wrapper depending on ``return_type``.

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

    if return_type == "dense":
        return data.to_dense_comb(check=check, atol=atol)
    if return_type == "mpo":
        return data.to_mpo_comb(
            compress_every=compress_every,
            tol=tol,
            max_bond_dim=max_bond_dim,
            n_sweeps=n_sweeps,
        )
    msg = f"Unknown return_type {return_type!r} (expected 'dense' or 'mpo')."
    raise ValueError(msg)
