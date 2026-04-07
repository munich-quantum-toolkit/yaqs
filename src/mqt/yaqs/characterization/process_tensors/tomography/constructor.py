# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Process-tensor tomography workflow: exhaustive discrete-basis simulation.

**Public** (see ``__all__`` in :mod:`mqt.yaqs.characterization.process_tensors.tomography`):
:func:`construct_process_tensor` (this module,
:mod:`mqt.yaqs.characterization.process_tensors.tomography.constructor`).

:func:`construct_process_tensor` is the high-level user entry point returning a comb directly
(:class:`~mqt.yaqs.characterization.process_tensors.tomography.combs.DenseComb` or
:class:`~mqt.yaqs.characterization.process_tensors.tomography.combs.MPOComb`).
The lower-level :func:`run_all_sequences` returns
:class:`~mqt.yaqs.characterization.process_tensors.tomography.data.SequenceData` covering all ``16^k``
Choi index sequences for ``k`` steps.

**Execution model** — Same pattern as :mod:`mqt.yaqs.simulator` and
:mod:`mqt.yaqs.characterization.process_tensors.surrogates.workflow`: build a picklable payload, optionally
install it as :data:`~mqt.yaqs.simulator.WORKER_CTX`, then dispatch with
:func:`~mqt.yaqs.simulator.run_backend_parallel` or run workers serially (optionally via
``threadpoolctl`` for BLAS restraint).

**Internals** — :func:`run_all_sequences` performs payload construction, aggregation, and
:func:`mqt.yaqs.characterization.process_tensors.tomography.basis._finalize_sequence_averages`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1) STANDARD LIBRARY
# ---------------------------------------------------------------------------
import copy
from typing import TYPE_CHECKING, Any, Literal

# ---------------------------------------------------------------------------
# 2) THIRD PARTY
# ---------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 3) LOCAL — core / simulator
# ---------------------------------------------------------------------------
from mqt.yaqs.simulator import WORKER_CTX, available_cpus, run_backend_parallel

from ..core.utils import (
    _evolve_backend_state,
    _get_rho_site_zero,
    _initialize_backend_state,
    _reprepare_backend_state_forced,
    make_mcwf_static_context,
)

# ---------------------------------------------------------------------------
# 4) LOCAL — tomography stack
# ---------------------------------------------------------------------------
from .basis import (
    _finalize_sequence_averages,
    calculate_dual_choi_basis,
    get_basis_states,
    get_choi_basis,
)
from .data import SequenceData

if TYPE_CHECKING:
    from collections.abc import Callable

    from mqt.yaqs.core.data_structures.networks import MPO
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

    from .basis import (
        TomographyBasis,
    )
    from .combs import DenseComb, MPOComb


# ---------------------------------------------------------------------------
# 5) PARALLEL JOB PAYLOAD
# ---------------------------------------------------------------------------
# Passed to ``run_backend_parallel`` (initializer → ``WORKER_CTX``) or directly to workers on the
# serial path. Keys must stay stable for pickling.
#
#   psi_pairs           list[list[(meas, prep)]] — one inner list per sequence (length k)
#   num_trajectories    MCWF trajectory split; must be 1 when ``noise_model`` is None
#   operator            Hamiltonian MPO
#   sim_params          :class:`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams`
#   timesteps           list[float] duration per step (length k)
#   noise_model         optional open-system noise
#   mcwf_static_ctx     from ``make_mcwf_static_context`` when solver is MCWF


# ---------------------------------------------------------------------------
# 6) WORKERS — one job index → one (sequence, trajectory) pair
# ---------------------------------------------------------------------------


def _sequence_worker(
    job_idx: int,
    payload: dict[str, Any] | None = None,
) -> tuple[int, int, np.ndarray, float]:
    """Single trajectory for one discrete-basis sequence: prep → evolve → site-0 density."""
    ctx = payload if payload is not None else WORKER_CTX
    num_trajectories: int = int(ctx["num_trajectories"])
    s_idx: int = int(job_idx // num_trajectories)
    traj_idx: int = int(job_idx % num_trajectories)

    psi_pairs = ctx["psi_pairs"][s_idx]
    operator = ctx["operator"]
    sim_params = ctx["sim_params"]
    timesteps: list[float] = ctx["timesteps"]
    noise_model = ctx["noise_model"]

    if noise_model is None:
        assert num_trajectories == 1, "num_trajectories must be 1 when noise_model is None."

    solver = sim_params.solver
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
        step_params.show_progress = False
        step_params.get_state = True
        n_steps = max(1, int(np.round(duration / step_params.dt)))
        step_params.times = np.linspace(0, n_steps * step_params.dt, n_steps + 1)

        current_state = _evolve_backend_state(
            current_state,
            operator,
            noise_model,
            step_params,
            solver,
            traj_idx=traj_idx,
            static_ctx=ctx.get("mcwf_static_ctx"),
        )

    rho_final = _get_rho_site_zero(current_state)
    return (s_idx, traj_idx, rho_final, float(weight))


def _call_backend_serial(backend: Callable[..., Any], *args: Any) -> Any:
    import contextlib  # noqa: PLC0415

    try:
        from threadpoolctl import threadpool_limits  # noqa: PLC0415
    except ImportError:
        return backend(*args)
    with contextlib.suppress(Exception), threadpool_limits(limits=1):
        return backend(*args)


# ---------------------------------------------------------------------------
# 7) ORCHESTRATION — build payload, run all ``16^k`` sequences, aggregate
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
) -> SequenceData:
    """Run the backend for every one of the ``16^k`` discrete Choi index sequences.

    Prefer :func:`construct_process_tensor` for the validated user entry; this routine assumes
    ``timesteps`` and
    solver compatibility are already correct.
    """
    local_params = copy.deepcopy(sim_params)
    local_params.get_state = True

    basis_set = get_basis_states(basis=basis, seed=basis_seed)
    choi_basis, choi_indices = get_choi_basis(basis=basis, seed=basis_seed)
    choi_duals = calculate_dual_choi_basis(choi_basis)

    k = len(timesteps)

    def _enumerate_sequences(k_in: int) -> list[tuple[int, ...]]:
        import itertools  # noqa: PLC0415

        return list(itertools.product(range(16), repeat=k_in))

    all_seqs = _enumerate_sequences(k)
    if len(all_seqs) == 0:
        msg = "No sequences for k=0."
        raise ValueError(msg)

    n_seq = len(all_seqs)
    samples_psi_pairs = [
        [(basis_set[choi_indices[a][1]][1], basis_set[choi_indices[a][0]][1]) for a in seq] for seq in all_seqs
    ]

    if noise_model is None:
        noise_model = local_params.noise_model
    if noise_model is None:
        num_trajectories = 1

    mcwf_static_ctx = None
    if local_params.solver == "MCWF":
        mcwf_static_ctx = make_mcwf_static_context(operator, local_params, noise_model=noise_model)
    elif local_params.solver != "TJM":
        msg = f"Tomography does not support solver {local_params.solver!r} (use MCWF or TJM)."
        raise ValueError(msg)

    total_jobs = n_seq * num_trajectories
    payload = {
        "psi_pairs": samples_psi_pairs,
        "num_trajectories": num_trajectories,
        "operator": operator,
        "sim_params": local_params,
        "timesteps": timesteps,
        "noise_model": noise_model,
        "mcwf_static_ctx": mcwf_static_ctx,
    }

    aggregated_outputs = [np.zeros((2, 2), dtype=np.complex128) for _ in range(n_seq)]
    aggregated_weights = np.zeros(n_seq, dtype=np.float64)

    if parallel and total_jobs > 1:
        max_workers = max(1, available_cpus() - 1)
        results_iterator = run_backend_parallel(
            worker_fn=_sequence_worker,
            payload=payload,
            n_jobs=total_jobs,
            max_workers=max_workers,
            show_progress=local_params.show_progress,
            desc=f"Simulating {n_seq} basis sequences (parallel)",
        )
        for _, (s_idx, _traj_idx, rho_final, weight) in results_iterator:
            aggregated_outputs[s_idx] += rho_final * weight
            aggregated_weights[s_idx] += weight
    else:
        disable_tqdm = not local_params.show_progress
        for job_idx in tqdm(
            range(total_jobs),
            desc=f"Simulating {n_seq} basis sequences (serial)",
            disable=disable_tqdm,
        ):
            (s_idx, _traj_idx, rho_final, weight) = _call_backend_serial(_sequence_worker, job_idx, payload)
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
# 8) PUBLIC ENTRY — high-level façade (cf. surrogate ``workflow`` / ``simulator.run``)
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
) -> SequenceData:
    """Validated data-construction path returning :class:`SequenceData`."""
    if timesteps is None:
        timesteps = [sim_params.elapsed_time]

    valid_solvers = {"MCWF", "TJM"}
    if sim_params.solver not in valid_solvers:
        msg = f"Tomography requires solvers {valid_solvers}, got {sim_params.solver!r}."
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
    )


def construct_process_tensor(
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
) -> DenseComb | MPOComb:
    """Construct a process tensor via exhaustive discrete-basis tomography.

    This simulates **every** ``16^k`` discrete basis sequence and returns a comb directly:

    - ``return_type="dense"``: reconstruct and return a :class:`DenseComb`.
    - ``return_type="mpo"``: build and return an :class:`MPOComb`.
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
