# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""High-level simulator module for using YAQS.

This module implements the common simulation routine for both circuit-based and Hamiltonian (analog) simulations.
It provides functions to run simulation trajectories in parallel using an MPS representation of the quantum state.
Depending on the type of simulation parameters provided (WeakSimParams, StrongSimParams, or AnalogSimParams),
the simulation is dispatched to the appropriate backend:
  - For circuit simulations, a QuantumCircuit is used and processed via the _run_circuit function.
  - For analog simulations, an MPO is used to represent the Hamiltonian and processed via the _run_analog function.

The module supports both strong and weak simulation schemes, including functionality for:
  - Initializing the state (MPS) to a canonical form (B normalized).
  - Running trajectories with noise (using a provided NoiseModel) and aggregating results.
  - Parallel execution of trajectories using a ProcessPoolExecutor with progress reporting via tqdm.

All simulation results (e.g., observables, measurements) are aggregated and returned as part of the simulation process.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 0) THREAD CAPS — must be set BEFORE importing numpy/scipy/qiskit/etc.
# We set conservative defaults to avoid worker processes each spawning a
# large number of BLAS/OpenMP threads (oversubscription → slowdown/crashes).
# Users can still override these by exporting the vars before import.
# ---------------------------------------------------------------------------
import os

for _k, _v in {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",  # harmless on Linux; relevant on macOS
    "BLIS_NUM_THREADS": "1",
}.items():
    os.environ.setdefault(_k, _v)

# ruff: noqa: E402

# ---------------------------------------------------------------------------
# 1) STANDARD/LIB IMPORTS (safe after thread-cap env is set)
# ---------------------------------------------------------------------------
import multiprocessing
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    Future,
    ProcessPoolExecutor,
    wait,
)
from typing import TYPE_CHECKING, Any, TypeVar

# Optional: extra control over threadpools inside worker processes.
# We keep references as optionals, set by a guarded import.
threadpool_limits: Callable[..., Any] | None
threadpool_info: Callable[[], Any] | None
try:
    from threadpoolctl import (  # type: ignore[import-not-found]
        threadpool_info as _threadpool_info,
    )
    from threadpoolctl import (
        threadpool_limits as _threadpool_limits,
    )
except ImportError:  # pragma: no cover - optional dependency
    threadpool_limits = None
    threadpool_info = None
else:
    threadpool_limits = _threadpool_limits
    threadpool_info = _threadpool_info

import contextlib
import copy
import importlib

# ---------------------------------------------------------------------------
# 2) THIRD-PARTY IMPORTS
# ---------------------------------------------------------------------------
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 3) LOCAL IMPORTS
# ---------------------------------------------------------------------------
from .analog.analog_tjm import analog_tjm_1, analog_tjm_2
from .core.data_structures.networks import MPO
from .core.data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams
from .digital.digital_tjm import digital_tjm

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from .core.data_structures.networks import MPS
    from .core.data_structures.noise_model import NoiseModel

__all__ = ["available_cpus", "run"]  # public API of this module

# ---------------------------------------------------------------------------
# 4) TYPE VARS FOR GENERIC PARALLEL RUNNERS
# ---------------------------------------------------------------------------
TArg = TypeVar("TArg")
TRes = TypeVar("TRes")


# ---------------------------------------------------------------------------
# 5) CPU DISCOVERY — be respectful of cgroups/SLURM/taskset limits.
# On Linux, processes may be constrained (containers, sched_setaffinity,
# SLURM). We try to detect the actual number of logical CPUs visible.
# ---------------------------------------------------------------------------
def available_cpus() -> int:
    """Determine the number of available CPU cores for parallel execution.

    This function checks if the SLURM_CPUS_ON_NODE environment variable is set (indicating a SLURM-managed cluster job).
    If so, it returns the number of CPUs specified by SLURM. Otherwise, it returns the total number of CPUs available
    on the machine as reported by multiprocessing.cpu_count().

    Returns:
        int: The number of available CPU cores for parallel execution.
    """
    # 1) SLURM hints first (explicit user/job request should win)
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(var, "").strip()
        if value:
            try:
                n = int(value)
                if n > 0:
                    return n
            except ValueError:
                # Ignore malformed values and continue
                pass

    # 2) Respect Linux affinity / cgroup limits if available
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        pass

    # 3) Fallback
    return multiprocessing.cpu_count() or 1


# ---------------------------------------------------------------------------
# 6) WORKER INITIALIZER — cap threads inside each worker process
# When a worker starts, we:
#   - Set environment caps (no-ops if already set)
#   - Try to cap numexpr and MKL explicitly if present
#   - Optionally use threadpoolctl to cap vendored OpenMP pools (OpenBLAS, MKL)
# ---------------------------------------------------------------------------
THREAD_ENV_VARS: dict[str, str] = {
    # OpenMP default thread count (covers any library compiled with OpenMP,
    # e.g., MKL, SciPy routines, numba-parallel, some Qiskit internals).
    "OMP_NUM_THREADS": "1",

    # OpenBLAS thread pool size (most Linux NumPy/SciPy wheels link to OpenBLAS).
    "OPENBLAS_NUM_THREADS": "1",

    # Intel MKL thread pool size (common in conda distributions of NumPy/SciPy).
    "MKL_NUM_THREADS": "1",

    # NumExpr parallelism (used by pandas.eval/query and some NumPy expressions).
    "NUMEXPR_NUM_THREADS": "1",

    # Apple vecLib/Accelerate framework (only relevant on macOS).
    "VECLIB_MAXIMUM_THREADS": "1",

    # BLIS BLAS implementation (used in some NumPy builds instead of OpenBLAS/MKL).
    "BLIS_NUM_THREADS": "1",
}


def _limit_worker_threads(n_threads: int = 1) -> None:
    """Initializer for worker processes to cap BLAS/OpenMP thread usage.

    This function is called once at worker process startup. It re-applies
    environment variable caps (e.g. ``OMP_NUM_THREADS``) inside the child
    process, and makes best-effort attempts to explicitly cap common math
    libraries that may spawn their own threadpools.

    Args:
        n_threads : Maximum number of threads per library to
            allow. Defaults to 1.

    Notes:
        - Affects OpenMP-based libraries, OpenBLAS, MKL, BLIS, NumExpr.
        - Uses ``threadpoolctl`` if available to force limits on vendored pools.
        - Has no effect if libraries are not present or ignore thread caps.
    """
    for k in THREAD_ENV_VARS:
        os.environ.setdefault(k, str(n_threads))
    os.environ.setdefault("OMP_DYNAMIC", "FALSE")
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")

    # Import optional libs safely without inline `import` statements
    with contextlib.suppress(Exception):
        numexpr = importlib.import_module("numexpr")
        numexpr.set_num_threads(n_threads)

    with contextlib.suppress(Exception):
        mkl = importlib.import_module("mkl")
        mkl.set_num_threads(n_threads)

    if threadpool_limits is not None:
        with contextlib.suppress(Exception):
            threadpool_limits(limits=n_threads)

    if os.environ.get("YAQS_THREAD_DEBUG", "") == "1" and threadpool_info is not None:
        with contextlib.suppress(Exception):
            threadpool_info()


# ---------------------------------------------------------------------------
# 7) SAFETY WRAPPER FOR BACKEND CALLS
# Wrap a single backend call in a context that (again) caps threadpools.
# This protects against libraries that spawn pools lazily during the call.
# ---------------------------------------------------------------------------
def _call_backend(backend: Callable[[TArg], TRes], arg: TArg) -> TRes:
    """Invoke a backend function under a strict temporary thread cap.

    Wraps a single backend call in a context that forces threadpool limits
    (if ``threadpoolctl`` is available). This ensures that even if a library
    lazily initializes its thread pool inside the backend call, it will still
    run single-threaded.

    Args:
        backend : The backend function to execute.
        arg : The argument to pass to the backend function.

    Returns:
        TRes: The result returned by the backend function.

    Notes:
        - If ``threadpoolctl`` is not available, falls back to direct call.
        - If enforcing thread limits fails, falls back silently to direct call.
    """
    if threadpool_limits is not None:
        # Caps any pools entered/created within the context
        with contextlib.suppress(Exception), threadpool_limits(limits=1):
            return backend(arg)
    # If threadpoolctl fails for any reason, fallback to direct call
    return backend(arg)


# ---------------------------------------------------------------------------
# 8) MULTIPROCESS "spawn" CONTEXT
# On Linux, using "fork" with heavy numerical libs can hang/crash due to
# non-fork-safe OpenMP/BLAS state. "spawn" is the robust cross-platform choice.
# ---------------------------------------------------------------------------
def _spawn_context() -> multiprocessing.context.BaseContext:
    """Return a multiprocessing context using the 'spawn' start method.

    The 'spawn' start method launches a fresh Python interpreter for each
    worker process. This is safer than 'fork' when working with OpenMP/BLAS
    libraries, which may leave non-fork-safe state (e.g., initialized thread
    pools) in the parent process.

    Returns:
        multiprocessing.context.BaseContext: A multiprocessing context
        configured to use 'spawn'.

    Notes:
        - On Linux, 'fork' is the default but can cause deadlocks/crashes
          with numerical libraries.
        - 'spawn' is slower to start but cross-platform safe.
    """
    return multiprocessing.get_context("spawn")


# ---------------------------------------------------------------------------
# 9) GENERIC PARALLEL EXECUTOR WITH RETRIES
# Submits one future per argument, tracks which index each future maps to,
# retries a small set of transient exceptions, and yields results in completion
# order while updating a tqdm progress bar.
# ---------------------------------------------------------------------------
def _run_backend_parallel(
    backend: Callable[[TArg], TRes],
    args: Sequence[TArg],
    *,
    max_workers: int,
    desc: str,
    total: int | None = None,
    max_retries: int = 10,
    retry_exceptions: tuple[type[BaseException], ...] = (CancelledError, TimeoutError, OSError),
) -> Iterator[tuple[int, TRes]]:
    """Execute backend calls in parallel with retry logic and progress reporting.

    Submits one future per argument to a worker pool, enforces per-worker
    thread caps, and yields results in completion order. Transient failures
    (e.g. worker cancellations, OS errors) are retried up to a fixed limit.

    Args:
        backend : The backend function to call.
        args: Sequence of argument objects, one per job.
        max_workers: Maximum number of worker processes.
        desc: Description to display in the tqdm progress bar.
        total: Expected total jobs for progress bar.
            Defaults to ``len(args)`` if None.
        max_retries: Maximum retry attempts per job.
            Defaults to 10.
        retry_exceptions:
            Exception types that should trigger a retry. Defaults to
            (CancelledError, TimeoutError, OSError).

    Yields:
        tuple[int, TRes]: A pair ``(i, result)``, where ``i`` is the
        index of the argument in ``args`` and ``result`` is the backend's
        return value.

    Notes:
        - Uses the 'spawn' start method for process safety with BLAS/OpenMP.
        - Caps threads per worker via the ``_limit_worker_threads`` initializer.
        - Retries only the specified transient exception types; others
          propagate immediately.
    """
    # Default progress bar length to number of arguments
    total = len(args) if total is None else total
    # Use a spawn context to avoid fork+OpenMP problems
    ctx = _spawn_context()

    # Create a pool of worker processes with per-worker thread caps
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_limit_worker_threads,
        initargs=(1,),  # enforce 1 thread per worker
    ) as ex, tqdm(total=total, desc=desc, ncols=80) as pbar:
        # Retry bookkeeping per index
        retries = dict.fromkeys(range(len(args)), 0)
        # Submit all tasks upfront; map Future -> index
        futures: dict[Future[TRes], int] = {ex.submit(backend, args[i]): i for i in range(len(args))}

        # Drain as futures complete
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                i = futures.pop(fut)
                try:
                    res = fut.result()
                except retry_exceptions:
                    # Retry a bounded number of times on selected transient errors
                    if retries[i] < max_retries:
                        retries[i] += 1
                        futures[ex.submit(backend, args[i])] = i
                        continue
                    # Exceeded retry budget → propagate
                    raise
                # Yield in completion order and update progress
                yield i, res
                pbar.update(1)


# ---------------------------------------------------------------------------
# 10) STRONG SIMULATION (circuit): returns observable trajectories
# - If noise is zero/absent → only 1 trajectory (deterministic).
# - If noise is present → multiple trajectories; cannot request final state.
# - Optionally count SAMPLE_OBSERVABLES layers (barriers with specific label).
# ---------------------------------------------------------------------------
def _run_strong_sim(
    initial_state: MPS,
    operator: QuantumCircuit,
    sim_params: StrongSimParams,
    noise_model: NoiseModel | None,
    *,
    parallel: bool,
) -> None:
    """Run strong simulation trajectories for a quantum circuit using a strong simulation scheme.

    This function executes circuit-based simulation trajectories using the 'digital_tjm' backend.
    If the noise model is absent or its strengths are all zero, only a single trajectory is executed.
    For each observable in sim_params.sorted_observables, the function initializes the observable,
    runs the simulation trajectories (in parallel if specified), and aggregates the results.

    Args:
        initial_state: The initial system state as an MPS.
        operator: The quantum circuit representing the operation to simulate.
        sim_params: Simulation parameters for strong simulation,
                                      including the number of trajectories (num_traj),
                                      time step (dt), and sorted observables.
        noise_model: The noise model applied during simulation.
        parallel: Flag indicating whether to run trajectories in parallel.
    """
    # digital_tjm signature: (traj_idx, MPS, NoiseModel|None, StrongSimParams, QuantumCircuit) -> list[obs_results]
    # We type as list[object] to keep mypy happy without over-constraining element types.
    backend: Callable[[tuple[int, MPS, NoiseModel | None, StrongSimParams, QuantumCircuit]], list[object]] = digital_tjm

    # If there's no noise at all, we don't need multiple trajectories
    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        sim_params.num_traj = 1
    else:
        # With stochastic noise, returning a final state is ill-defined
        assert not sim_params.get_state, "Cannot return state in noisy circuit simulation due to stochastics."

    # If requested, count mid-measurement sampling barriers (optional feature)
    if sim_params.sample_layers:
        dag = circuit_to_dag(operator)
        sim_params.num_mid_measurements = sum(
            1
            for n in dag.op_nodes()
            if n.op.name == "barrier" and str(getattr(n.op, "label", "")).strip().upper() == "SAMPLE_OBSERVABLES"
        )

    # Observables set up their own trajectory storage
    for observable in sim_params.sorted_observables:
        observable.initialize(sim_params)

    # Pack per-trajectory argument tuples (index used to place results later)
    args: list[tuple[int, MPS, NoiseModel | None, StrongSimParams, QuantumCircuit]] = [
        (i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)
    ]

    if parallel and sim_params.num_traj > 1:
        # Reserve one logical CPU for the parent; use the rest for workers
        max_workers = max(1, available_cpus() - 1)
        # Submit all trajectories in parallel and stitch results back in place
        for i, result in _run_backend_parallel(
            backend=backend,
            args=args,
            max_workers=max_workers,
            desc="Running trajectories",
            total=sim_params.num_traj,
            max_retries=10,
            retry_exceptions=(CancelledError, TimeoutError, OSError),
        ):
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                assert observable.trajectories is not None, "Trajectories should have been initialized"
                observable.trajectories[i] = result[obs_index]
    else:
        # Serial path (debugging/single-core/short runs)
        for i, arg in enumerate(args):
            result = _call_backend(backend, arg)
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                assert observable.trajectories is not None, "Trajectories should have been initialized"
                observable.trajectories[i] = result[obs_index]

    # Reduce per-trajectory results into final arrays/statistics per observable
    sim_params.aggregate_trajectories()


# ---------------------------------------------------------------------------
# 11) WEAK SIMULATION (circuit): returns measurement results per trajectory
# - With noise: trajectories = shots; we set shots=1 so each trajectory
#   measures once and we aggregate externally.
# ---------------------------------------------------------------------------
def _run_weak_sim(
    initial_state: MPS,
    operator: QuantumCircuit,
    sim_params: WeakSimParams,
    noise_model: NoiseModel | None,
    *,
    parallel: bool,
) -> None:
    """Run weak simulation trajectories for a quantum circuit using a weak simulation scheme.

    This function executes circuit-based simulation trajectories using the 'digital_tjm' backend
    in weak simulation mode. In this mode, the outputs are raw measurement results rather than
    observable expectation values. If the noise model is absent or its strengths are all zero,
    only a single trajectory is executed. If noise is present, the number of trajectories is set
    equal to the number of shots, and each trajectory corresponds to one measurement sample
    (with sim_params.shots forced to 1 internally).

    The trajectories are executed (in parallel if specified) and the measurement results
    are aggregated into the requested statistics or histograms.

    Args:
        initial_state : The initial system state as an MPS.
        operator: The quantum circuit representing the operation to simulate.
        sim_params: Simulation parameters for weak simulation,
                                    including number of shots, trajectory count,
                                    and storage for measurements.
        noise_model: The noise model applied during simulation.
        parallel: Flag indicating whether to run trajectories in parallel.
    """
    # digital_tjm returns a measurement outcome structure for weak sim
    backend: Callable[[tuple[int, MPS, NoiseModel | None, WeakSimParams, QuantumCircuit]], dict[int, int]] = digital_tjm

    # Trajectory count policy
    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        sim_params.num_traj = 1
    else:
        # Map "shots" to "independent trajectories of length 1"
        sim_params.num_traj = sim_params.shots
        sim_params.shots = 1
        assert not sim_params.get_state, "Cannot return state in noisy circuit simulation due to stochastics."

    # Bundle args for each trajectory
    args: list[tuple[int, MPS, NoiseModel | None, WeakSimParams, QuantumCircuit]] = [
        (i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)
    ]

    if parallel and sim_params.num_traj > 1:
        max_workers = max(1, available_cpus() - 1)
        for i, result in _run_backend_parallel(
            backend=backend,
            args=args,
            max_workers=max_workers,
            desc="Running trajectories",
            total=sim_params.num_traj,
            max_retries=10,
            retry_exceptions=(CancelledError, TimeoutError, OSError),
        ):
            # For weak sim, write the raw per-trajectory measurement structure
            sim_params.measurements[i] = result
    else:
        # Serial path
        for i, arg in enumerate(args):
            result = _call_backend(backend, arg)
            sim_params.measurements[i] = result

    # Aggregate individual measurements into the requested statistics/histograms
    sim_params.aggregate_measurements()


# ---------------------------------------------------------------------------
# 12) CIRCUIT DISPATCHER — reverse bits for internal convention, then route
#     to strong or weak simulation path based on the sim params type.
# ---------------------------------------------------------------------------
def _run_circuit(
    initial_state: MPS,
    operator: QuantumCircuit,
    sim_params: WeakSimParams | StrongSimParams,
    noise_model: NoiseModel | None,
    *,
    parallel: bool,
) -> None:
    """Run circuit-based simulation trajectories.

    This function validates that the number of qubits in the quantum circuit matches the length of the MPS,
    reverses the bit order of the circuit, and dispatches the simulation to the appropriate backend based on
    whether the simulation parameters indicate strong or weak simulation.

    Args:
        initial_state: The initial system state as an MPS.
        operator: The quantum circuit to simulate.
        sim_params: Simulation parameters for circuit simulation.
        noise_model: The noise model applied during simulation.
        parallel: Flag indicating whether to run trajectories in parallel.


    """
    # Sanity check: MPS length must equal circuit qubit count
    assert initial_state.length == operator.num_qubits, "State and circuit qubit counts do not match."
    # Internal convention expects qubit order reversed (if applicable)
    operator = copy.deepcopy(operator.reverse_bits())

    if isinstance(sim_params, StrongSimParams):
        _run_strong_sim(initial_state, operator, sim_params, noise_model, parallel=parallel)
    elif isinstance(sim_params, WeakSimParams):
        _run_weak_sim(initial_state, operator, sim_params, noise_model, parallel=parallel)


# ---------------------------------------------------------------------------
# 13) ANALOG (HAMILTONIAN) SIMULATION — similar to strong sim:
#     choose 1st/2nd-order integrator backend, run trajectories, collect
#     observable trajectories, and aggregate.
# ---------------------------------------------------------------------------
def _run_analog(
    initial_state: MPS,
    operator: MPO,
    sim_params: AnalogSimParams,
    noise_model: NoiseModel | None,
    *,
    parallel: bool,
) -> None:
    """Run analog simulation trajectories for Hamiltonian evolution.

    This function selects the appropriate analog simulation backend based on sim_params.order
    (either one-site or two-site evolution) and runs the simulation trajectories for the given Hamiltonian
    (represented as an MPO). The trajectories are executed (in parallel if specified) and the results are aggregated.

    Args:
        initial_state: The initial system state as an MPS.
        operator: The Hamiltonian operator represented as an MPO.
        sim_params: Simulation parameters for analog simulation,
                                       including time step and evolution order.
        noise_model: The noise model applied during simulation.
        parallel: Flag indicating whether to run trajectories in parallel.


    """
    # Choose integrator order (1 or 2) for the analog TJM backend
    backend: Callable[[tuple[int, MPS, NoiseModel | None, AnalogSimParams, MPO]], list[object]] = (
        analog_tjm_1 if sim_params.order == 1 else analog_tjm_2
    )

    # If no noise, determinism implies a single trajectory suffices
    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        sim_params.num_traj = 1
    else:
        # With stochastic noise, returning final state is ill-defined
        assert not sim_params.get_state, "Cannot return state in noisy analog simulation due to stochastics."

    # Observable storage preparation
    for observable in sim_params.sorted_observables:
        observable.initialize(sim_params)

    # Argument bundles per trajectory
    args: list[tuple[int, MPS, NoiseModel | None, AnalogSimParams, MPO]] = [
        (i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)
    ]

    if parallel and sim_params.num_traj > 1:
        max_workers = max(1, available_cpus() - 1)
        for i, result in _run_backend_parallel(
            backend=backend,
            args=args,
            max_workers=max_workers,
            desc="Running trajectories",
            total=sim_params.num_traj,
            max_retries=10,
            retry_exceptions=(CancelledError, TimeoutError, OSError),
        ):
            # Stitch each observable's i-th trajectory back into place
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                assert observable.trajectories is not None, "Trajectories should have been initialized"
                observable.trajectories[i] = result[obs_index]
    else:
        # Serial fallback
        for i, arg in enumerate(args):
            result = _call_backend(backend, arg)
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                assert observable.trajectories is not None, "Trajectories should have been initialized"
                observable.trajectories[i] = result[obs_index]

    # Aggregate per-trajectory data into final arrays/statistics
    sim_params.aggregate_trajectories()


# ---------------------------------------------------------------------------
# 14) PUBLIC ENTRY POINT — normalize MPS to B-canonical, then dispatch to
#     circuit or analog engines based on sim_params type.
# ---------------------------------------------------------------------------
def run(
    initial_state: MPS,
    operator: MPO | QuantumCircuit,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    noise_model: NoiseModel | None = None,
    *,
    parallel: bool = True,
) -> None:
    """Execute the common simulation routine for both circuit and Hamiltonian simulations.

    This function first normalizes the initial state (MPS) to B normalization, then dispatches the simulation
    to the appropriate backend based on the type of simulation parameters provided. For circuit-based simulations,
    the operator must be a QuantumCircuit; for Hamiltonian simulations, the operator must be an MPO.

    Args:
        initial_state: The initial state of the system as an MPS. Must be B normalized.
        operator: The operator representing the evolution; an MPO for analog simulations
            or a QuantumCircuit for circuit simulations.
        sim_params: Simulation parameters specifying
                                                                         the simulation mode and settings.
        noise_model: The noise model to apply during simulation.
        parallel: Whether to run trajectories in parallel. Defaults to True.

    """
    # Ensure the state is in B-normalization before any evolution
    initial_state.normalize("B")

    if isinstance(sim_params, (StrongSimParams, WeakSimParams)):
        assert isinstance(operator, QuantumCircuit)
        _run_circuit(initial_state, operator, sim_params, noise_model, parallel=parallel)
    elif isinstance(sim_params, AnalogSimParams):
        assert isinstance(operator, MPO)
        _run_analog(initial_state, operator, sim_params, noise_model, parallel=parallel)
