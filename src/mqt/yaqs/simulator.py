# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
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

# ruff: noqa: E402
# ---------------------------------------------------------------------------
# 1) STANDARD/LIB IMPORTS (safe after thread-cap env is set)
# ---------------------------------------------------------------------------
import multiprocessing

# ---------------------------------------------------------------------------
# 0) IMPORTS
# Thread caps are NOT set at module level to allow single-trajectory
# simulations to use multi-threading via threadpoolctl.
# Thread limits are enforced in worker processes via _limit_worker_threads()
# and in backend calls via _call_backend() with threadpoolctl.
# ---------------------------------------------------------------------------
import os
from collections.abc import Callable, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    ProcessPoolExecutor,
    wait,
)
import concurrent
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

# Optional: extra control over threadpools inside worker processes.
# We keep references as optionals, set by a guarded import.
threadpool_limits: Callable[..., Any] | None
threadpool_info: Callable[[], Any] | None
try:
    from threadpoolctl import threadpool_info as _threadpool_info
    from threadpoolctl import threadpool_limits as _threadpool_limits
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
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 3) LOCAL IMPORTS
# ---------------------------------------------------------------------------
from .core.data_structures.networks import MPO
from .core.data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from concurrent.futures import Future

    import numpy as np
    from numpy.typing import NDArray

    from .core.data_structures.networks import MPS
    from .core.data_structures.noise_model import NoiseModel
    from qiskit.circuit import QuantumCircuit

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

    This function checks if the PYTEST_XDIST_WORKER environment variable is set. If so, it returns 1 to avoid
    nested parallelism during tests.
    Next, it checks if the SLURM_CPUS_ON_NODE environment variable is set (indicating a SLURM-managed cluster job).
    If so, it returns the number of CPUs specified by SLURM. Otherwise, it returns the total number of CPUs available
    on the machine as reported by multiprocessing.cpu_count().

    Returns:
        int: The number of available CPU cores for parallel execution.
    """
    # 0) Priority Override: YAQS_MAX_WORKERS
    if "YAQS_MAX_WORKERS" in os.environ:
        try:
            val = int(os.environ["YAQS_MAX_WORKERS"])
            if val > 0:
                return val
        except ValueError:
            pass

    # 1) Detect xdist: running inside a pytest worker?
    if os.environ.get("PYTEST_XDIST_WORKER", ""):
        return 1

    # 2) SLURM hints (explicit user/job request should win)
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

    # 3) Respect Linux affinity / cgroup limits if available
    fn = getattr(os, "sched_getaffinity", None)
    if fn is not None:
        try:
            sched_getaffinity = cast("Callable[[int], set[int]]", fn)
            n = len(sched_getaffinity(0))
            if n > 0:
                return n
        except OSError:
            # System call failed; fall through to next fallback
            pass

    # 4) Fallback
    count = 0
    try:
        count = os.cpu_count() or multiprocessing.cpu_count() or 1
    except (NotImplementedError, OSError):
        count = 1

    return count



# ---------------------------------------------------------------------------
# 6) WORKER INITIALIZER — cap threads inside each worker process
# When a worker starts, we:
#   - Set environment caps (no-ops if already set)
#   - Try to cap numexpr and MKL explicitly if present
#   - Optionally use threadpoolctl to cap vendored OpenMP pools (OpenBLAS, MKL)
#   - Initialize the worker-global context with large objects (e.g. MPS, NoiseModel)
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


# Global worker state (initialized once per process)
_WORKER_CTX: dict[str, Any] = {}


def _worker_init(payload: dict[str, Any], n_threads: int = 1) -> None:
    """Initialize the worker process state.

    This function is called once per worker process upon startup. It enforces
    thread limits for numerical libraries (BLAS, OpenMP, etc.) and populates
    the global `_WORKER_CTX` dictionary with shared simulation objects. This
    strategy avoids repeated pickling of large objects for every task.

    Args:
        payload: A dictionary containing large, read-only objects (e.g., MPS,
            MPO, NoiseModel, SimParams) to be stored in the global worker context.
        n_threads: The maximum number of threads allowed for this worker process.
            Defaults to 1 to prevent thread oversubscription.
    """
    # 1. Thread Capping
    _limit_worker_threads(n_threads)

    # 2. Context Initialization
    _WORKER_CTX.clear()
    _WORKER_CTX.update(payload)


def _limit_worker_threads(n_threads: int = 1) -> None:
    """Limit the number of threads used by numerical libraries in the current process.

    This helper sets environment variables (OMP_NUM_THREADS, MKL_NUM_THREADS, etc.)
    and calls runtime configuration functions for libraries like `numexpr`, `mkl`,
    and `threadpoolctl` to prevent thread oversubscription when running many
    worker processes in parallel.

    Args:
        n_threads: The maximum number of threads to allow. Defaults to 1.
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

    with contextlib.suppress(Exception):
        numba = importlib.import_module("numba")
        numba.set_num_threads(n_threads)

    if threadpool_limits is not None:
        with contextlib.suppress(Exception):
            threadpool_limits(limits=n_threads)

    if os.environ.get("YAQS_THREAD_DEBUG", "") == "1" and threadpool_info is not None:
        with contextlib.suppress(Exception):
            threadpool_info()



# ---------------------------------------------------------------------------
# 7) WORKER WRAPPERS
# These functions are pickled and sent to workers. They retrieve large objects
# from the global _WORKER_CTX instead of receiving them as arguments.
# ---------------------------------------------------------------------------
def _digital_strong_worker(traj_idx: int) -> Any:
    """Execute a single digital strong simulation trajectory.

    Retrieves the required simulation objects (initial state, noise model,
    parameters, circuit) from the global `_WORKER_CTX` and delegates to the
    `digital_tjm` backend.

    Args:
        traj_idx: The integer index of the trajectory to execute.

    Returns:
        Any: The result of the single-trajectory simulation (typically an
        observable trajectory or final state).
    """
    from .digital.digital_tjm import digital_tjm

    return digital_tjm(
        (
            traj_idx,
            _WORKER_CTX["initial_state"],
            _WORKER_CTX["noise_model"],
            _WORKER_CTX["sim_params"],
            _WORKER_CTX["operator"],
        )
    )


def _digital_weak_worker(traj_idx: int) -> dict[int, int]:
    """Execute a single digital weak simulation trajectory.

    Retrieves simulation objects from `_WORKER_CTX` and executes a 'shots=1'
    weak simulation using `digital_tjm`.

    Args:
        traj_idx: The integer index of the trajectory (effectively a shot index).

    Returns:
        dict[int, int]: A dictionary mapping outcome bitstrings (as integers)
        to counts (typically {outcome: 1} for a single shot).
    """
    from .digital.digital_tjm import digital_tjm

    return cast(
        "dict[int, int]",
        digital_tjm(
            (
                traj_idx,
                _WORKER_CTX["initial_state"],
                _WORKER_CTX["noise_model"],
                _WORKER_CTX["sim_params"],
                _WORKER_CTX["operator"],
            )
        ),
    )


def _analog_worker(traj_idx: int) -> Any:
    """Execute a single analog simulation trajectory (TJM or Lindblad).

    Retrieves the appropriate backend function and simulation arguments from
    `_WORKER_CTX` and executes the trajectory.

    Args:
        traj_idx: The integer index of the trajectory to execute.

    Returns:
        Any: The result of the simulation (typically observable values over time).
    """
    # backend is chosen in _run_analog and stored in context
    backend = _WORKER_CTX["backend"]
    return backend(
        (
            traj_idx,
            _WORKER_CTX["initial_state"],
            _WORKER_CTX["noise_model"],
            _WORKER_CTX["sim_params"],
            _WORKER_CTX["operator"],
        )
    )


def _mcwf_worker(traj_idx: int) -> Any:
    """Execute a single Monte Carlo Wavefunction (MCWF) trajectory.

    Retrieves the preprocessed MCWF context from `_WORKER_CTX` and executes
    the trajectory.

    Args:
        traj_idx: The integer index of the trajectory to execute.

    Returns:
        Any: The result of the MCWF trajectory.
    """
    from .analog.mcwf import mcwf

    return mcwf((traj_idx, _WORKER_CTX["ctx"]))


# ---------------------------------------------------------------------------
# 8) SAFETY WRAPPER FOR SERIAL BACKEND CALLS
# Wrap a single backend call in a context that (again) caps threadpools.
# This protects against libraries that spawn pools lazily during the call.
# ---------------------------------------------------------------------------
def _call_backend(backend: Callable[[Any], TRes], arg: Any, n_threads: int = 1) -> TRes:  # noqa: ANN401
    """Invoke a backend function under a strict temporary thread cap.

    Wraps a single backend call in a context that forces threadpool limits
    (if ``threadpoolctl`` is available). This ensures that even if a library
    lazily initializes its thread pool inside the backend call, it will still
    run single-threaded (or with the specified number of threads).

    Args:
        backend : The backend function to execute.
        arg : The argument to pass to the backend function.
        n_threads: The maximum number of threads to allow. Defaults to 1.

    Returns:
        TRes: The result returned by the backend function.

    Notes:
        - If ``threadpoolctl`` is not available, falls back to direct call.
        - If enforcing thread limits fails, falls back silently to direct call.
    """
    if threadpool_limits is not None:
        # Caps any pools entered/created within the context
        with contextlib.suppress(Exception), threadpool_limits(limits=n_threads):
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


def _run_backend_parallel(
    worker_fn: Callable[[int], TRes],
    *,
    payload: dict[str, Any] | None,
    n_jobs: int,
    max_workers: int,
    show_progress: bool = True,
    desc: str,
    max_retries: int = 10,
    retry_exceptions: tuple[type[BaseException], ...] = (CancelledError, TimeoutError, OSError),
) -> Iterator[tuple[int, TRes]]:
    """Execute backend calls in parallel with bounded submission and retry logic.

    This function manages the parallel execution of tasks using a `ProcessPoolExecutor`.
    refactored to prevent task flooding and memory exhaustion:
    1.  **Worker-Global State**: Uses `_worker_init` to initialize large objects
        once per worker, avoiding per-task pickling overhead.
    2.  **Bounded In-Flight**: Submits tasks in a queue-like manner, keeping
        only a limited number of futures active (2 * max_workers) at any time.

    Args:
        worker_fn: The worker function to execute. It must accept a single
            integer argument (the job index) and return a result of type `TRes`.
        payload: A dictionary of large objects (e.g., MPS, NoiseModel) to be
            initialized in the global worker context. passed to `_worker_init`.
        n_jobs: The total number of jobs to execute (indices 0 to n_jobs-1).
        max_workers: The maximum number of worker processes to use.
        show_progress: If True, displays a tqdm progress bar. Defaults to True.
        desc: The description string for the progress bar.
        max_retries: The maximum number of retry attempts for transient errors
            (e.g., TimeoutError). Defaults to 10.
        retry_exceptions: A tuple of exception types that trigger a retry.
            Defaults to (CancelledError, TimeoutError, OSError).

    Yields:
        tuple[int, TRes]: A tuple containing the job index and its result,
        yielded in the order of completion.
    """
    # Use a spawn context to avoid fork+OpenMP problems
    ctx = _spawn_context()

    # Bounded in-flight factor (keep 2-4x workers busy to hide latency)
    inflight_factor = 2
    max_inflight = max_workers * inflight_factor

    # Create a pool of worker processes with per-worker thread caps
    # and global context initialization
    with (
        ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(payload or {}, 1),  # enforce 1 thread per worker
        ) as ex,
        tqdm(total=n_jobs, desc=desc, ncols=80, disable=(not show_progress)) as pbar,
    ):
        # Retry bookkeeping per index
        retries = dict.fromkeys(range(n_jobs), 0)
        
        # In-flight tracking
        futures: dict[Future[TRes], int] = {}
        next_job_idx = 0

        def submit_job(idx: int) -> None:
            """Submit a job for the given index."""
            futures[ex.submit(worker_fn, idx)] = idx

        # Initial batch submission (up to max_inflight)
        while next_job_idx < n_jobs and len(futures) < max_inflight:
            submit_job(next_job_idx)
            next_job_idx += 1

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
                        submit_job(i)
                        continue
                    # Exceeded retry budget → propagate
                    raise
                
                # Yield in completion order and update progress
                yield i, res
                pbar.update(1)

                # Submit next job if available
                if next_job_idx < n_jobs:
                    submit_job(next_job_idx)
                    next_job_idx += 1


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
    # digital_tjm signature: (traj_idx, MPS, NoiseModel | None, StrongSimParams, QuantumCircuit) -> NDArray[np.float64]
    # We type as Any to keep ty happy without over-constraining element types.
    from .digital.digital_tjm import digital_tjm

    backend: Callable[[tuple[int, MPS, NoiseModel | None, StrongSimParams, QuantumCircuit]], Any] = digital_tjm

    # If there's no noise at all, we don't need multiple trajectories
    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        sim_params.num_traj = 1
    else:
        # With stochastic noise, returning a final state is ill-defined
        assert not sim_params.get_state, "Cannot return state in noisy circuit simulation due to stochastics."

    # If requested, count mid-measurement sampling barriers (optional feature)
    if sim_params.sample_layers:
        from qiskit.converters import circuit_to_dag

        dag = circuit_to_dag(operator)
        sim_params.num_mid_measurements = sum(
            1
            for n in dag.op_nodes()
            if n.op.name == "barrier" and str(getattr(n.op, "label", "")).strip().upper() == "SAMPLE_OBSERVABLES"
        )

    # Observables set up their own trajectory storage
    for observable in sim_params.sorted_observables:
        observable.initialize(sim_params)

    # Create worker-global payload
    payload: dict[str, Any] = {
        "initial_state": initial_state,
        "noise_model": noise_model,
        "sim_params": sim_params,
        "operator": operator,
    }

    if parallel and sim_params.num_traj > 1:
        # Reserve one logical CPU for the parent; use the rest for workers
        max_workers = max(1, available_cpus() - 1)
        # Submit task indices in parallel and stitch results back in place
        for i, result in _run_backend_parallel(
            worker_fn=_digital_strong_worker,
            payload=payload,
            n_jobs=sim_params.num_traj,
            max_workers=max_workers,
            show_progress=sim_params.show_progress,
            desc="Running trajectories",
            max_retries=10,
            retry_exceptions=(CancelledError, TimeoutError, OSError),
        ):
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                assert observable.trajectories is not None, "Trajectories should have been initialized"
                observable.trajectories[i] = result[obs_index]
    else:
        # Serial path (debugging/single-core/short runs)
        # Use all available cores for multithreading in serial mode
        n_threads = available_cpus()
        
        # Reconstruct args locally for serial execution
        args: list[tuple[int, MPS, NoiseModel | None, StrongSimParams, QuantumCircuit]] = [
            (i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)
        ]

        # Use tqdm if show_progress is True
        iterator = tqdm(args, desc="Running trajectories", ncols=80, disable=not sim_params.show_progress)
        
        for i, arg in enumerate(iterator):
            result = _call_backend(backend, arg, n_threads=n_threads)
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
        sim_params: Simulation parameters for weak simulation, including number of shots,
            trajectory count, and storage for measurements.
        noise_model: The noise model applied during simulation.
        parallel: Flag indicating whether to run trajectories in parallel.

    Raises:
        TypeError: If a measurement result is not of the expected type.
    """
    # digital_tjm returns a measurement outcome structure for weak sim
    from .digital.digital_tjm import digital_tjm

    backend: Callable[[tuple[int, MPS, NoiseModel | None, WeakSimParams, QuantumCircuit]], Any] = digital_tjm

    # Trajectory count policy
    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        sim_params.num_traj = 1
    else:
        # Map "shots" to "independent trajectories of length 1"
        sim_params.num_traj = sim_params.shots
        sim_params.shots = 1
        assert not sim_params.get_state, "Cannot return state in noisy circuit simulation due to stochastics."

    # Create worker-global payload
    payload: dict[str, Any] = {
        "initial_state": initial_state,
        "noise_model": noise_model,
        "sim_params": sim_params,
        "operator": operator,
    }

    if parallel and sim_params.num_traj > 1:
        max_workers = max(1, available_cpus() - 1)
        for i, result in _run_backend_parallel(
            worker_fn=_digital_weak_worker,
            payload=payload,
            n_jobs=sim_params.num_traj,
            max_workers=max_workers,
            show_progress=sim_params.show_progress,
            desc="Running trajectories",
            max_retries=10,
            retry_exceptions=(CancelledError, TimeoutError, OSError),
        ):
            # For weak sim, write the raw per-trajectory measurement structure
            if not isinstance(result, dict):
                msg = f"Expected measurement result to be dict[int, int], got {type(result).__name__}."
                raise TypeError(msg)
            sim_params.measurements[i] = cast("dict[int, int]", result)
    else:
        # Serial path
        # Use all available cores for multithreading in serial mode
        n_threads = available_cpus()
        
        # Reconstruct args locally
        args: list[tuple[int, MPS, NoiseModel | None, WeakSimParams, QuantumCircuit]] = [
            (i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)
        ]

        # Use tqdm if show_progress is True
        iterator = tqdm(args, desc="Running trajectories", ncols=80, disable=not sim_params.show_progress)

        for i, arg in enumerate(iterator):
            result = _call_backend(backend, arg, n_threads=n_threads)
            if not isinstance(result, dict):
                msg = f"Expected measurement result to be dict[int, int], got {type(result).__name__}."
                raise TypeError(msg)
            sim_params.measurements[i] = cast("dict[int, int]", result)

    # Reset shots back from trajectories
    if not (noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes)):
        sim_params.shots = sim_params.num_traj

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
        sim_params: Simulation parameters for analog simulation, including time step and evolution order.
        noise_model: The noise model applied during simulation.
        parallel: Flag indicating whether to run trajectories in parallel.
    """
    # Choose integrator order (1 or 2) for the analog TJM backend
    from .analog.analog_tjm import analog_tjm_1, analog_tjm_2
    from .analog.lindblad import lindblad
    from .analog.mcwf import mcwf, preprocess_mcwf

    backend: Callable[[Any], NDArray[np.float64]]
    if sim_params.solver == "Lindblad":
        backend = lindblad
    elif sim_params.solver == "MCWF":
        backend = mcwf
    elif sim_params.order == 1:
        backend = analog_tjm_1
    else:
        backend = analog_tjm_2

    # If no noise, determinism implies a single trajectory suffices
    if (
        noise_model is None
        or all(proc["strength"] == 0 for proc in noise_model.processes)
        or sim_params.solver == "Lindblad"
    ):
        sim_params.num_traj = 1
    else:
        # With stochastic noise, returning final state is ill-defined
        assert not sim_params.get_state, "Cannot return state in noisy analog simulation due to stochastics."

    # Observable storage preparation
    for observable in sim_params.sorted_observables:
        observable.initialize(sim_params)

    # Argument bundles per trajectory
    # args: list[Any]
    payload: dict[str, Any]
    worker_fn: Callable[[int], Any]

    if sim_params.solver == "MCWF":
        # Optimization: Pre-compute dense operators once
        ctx = preprocess_mcwf(initial_state, operator, noise_model, sim_params)
        # args = [(i, ctx) for i in range(sim_params.num_traj)]
        payload = {"ctx": ctx}
        worker_fn = _mcwf_worker
    else:
        # Standard TJM/Lindblad arguments
        # args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)]
        payload = {
            "initial_state": initial_state,
            "noise_model": noise_model,
            "sim_params": sim_params,
            "operator": operator,
            "backend": backend,
        }
        worker_fn = _analog_worker

    if parallel and sim_params.num_traj > 1:
        max_workers = max(1, available_cpus() - 1)
        for i, result in _run_backend_parallel(
            worker_fn=worker_fn,
            payload=payload,
            n_jobs=sim_params.num_traj,
            max_workers=max_workers,
            show_progress=sim_params.show_progress,
            desc="Running trajectories",
            max_retries=10,
            retry_exceptions=(CancelledError, TimeoutError, OSError),
        ):
            # Stitch each observable's i-th trajectory back into place
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                assert observable.trajectories is not None, "Trajectories should have been initialized"
                observable.trajectories[i] = result[obs_index]
    else:
        # Serial fallback
        # Use all available cores for multithreading in serial mode
        n_threads = available_cpus()
        
        # Reconstruct args locally for serial execution
        args: list[Any]
        if sim_params.solver == "MCWF":
            # For MCWF serial, we still use the pre-computed ctx
            # ctx is already in local scope from above if block
            args = [(i, ctx) for i in range(sim_params.num_traj)]
        else:
             args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)]

        # Use tqdm if show_progress is True
        iterator = tqdm(args, desc="Running trajectories", ncols=80, disable=not sim_params.show_progress)

        for i, arg in enumerate(iterator):
            result = _call_backend(backend, arg, n_threads=n_threads)
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
        noise_model: The noise model to apply during simulation. If provided, it is sampled once
            at the beginning of the run to generate a concrete noise realization (static disorder).
            The sampled noise model is then saved to `sim_params.noise_model`.
        parallel: Whether to run trajectories in parallel. Defaults to True.

    """
    # Ensure the state is in B-normalization before any evolution
    initial_state.normalize("B")

    # Sample a concrete noise model once for this run (static disorder)
    if noise_model is not None:
        noise_model = noise_model.sample()
    sim_params.noise_model = noise_model
    
    if isinstance(sim_params, (StrongSimParams, WeakSimParams)):
        from qiskit.circuit import QuantumCircuit
        
        assert isinstance(operator, QuantumCircuit)
        _run_circuit(initial_state, operator, sim_params, noise_model, parallel=parallel)
    elif isinstance(sim_params, AnalogSimParams):
        assert isinstance(operator, MPO)
        _run_analog(initial_state, operator, sim_params, noise_model, parallel=parallel)
