# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""High-level :class:`Simulator` class for YAQS.

This module implements the common simulation routine for Hamiltonian (analog) and
circuit-based simulations as the public :class:`Simulator` class. Analog evolution is
the primary mode supported by YAQS; circuit (strong/weak) simulation reuses the same
execution machinery.

The :class:`Simulator` class owns the execution-side configuration (parallel vs.
serial execution, worker count, progress reporting, multiprocessing context, and
retry policy), while the physics inputs are passed to :meth:`Simulator.run` as a
:class:`~mqt.yaqs.core.data_structures.state.State` and either a
:class:`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` (analog) or a
:class:`~qiskit.circuit.QuantumCircuit` (digital) together with a simulation
parameter object. Depending on the type of simulation parameters provided
(``AnalogSimParams``, ``StrongSimParams``, or ``WeakSimParams``), the simulation is
dispatched to the appropriate backend:

  - For analog simulations, a ``Hamiltonian`` is validated and materialized for the
    selected state representation, then processed via the analog dispatch. Passing a
    ``list[State]`` triggers the deterministic unitary ensemble path.
  - For circuit simulations, a ``QuantumCircuit`` is used and processed via the
    circuit dispatch (strong for observables/trajectories, weak for measurement
    counts).

The module supports analog (TJM / MCWF / Lindblad / unitary ensemble) and digital
(strong / weak) simulation, including functionality for:

  - Initializing the state (``State``) to a canonical form (B normalized).
  - Running trajectories with noise (using a provided ``NoiseModel``) and aggregating results.
  - Parallel execution of trajectories using a ``ProcessPoolExecutor`` with progress reporting via tqdm.

:meth:`Simulator.run` returns a :class:`~mqt.yaqs.core.data_structures.result.Result`
holding every simulation output (aggregated expectation values, per-trajectory data,
shared time grid, optional output state, measurement counts, and the sampled noise
model). The ``*SimParams`` object passed in is never mutated; ``Result.sim_params``
references it unchanged.
"""

from __future__ import annotations

import contextlib
import copy
import importlib

# ruff: noqa: E402
# ---------------------------------------------------------------------------
# 0) IMPORTS
# Thread caps are NOT set at module level to allow single-trajectory
# simulations to use multi-threading via threadpoolctl.
# Thread limits are enforced in worker processes via limit_worker_threads()
# and in backend calls via _call_backend() with threadpoolctl.
# ---------------------------------------------------------------------------
from concurrent.futures import (
    FIRST_COMPLETED,
    CancelledError,
    ProcessPoolExecutor,
    wait,
)
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from .core.data_structures.hamiltonian import Representation
    from .core.data_structures.mpo import MPO
    from .core.data_structures.mps import MPS

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


# ---------------------------------------------------------------------------
# 2) THIRD-PARTY IMPORTS
# ---------------------------------------------------------------------------
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 3) LOCAL IMPORTS
# ---------------------------------------------------------------------------
from .core.data_structures.hamiltonian import Hamiltonian
from .core.data_structures.result import (
    Result,
    aggregate_counts,
    aggregate_diagnostics,
    aggregate_trajectories,
    allocate_diagnostic_buffers,
    allocate_observable_buffers,
)
from .core.data_structures.simulation_parameters import (
    AnalogSimParams,
    StrongSimParams,
    WeakSimParams,
    _prepare_observable_ordering,
)
from .core.data_structures.state import State
from .parallel_utils import MPContext, available_cpus, get_parallel_context, limit_worker_threads

if TYPE_CHECKING:
    from collections.abc import Iterator
    from concurrent.futures import Future

    from numpy.typing import NDArray

    from .core.data_structures.noise_model import NoiseModel

from pathlib import Path

from .analog.analog_tjm import analog_tjm_1, analog_tjm_2
from .analog.ensemble import ensemble_member_worker
from .analog.lindblad import lindblad_evolve, preprocess_lindblad
from .analog.mcwf import mcwf, preprocess_mcwf
from .digital.digital_tjm import digital_tjm
from .digital.utils.qasm_utils import load_circuit

__all__ = ["Simulator", "available_cpus", "run_backend_parallel"]


# ---------------------------------------------------------------------------
# 4) TYPE VARS FOR GENERIC PARALLEL RUNNERS
# ---------------------------------------------------------------------------
TArg = TypeVar("TArg")
TRes = TypeVar("TRes")

# Backward-compatible alias for tests and docs that import the private name.
_get_parallel_context = get_parallel_context


# ---------------------------------------------------------------------------
# 5) WORKER INITIALIZER — cap threads inside each worker process
# When a worker starts, we:
#   - Set environment caps (no-ops if already set)
#   - Try to cap numexpr and MKL explicitly if present
#   - Optionally use threadpoolctl to cap vendored OpenMP pools (OpenBLAS, MKL)
#   - Initialize the worker-global context with large objects (e.g. State, NoiseModel)
# ---------------------------------------------------------------------------
# Global worker state (initialized once per process)
WORKER_CTX: dict[str, Any] = {}


def worker_init(payload: dict[str, Any], n_threads: int = 1) -> None:
    """Initialize the worker process state.

    This function is called once per worker process upon startup. It enforces
    thread limits for numerical libraries (BLAS, OpenMP, etc.) and populates
    the global `WORKER_CTX` dictionary with shared simulation objects. This
    strategy avoids repeated pickling of large objects for every task.

    Args:
        payload: A dictionary containing large, read-only objects (e.g., State,
            MPO, NoiseModel, SimParams) to be stored in the global worker context.
        n_threads: The maximum number of threads allowed for this worker process.
            Defaults to 1 to prevent thread oversubscription.
    """
    # 1. Thread Capping
    limit_worker_threads(n_threads)

    # 2. Context Initialization
    WORKER_CTX.clear()
    WORKER_CTX.update(payload)

    # 3. Numba Threading (Runtime)
    # Some Numba versions ignore env vars if imported before.
    try:
        import numba  # noqa: PLC0415

        numba.set_num_threads(n_threads)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 6) WORKER WRAPPERS
# These functions are pickled and sent to workers. They retrieve large objects
# from the global _WORKER_CTX instead of receiving them as arguments. Analog
# workers come first (primary simulation mode), followed by the digital
# strong/weak workers, with the unitary ensemble worker last.
# ---------------------------------------------------------------------------
def _analog_worker(traj_idx: int) -> tuple[NDArray[np.float64], NDArray[np.float64] | None, MPS | None]:
    """Execute a single analog simulation trajectory (TJM or Lindblad).

    Retrieves the appropriate backend function and simulation arguments from
    `WORKER_CTX` and executes the trajectory.

    Args:
        traj_idx: The integer index of the trajectory to execute.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64] | None, MPS | None]:
            Observable data, optional diagnostics, and optional final MPS.
    """
    # backend is chosen by Simulator._run_analog and stored in context
    backend = WORKER_CTX["backend"]
    return backend((
        traj_idx,
        WORKER_CTX["initial_state"],
        WORKER_CTX["noise_model"],
        WORKER_CTX["sim_params"],
        WORKER_CTX["operator"],
    ))


def _mcwf_worker(traj_idx: int) -> tuple[NDArray[np.float64], None, np.ndarray | None]:
    """Execute a single Monte Carlo Wavefunction (MCWF) trajectory.

    Retrieves the preprocessed MCWF context from `WORKER_CTX` and executes
    the trajectory.

    Args:
        traj_idx: The integer index of the trajectory to execute.

    Returns:
        NDArray[np.float64]: The result of the MCWF trajectory.
    """
    return mcwf((traj_idx, WORKER_CTX["ctx"]))


def _lindblad_ctx_worker(_traj_idx: int) -> tuple[NDArray[np.float64], None, None]:
    """Execute Lindblad evolution from a preprocessed context in `WORKER_CTX`.

    Returns:
        Observable expectation values over time for one trajectory.
    """
    return lindblad_evolve(WORKER_CTX["ctx"])


def _digital_strong_worker(
    traj_idx: int,
) -> tuple[NDArray[np.float64] | dict[int, int], NDArray[np.float64] | None, MPS | None]:
    """Execute a single digital strong simulation trajectory.

    Retrieves the required simulation objects (initial state, noise model,
    parameters, circuit) from the global `WORKER_CTX` and delegates to the
    `digital_tjm` backend.

    Args:
        traj_idx: The integer index of the trajectory to execute.

    Returns:
        tuple[NDArray[np.float64] | dict[int, int], NDArray[np.float64] | None, MPS | None]:
            Observable data or shot counts, optional diagnostics, optional final MPS.
    """
    return digital_tjm((
        traj_idx,
        WORKER_CTX["initial_state"],
        WORKER_CTX["noise_model"],
        WORKER_CTX["sim_params"],
        WORKER_CTX["operator"],
    ))


def _digital_weak_worker(traj_idx: int) -> tuple[dict[int, int], None, MPS | None]:
    """Execute a single digital weak simulation trajectory.

    Retrieves simulation objects from `WORKER_CTX` and executes a 'shots=1'
    weak simulation using `digital_tjm`.

    Args:
        traj_idx: The integer index of the trajectory (effectively a shot index).

    Returns:
        tuple[dict[int, int], MPS | None]: Shot counts and optional final MPS.
    """
    return cast(
        "tuple[dict[int, int], None, MPS | None]",
        digital_tjm((
            traj_idx,
            WORKER_CTX["initial_state"],
            WORKER_CTX["noise_model"],
            WORKER_CTX["sim_params"],
            WORKER_CTX["operator"],
        )),
    )


def _ensemble_worker(
    job_idx: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128] | None]:
    """Execute one deterministic unitary ensemble member trajectory.

    Uses :data:`WORKER_CTX` keys ``initial_states``, ``sim_params``, and ``operator``.

    Args:
        job_idx: Index of this member; selects ``WORKER_CTX["initial_states"][job_idx]``.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128] | None]:
            Observable trajectories, diagnostics, and optional multi-time block for one member.
    """
    return ensemble_member_worker((
        job_idx,
        WORKER_CTX["initial_states"][job_idx],
        WORKER_CTX["sim_params"],
        WORKER_CTX["operator"],
    ))


def _materialized_mps(state: State) -> MPS | None:
    """Return the encoded MPS if present, else ``None``."""
    try:
        return state.mps
    except RuntimeError:
        return None


def _hamiltonian_backend_target(state_rep: str) -> str:
    """Internal storage target for ``Hamiltonian`` given ``State.representation``.

    Returns:
        ``"sparse"`` for vector or density-matrix states, otherwise ``"mpo"``.
    """
    if state_rep in {"vector", "density_matrix"}:
        return "sparse"
    return "mpo"


def _validate_state_hamiltonian_pairing(state: State, hamiltonian: Hamiltonian) -> None:
    """Check ``State`` and ``Hamiltonian`` can be evolved together.

    Raises:
        ValueError: If representations or lengths are incompatible.
    """
    if state.length != hamiltonian.length:
        msg = f"State.length={state.length} does not match Hamiltonian.length={hamiltonian.length}."
        raise ValueError(msg)
    if state.representation == "mps" and hamiltonian.representation != "mpo":
        msg = (
            "TJM simulation requires Hamiltonian.representation='mpo'. "
            "Use State.representation='vector' or 'density_matrix' for matrix Hamiltonians."
        )
        raise ValueError(msg)


def _prepare_hamiltonian_for_run(
    hamiltonian: Hamiltonian,
    state_rep: str,
) -> tuple[MPO | None, Any]:
    """Ensure ``hamiltonian`` is encoded for the backend matching ``state_rep``.

    Returns:
        ``(mpo, h_sparse)`` with one entry set for the active backend.
    """
    target = _hamiltonian_backend_target(state_rep)
    hamiltonian.ensure_encoded(cast("Representation", target))
    if target == "mpo":
        return hamiltonian.mpo, None
    return None, hamiltonian.sparse_matrix


def _prepare_result_observables(
    result: Result,
    sim_params: AnalogSimParams | StrongSimParams,
    *,
    num_traj: int,
    num_mid_measurements: int | None = None,
) -> None:
    """Deep-copy user-ordered observables onto ``result`` and allocate output buffers."""
    result.observables = [copy.deepcopy(obs) for obs in sim_params.observables]
    trajectories, expectation_values, times = allocate_observable_buffers(
        sim_params,
        len(result.observables),
        num_traj=num_traj,
        num_mid_measurements=num_mid_measurements,
    )
    result.trajectories = trajectories
    result.expectation_values = expectation_values
    result.times = times


def _worker_sim_params(
    sim_params: AnalogSimParams | StrongSimParams,
) -> AnalogSimParams | StrongSimParams:
    """Build worker-visible params that expose sorted observables for measurement.

    Returns:
        A deep copy of ``sim_params`` whose observable lists are ordered for worker evaluation.
    """
    worker_params = copy.deepcopy(sim_params)
    # Workers evaluate in sorted order for efficiency; Result retains user order.
    sorted_obs, _ = _prepare_observable_ordering(sim_params.observables)
    worker_params.observables = [copy.deepcopy(obs) for obs in sorted_obs]
    return worker_params


def _store_observable_trajectory(
    result: Result,
    sim_params: AnalogSimParams | StrongSimParams,
    *,
    traj_index: int,
    sorted_traj_data: NDArray[np.float64] | NDArray[np.complex128],
) -> None:
    """Store one trajectory's observable data into result buffers in user order."""
    _, observable_sorted_indices = _prepare_observable_ordering(sim_params.observables)
    for user_i, sorted_i in enumerate(observable_sorted_indices):
        result.trajectories[user_i][traj_index] = sorted_traj_data[sorted_i]


def _store_final_mps(result: Result, final_mps: MPS | None) -> None:
    if final_mps is not None:
        result.output_state = State.from_mps(final_mps)


def _store_mcwf_final_state(result: Result, psi: np.ndarray | None) -> None:
    if psi is not None:
        result.output_state = State(vector=psi)


def _expect_shot_counts(payload: NDArray[np.float64] | dict[int, int]) -> dict[int, int]:
    """Return weak-simulation shot counts from a digital backend payload.

    Raises:
        TypeError: If ``payload`` is not a shot-count dictionary.
    """
    if not isinstance(payload, dict):
        msg = f"Expected measurement result to be dict[int, int], got {type(payload).__name__}."
        raise TypeError(msg)
    return cast("dict[int, int]", payload)


def _looks_like_qudit_circuit(operator: object) -> bool:
    return (
        hasattr(operator, "dimensions") and hasattr(operator, "num_qudits") and not isinstance(operator, QuantumCircuit)
    )


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
    # Numba threading must be set BEFORE the threadpoolctl context limits backend threads,
    try:
        import numba  # noqa: PLC0415

        numba.set_num_threads(n_threads)
    except (ImportError, AttributeError):
        pass

    if threadpool_limits is not None:
        # Caps any pools entered/created within the context
        with contextlib.suppress(Exception), threadpool_limits(limits=n_threads):
            return backend(arg)

    # If threadpoolctl fails or is missing, fallback to direct call
    return backend(arg)


def run_backend_parallel(
    worker_fn: Callable[[int], TRes],
    *,
    payload: dict[str, Any] | None,
    n_jobs: int,
    max_workers: int,
    show_progress: bool = True,
    desc: str,
    max_retries: int = 10,
    retry_exceptions: tuple[type[BaseException], ...] = (CancelledError, TimeoutError, OSError),
    mp_context: MPContext = "auto",
) -> Iterator[tuple[int, TRes]]:
    """Execute backend calls in parallel with bounded submission and retry logic.

    This function manages the parallel execution of tasks using a
    ``ProcessPoolExecutor``, refactored to prevent task flooding and memory
    exhaustion:

    1. **Worker-Global State**: Uses ``_worker_init`` to initialize large objects
       once per worker, avoiding per-task pickling overhead.
    2. **Bounded In-Flight**: Submits tasks in a queue-like manner, keeping only
       a limited number of futures active (``2 * max_workers``) at any time.

    Args:
        worker_fn: The worker function to execute. It must accept a single
            integer argument (the job index) and return a result of type `TRes`.
        payload: A dictionary of large objects (e.g., State, NoiseModel) to be
            initialized in the global worker context. passed to `_worker_init`.
        n_jobs: The total number of jobs to execute (indices 0 to n_jobs-1).
        max_workers: The maximum number of worker processes to use.
        show_progress: If True, displays a tqdm progress bar. Defaults to True.
        desc: The description string for the progress bar.
        max_retries: The maximum number of retry attempts for transient errors
            (e.g., TimeoutError). Defaults to 10.
        retry_exceptions: A tuple of exception types that trigger a retry.
            Defaults to (CancelledError, TimeoutError, OSError).
        mp_context: Multiprocessing context selector; see :func:`~mqt.yaqs.parallel_utils.get_parallel_context`.

    Yields:
        tuple[int, TRes]: A tuple containing the job index and its result,
        yielded in the order of completion.
    """
    ctx = get_parallel_context(mp_context)

    # Bounded in-flight factor (keep 2-4x workers busy to hide latency)
    inflight_factor = 2
    max_inflight = max_workers * inflight_factor

    with (
        ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=worker_init,
            initargs=(payload or {}, 1),
        ) as ex,
        tqdm(total=n_jobs, desc=desc, ncols=80, disable=(not show_progress)) as pbar,
    ):
        retries = dict.fromkeys(range(n_jobs), 0)

        futures: dict[Future[TRes], int] = {}
        next_job_idx = 0

        def submit_job(idx: int) -> None:
            """Submit a job for the given index."""
            futures[ex.submit(worker_fn, idx)] = idx

        while next_job_idx < n_jobs and len(futures) < max_inflight:
            submit_job(next_job_idx)
            next_job_idx += 1

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                i = futures.pop(fut)
                try:
                    res = fut.result()
                except retry_exceptions:
                    if retries[i] < max_retries:
                        retries[i] += 1
                        submit_job(i)
                        continue
                    raise

                yield i, res
                pbar.update(1)

                if next_job_idx < n_jobs:
                    submit_job(next_job_idx)
                    next_job_idx += 1


# ---------------------------------------------------------------------------
# 10) SIMULATOR — public entry point
# Owns the execution-side configuration (parallel/serial, workers, progress,
# multiprocessing context, retry policy) and dispatches to circuit or analog
# engines based on the sim_params type.
# ---------------------------------------------------------------------------
class Simulator:
    """Public entry point for running YAQS simulations.

    A :class:`Simulator` owns the execution-side configuration: how trajectories
    are parallelized, how many workers to use, whether to display a progress bar,
    which multiprocessing context to use, and the retry policy for transient
    worker errors. The physics inputs (initial state, operator, simulation
    parameters, optional noise model) are passed per call to :meth:`run`.

    Multiple :meth:`run` calls share the same configuration. Each call constructs
    its own short-lived process pool when ``parallel=True``; the pool is not
    persisted across runs in the current implementation.

    Attributes:
        parallel: Whether to execute trajectories in parallel via a process pool.
        max_workers: Maximum number of worker processes when ``parallel=True``.
            Defaults to ``max(1, available_cpus() - 1)``.
        show_progress: Whether to display a tqdm progress bar.
        mp_context: Multiprocessing context: ``"auto"`` (default), ``"fork"``,
            or ``"spawn"``. ``"auto"`` selects ``"fork"`` on Linux and ``"spawn"`` elsewhere.
        max_retries: Maximum retry attempts for transient worker errors.
        retry_exceptions: Exception types that trigger a retry.
    """

    def __init__(
        self,
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        show_progress: bool = True,
        mp_context: MPContext = "auto",
        max_retries: int = 10,
        retry_exceptions: tuple[type[BaseException], ...] = (CancelledError, TimeoutError, OSError),
    ) -> None:
        """Initialize the simulator with execution-side configuration.

        Args:
            parallel: If ``True`` (default), use a process pool for multi-trajectory runs.
            max_workers: Maximum worker processes when running in parallel. ``None`` (default)
                resolves to ``max(1, available_cpus() - 1)``.
            show_progress: Show a tqdm progress bar during trajectory execution.
            mp_context: Multiprocessing start method (``"auto"``, ``"fork"``, or ``"spawn"``).
            max_retries: Maximum retries for transient worker errors.
            retry_exceptions: Exception types that trigger a retry.
        """
        self.parallel = parallel
        self.max_workers = max_workers if max_workers is not None else max(1, available_cpus() - 1)
        self.show_progress = show_progress
        self.mp_context: MPContext = mp_context
        self.max_retries = max_retries
        self.retry_exceptions = retry_exceptions

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def run(
        self,
        initial_state: State | list[State],
        operator: Hamiltonian | QuantumCircuit | str | Path,
        sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
        noise_model: NoiseModel | None = None,
    ) -> Result:
        """Execute the common simulation routine for Hamiltonian (analog) and circuit simulations.

        Dispatches the simulation to the appropriate backend based on the type of simulation
        parameters provided. For analog simulations, the
        :class:`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` and ``State`` are
        materialized for the active representation as needed (``list[State]`` triggers the
        deterministic unitary ensemble path). For circuit-based simulations, the initial
        :class:`~mqt.yaqs.core.data_structures.state.State` must use ``representation="mps"``.

        Args:
            initial_state: The initial state as a :class:`~mqt.yaqs.core.data_structures.state.State`,
                or a list of states for deterministic analog unitary ensemble evolution
                (``AnalogSimParams`` only).
            operator: :class:`~mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian` for analog
                simulations, or a :class:`~qiskit.circuit.QuantumCircuit`, raw QASM ``str``, or
                ``Path`` to a ``.qasm`` file for circuit simulations.
            sim_params: Simulation parameters specifying the simulation mode and settings.
            noise_model: The noise model to apply. If provided, it is sampled once at the
                beginning of the run to generate a concrete noise realization (static disorder).
                The sampled noise model is stored on the returned :class:`~mqt.yaqs.Result`.

        Returns:
            A :class:`~mqt.yaqs.core.data_structures.result.Result` holding all
            simulation outputs. The supplied ``sim_params`` is not mutated;
            ``Result.sim_params`` references the original configuration object.

        Raises:
            ValueError: If no output is specified (neither observables nor ``get_state``).
            TypeError: If the provided ``initial_state`` type is incompatible with the
                selected simulation mode.
        """
        if not isinstance(sim_params, AnalogSimParams) and isinstance(operator, (str, Path)):
            operator = load_circuit(operator)

        if isinstance(initial_state, list) and any(not isinstance(state, State) for state in initial_state):
            msg = "initial_state list must contain only State objects."
            raise TypeError(msg)

        if noise_model is not None:
            sample_seed = getattr(sim_params, "random_seed", None)
            noise_model = noise_model.sample(rng=sample_seed)

        result = Result(sim_params=sim_params, noise_model=noise_model)

        if (
            isinstance(sim_params, AnalogSimParams)
            and not sim_params.get_state
            and not sim_params.observables
            and not sim_params.multi_time_observables
        ):
            msg = "No output specified: either observables or get_state must be set."
            raise ValueError(msg)
        if isinstance(sim_params, StrongSimParams) and not sim_params.get_state and not sim_params.observables:
            msg = "No output specified: either observables or get_state must be set."
            raise ValueError(msg)

        if isinstance(sim_params, AnalogSimParams):
            if not isinstance(operator, Hamiltonian):
                msg = "Analog simulation requires a Hamiltonian operator."
                raise TypeError(msg)
            if not isinstance(initial_state, (State, list)):
                msg = "Analog simulation requires initial_state to be a list or State."
                raise TypeError(msg)
            self._run_analog(initial_state, operator, sim_params, noise_model, result)
        elif isinstance(sim_params, (StrongSimParams, WeakSimParams)):
            if isinstance(initial_state, list):
                msg = "Circuit simulation requires a single State initial_state."
                raise TypeError(msg)
            if not isinstance(initial_state, State):
                msg = "Circuit simulation requires a State initial_state."
                raise TypeError(msg)
            if _looks_like_qudit_circuit(operator):
                self._run_circuit_qudit(initial_state, operator, sim_params, noise_model, result)
            elif isinstance(operator, QuantumCircuit):
                self._run_circuit(initial_state, operator, sim_params, noise_model, result)
            else:
                msg = "Circuit simulation requires a QuantumCircuit or qudit QuantumCircuit operator."
                raise TypeError(msg)

        return result

    # -----------------------------------------------------------------------
    # Analog (Hamiltonian) simulation -- primary YAQS simulation mode
    # -----------------------------------------------------------------------
    def _run_analog(
        self,
        initial_state: State | list[State],
        operator: Hamiltonian,
        sim_params: AnalogSimParams,
        noise_model: NoiseModel | None,
        result: Result,
    ) -> None:
        """Run analog simulation trajectories for Hamiltonian evolution.

        Selects the appropriate analog simulation backend based on ``sim_params.order``
        (either one-site or two-site evolution) and runs the simulation trajectories for the given
        Hamiltonian. The trajectories are executed and the results are aggregated.

        A ``list[State]`` ``initial_state`` triggers the deterministic unitary ensemble
        path via :meth:`_run_ensemble`.

        Args:
            initial_state: The initial system state as a :class:`State`, or a list of
                :class:`State` objects for deterministic unitary analog ensemble evolution.
            operator: The Hamiltonian specification (materialized once per ``run`` call).
            sim_params: Simulation parameters for analog simulation.
            noise_model: The noise model applied during simulation.
            result: Output container populated during this run.

        Raises:
            ValueError: If ``get_state=True`` with ``State.representation='density_matrix'``,
                or if ``get_state=True`` is combined with a non-trivial noise model
                (the trajectory ensemble has no single representative state).
        """
        if isinstance(initial_state, list):
            initial_state_list = cast("list[State]", initial_state)
            if any(spec.representation != "mps" for spec in initial_state_list):
                msg = "list[State] analog ensemble currently supports only State.representation='mps'."
                raise ValueError(msg)
            operator.ensure_encoded("mpo")
            for spec in initial_state_list:
                spec.ensure_encoded("mps")
                _validate_state_hamiltonian_pairing(spec, operator)
            self._run_ensemble(
                [spec.mps for spec in initial_state_list],
                operator.mpo,
                sim_params,
                noise_model,
                result,
            )
            return

        initial_state.ensure_encoded(initial_state.representation)
        mps = _materialized_mps(initial_state)
        state_rep = initial_state.representation
        _validate_state_hamiltonian_pairing(initial_state, operator)
        mpo_op, h_sparse = _prepare_hamiltonian_for_run(operator, state_rep)

        backend: Callable[..., tuple[NDArray[np.float64], Any, Any]]
        if state_rep == "density_matrix":
            backend = lindblad_evolve
        elif state_rep == "vector":
            backend = mcwf
        elif sim_params.order == 1:
            backend = analog_tjm_1
        else:
            backend = analog_tjm_2

        if state_rep == "density_matrix" and sim_params.get_state:
            msg = "get_state=True is not supported for State.representation='density_matrix'."
            raise ValueError(msg)

        if (
            noise_model is None
            or all(proc["strength"] == 0 for proc in noise_model.processes)
            or state_rep == "density_matrix"
        ):
            effective_num_traj = 1
        else:
            if sim_params.get_state:
                msg = "Cannot return state in noisy analog simulation due to stochastics."
                raise ValueError(msg)
            effective_num_traj = sim_params.num_traj

        _prepare_result_observables(result, sim_params, num_traj=effective_num_traj)
        worker_params = cast("AnalogSimParams", _worker_sim_params(sim_params))

        diag_per_traj: NDArray[np.float64] | None = None
        if state_rep == "mps":
            diag_per_traj, _ = allocate_diagnostic_buffers(sim_params, num_traj=effective_num_traj)

        payload: dict[str, Any]
        worker_fn: Callable[[int], Any]

        if state_rep == "vector":
            ctx = preprocess_mcwf(
                mps,
                mpo_op,
                noise_model,
                worker_params,
                psi_initial=None if mps is not None else initial_state.vector,
                num_sites=initial_state.length if mps is None else None,
                h_sparse=h_sparse,
            )
            payload = {"ctx": ctx}
            worker_fn = _mcwf_worker
        elif state_rep == "density_matrix":
            lindblad_ctx = preprocess_lindblad(
                mps,
                mpo_op,
                noise_model,
                worker_params,
                rho_initial=initial_state.density_matrix,
                num_sites=initial_state.length,
                h_sparse=h_sparse,
            )
            payload = {"ctx": lindblad_ctx}
            worker_fn = _lindblad_ctx_worker
        else:
            assert mps is not None, "MPS representation requires a materialized MPS."
            assert mpo_op is not None
            payload = {
                "initial_state": mps,
                "noise_model": noise_model,
                "sim_params": worker_params,
                "operator": mpo_op,
                "backend": backend,
            }
            worker_fn = _analog_worker

        final_mps: MPS | None = None
        final_psi: np.ndarray | None = None

        if self.parallel and effective_num_traj > 1:
            for i, traj_payload in run_backend_parallel(
                worker_fn=worker_fn,
                payload=payload,
                n_jobs=effective_num_traj,
                max_workers=self.max_workers,
                show_progress=self.show_progress,
                desc="Running trajectories",
                max_retries=self.max_retries,
                retry_exceptions=self.retry_exceptions,
                mp_context=self.mp_context,
            ):
                traj_data, traj_diag, traj_final = traj_payload
                _store_observable_trajectory(result, sim_params, traj_index=i, sorted_traj_data=traj_data)
                if traj_diag is not None and diag_per_traj is not None:
                    diag_per_traj[:, i, :] = traj_diag
                if traj_final is not None:
                    if state_rep == "vector":
                        final_psi = cast("np.ndarray", traj_final)
                    else:
                        final_mps = cast("MPS", traj_final)
        else:
            n_threads = available_cpus()

            args: list[Any]
            if state_rep == "vector":
                args = [(i, copy.copy(ctx)) for i in range(effective_num_traj)]
            elif state_rep == "density_matrix":
                args = [lindblad_ctx for _ in range(effective_num_traj)]
            else:
                args = [(i, mps, noise_model, worker_params, mpo_op) for i in range(effective_num_traj)]

            iterator = tqdm(args, desc="Running trajectories", ncols=80, disable=not self.show_progress)

            for i, arg in enumerate(iterator):
                traj_data, traj_diag, traj_final = _call_backend(backend, arg, n_threads=n_threads)
                _store_observable_trajectory(result, sim_params, traj_index=i, sorted_traj_data=traj_data)
                if traj_diag is not None and diag_per_traj is not None:
                    diag_per_traj[:, i, :] = traj_diag
                if traj_final is not None:
                    if state_rep == "vector":
                        final_psi = cast("np.ndarray", traj_final)
                    else:
                        final_mps = cast("MPS", traj_final)

        if state_rep == "vector":
            _store_mcwf_final_state(result, final_psi)
        else:
            _store_final_mps(result, final_mps)

        if diag_per_traj is not None:
            result.runtime_cost, result.max_bond, result.total_bond = aggregate_diagnostics(diag_per_traj)
        aggregate_trajectories(result)

    # -----------------------------------------------------------------------
    # Strong simulation (circuit): observable trajectories
    # -----------------------------------------------------------------------
    def _run_strong_sim(
        self,
        initial_state: MPS,
        operator: QuantumCircuit,
        sim_params: StrongSimParams,
        noise_model: NoiseModel | None,
        result: Result,
    ) -> None:
        """Run strong circuit simulation trajectories.

        Executes circuit-based simulation trajectories using the ``digital_tjm`` backend.
        If the noise model is absent or its strengths are all zero, only a single trajectory
        is executed. For each observable in ``sim_params.sorted_observables``, the function
        initializes the observable, runs the simulation trajectories, and aggregates the results.

        Args:
            initial_state: The initial system state as an MPS.
            operator: The quantum circuit representing the operation to simulate.
            sim_params: Simulation parameters for strong simulation.
            noise_model: The noise model applied during simulation.
            result: Output container populated during this run.

        Raises:
            ValueError: If ``sim_params.get_state`` is ``True`` while a non-trivial
                noise model is supplied (the trajectory ensemble has no single
                representative state).
        """
        backend: Callable[[tuple[int, MPS, NoiseModel | None, StrongSimParams, QuantumCircuit]], Any] = digital_tjm

        if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
            effective_num_traj = 1
        else:
            if sim_params.get_state:
                msg = "Cannot return state in noisy circuit simulation due to stochastics."
                raise ValueError(msg)
            effective_num_traj = sim_params.num_traj

        effective_num_mid_measurements = sim_params.num_mid_measurements
        if sim_params.sample_layers:
            dag = circuit_to_dag(operator)
            effective_num_mid_measurements = sum(
                1
                for n in dag.op_nodes()
                if n.op.name == "barrier" and str(getattr(n.op, "label", "")).strip().upper() == "SAMPLE_OBSERVABLES"
            )

        _prepare_result_observables(
            result,
            sim_params,
            num_traj=effective_num_traj,
            num_mid_measurements=effective_num_mid_measurements,
        )
        worker_params = cast("StrongSimParams", _worker_sim_params(sim_params))
        if sim_params.sample_layers:
            worker_params.num_mid_measurements = effective_num_mid_measurements

        diag_per_traj, _ = allocate_diagnostic_buffers(
            sim_params,
            num_traj=effective_num_traj,
            num_mid_measurements=effective_num_mid_measurements,
        )

        payload: dict[str, Any] = {
            "initial_state": initial_state,
            "noise_model": noise_model,
            "sim_params": worker_params,
            "operator": operator,
        }

        final_mps: MPS | None = None

        if self.parallel and effective_num_traj > 1:
            for i, traj_payload in run_backend_parallel(
                worker_fn=_digital_strong_worker,
                payload=payload,
                n_jobs=effective_num_traj,
                max_workers=self.max_workers,
                show_progress=self.show_progress,
                desc="Running trajectories",
                max_retries=self.max_retries,
                retry_exceptions=self.retry_exceptions,
                mp_context=self.mp_context,
            ):
                traj_data, traj_diag, traj_final = traj_payload
                traj_data = cast("NDArray[np.float64] | NDArray[np.complex128]", traj_data)
                _store_observable_trajectory(result, sim_params, traj_index=i, sorted_traj_data=traj_data)
                if traj_diag is not None:
                    diag_per_traj[:, i, :] = traj_diag
                if traj_final is not None:
                    final_mps = traj_final
        else:
            n_threads = available_cpus()

            args: list[tuple[int, MPS, NoiseModel | None, StrongSimParams, QuantumCircuit]] = [
                (i, initial_state, noise_model, worker_params, operator) for i in range(effective_num_traj)
            ]

            iterator = tqdm(args, desc="Running trajectories", ncols=80, disable=not self.show_progress)

            for i, arg in enumerate(iterator):
                traj_data, traj_diag, traj_final = _call_backend(backend, arg, n_threads=n_threads)
                traj_data = cast("NDArray[np.float64] | NDArray[np.complex128]", traj_data)
                _store_observable_trajectory(result, sim_params, traj_index=i, sorted_traj_data=traj_data)
                if traj_diag is not None:
                    diag_per_traj[:, i, :] = traj_diag
                if traj_final is not None:
                    final_mps = traj_final

        _store_final_mps(result, final_mps)
        result.runtime_cost, result.max_bond, result.total_bond = aggregate_diagnostics(diag_per_traj)
        aggregate_trajectories(result)

    # -----------------------------------------------------------------------
    # Weak simulation (circuit): measurement counts
    # -----------------------------------------------------------------------
    def _run_weak_sim(
        self,
        initial_state: MPS,
        operator: QuantumCircuit,
        sim_params: WeakSimParams,
        noise_model: NoiseModel | None,
        result: Result,
    ) -> None:
        """Run weak circuit simulation trajectories.

        Executes circuit-based simulation trajectories using the ``digital_tjm`` backend in weak
        simulation mode. The outputs are raw measurement results rather than observable
        expectation values. If the noise model is absent or its strengths are all zero, only a
        single trajectory is executed. If noise is present, the number of trajectories is set
        equal to the number of shots, and each trajectory corresponds to one measurement sample
        (with ``sim_params.shots`` forced to 1 internally).

        Args:
            initial_state: The initial system state as an MPS.
            operator: The quantum circuit representing the operation to simulate.
            sim_params: Simulation parameters for weak simulation.
            noise_model: The noise model applied during simulation.
            result: Output container populated during this run.

        Raises:
            ValueError: If ``sim_params.get_state`` is ``True`` while a non-trivial
                noise model is supplied (the trajectory ensemble has no single
                representative state).
        """
        backend: Callable[[tuple[int, MPS, NoiseModel | None, WeakSimParams, QuantumCircuit]], Any] = digital_tjm

        noisy = not (noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes))
        if noisy:
            if sim_params.get_state:
                msg = "Cannot return state in noisy circuit simulation due to stochastics."
                raise ValueError(msg)
            effective_num_traj = sim_params.shots
            per_call_shots = 1
        else:
            effective_num_traj = 1
            per_call_shots = sim_params.shots

        if noisy:
            result.measurements = [None] * effective_num_traj
        else:
            result.measurements = [None]

        worker_params = copy.deepcopy(sim_params)
        payload: dict[str, Any] = {
            "initial_state": initial_state,
            "noise_model": noise_model,
            "sim_params": worker_params,
            "operator": operator,
            "per_call_shots": per_call_shots,
        }
        WORKER_CTX["per_call_shots"] = per_call_shots

        final_mps: MPS | None = None

        if self.parallel and effective_num_traj > 1:
            for i, traj_payload in run_backend_parallel(
                worker_fn=_digital_weak_worker,
                payload=payload,
                n_jobs=effective_num_traj,
                max_workers=self.max_workers,
                show_progress=self.show_progress,
                desc="Running trajectories",
                max_retries=self.max_retries,
                retry_exceptions=self.retry_exceptions,
                mp_context=self.mp_context,
            ):
                shot_counts, _traj_diag, traj_final = traj_payload
                result.measurements[i] = _expect_shot_counts(shot_counts)
                if traj_final is not None:
                    final_mps = traj_final
        else:
            n_threads = available_cpus()

            args: list[Any] = [
                (i, initial_state, noise_model, worker_params, operator) for i in range(effective_num_traj)
            ]

            iterator = tqdm(args, desc="Running trajectories", ncols=80, disable=not self.show_progress)

            for i, arg in enumerate(iterator):
                shot_counts, _traj_diag, traj_final = _call_backend(backend, arg, n_threads=n_threads)
                counts_dict = _expect_shot_counts(shot_counts)
                if noisy:
                    result.measurements[i] = counts_dict
                else:
                    result.measurements[0] = counts_dict
                if traj_final is not None:
                    final_mps = traj_final

        WORKER_CTX.pop("per_call_shots", None)
        _store_final_mps(result, final_mps)
        aggregate_counts(result)

    # -----------------------------------------------------------------------
    # Circuit dispatcher
    # -----------------------------------------------------------------------
    def _run_circuit(
        self,
        initial_state: State,
        operator: QuantumCircuit,
        sim_params: WeakSimParams | StrongSimParams,
        noise_model: NoiseModel | None,
        result: Result,
    ) -> None:
        """Run circuit-based simulation trajectories.

        Requires :attr:`~mqt.yaqs.core.data_structures.state.State.representation` ``"mps"``,
        materializes the state, validates that the number of qubits in the quantum circuit
        matches the MPS length, and dispatches the simulation to the appropriate backend
        based on whether the simulation parameters indicate strong or weak simulation.

        Args:
            initial_state: The initial system state (must use MPS representation).
            operator: The quantum circuit to simulate.
            sim_params: Simulation parameters for circuit simulation.
            noise_model: The noise model applied during simulation.
            result: Output container populated during this run.

        Raises:
            ValueError: If ``initial_state.representation`` is not ``"mps"``.
        """
        if initial_state.representation != "mps":
            msg = (
                "Circuit simulation requires State.representation='mps'. "
                "Use representation='vector' or 'density_matrix' only for analog Hamiltonian runs."
            )
            raise ValueError(msg)
        initial_state.ensure_encoded("mps")
        mps = initial_state.mps

        if mps.length != operator.num_qubits:
            msg = "State and circuit qubit counts do not match."
            raise ValueError(msg)

        if isinstance(sim_params, StrongSimParams):
            self._run_strong_sim(mps, operator, sim_params, noise_model, result)
        elif isinstance(sim_params, WeakSimParams):
            self._run_weak_sim(mps, operator, sim_params, noise_model, result)

    # -----------------------------------------------------------------------
    # Qudit circuit simulation (mqt.yaqs.digital.qudit_tjm, optional mqt-qudits dependency)
    # -----------------------------------------------------------------------
    @staticmethod
    def _run_circuit_qudit(
        initial_state: State,
        operator: object,
        sim_params: StrongSimParams | WeakSimParams,
        noise_model: NoiseModel | None,
        result: Result,
    ) -> None:
        """Run a single-trajectory qudit circuit simulation.

        ``operator`` is loosely typed (``object``) on purpose: this method must not statically
        import anything from ``mqt.yaqs.digital.qudit_tjm``, which transitively imports the
        optional ``mqt-qudits`` package. A static import here fails to resolve under ``ty``
        whenever ``mqt-qudits`` is not installed. ``mqt.yaqs.digital.qudit_tjm`` is loaded
        dynamically via :mod:`importlib` instead.

        Args:
            initial_state: The initial system state (must use MPS representation).
            operator: An ``mqt.qudits.quantum_circuit.QuantumCircuit``.
            sim_params: Simulation parameters for circuit simulation.
            noise_model: The noise model applied during simulation.
            result: Output container populated during this run.

        Raises:
            ValueError: If ``initial_state.representation`` is not ``"mps"``, or if the
                state's site count/physical dimensions do not match ``operator``.
        """
        if initial_state.representation != "mps":
            msg = (
                "Circuit simulation requires State.representation='mps'. "
                "Use representation='vector' or 'density_matrix' only for analog Hamiltonian runs."
            )
            raise ValueError(msg)
        initial_state.ensure_encoded("mps")
        mps = initial_state.mps

        num_qudits = getattr(operator, "num_qudits")  # noqa: B009
        dimensions = getattr(operator, "dimensions")  # noqa: B009
        if mps.length != num_qudits:
            msg = "State and circuit qudit counts do not match."
            raise ValueError(msg)
        if list(mps.physical_dimensions) != list(dimensions):
            msg = "State physical dimensions do not match the circuit's qudit dimensions."
            raise ValueError(msg)

        qudit_tjm_module = importlib.import_module("mqt.yaqs.digital.qudit_tjm")

        if isinstance(sim_params, WeakSimParams):
            noisy = noise_model is not None and any(proc["strength"] > 0 for proc in noise_model.processes)
            if noisy:
                if sim_params.get_state:
                    msg = "Cannot return state in noisy circuit simulation due to stochastics."
                    raise ValueError(msg)
                effective_num_traj = sim_params.shots
                per_call_shots = 1
            else:
                effective_num_traj = 1
                per_call_shots = sim_params.shots

            result.measurements = [None] * effective_num_traj if noisy else [None]

            WORKER_CTX["per_call_shots"] = per_call_shots
            final_mps: MPS | None = None
            for traj_idx in range(effective_num_traj):
                shot_counts, _, traj_final_mps = qudit_tjm_module.qudit_tjm((
                    traj_idx,
                    mps,
                    noise_model,
                    sim_params,
                    operator,
                ))
                if noisy:
                    result.measurements[traj_idx] = _expect_shot_counts(shot_counts)
                else:
                    result.measurements[0] = _expect_shot_counts(shot_counts)
                if traj_final_mps is not None:
                    final_mps = traj_final_mps
            WORKER_CTX.pop("per_call_shots", None)

            _store_final_mps(result, final_mps)
            aggregate_counts(result)
            return

        has_noise = noise_model is not None and any(proc["strength"] > 0 for proc in noise_model.processes)
        if has_noise:
            if sim_params.get_state:
                msg = "Cannot return state in noisy circuit simulation due to stochastics."
                raise ValueError(msg)
            num_traj = sim_params.num_traj
        else:
            num_traj = 1

        _prepare_result_observables(result, sim_params, num_traj=num_traj)

        final_mps = None
        for traj_idx in range(num_traj):
            results, _, traj_final_mps = qudit_tjm_module.qudit_tjm((traj_idx, mps, noise_model, sim_params, operator))
            results = cast("NDArray[np.float64]", results)
            _store_observable_trajectory(result, sim_params, traj_index=traj_idx, sorted_traj_data=results[:, 0])

            if traj_final_mps is not None:
                final_mps = traj_final_mps

        _store_final_mps(result, final_mps)
        aggregate_trajectories(result)

    # -----------------------------------------------------------------------
    # Unitary ensemble (deterministic, no noise)
    # -----------------------------------------------------------------------
    def _run_ensemble(
        self,
        initial_states: list[MPS],
        operator: MPO,
        sim_params: AnalogSimParams,
        noise_model: NoiseModel | None,
        result: Result,
    ) -> None:
        """Run deterministic unitary evolution for an ensemble of initial MPS states.

        This mode is intentionally separate from stochastic-noise trajectories: users may either
        provide a list of initial states with no noise (this mode), or a single initial state with
        noise (standard TJM / Lindblad / MCWF paths).

        Args:
            initial_states: One MPS per ensemble member; lengths must match ``operator.length``.
            operator: Hamiltonian as an MPO shared by all members.
            sim_params: Analog parameters; ``num_traj`` is set to ``len(initial_states)``.
            noise_model: Must be absent or contain only zero-strength processes.
            result: Output container populated during this run.

        Raises:
            ValueError: If noisy simulation is requested with a list of states, if
                ``representation`` is unsupported in list mode, if the list is empty, or if state
                lengths do not match the MPO.
        """
        if noise_model is not None and any(proc["strength"] > 0 for proc in noise_model.processes):
            msg = (
                "list[State] with noisy analog simulation is not supported yet. "
                "Use list[State] with no noise for unitary ensembles, or use a single State for noisy simulation."
            )
            raise ValueError(msg)
        if not initial_states:
            msg = "initial_state list must not be empty."
            raise ValueError(msg)

        if any(state.length != operator.length for state in initial_states):
            msg = "All initial states in the list must match the MPO length."
            raise ValueError(msg)
        if sim_params.get_state:
            msg = "get_state=True is not supported for list[State] analog ensemble mode."
            raise ValueError(msg)

        effective_num_traj = len(initial_states)

        _prepare_result_observables(result, sim_params, num_traj=effective_num_traj)
        worker_params = cast("AnalogSimParams", _worker_sim_params(sim_params))
        diag_per_traj, _ = allocate_diagnostic_buffers(sim_params, num_traj=effective_num_traj)

        n_pairs = len(sim_params.multi_time_observables)
        n_cols = len(sim_params.times) if sim_params.sample_timesteps else 1
        multi_time_matrix: NDArray[np.complex128] | None = None
        if n_pairs > 0:
            multi_time_matrix = np.zeros((len(initial_states), n_pairs, n_cols), dtype=np.complex128)
            result.multi_time_times = (
                sim_params.times
                if sim_params.sample_timesteps
                else np.array([sim_params.elapsed_time], dtype=np.float64)
            )

        payload: dict[str, Any] = {
            "initial_states": initial_states,
            "sim_params": worker_params,
            "operator": operator,
        }

        if self.parallel and len(initial_states) > 1:
            for i, (obs_result, traj_diag, multi_time_result) in run_backend_parallel(
                worker_fn=_ensemble_worker,
                payload=payload,
                n_jobs=len(initial_states),
                max_workers=self.max_workers,
                show_progress=self.show_progress,
                desc="Running unitary ensemble",
                max_retries=self.max_retries,
                retry_exceptions=self.retry_exceptions,
                mp_context=self.mp_context,
            ):
                _store_observable_trajectory(result, sim_params, traj_index=i, sorted_traj_data=obs_result)
                diag_per_traj[:, i, :] = traj_diag
                if multi_time_matrix is not None:
                    assert multi_time_result is not None
                    multi_time_matrix[i] = multi_time_result
        else:
            n_threads = available_cpus()
            args = [(i, initial_states[i], worker_params, operator) for i in range(len(initial_states))]
            iterator = tqdm(args, desc="Running unitary ensemble", ncols=80, disable=not self.show_progress)
            for i, arg in enumerate(iterator):
                obs_result, traj_diag, multi_time_result = _call_backend(
                    ensemble_member_worker, arg, n_threads=n_threads
                )
                _store_observable_trajectory(result, sim_params, traj_index=i, sorted_traj_data=obs_result)
                diag_per_traj[:, i, :] = traj_diag
                if multi_time_matrix is not None:
                    assert multi_time_result is not None
                    multi_time_matrix[i] = multi_time_result

        result.runtime_cost, result.max_bond, result.total_bond = aggregate_diagnostics(diag_per_traj)
        aggregate_trajectories(result)
        if multi_time_matrix is not None:
            result.multi_time_results = np.mean(multi_time_matrix, axis=0)
