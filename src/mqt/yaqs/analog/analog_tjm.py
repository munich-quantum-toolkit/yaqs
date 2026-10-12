# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""First- and second-order Tensor Jump Method trajectories for MPS evolution.

Both schemes use TDVP or BUG for unitary intervals and return observable
buffers, diagnostics, and an optional final state.
"""

# Simulation workers share private MPS buffer writers.

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from ..core.data_structures.mpo import MPO
from ..core.data_structures.result import _allocate_worker_observable_buffer
from ..core.methods.dissipation import apply_dissipation
from ..core.methods.scheduled_jumps import apply_scheduled_jumps, has_scheduled_jump
from ..core.methods.stochastic_process import stochastic_process
from ..core.random_utils import make_sample_rng, make_trajectory_rng
from .evolution import apply_unitary_evolution

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ..core.data_structures.mps import MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import AnalogSimParams

HamiltonianOperator = MPO | tuple[MPO, ...]


def _mpo_for_interval(hamiltonian: HamiltonianOperator, interval_index: int) -> MPO:
    """Return the MPO used on analog interval ``interval_index``."""
    if not isinstance(hamiltonian, tuple):
        return hamiltonian
    mpo = hamiltonian[interval_index]
    assert isinstance(mpo, MPO)
    return mpo


def _measure_at(
    j: int,
    *,
    n_times: int,
    sample_timesteps: bool,
    sample_at: frozenset[int] | None,
) -> bool:
    """Return whether analog TJM should record observables at time index ``j``."""
    if sample_at is not None:
        return j in sample_at
    if sample_timesteps:
        return True
    return j == n_times - 1


def _validate_sample_at(sim_params: AnalogSimParams, sample_at: Sequence[int] | None) -> None:
    """Reject invalid ``sample_at`` indices and multi-index one-column buffers.

    Raises:
        ValueError: If an index is outside ``[0, len(times))``, or more than one
            index is requested without ``sample_timesteps``.
    """
    if sample_at is None:
        return
    n_times = len(sim_params.times)
    for index in sample_at:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0 or index >= n_times:
            msg = f"sample_at index {index!r} is outside the time grid [0, {n_times})."
            raise ValueError(msg)
    if not sim_params.sample_timesteps and len(frozenset(sample_at)) > 1:
        msg = "Selecting multiple sample_at indices requires sample_timesteps=True."
        raise ValueError(msg)


def initialize(
    state: MPS, noise_model: NoiseModel | None, sim_params: AnalogSimParams, rng: np.random.Generator | None = None
) -> MPS:
    """Apply the initial half-step of dissipation for second-order TJM.

    A scheduled jump at the initial time replaces stochastic jump sampling.

    Args:
        state: Initial MPS, updated in place.
        noise_model: Optional noise model.
        sim_params: Analog settings, including ``dt`` and the time grid.
        rng: Generator for stochastic jump decisions.

    Returns:
        The same MPS after dissipation and a scheduled or sampled jump.
    """
    apply_dissipation(state, noise_model, sim_params.dt / 2, sim_params)
    # Check for scheduled jumps at start time
    current_time = sim_params.times[0]
    if has_scheduled_jump(noise_model, current_time, sim_params.dt):
        return apply_scheduled_jumps(state, noise_model, current_time, sim_params)
    return stochastic_process(state, noise_model, sim_params.dt, sim_params, rng=rng)


def step_through(
    state: MPS,
    hamiltonian: MPO,
    noise_model: NoiseModel | None,
    sim_params: AnalogSimParams,
    current_time: float,
    rng: np.random.Generator | None = None,
) -> MPS:
    """Advance a second-order TJM trajectory between measurement copies.

    Applies a unitary step, full-step dissipation, and a jump update.
    A scheduled jump at ``current_time`` replaces stochastic jump sampling.

    Args:
        state: Trajectory MPS, updated in place.
        hamiltonian: MPO for the interval.
        noise_model: Optional noise model.
        sim_params: Analog evolution settings.
        current_time: Time used to check scheduled jumps.
        rng: Generator for stochastic jump decisions.

    Returns:
        The same MPS after the update.
    """
    apply_unitary_evolution(state, hamiltonian, sim_params)
    apply_dissipation(state, noise_model, sim_params.dt, sim_params)

    if has_scheduled_jump(noise_model, current_time, sim_params.dt):
        return apply_scheduled_jumps(state, noise_model, current_time, sim_params)
    return stochastic_process(state, noise_model, sim_params.dt, sim_params, rng=rng)


def sample(
    phi: MPS,
    hamiltonian: MPO,
    noise_model: NoiseModel | None,
    sim_params: AnalogSimParams,
    results: NDArray[np.float64],
    j: int,
    rng: np.random.Generator | None = None,
    diagnostics: NDArray[np.float64] | None = None,
    *,
    write_results: bool = True,
) -> MPS | None:
    """Evolve a measurement copy and record second-order TJM outputs.

    Leaves ``phi`` unchanged. The copy receives one unitary step, a half-step
    of dissipation, and a scheduled or sampled jump.

    Args:
        phi: Trajectory MPS to copy.
        hamiltonian: MPO for the interval.
        noise_model: Optional noise model.
        sim_params: Evolution and output settings.
        results: Observable buffer; column ``j`` for time sampling, otherwise column zero.
        j: Measurement time-grid index.
        rng: Jump generator for the copy. Keep independent of the trajectory
            generator so intermediate measurements do not change later evolution.
        diagnostics: Optional ``(3, T)`` buffer for cost, maximum bond, and total bond.
        write_results: Whether to write observables and diagnostics. ``False``
            still evolves the copy for final-state output.

    Returns:
        The copy at the final time if ``get_state=True``, otherwise ``None``.
    """
    psi = copy.deepcopy(phi)
    apply_unitary_evolution(psi, hamiltonian, sim_params)
    apply_dissipation(psi, noise_model, sim_params.dt / 2, sim_params)

    current_time = sim_params.times[j]
    if has_scheduled_jump(noise_model, current_time, sim_params.dt):
        psi = apply_scheduled_jumps(psi, noise_model, current_time, sim_params)
    else:
        psi = stochastic_process(psi, noise_model, sim_params.dt, sim_params, rng=rng)
    if write_results:
        col = j if sim_params.sample_timesteps else 0
        if diagnostics is not None:
            psi._record_diagnostics(diagnostics, col)  # ruff: ignore[private-member-access]
        if sim_params.sample_timesteps:
            psi._evaluate_observables(sim_params, results, j)  # ruff: ignore[private-member-access]
        else:
            psi._evaluate_observables(sim_params, results)  # ruff: ignore[private-member-access]

    if j == len(sim_params.times) - 1 and sim_params.get_state:
        return psi
    return None


def _diagnostic_num_columns(sim_params: AnalogSimParams) -> int:
    return len(sim_params.times) if sim_params.sample_timesteps else 1


def analog_tjm_2(
    args: tuple[int, MPS, NoiseModel | None, AnalogSimParams, HamiltonianOperator],
    *,
    copy_initial_state: bool = True,
    rng: np.random.Generator | None = None,
    sample_timestep_offset: int = 0,
    use_trajectory_rng_for_final_sample: bool = False,
    return_trajectory_state: bool = False,
    continue_trajectory: bool = False,
    sample_at: Sequence[int] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], MPS | None]:
    """Run a second-order TJM trajectory with TDVP or BUG unitary evolution.

    Measurement copies use separate random streams from the evolving trajectory.
    Continuation can hand off the staggered trajectory state between program segments.

    Args:
        args: ``(trajectory_id, initial_mps, noise_model, sim_params, hamiltonian)``.
            The Hamiltonian is a static MPO or one MPO per analog interval.
        copy_initial_state: Copy before evolution; ``False`` permits input mutation.
        rng: External trajectory generator; omit to derive one from the trajectory
            ID and ``sim_params.random_seed``.
        sample_timestep_offset: Offset for measurement-copy RNG indices, allowing
            split segments to share the continuous run's sample timeline.
        use_trajectory_rng_for_final_sample: Use an external ``rng`` for the last
            measurement copy instead of a separate sample stream.
        return_trajectory_state: Return the staggered trajectory MPS for continuation,
            instead of the final measurement copy, regardless of ``get_state``.
        continue_trajectory: Skip initialization for a handed-off trajectory with
            the same static Hamiltonian and time step.
        sample_at: Optional time-grid indices to measure. Omit to follow
            ``sample_timesteps``; multiple indices require time sampling.

    Returns:
        Observable buffer, diagnostics ``(3, T)``, and optional final MPS.
        ``T`` is the grid length for time sampling, otherwise one.
    """
    traj_idx, initial_state, noise_model, sim_params, hamiltonian = args

    base_seed = sim_params.random_seed
    external_rng = rng is not None
    if rng is None:
        rng = make_trajectory_rng(traj_idx, base_seed=sim_params.random_seed)
    n_times = len(sim_params.times)
    _validate_sample_at(sim_params, sample_at)
    measure_at = frozenset(sample_at) if sample_at is not None else None
    last = n_times - 1
    need_physical_final = sim_params.get_state and not return_trajectory_state

    def record(j: int) -> bool:
        return _measure_at(
            j,
            n_times=n_times,
            sample_timesteps=sim_params.sample_timesteps,
            sample_at=measure_at,
        )

    def measurement_rng(timestep: int) -> np.random.Generator:
        """Return the RNG for one measurement-copy sample at ``timestep``."""
        if use_trajectory_rng_for_final_sample and external_rng and timestep == n_times - 1:
            return rng
        return make_sample_rng(
            traj_idx,
            base_seed=base_seed,
            timestep=timestep + sample_timestep_offset,
        )

    state = copy.deepcopy(initial_state) if copy_initial_state else initial_state
    num_cols = _diagnostic_num_columns(sim_params)
    diagnostics = np.zeros((3, num_cols), dtype=np.float64)
    if sim_params.sample_timesteps:
        results = _allocate_worker_observable_buffer(sim_params, len(sim_params.times))
    else:
        results = _allocate_worker_observable_buffer(sim_params, 1)

    final_state: MPS | None = None

    def capture_sample(current_phi: MPS, j: int, operator: MPO) -> MPS | None:
        should_record = record(j)
        if not should_record and not (need_physical_final and j == last):
            return None
        extra: dict[str, bool] = {}
        if not should_record:
            extra["write_results"] = False
        return sample(
            current_phi,
            operator,
            noise_model,
            sim_params,
            results,
            j,
            rng=measurement_rng(j),
            diagnostics=diagnostics if should_record else None,
            **extra,
        )

    # Zero-duration runs: evaluate the initial state before any noise/evolution (F0).
    if n_times == 1:
        if record(0):
            state._record_diagnostics(diagnostics, 0)  # ruff: ignore[private-member-access]
            if sim_params.sample_timesteps:
                state._evaluate_observables(sim_params, results, 0)  # ruff: ignore[private-member-access]
            else:
                state._evaluate_observables(sim_params, results)  # ruff: ignore[private-member-access]
        return results, diagnostics, state if (sim_params.get_state or return_trajectory_state) else None

    if continue_trajectory:
        # Mid-Trotter handoff: remeasure the junction with the global sample stream
        # (local 0 + offset matches the prior segment's last sample) without
        # replacing ``phi``, then continue ``step_through``.
        # Program continuation is static-MPO only; interval 0 is the sole operator.
        phi = state
        continued = _mpo_for_interval(hamiltonian, 0)
        capture_sample(phi, 0, continued)
        for j, _ in enumerate(sim_params.times[1:], start=1):
            phi = step_through(phi, continued, noise_model, sim_params, sim_params.times[j], rng=rng)
            sampled_state = capture_sample(phi, j, continued)
            if sampled_state is not None:
                final_state = sampled_state
    else:
        if record(0):
            state._record_diagnostics(diagnostics, 0)  # ruff: ignore[private-member-access]
            state._evaluate_observables(sim_params, results, 0)  # ruff: ignore[private-member-access]

        phi = initialize(state, noise_model, sim_params, rng=rng)

        # Sample at times[1] whenever it is requested, is the final time, or
        # ``get_state`` needs the physical copy at that last time.
        sampled_state = capture_sample(phi, 1, _mpo_for_interval(hamiltonian, 0))
        if sampled_state is not None:
            final_state = sampled_state

        for j, _ in enumerate(sim_params.times[2:], start=2):
            phi = step_through(
                phi,
                _mpo_for_interval(hamiltonian, j - 2),
                noise_model,
                sim_params,
                sim_params.times[j],
                rng=rng,
            )
            sampled_state = capture_sample(phi, j, _mpo_for_interval(hamiltonian, j - 1))
            if sampled_state is not None:
                final_state = sampled_state

    if return_trajectory_state:
        return results, diagnostics, phi
    return results, diagnostics, final_state


def analog_tjm_1(
    args: tuple[int, MPS, NoiseModel | None, AnalogSimParams, HamiltonianOperator],
    *,
    copy_initial_state: bool = True,
    rng: np.random.Generator | None = None,
    sample_at: Sequence[int] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], MPS | None]:
    """Run a first-order TJM trajectory with TDVP or BUG unitary evolution.

    Each interval has a unitary step followed by dissipation and a jump update.
    Observables read the trajectory directly without extra jump draws.

    Args:
        args: ``(trajectory_id, initial_mps, noise_model, sim_params, hamiltonian)``.
            The Hamiltonian is a static MPO or one MPO per analog interval.
        copy_initial_state: Copy before evolution; ``False`` permits input mutation.
        rng: External jump generator; omit to derive one from the trajectory ID
            and ``sim_params.random_seed``.
        sample_at: Optional time-grid indices to measure. Omit to follow
            ``sample_timesteps``; multiple indices require time sampling.

    Returns:
        Observable buffer, diagnostics ``(3, T)``, and final MPS if ``get_state=True``.
        ``T`` is the grid length for time sampling, otherwise one.
    """
    traj_idx, initial_state, noise_model, sim_params, hamiltonian = args

    if rng is None:
        rng = make_trajectory_rng(traj_idx, base_seed=sim_params.random_seed)

    n_times = len(sim_params.times)
    _validate_sample_at(sim_params, sample_at)
    measure_at = frozenset(sample_at) if sample_at is not None else None

    def record(j: int) -> bool:
        return _measure_at(
            j,
            n_times=n_times,
            sample_timesteps=sim_params.sample_timesteps,
            sample_at=measure_at,
        )

    state = copy.deepcopy(initial_state) if copy_initial_state else initial_state
    num_cols = _diagnostic_num_columns(sim_params)
    diagnostics = np.zeros((3, num_cols), dtype=np.float64)

    if sim_params.sample_timesteps:
        results = _allocate_worker_observable_buffer(sim_params, len(sim_params.times))
    else:
        results = _allocate_worker_observable_buffer(sim_params, 1)

    # Apply scheduled jumps at t=times[0] before the initial sample so observables
    # and get_state agree (later timesteps also sample after the jump event).
    if noise_model is not None and has_scheduled_jump(noise_model, sim_params.times[0], sim_params.dt):
        state = apply_scheduled_jumps(state, noise_model, sim_params.times[0], sim_params)

    measure_initial = 0 in measure_at if measure_at is not None else sim_params.sample_timesteps
    if measure_initial:
        state._record_diagnostics(diagnostics, 0)  # ruff: ignore[private-member-access]
        state._evaluate_observables(sim_params, results, 0)  # ruff: ignore[private-member-access]

    for j, _ in enumerate(sim_params.times[1:], start=1):
        apply_unitary_evolution(state, _mpo_for_interval(hamiltonian, j - 1), sim_params)
        if noise_model is not None:
            apply_dissipation(state, noise_model, sim_params.dt, sim_params)
            current_time = sim_params.times[j]
            if has_scheduled_jump(noise_model, current_time, sim_params.dt):
                state = apply_scheduled_jumps(state, noise_model, current_time, sim_params)
            else:
                state = stochastic_process(state, noise_model, sim_params.dt, sim_params, rng=rng)

        if record(j):
            col = j if sim_params.sample_timesteps else 0
            state._record_diagnostics(diagnostics, col)  # ruff: ignore[private-member-access]
            if sim_params.sample_timesteps:
                state._evaluate_observables(sim_params, results, j)  # ruff: ignore[private-member-access]
            else:
                state._evaluate_observables(sim_params, results)  # ruff: ignore[private-member-access]

    # Final-only runs with elapsed_time=0 never enter the loop above.
    if not measure_initial and not sim_params.sample_timesteps and n_times <= 1:
        state._record_diagnostics(diagnostics, 0)  # ruff: ignore[private-member-access]
        state._evaluate_observables(sim_params, results)  # ruff: ignore[private-member-access]

    final_state = state if sim_params.get_state else None
    return results, diagnostics, final_state
