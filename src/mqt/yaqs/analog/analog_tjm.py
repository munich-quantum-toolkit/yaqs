# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Hamiltonian simulation of quantum many-body systems using the Tensor Jump Method (TJM).

This module implements the Tensor Jump Method (TJM) for simulating the dynamics of quantum many-body systems.
It provides functions for initializing the sampling state with noise (via dissipation and stochastic processes),
evolving the state with the configured unitary evolution mode (TDVP or BUG), and sampling observable
measurements over time. The functions analog_tjm_2 and analog_tjm_1 correspond to second-order and
first-order TJM schemes, respectively, and return trajectories of expectation values for further analysis.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from ..core.data_structures.hamiltonian_schedule import HamiltonianSchedule
from ..core.data_structures.mpo import MPO
from ..core.methods.dissipation import apply_dissipation
from ..core.methods.scheduled_jumps import apply_scheduled_jumps, has_scheduled_jump
from ..core.methods.stochastic_process import stochastic_process
from ..core.methods.tdvp import tdvp
from ..core.random_utils import make_sample_rng, make_trajectory_rng
from .evolution import apply_unitary_evolution

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.mps import MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import AnalogSimParams

HamiltonianOperator = MPO | HamiltonianSchedule


def _interval_duration(sim_params: AnalogSimParams, interval_index: int) -> float:
    """Return the exact duration of one physical interval."""
    return float(sim_params.times[interval_index + 1] - sim_params.times[interval_index])


def _evolve_interval(
    state: MPS,
    hamiltonian: HamiltonianOperator,
    sim_params: AnalogSimParams,
    interval_index: int,
) -> None:
    """Apply one static or midpoint-resolved unitary interval in place."""
    duration = _interval_duration(sim_params, interval_index)
    if isinstance(hamiltonian, HamiltonianSchedule):
        interval = hamiltonian.intervals[interval_index]
        for substep in interval.substeps:
            tdvp(
                state,
                hamiltonian.resolve(substep),
                sim_params,
                step_duration=substep.duration,
                num_sweeps=1,
            )
        return
    apply_unitary_evolution(state, hamiltonian, sim_params, step_duration=duration)


def initialize(
    state: MPS,
    noise_model: NoiseModel | None,
    sim_params: AnalogSimParams,
    rng: np.random.Generator | None = None,
    *,
    step_duration: float | None = None,
) -> MPS:
    """Initialize the sampling MPS for second-order Trotterization.

    This function prepares the initial sampling MPS (denoted as Phi(0)) by applying a half time step of dissipation
    followed by a stochastic process. It corresponds to F0 in the TJM paper.

    Args:
        state (MPS): The initial state of the system.
        noise_model (NoiseModel | None): The noise model to apply to the system.
        sim_params (AnalogSimParams): Simulation parameters including the time step (dt).
        rng: The random number generator to use.
        step_duration: Exact duration represented by this half-step initialization.

    Returns:
        MPS: The initialized sampling MPS Phi(0).
    """
    duration = sim_params.dt if step_duration is None else step_duration
    apply_dissipation(state, noise_model, duration / 2, sim_params)
    # Check for scheduled jumps at start time
    current_time = sim_params.times[0]
    if has_scheduled_jump(noise_model, current_time, duration):
        return apply_scheduled_jumps(state, noise_model, current_time, sim_params, dt=duration)
    return stochastic_process(state, noise_model, duration, sim_params, rng=rng)


def step_through(
    state: MPS,
    hamiltonian: HamiltonianOperator,
    noise_model: NoiseModel | None,
    sim_params: AnalogSimParams,
    current_time: float,
    rng: np.random.Generator | None = None,
    *,
    interval_index: int = 0,
    next_interval_index: int | None = None,
) -> MPS:
    """Perform a single time step evolution of the system state using the TJM.

    Corresponding to Fj in the TJM paper, this function evolves the state by applying the configured
    unitary evolution mode (TDVP or BUG), dissipation, and a stochastic process in sequence.

    Args:
        state (MPS): The current state of the system.
        hamiltonian (MPO): The Hamiltonian operator for the system.
        noise_model (NoiseModel | None): The noise model to apply to the system.
        sim_params (AnalogSimParams): Simulation parameters including the time step and measurement settings.
        current_time (float): The current simulation time.
        rng: The random number generator to use.
        interval_index: Interval evolved before applying noise.
        next_interval_index: Optional following interval used to center the
            order-2 noise bridge across unequal durations.

    Returns:
        MPS: The updated state after one time step evolution.
    """
    _evolve_interval(state, hamiltonian, sim_params, interval_index)
    duration = _interval_duration(sim_params, interval_index)
    if next_interval_index is not None:
        duration = 0.5 * (duration + _interval_duration(sim_params, next_interval_index))
    apply_dissipation(state, noise_model, duration, sim_params)

    if has_scheduled_jump(noise_model, current_time, duration):
        return apply_scheduled_jumps(state, noise_model, current_time, sim_params, dt=duration)
    return stochastic_process(state, noise_model, duration, sim_params, rng=rng)


def sample(
    phi: MPS,
    hamiltonian: HamiltonianOperator,
    noise_model: NoiseModel | None,
    sim_params: AnalogSimParams,
    results: NDArray[np.float64],
    j: int,
    rng: np.random.Generator | None = None,
    diagnostics: NDArray[np.float64] | None = None,
    *,
    interval_index: int | None = None,
) -> MPS | None:
    """Sample the quantum state and record observable measurements from the sampling MPS.

    This function evolves a deep copy of the sampling MPS, applies dissipation and a stochastic process,
    and then measures the observables specified in sim_params. The measured values are stored in the provided
    results array at index j (or at index 0 if only one measurement is taken).

    Args:
        phi (MPS): The sampling MPS prior to measurement.
        hamiltonian (MPO): The Hamiltonian operator for the system.
        noise_model (NoiseModel | None): The noise model to apply during evolution.
        sim_params (AnalogSimParams): Simulation parameters including time step and measurement settings.
        results (NDArray[np.float64]): An array to store the measured observable values.
        j (int): The time step or shot index at which the measurement is recorded.
        rng: RNG for jump decisions on the measurement copy. Must be independent of the
            trajectory RNG used for ``initialize`` / ``step_through`` so intermediate
            sampling does not alter subsequent evolution.
        diagnostics: Optional ``(3, T)`` buffer for runtime cost, max bond, and total bond.
        interval_index: Exact interval evolved on the measurement copy.

    Returns:
        The evolved MPS when this is the final time step and ``get_state=True``, else ``None``.
    """
    psi = copy.deepcopy(phi)
    resolved_interval_index = max(j - 1, 0) if interval_index is None else interval_index
    _evolve_interval(psi, hamiltonian, sim_params, resolved_interval_index)
    duration = _interval_duration(sim_params, resolved_interval_index)
    apply_dissipation(psi, noise_model, duration / 2, sim_params)

    current_time = sim_params.times[j]
    if has_scheduled_jump(noise_model, current_time, duration):
        psi = apply_scheduled_jumps(psi, noise_model, current_time, sim_params, dt=duration)
    else:
        psi = stochastic_process(psi, noise_model, duration, sim_params, rng=rng)
    col = j if sim_params.sample_timesteps else 0
    if diagnostics is not None:
        psi.record_diagnostics(diagnostics, col)
    if sim_params.sample_timesteps:
        psi.evaluate_observables(sim_params, results, j)
    else:
        psi.evaluate_observables(sim_params, results)

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
) -> tuple[NDArray[np.float64], NDArray[np.float64], MPS | None]:
    """Run a single trajectory of the TJM using the configured unitary evolution mode.

    This function executes a full trajectory by evolving the initial state,
    sampling observable measurements over time, and recording the results.
    It corresponds to the second-order TJM scheme; unitary intervals use
    ``sim_params.evolution_mode`` (TDVP or BUG).

    Args:
        args: A tuple containing:
            - Trajectory identifier.
            - The initial MPS.
            - Optional noise model.
            - Simulation parameters (time step, SVD threshold, etc.).
            - Hamiltonian MPO.
        copy_initial_state: Whether to deep-copy the input MPS before evolution.
        rng: Optional externally managed trajectory RNG. When omitted, the
            standalone trajectory seed behavior is preserved.
        sample_timestep_offset: Added to each local measurement timestep when
            deriving sample RNG streams. Program execution accumulates this
            across prior analog segments so split evolutions share one global
            sample timeline with a single continuous run.
        use_trajectory_rng_for_final_sample: When True with an external ``rng``,
            the last measurement copy draws from the trajectory RNG. Programs
            enable this only on the final order-2 instruction.
        return_trajectory_state: When True, return the trajectory MPS (``phi``)
            for handoff instead of the last measurement copy. Programs always
            enable this so later segments continue the physical trajectory.
        continue_trajectory: When True, skip ``initialize`` and continue
            ``step_through`` from the handed-off trajectory state. Programs set
            this for order-2 segments after the first.

    Returns:
        Observable data, diagnostics ``(3, T)``, and optional final MPS.
    """
    traj_idx, initial_state, noise_model, sim_params, hamiltonian = args

    base_seed = sim_params.random_seed
    external_rng = rng is not None
    if rng is None:
        rng = make_trajectory_rng(traj_idx, base_seed=sim_params.random_seed)
    n_times = len(sim_params.times)

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
        results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)))
    else:
        results = np.zeros((len(sim_params.sorted_observables), 1))

    final_state: MPS | None = None

    # Zero-duration runs: evaluate the initial state before any noise/evolution (F0).
    if n_times == 1:
        state.record_diagnostics(diagnostics, 0)
        if sim_params.sample_timesteps:
            state.evaluate_observables(sim_params, results, 0)
        else:
            state.evaluate_observables(sim_params, results)
        return results, diagnostics, state if (sim_params.get_state or return_trajectory_state) else None

    if continue_trajectory:
        # Mid-Trotter handoff: remeasure the junction with the global sample stream
        # (local 0 + offset matches the prior segment's last sample) without
        # replacing ``phi``, then continue ``step_through``.
        phi = state
        if sim_params.sample_timesteps:
            sample(
                phi,
                hamiltonian,
                noise_model,
                sim_params,
                results,
                j=0,
                rng=measurement_rng(0),
                diagnostics=diagnostics,
            )
        for j, _ in enumerate(sim_params.times[1:], start=1):
            phi = step_through(phi, hamiltonian, noise_model, sim_params, sim_params.times[j], rng=rng)
            if sim_params.sample_timesteps or j == n_times - 1:
                sampled_state = sample(
                    phi,
                    hamiltonian,
                    noise_model,
                    sim_params,
                    results,
                    j,
                    rng=measurement_rng(j),
                    diagnostics=diagnostics,
                )
                if sampled_state is not None:
                    final_state = sampled_state
    else:
        if sim_params.sample_timesteps:
            state.record_diagnostics(diagnostics, 0)
            state.evaluate_observables(sim_params, results, 0)

        first_duration = _interval_duration(sim_params, 0)
        if np.isclose(first_duration, sim_params.dt, rtol=0.0, atol=np.spacing(sim_params.dt) * 8):
            phi = initialize(state, noise_model, sim_params, rng=rng)
        else:
            phi = initialize(state, noise_model, sim_params, rng=rng, step_duration=first_duration)

        # Sample at times[1] whenever it is requested or is the final time (len==2 final-only).
        # Per-timestep sample RNGs so intermediate draws cannot change the final measurement.
        if sim_params.sample_timesteps or n_times == 2:
            sampled_state = sample(
                phi,
                hamiltonian,
                noise_model,
                sim_params,
                results,
                j=1,
                rng=measurement_rng(1),
                diagnostics=diagnostics,
                interval_index=0,
            )
            if sampled_state is not None:
                final_state = sampled_state

        for j, _ in enumerate(sim_params.times[2:], start=2):
            phi = step_through(
                phi,
                hamiltonian,
                noise_model,
                sim_params,
                sim_params.times[j],
                rng=rng,
                interval_index=j - 2,
                next_interval_index=j - 1,
            )
            if sim_params.sample_timesteps or j == n_times - 1:
                sampled_state = sample(
                    phi,
                    hamiltonian,
                    noise_model,
                    sim_params,
                    results,
                    j,
                    rng=measurement_rng(j),
                    diagnostics=diagnostics,
                    interval_index=j - 1,
                )
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], MPS | None]:
    """Run a single trajectory of the TJM using a first-order evolution scheme.

    This function evolves the state with one unitary update per interval
    (TDVP or BUG according to ``sim_params.evolution_mode``), applying noise
    (if provided) and taking observable measurements over time.

    Args:
        args (tuple): A tuple containing:
            - int: Trajectory identifier.
            - MPS: The initial state of the system.
            - NoiseModel | None: The noise model to be applied (if any).
            - AnalogSimParams: Simulation parameters including the time step and measurement settings.
            - MPO: The Hamiltonian operator represented as an MPO.
        copy_initial_state: Whether to deep-copy the input MPS before evolution.
        rng: Optional externally managed trajectory RNG. When omitted, the
            standalone trajectory seed behavior is preserved.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64], MPS | None]:
            Observable data, diagnostics ``(3, T)``, and optional final MPS.
    """
    traj_idx, initial_state, noise_model, sim_params, hamiltonian = args

    if rng is None:
        rng = make_trajectory_rng(traj_idx, base_seed=sim_params.random_seed)

    state = copy.deepcopy(initial_state) if copy_initial_state else initial_state
    num_cols = _diagnostic_num_columns(sim_params)
    diagnostics = np.zeros((3, num_cols), dtype=np.float64)

    if sim_params.sample_timesteps:
        results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)), dtype=object)
    else:
        results = np.zeros((len(sim_params.sorted_observables), 1), dtype=object)

    # Apply scheduled jumps at t=times[0] before the initial sample so observables
    # and get_state agree (later timesteps also sample after the jump event).
    initial_match_duration = _interval_duration(sim_params, 0) if len(sim_params.times) > 1 else sim_params.dt
    if noise_model is not None and has_scheduled_jump(noise_model, sim_params.times[0], initial_match_duration):
        state = apply_scheduled_jumps(
            state,
            noise_model,
            sim_params.times[0],
            sim_params,
            dt=initial_match_duration,
        )

    if sim_params.sample_timesteps:
        state.record_diagnostics(diagnostics, 0)
        state.evaluate_observables(sim_params, results, 0)

    for j, _ in enumerate(sim_params.times[1:], start=1):
        interval_index = j - 1
        duration = _interval_duration(sim_params, interval_index)
        _evolve_interval(state, hamiltonian, sim_params, interval_index)
        if noise_model is not None:
            apply_dissipation(state, noise_model, duration, sim_params)
            current_time = sim_params.times[j]
            if has_scheduled_jump(noise_model, current_time, duration):
                state = apply_scheduled_jumps(state, noise_model, current_time, sim_params, dt=duration)
            else:
                state = stochastic_process(state, noise_model, duration, sim_params, rng=rng)

        if sim_params.sample_timesteps:
            state.record_diagnostics(diagnostics, j)
            state.evaluate_observables(sim_params, results, j)
        elif j == len(sim_params.times) - 1:
            state.record_diagnostics(diagnostics, 0)
            state.evaluate_observables(sim_params, results)

    # Final-only runs with elapsed_time=0 never enter the loop above.
    if not sim_params.sample_timesteps and len(sim_params.times) <= 1:
        state.record_diagnostics(diagnostics, 0)
        state.evaluate_observables(sim_params, results)

    final_state = state if sim_params.get_state else None
    return results, diagnostics, final_state
