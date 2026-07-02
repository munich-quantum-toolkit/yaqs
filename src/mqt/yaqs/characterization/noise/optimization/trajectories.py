# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reference trajectories and loss assembly for analytical noise optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.characterization.noise.optimization.loss import TrajectoryLoss
from mqt.yaqs.characterization.noise.shared.propagation import Propagator
from mqt.yaqs.characterization.noise.shared.representation import (
    DEFAULT_LINDBLAD_MAX_QUBITS,
    DEFAULT_VECTOR_MAX_QUBITS,
    prepare_state_for_representation,
    resolve_noise_representation,
)
from mqt.yaqs.simulator import Simulator

if TYPE_CHECKING:
    from mqt.yaqs.characterization.noise.shared.representation import NoiseRepresentation
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.data_structures.state import State
    from mqt.yaqs.core.parallel_utils import ExecutionConfig


def build_simulator(execution: ExecutionConfig) -> Simulator:
    """Construct a :class:`~mqt.yaqs.Simulator` from execution settings.

    Args:
        execution: Parallelism and progress configuration.

    Returns:
        Configured simulator instance.
    """
    return Simulator(
        parallel=execution.parallel,
        max_workers=execution.max_workers,
        show_progress=execution.show_progress,
        mp_context=execution.mp_context,
        max_retries=execution.max_retries,
        retry_exceptions=execution.retry_exceptions,
    )


def resolve_prepared_state(
    hamiltonian: Hamiltonian,
    init_state: State,
    representation: NoiseRepresentation,
    *,
    lindblad_max_qubits: int,
    vector_max_qubits: int,
) -> State:
    """Resolve representation and encode the initial state.

    Args:
        hamiltonian: System Hamiltonian (used for chain length).
        init_state: User-supplied initial state.
        representation: Forward-model selection.
        lindblad_max_qubits: Auto cutover to Lindblad evolution.
        vector_max_qubits: Auto cutover from MCWF to TJM.

    Returns:
        Initial state encoded for the resolved forward backend.
    """
    resolved = resolve_noise_representation(
        hamiltonian.length,
        representation,
        lindblad_max_qubits=lindblad_max_qubits,
        vector_max_qubits=vector_max_qubits,
    )
    return prepare_state_for_representation(init_state, resolved)


def simulate_observable_trajectories(
    *,
    sim_params: AnalogSimParams,
    hamiltonian: Hamiltonian,
    init_state: State,
    noise_model: NoiseModel,
    observables: list[Observable],
    simulator: Simulator | None = None,
    representation: NoiseRepresentation = "auto",
    lindblad_max_qubits: int = DEFAULT_LINDBLAD_MAX_QUBITS,
    vector_max_qubits: int = DEFAULT_VECTOR_MAX_QUBITS,
) -> tuple[np.ndarray, np.ndarray, State]:
    """Simulate observable expectation trajectories under a noise model.

    Args:
        sim_params: Analog simulation parameters.
        hamiltonian: System Hamiltonian.
        init_state: Initial state.
        noise_model: Noise model whose strengths are propagated.
        observables: Observables to track.
        simulator: Optional simulator instance.
        representation: Forward-model selection.
        lindblad_max_qubits: Auto cutover to Lindblad evolution.
        vector_max_qubits: Auto cutover from MCWF to TJM.

    Returns:
        Tuple ``(expectations, times, prepared_state)`` with expectations shaped
        ``(n_obs, n_times)``.
    """
    prepared_state = resolve_prepared_state(
        hamiltonian,
        init_state,
        representation,
        lindblad_max_qubits=lindblad_max_qubits,
        vector_max_qubits=vector_max_qubits,
    )
    fit_simulator = simulator or Simulator(show_progress=False)
    propagator = Propagator(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        noise_model=noise_model,
        init_state=prepared_state,
        simulator=fit_simulator,
    )
    propagator.set_observable_list(observables)
    propagator.run(noise_model)
    return (
        np.asarray(propagator.obs_array, dtype=float),
        np.asarray(propagator.times, dtype=float),
        prepared_state,
    )


def resolve_reference_expectations(
    *,
    sim_params: AnalogSimParams,
    hamiltonian: Hamiltonian,
    init_state: State,
    observables: list[Observable],
    reference_model: NoiseModel | None,
    ref_expectations: np.ndarray | None,
    simulator: Simulator,
    representation: NoiseRepresentation,
    lindblad_max_qubits: int,
    vector_max_qubits: int,
) -> tuple[np.ndarray, np.ndarray, State | None]:
    """Build or validate the reference trajectory used for fitting.

    Args:
        sim_params: Analog simulation parameters.
        hamiltonian: System Hamiltonian.
        init_state: Initial state.
        observables: Fitting observables.
        reference_model: Optional model used to simulate the reference.
        ref_expectations: Optional precomputed experimental trajectories.
        simulator: Simulator used when simulating ``reference_model``.
        representation: Forward-model selection.
        lindblad_max_qubits: Auto cutover to Lindblad evolution.
        vector_max_qubits: Auto cutover from MCWF to TJM.

    Returns:
        Tuple ``(ref_expectations, times, prepared_state)``. ``prepared_state`` is
        set when ``reference_model`` was simulated internally.

    Raises:
        ValueError: If neither or both reference sources are supplied, or shapes mismatch.
    """
    if (reference_model is None) == (ref_expectations is None):
        msg = "Specify exactly one of reference_model= or ref_expectations=."
        raise ValueError(msg)

    if ref_expectations is not None:
        ref_array = np.asarray(ref_expectations, dtype=float)
        if ref_array.ndim != 2:
            msg = f"ref_expectations must be 2-D, got shape {ref_array.shape}."
            raise ValueError(msg)
        if ref_array.shape[0] != len(observables):
            msg = (
                f"ref_expectations has {ref_array.shape[0]} rows but {len(observables)} fitting observables were given."
            )
            raise ValueError(msg)
        times = np.asarray(sim_params.times, dtype=float)
        if ref_array.shape[1] != len(times):
            msg = f"ref_expectations has {ref_array.shape[1]} columns but sim_params defines {len(times)} time samples."
            raise ValueError(msg)
        return ref_array, times, None

    if reference_model is None:
        msg = "reference_model is required when ref_expectations is omitted."
        raise ValueError(msg)
    ref_array, times, prepared_state = simulate_observable_trajectories(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        noise_model=reference_model,
        observables=observables,
        simulator=simulator,
        representation=representation,
        lindblad_max_qubits=lindblad_max_qubits,
        vector_max_qubits=vector_max_qubits,
    )
    return ref_array, times, prepared_state


def build_trajectory_loss(
    *,
    sim_params: AnalogSimParams,
    hamiltonian: Hamiltonian,
    init_state: State,
    init_guess: NoiseModel,
    observables: list[Observable],
    ref_expectations: np.ndarray,
    simulator: Simulator,
    representation: NoiseRepresentation,
    lindblad_max_qubits: int,
    vector_max_qubits: int,
    prepared_state: State | None = None,
) -> tuple[TrajectoryLoss, Propagator]:
    """Wire a trajectory loss and fit propagator for optimization.

    Args:
        sim_params: Analog simulation parameters.
        hamiltonian: System Hamiltonian.
        init_state: Initial state.
        init_guess: Initial noise guess defining the fit topology.
        observables: Fitting observables.
        ref_expectations: Target trajectories with shape ``(n_obs, n_times)``.
        simulator: Simulator used for forward propagation.
        representation: Forward-model selection.
        lindblad_max_qubits: Auto cutover to Lindblad evolution.
        vector_max_qubits: Auto cutover from MCWF to TJM.
        prepared_state: Optional state already encoded for the forward backend.

    Returns:
        Tuple of loss and fit propagator.
    """
    if prepared_state is None:
        prepared_state = resolve_prepared_state(
            hamiltonian,
            init_state,
            representation,
            lindblad_max_qubits=lindblad_max_qubits,
            vector_max_qubits=vector_max_qubits,
        )
    fit_propagator = Propagator(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        noise_model=init_guess,
        init_state=prepared_state,
        simulator=simulator,
    )
    fit_propagator.set_observable_list(observables)
    loss = TrajectoryLoss(
        ref_expectations=np.asarray(ref_expectations, dtype=float),
        propagator=fit_propagator,
    )
    return loss, fit_propagator
