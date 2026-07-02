# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Reference trajectories and loss assembly for trajectory matching."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.characterization.noise.shared.loss import TrajectoryLoss
from mqt.yaqs.characterization.noise.shared.propagation import Propagator
from mqt.yaqs.characterization.noise.shared.representation import (
    ResolvedNoiseRepresentation,
    prepare_state_for_representation,
    resolve_noise_representation,
)
from mqt.yaqs.simulator import Simulator

if TYPE_CHECKING:
    from mqt.yaqs.characterization.noise.shared.representation import NoiseRepresentation
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
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
) -> tuple[ResolvedNoiseRepresentation, State]:
    """Resolve representation and encode the initial state.

    Args:
        hamiltonian: System Hamiltonian (used for chain length).
        init_state: User-supplied initial state.
        representation: Forward-model selection.
        lindblad_max_qubits: Auto cutover to Lindblad evolution.
        vector_max_qubits: Auto cutover from MCWF to TJM.

    Returns:
        Tuple of resolved representation and prepared state.
    """
    resolved = resolve_noise_representation(
        hamiltonian.length,
        representation,
        lindblad_max_qubits=lindblad_max_qubits,
        vector_max_qubits=vector_max_qubits,
    )
    prepared_state = prepare_state_for_representation(init_state, resolved)
    return resolved, prepared_state


def simulate_observable_trajectories(
    *,
    sim_params: AnalogSimParams,
    hamiltonian: Hamiltonian,
    init_state: State,
    noise_model: CompactNoiseModel,
    observables: list[Observable],
    simulator: Simulator | None = None,
    representation: NoiseRepresentation = "auto",
    lindblad_max_qubits: int = 8,
    vector_max_qubits: int = 10,
) -> tuple[np.ndarray, np.ndarray, ResolvedNoiseRepresentation]:
    """Simulate observable expectation trajectories under a compact noise model.

    Args:
        sim_params: Analog simulation parameters.
        hamiltonian: System Hamiltonian.
        init_state: Initial state.
        noise_model: Compact noise model whose strengths are propagated.
        observables: Observables to track.
        simulator: Optional simulator instance.
        representation: Forward-model selection.
        lindblad_max_qubits: Auto cutover to Lindblad evolution.
        vector_max_qubits: Auto cutover from MCWF to TJM.

    Returns:
        Tuple ``(expectations, times, resolved_representation)`` with expectations
        shaped ``(n_obs, n_times)``.
    """
    resolved, prepared_state = resolve_prepared_state(
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
        compact_noise_model=noise_model,
        init_state=prepared_state,
        simulator=fit_simulator,
    )
    propagator.set_observable_list(observables)
    propagator.run(noise_model)
    return (
        np.asarray(propagator.obs_array, dtype=float),
        np.asarray(propagator.times, dtype=float),
        resolved,
    )


def resolve_reference_expectations(
    *,
    sim_params: AnalogSimParams,
    hamiltonian: Hamiltonian,
    init_state: State,
    observables: list[Observable],
    reference_model: CompactNoiseModel | None,
    ref_expectations: np.ndarray | None,
    simulator: Simulator,
    representation: NoiseRepresentation,
    lindblad_max_qubits: int,
    vector_max_qubits: int,
) -> tuple[np.ndarray, np.ndarray, ResolvedNoiseRepresentation]:
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
        Tuple ``(ref_expectations, times, resolved_representation)``.

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
        resolved = resolve_noise_representation(
            hamiltonian.length,
            representation,
            lindblad_max_qubits=lindblad_max_qubits,
            vector_max_qubits=vector_max_qubits,
        )
        return ref_array, times, resolved

    if reference_model is None:
        msg = "reference_model is required when ref_expectations is omitted."
        raise ValueError(msg)
    ref_array, times, resolved = simulate_observable_trajectories(
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
    return ref_array, times, resolved


def build_trajectory_loss(
    *,
    sim_params: AnalogSimParams,
    hamiltonian: Hamiltonian,
    init_state: State,
    init_guess: CompactNoiseModel,
    observables: list[Observable],
    ref_expectations: np.ndarray,
    simulator: Simulator,
    representation: NoiseRepresentation,
    lindblad_max_qubits: int,
    vector_max_qubits: int,
) -> tuple[TrajectoryLoss, Propagator, ResolvedNoiseRepresentation]:
    """Wire a trajectory loss and fit propagator for optimization.

    Args:
        sim_params: Analog simulation parameters.
        hamiltonian: System Hamiltonian.
        init_state: Initial state.
        init_guess: Initial compact noise guess defining the fit topology.
        observables: Fitting observables.
        ref_expectations: Target trajectories with shape ``(n_obs, n_times)``.
        simulator: Simulator used for forward propagation.
        representation: Forward-model selection.
        lindblad_max_qubits: Auto cutover to Lindblad evolution.
        vector_max_qubits: Auto cutover from MCWF to TJM.

    Returns:
        Tuple of loss, fit propagator, and resolved representation.
    """
    resolved, prepared_state = resolve_prepared_state(
        hamiltonian,
        init_state,
        representation,
        lindblad_max_qubits=lindblad_max_qubits,
        vector_max_qubits=vector_max_qubits,
    )
    fit_propagator = Propagator(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        compact_noise_model=init_guess,
        init_state=prepared_state,
        simulator=simulator,
    )
    fit_propagator.set_observable_list(observables)
    loss = TrajectoryLoss(
        ref_expectations=np.asarray(ref_expectations, dtype=float),
        propagator=fit_propagator,
    )
    return loss, fit_propagator, resolved
