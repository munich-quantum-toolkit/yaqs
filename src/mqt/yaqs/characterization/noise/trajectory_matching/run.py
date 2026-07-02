# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Trajectory-matching orchestration for Markovian noise characterization."""

# ruff: noqa: ANN401 -- optimizer kwargs forwarded to CMA-ES

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.yaqs.characterization.noise.backends.gradient_free.cma import cma_opt
from mqt.yaqs.characterization.noise.shared.representation import (
    DEFAULT_LINDBLAD_MAX_QUBITS,
    DEFAULT_VECTOR_MAX_QUBITS,
    NoiseRepresentation,
    ResolvedNoiseRepresentation,
)
from mqt.yaqs.characterization.noise.trajectory_matching.reference import (
    build_simulator,
    build_trajectory_loss,
    resolve_reference_expectations,
)
from mqt.yaqs.characterization.noise.trajectory_matching.results import NoiseCharacterizationResult

if TYPE_CHECKING:
    from mqt.yaqs.characterization.noise.shared.loss import TrajectoryLoss
    from mqt.yaqs.characterization.noise.shared.propagation import Propagator
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.data_structures.state import State
    from mqt.yaqs.core.parallel_utils import ExecutionConfig


def _finalize_result(
    *,
    loss: TrajectoryLoss,
    propagator: Propagator,
    x_best: np.ndarray,
    best_loss: float,
    loss_history: list[float],
    parameter_history: list[np.ndarray],
    ref_traj: np.ndarray,
    times: np.ndarray,
    resolved_representation: ResolvedNoiseRepresentation,
) -> NoiseCharacterizationResult:
    """Build the characterization result after CMA-ES completes.

    Args:
        loss: Wired trajectory loss used during optimization.
        propagator: Forward model used for the final fit trajectory.
        x_best: Best parameter vector found by the optimizer.
        best_loss: Best scalar loss value.
        loss_history: Per-evaluation loss trace.
        parameter_history: Per-evaluation parameter trace.
        ref_traj: Reference trajectories matched during fitting.
        times: Simulation time grid.
        resolved_representation: Forward backend used for propagation.

    Returns:
        Structured optimization result including fitted trajectories.
    """
    optimal_model = loss.x_to_noise_model(x_best)
    propagator.run(optimal_model)
    fit_traj = np.asarray(propagator.obs_array, dtype=float)

    return NoiseCharacterizationResult(
        optimal_model=optimal_model,
        best_loss=float(best_loss),
        best_parameters=np.asarray(x_best, dtype=float),
        loss_history=loss_history,
        parameter_history=parameter_history,
        ref_traj=ref_traj,
        fit_traj=fit_traj,
        times=times,
        resolved_representation=resolved_representation,
    )


def run_trajectory_characterization(
    *,
    hamiltonian: Hamiltonian,
    sim_params: AnalogSimParams,
    init_state: State,
    init_guess: NoiseModel,
    observables: list[Observable],
    x_low: np.ndarray,
    x_up: np.ndarray,
    reference_model: NoiseModel | None = None,
    ref_expectations: np.ndarray | None = None,
    execution: ExecutionConfig,
    representation: NoiseRepresentation = "auto",
    lindblad_max_qubits: int = DEFAULT_LINDBLAD_MAX_QUBITS,
    vector_max_qubits: int = DEFAULT_VECTOR_MAX_QUBITS,
    **optimizer_kwargs: Any,
) -> NoiseCharacterizationResult:
    """Fit noise strengths by matching observable trajectories.

    Args:
        hamiltonian: System Hamiltonian.
        sim_params: Analog simulation parameters.
        init_state: Initial state.
        init_guess: Initial noise guess.
        observables: Fitting observables whose trajectories are matched.
        x_low: Lower parameter bounds.
        x_up: Upper parameter bounds.
        reference_model: Optional reference model to simulate target trajectories.
        ref_expectations: Optional experimental trajectories with shape ``(n_obs, n_times)``.
        execution: Parallel execution configuration.
        representation: Forward-model selection.
        lindblad_max_qubits: Auto cutover to Lindblad evolution.
        vector_max_qubits: Auto cutover from MCWF to TJM.
        **optimizer_kwargs: Keyword arguments forwarded to the CMA-ES backend.

    Returns:
        Structured optimization result including optional trajectory arrays.
    """
    simulator = build_simulator(execution)
    ref_array, times, resolved, prepared_state = resolve_reference_expectations(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        observables=observables,
        reference_model=reference_model,
        ref_expectations=ref_expectations,
        simulator=simulator,
        representation=representation,
        lindblad_max_qubits=lindblad_max_qubits,
        vector_max_qubits=vector_max_qubits,
    )
    loss, propagator, resolved = build_trajectory_loss(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        init_guess=init_guess,
        observables=observables,
        ref_expectations=ref_array,
        simulator=simulator,
        representation=representation,
        lindblad_max_qubits=lindblad_max_qubits,
        vector_max_qubits=vector_max_qubits,
        prepared_state=prepared_state,
    )

    x_best, best_loss, loss_history, parameter_history = cma_opt(
        loss,
        np.array([proc["strength"] for proc in init_guess.processes], dtype=float),
        x_low=x_low,
        x_up=x_up,
        **optimizer_kwargs,
    )

    return _finalize_result(
        loss=loss,
        propagator=propagator,
        x_best=x_best,
        best_loss=best_loss,
        loss_history=loss_history,
        parameter_history=parameter_history,
        ref_traj=ref_array,
        times=times,
        resolved_representation=resolved,
    )
