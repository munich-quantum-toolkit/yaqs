# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Trajectory-matching orchestration for Markovian noise characterization."""

# ruff: noqa: ANN401 -- optimizer kwargs forwarded to CMA-ES

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from mqt.yaqs.characterization.noise.backends.gradient_free.cma import cma_opt
from mqt.yaqs.characterization.noise.trajectory_matching.reference import (
    build_simulator,
    build_trajectory_loss,
    resolve_reference_expectations,
    simulate_observable_trajectories,
)
from mqt.yaqs.characterization.noise.trajectory_matching.results import NoiseCharacterizationResult
from mqt.yaqs.characterization.noise.shared.representation import (
    DEFAULT_LINDBLAD_MAX_QUBITS,
    DEFAULT_VECTOR_MAX_QUBITS,
    NoiseRepresentation,
)

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.data_structures.state import State
    from mqt.yaqs.core.parallel_utils import ExecutionConfig


def run_trajectory_characterization(
    *,
    hamiltonian: Hamiltonian,
    sim_params: AnalogSimParams,
    init_state: State,
    init_guess: CompactNoiseModel,
    observables: list[Observable],
    x_low: np.ndarray,
    x_up: np.ndarray,
    reference_model: CompactNoiseModel | None = None,
    ref_expectations: np.ndarray | None = None,
    optimizer: Literal["cma"] = "cma",
    execution: ExecutionConfig,
    representation: NoiseRepresentation = "auto",
    lindblad_max_qubits: int = DEFAULT_LINDBLAD_MAX_QUBITS,
    vector_max_qubits: int = DEFAULT_VECTOR_MAX_QUBITS,
    **optimizer_kwargs: Any,
) -> NoiseCharacterizationResult:
    """Fit compact noise strengths by matching observable trajectories.

    Args:
        hamiltonian: System Hamiltonian.
        sim_params: Analog simulation parameters.
        init_state: Initial state.
        init_guess: Initial compact noise guess.
        observables: Fitting observables whose trajectories are matched.
        x_low: Lower parameter bounds.
        x_up: Upper parameter bounds.
        reference_model: Optional reference model to simulate target trajectories.
        ref_expectations: Optional experimental trajectories with shape ``(n_obs, n_times)``.
        optimizer: Optimizer backend (``"cma"`` only).
        execution: Parallel execution configuration.
        representation: Forward-model selection.
        lindblad_max_qubits: Auto cutover to Lindblad evolution.
        vector_max_qubits: Auto cutover from MCWF to TJM.
        **optimizer_kwargs: Keyword arguments forwarded to the optimizer backend.

    Returns:
        Structured optimization result including optional trajectory arrays.

    Raises:
        ValueError: If reference inputs are invalid or an unsupported optimizer is requested.
    """
    if optimizer != "cma":
        msg = f"optimizer must be 'cma', got {optimizer!r}."
        raise ValueError(msg)

    simulator = build_simulator(execution)
    ref_array, times, resolved = resolve_reference_expectations(
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
    )

    x_best, best_loss, loss_history, parameter_history = cma_opt(
        loss,
        init_guess.strength_list.copy(),
        x_low=x_low,
        x_up=x_up,
        **optimizer_kwargs,
    )
    optimal_model = loss.x_to_noise_model(x_best)
    propagator.run(optimal_model)
    fit_traj = np.asarray(propagator.obs_array, dtype=float)

    return NoiseCharacterizationResult(
        optimal_model=optimal_model,
        best_loss=float(best_loss),
        best_parameters=np.asarray(x_best, dtype=float),
        loss_history=loss_history,
        parameter_history=parameter_history,
        ref_traj=ref_array,
        fit_traj=fit_traj,
        times=times,
        resolved_representation=resolved,
        fitting_observables=list(observables),
    )


def simulate_fit_trajectory(
    *,
    hamiltonian: Hamiltonian,
    sim_params: AnalogSimParams,
    init_state: State,
    noise_model: CompactNoiseModel,
    observables: list[Observable],
    execution: ExecutionConfig,
    representation: NoiseRepresentation = "auto",
    lindblad_max_qubits: int = DEFAULT_LINDBLAD_MAX_QUBITS,
    vector_max_qubits: int = DEFAULT_VECTOR_MAX_QUBITS,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate trajectories for a candidate noise model (evaluation helper).

    Returns:
        Tuple ``(expectations, times)``.
    """
    simulator = build_simulator(execution)
    expectations, times, _ = simulate_observable_trajectories(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        noise_model=noise_model,
        observables=observables,
        simulator=simulator,
        representation=representation,
        lindblad_max_qubits=lindblad_max_qubits,
        vector_max_qubits=vector_max_qubits,
    )
    return expectations, times
