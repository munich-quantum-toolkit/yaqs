# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing entry point for Markovian noise-parameter characterization."""

# ruff: noqa: ANN401 -- optimizer kwargs forwarded to CMA-ES

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.yaqs.characterization.noise.backends.gradient_free.cma import cma_opt
from mqt.yaqs.characterization.noise.protocol.results import NoiseCharacterizationResult
from mqt.yaqs.characterization.noise.shared.loss import TrajectoryLoss
from mqt.yaqs.characterization.noise.shared.propagation import Propagator
from mqt.yaqs.characterization.noise.shared.representation import (
    DEFAULT_LINDBLAD_MAX_QUBITS,
    DEFAULT_VECTOR_MAX_QUBITS,
    NoiseRepresentation,
    ResolvedNoiseRepresentation,
    prepare_state_for_representation,
    resolve_noise_representation,
)
from mqt.yaqs.simulator import Simulator

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.data_structures.state import State


class NoiseCharacterizer:
    """Fit compact Lindblad jump rates by matching observable trajectories."""

    def __init__(
        self,
        *,
        propagator: Propagator,
        init_guess: CompactNoiseModel,
        loss: TrajectoryLoss,
        representation: NoiseRepresentation = "auto",
        resolved_representation: ResolvedNoiseRepresentation | None = None,
        lindblad_max_qubits: int = DEFAULT_LINDBLAD_MAX_QUBITS,
        vector_max_qubits: int = DEFAULT_VECTOR_MAX_QUBITS,
        simulator: Simulator | None = None,
    ) -> None:
        """Wire together the forward model and trajectory loss.

        Args:
            propagator: Forward model used during optimization.
            init_guess: Initial compact noise model.
            loss: Trajectory-matching loss built from a reference run.
            representation: Forward-model selection (``"auto"`` prefers Lindblad on small chains).
            resolved_representation: Concrete backend used for propagation.
            lindblad_max_qubits: Auto cutover to Lindblad master-equation evolution.
            vector_max_qubits: Auto cutover from MCWF to TJM.
            simulator: Optional :class:`~mqt.yaqs.Simulator` used during fitting.
        """
        self.propagator = propagator
        self.init_guess = copy.deepcopy(init_guess)
        self.loss = loss
        self.init_x = self.init_guess.strength_list.copy()
        self.representation = representation
        self.resolved_representation = resolved_representation or propagator.representation
        self.lindblad_max_qubits = int(lindblad_max_qubits)
        self.vector_max_qubits = int(vector_max_qubits)
        self.simulator = simulator
        self.result: NoiseCharacterizationResult | None = None

    @classmethod
    def from_reference(
        cls,
        *,
        sim_params: AnalogSimParams,
        hamiltonian: Hamiltonian,
        init_state: State,
        reference_model: CompactNoiseModel,
        init_guess: CompactNoiseModel,
        observables: list[Observable],
        representation: NoiseRepresentation = "auto",
        lindblad_max_qubits: int = DEFAULT_LINDBLAD_MAX_QUBITS,
        vector_max_qubits: int = DEFAULT_VECTOR_MAX_QUBITS,
        parallel: bool = False,
        max_workers: int | None = None,
        show_progress: bool = False,
        simulator: Simulator | None = None,
    ) -> NoiseCharacterizer:
        """Build a characterizer from a reference noise model and observable set.

        Args:
            sim_params: Analog simulation parameters.
            hamiltonian: System Hamiltonian.
            init_state: Initial state.
            reference_model: Known noise model used to generate the target trajectory.
            init_guess: Initial optimization guess.
            observables: Observables whose trajectories are matched.
            representation: ``"density_matrix"`` (Lindblad), ``"vector"`` (MCWF), ``"mps"`` (TJM),
                or ``"auto"`` (Lindblad-first by chain length).
            lindblad_max_qubits: Auto cutover to Lindblad master-equation evolution.
            vector_max_qubits: Auto cutover from MCWF to TJM.
            parallel: Whether to parallelize trajectory execution in :class:`~mqt.yaqs.Simulator`.
            max_workers: Worker cap when ``parallel=True``.
            show_progress: Whether to show tqdm progress during propagation.
            simulator: Optional simulator instance.

        Returns:
            Configured :class:`NoiseCharacterizer`.
        """
        resolved = resolve_noise_representation(
            hamiltonian.length,
            representation,
            lindblad_max_qubits=lindblad_max_qubits,
            vector_max_qubits=vector_max_qubits,
        )
        prepared_state = prepare_state_for_representation(init_state, resolved)
        fit_simulator = simulator or Simulator(
            parallel=parallel,
            max_workers=max_workers,
            show_progress=show_progress,
        )

        reference_propagator = Propagator(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            compact_noise_model=reference_model,
            init_state=prepared_state,
            simulator=fit_simulator,
        )
        reference_propagator.set_observable_list(observables)
        reference_propagator.run(reference_model)

        fit_propagator = Propagator(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            compact_noise_model=init_guess,
            init_state=prepared_state,
            simulator=fit_simulator,
        )
        fit_propagator.set_observable_list(observables)
        loss = TrajectoryLoss(
            ref_expectations=np.asarray(reference_propagator.obs_array, dtype=float),
            propagator=fit_propagator,
        )
        return cls(
            propagator=loss.propagator,
            init_guess=init_guess,
            loss=loss,
            representation=representation,
            resolved_representation=resolved,
            lindblad_max_qubits=lindblad_max_qubits,
            vector_max_qubits=vector_max_qubits,
            simulator=fit_simulator,
        )

    def optimize(
        self,
        *,
        x_low: np.ndarray,
        x_up: np.ndarray,
        **kwargs: Any,
    ) -> NoiseCharacterizationResult:
        """Run CMA-ES to minimize the trajectory-matching loss.

        Args:
            x_low: Lower parameter bounds.
            x_up: Upper parameter bounds.
            **kwargs: Keyword arguments forwarded to
                :func:`~mqt.yaqs.characterization.noise.backends.gradient_free.cma.cma_opt`.

        Returns:
            Structured optimization result.
        """
        x_best, best_loss, loss_history, parameter_history = cma_opt(
            self.loss,
            self.init_x,
            x_low=x_low,
            x_up=x_up,
            **kwargs,
        )

        optimal_model = self.loss.x_to_noise_model(x_best)
        self.result = NoiseCharacterizationResult(
            optimal_model=optimal_model,
            best_loss=float(best_loss),
            best_parameters=np.asarray(x_best, dtype=float),
            loss_history=loss_history,
            parameter_history=parameter_history,
        )
        return self.result
