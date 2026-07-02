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

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.data_structures.state import State
    from mqt.yaqs.simulator import Simulator


class NoiseCharacterizer:
    """Fit compact Lindblad jump rates by matching observable trajectories."""

    def __init__(
        self,
        *,
        propagator: Propagator,
        init_guess: CompactNoiseModel,
        loss: TrajectoryLoss,
    ) -> None:
        """Wire together the forward model and trajectory loss.

        Args:
            propagator: Forward model used during optimization.
            init_guess: Initial compact noise model.
            loss: Trajectory-matching loss built from a reference run.
        """
        self.propagator = propagator
        self.init_guess = copy.deepcopy(init_guess)
        self.loss = loss
        self.init_x = self.init_guess.strength_list.copy()
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
            simulator: Optional simulator instance.

        Returns:
            Configured :class:`NoiseCharacterizer`.
        """
        reference_propagator = Propagator(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            compact_noise_model=reference_model,
            init_state=init_state,
            simulator=simulator,
        )
        reference_propagator.set_observable_list(observables)
        reference_propagator.run(reference_model)

        fit_propagator = Propagator(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            compact_noise_model=init_guess,
            init_state=init_state,
            simulator=simulator,
        )
        fit_propagator.set_observable_list(observables)
        loss = TrajectoryLoss(
            ref_expectations=np.asarray(reference_propagator.obs_array, dtype=float),
            propagator=fit_propagator,
        )
        return cls(propagator=loss.propagator, init_guess=init_guess, loss=loss)

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
            **kwargs: Keyword arguments forwarded to :func:`~mqt.yaqs.characterization.noise.backends.gradient_free.cma.cma_opt`.

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
