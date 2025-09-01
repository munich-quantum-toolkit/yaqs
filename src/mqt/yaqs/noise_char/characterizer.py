# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the optimization routines for noise characterization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mqt.yaqs.noise_char.optimization import LossClass, adam_optimizer
from mqt.yaqs.noise_char.propagation import PropagatorWithGradients

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from mqt.yaqs.core.data_structures.networks import MPO, MPS
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable


class Characterizer:
    loss: LossClass

    traj_gradients: PropagatorWithGradients

    init_guess: NoiseModel

    init_x: np.ndarray

    loss_history: list[float]

    x_history: list[np.ndarray]

    x_avg_history: list[np.ndarray]

    times: np.ndarray

    observable_traj: np.ndarray

    def __init__(
        self,
        *,
        sim_params: AnalogSimParams,
        hamiltonian: MPO,
        init_guess: NoiseModel,
        init_state: MPS,
        ref_traj: list[Observable],
        print_to_file: bool = True,
    ) -> None:
        self.init_guess = init_guess

        self.traj_gradients = PropagatorWithGradients(
            sim_params=sim_params, hamiltonian=hamiltonian, noise_model=init_guess, init_state=init_state
        )

        self.loss = LossClass(ref_traj=ref_traj, traj_gradients=self.traj_gradients, print_to_file=print_to_file)

        self.init_x = self.loss.noise_model_to_x(self.init_guess)

    def adam_optimize(
        self,
        *,
        alpha: float = 0.05,
        max_iterations: int = 100,
        threshold: float = 5e-4,
        max_n_convergence: int = 50,
        tolerance: float = 1e-8,
        beta1: float = 0.5,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        restart: bool = False,
        restart_file: Path | None = None,
    ) -> None:
        self.loss_history, self.x_history, self.x_avg_history, self.times, self.observable_traj = adam_optimizer(
            self.loss,
            self.init_x,
            alpha=alpha,
            max_iterations=max_iterations,
            threshold=threshold,
            max_n_convergence=max_n_convergence,
            tolerance=tolerance,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            restart=restart,
            restart_file=restart_file,
        )
