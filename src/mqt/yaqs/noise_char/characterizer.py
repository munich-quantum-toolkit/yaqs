# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the optimization routines for noise characterization."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from mqt.yaqs.noise_char.optimization import adam_optimizer, gradient_descent_optimizer
from mqt.yaqs.noise_char.algorithms.cma import cma_opt
from mqt.yaqs.noise_char.algorithms.bayesian_optimization import bayesian_opt
from mqt.yaqs.noise_char.algorithms.mcmc import mcmc_opt


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
    from mqt.yaqs.noise_char.optimization import LossClass
    from mqt.yaqs.noise_char.propagation import PropagatorWithGradients, Propagator


class Characterizer:
    """High-level helper to estimate parameters of a compact noise model.

    This class wraps a trajectory propagator and a loss evaluator to perform
    gradient-based optimization of a compact noise model's parameters.

    Stored objects
    - propagator (PropagatorWithGradients): Propagator that can produce
      observable trajectories and gradients with respect to compact noise-model
      parameters.
    - init_guess (CompactNoiseModel): Deep copy of the initial noise-model
      guess used to initialize the optimizer.
    - loss (LossClass): Loss object that compares simulated trajectories to a
      reference and provides conversions between CompactNoiseModel and a flat
      optimization vector.

    Key attributes populated by optimization
    - init_x: flattened parameter vector derived from init_guess
    - loss_history, x_history, x_avg_history, times, observable_traj: optimizer
      histories produced by adam_optimize
    - optimal_model: CompactNoiseModel reconstructed from the final averaged
      parameter vector via loss.x_to_noise_model

    The actual numerical optimization is delegated to the external
    adam_optimizer routine. See adam_optimize for usage and available options.
    """

    loss: LossClass

    propagator: PropagatorWithGradients | Propagator

    init_guess: CompactNoiseModel

    init_x: np.ndarray

    loss_history: list[float] | None = None

    x_history: list[np.ndarray] | None = None

    x_avg_history: list[np.ndarray] | None = None

    times: np.ndarray | None = None

    observable_traj: np.ndarray | None = None

    optimal_model: CompactNoiseModel | None = None

    def __init__(
        self,
        *,
        propagator: PropagatorWithGradients | Propagator,
        init_guess: CompactNoiseModel,
        loss: LossClass,
    ) -> None:
        """Initialize the noise characterizer.

        Parameters
        ----------
        propagator : PropagatorWithGradients
            Propagator capable of producing trajectories and gradients with respect
            to the compact noise-model parameters.
        init_guess : CompactNoiseModel
            Initial guess for the compact noise model. A deep copy of this object
            will be stored on the instance.
        loss : LossClass
            Loss object that compares simulated trajectories to a reference and
            provides utilities to convert between compact noise models and flat
            optimization vectors.

        The constructor stores a deep copy of init_guess, assigns propagator
        and loss, and initializes self.init_x from init_guess.strength_list.
        """
        self.init_guess = copy.deepcopy(init_guess)
        self.propagator = propagator

        self.loss = loss

        self.init_x = self.init_guess.strength_list.copy()

    def adam_optimize(
        self,
        *,
        x_low: np.ndarray = None,
        x_up: np.ndarray = None,
        alpha: float = 0.05,
        max_iter: int = 500,
        h: float = 1e-3,
        threshold: float = 5e-4,
        max_n_convergence: int = 50,
        tolerance: float = 1e-8,
        beta1: float = 0.5,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        restart: bool = False,
        restart_file: Path | None = None,
    ) -> None:
        """Run an ADAM-based optimization to fit the noise model parameters.

        This method runs the external `adam_optimizer` routine to minimize the
        callable `self.loss` starting from `self.init_x`. On completion (or
        early termination), several instance attributes are updated with the
        optimizer history and the recovered optimal noise model.
        Parameters.
        ----------
        alpha (float, optional): Learning rate for Adam optimizer. Default is 0.05.
        max_iter (int, optional): Maximum number of optimization iterations. Default is 1000.
        threshold (float, optional): Threshold for parameter convergence check. Default is 5e-4.
        max_n_convergence (int, optional): Number of consecutive iterations to check for convergence. Default is 50.
        tolerance (float, optional): Absolute loss tolerance for early stopping. Default is 1e-8.
        beta1 (float, optional): Exponential decay rate for the first moment estimates. Default is 0.5.
        beta2 (float, optional): Exponential decay rate for the second moment estimates. Default is 0.999.
        epsilon (float, optional): Small constant for numerical stability. Default is 1e-8.
        restart (bool, optional): Whether to restart optimization from a checkpoint. Default is False.
        restart_file (str, optional): Path to a specific checkpoint file to restart from.
        If None, the latest checkpoint in the working directory is used.

        Notes:
        -----
        - The method delegates the numerical optimization to the external
          `adam_optimizer` function and forwards the provided hyperparameters to it.
        - Convergence and restart semantics depend on the `adam_optimizer` implementation.
        - After optimization, the final noise model is constructed by calling
          `self.loss.x_to_noise_model` with the final averaged parameters.

        Raises:
        ------
        ValueError
            If one or more numeric hyperparameters are out of valid ranges (e.g.,
            non-positive learning rate or non-positive max_iter).
        FileNotFoundError
            If `restart` is True and `restart_file` is provided but does not exist.
        RuntimeError
            If the underlying optimizer reports a failure to converge when such
            failures are surfaced as exceptions.

        Examples:
        --------
        >>> # run with default hyperparameters
        >>> obj.adam_optimize()
        >>> # run with a smaller learning rate and more iterations
        >>> obj.adam_optimize(alpha=0.01, max_iter=1000)
        """

        self.loss.return_numeric_gradients = True

        self.loss.epsilon = h

        self.loss_history, self.x_history, self.x_avg_history, self.times, self.observable_traj = adam_optimizer(
            self.loss,
            self.init_x,
            x_low=x_low,
            x_up=x_up,
            alpha=alpha,
            max_iter=max_iter,
            threshold=threshold,
            max_n_convergence=max_n_convergence,
            tolerance=tolerance,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            restart=restart,
            restart_file=restart_file,
        )

        self.optimal_model = self.loss.x_to_noise_model(self.x_avg_history[-1])


    def gradient_descent_optimize(
        self,
        *,
        x_low: np.ndarray = None,
        x_up: np.ndarray = None,
        alpha: float = 0.05,
        max_iter: int = 100,
        threshold: float = 5e-4,
        max_n_convergence: int = 50,
        h: float = 1e-2,
        tolerance: float = 1e-8,
        restart: bool = False,
        restart_file: Path | None = None,
    ) -> None:
        """Run an ADAM-based optimization to fit the noise model parameters.

        This method runs the external `adam_optimizer` routine to minimize the
        callable `self.loss` starting from `self.init_x`. On completion (or
        early termination), several instance attributes are updated with the
        optimizer history and the recovered optimal noise model.
        Parameters.
        ----------
        alpha (float, optional): Learning rate for Adam optimizer. Default is 0.05.
        max_iter (int, optional): Maximum number of optimization iterations. Default is 1000.
        threshold (float, optional): Threshold for parameter convergence check. Default is 5e-4.
        max_n_convergence (int, optional): Number of consecutive iterations to check for convergence. Default is 50.
        tolerance (float, optional): Absolute loss tolerance for early stopping. Default is 1e-8.
        restart (bool, optional): Whether to restart optimization from a checkpoint. Default is False.
        restart_file (str, optional): Path to a specific checkpoint file to restart from.
        If None, the latest checkpoint in the working directory is used.

        Notes:
        -----
        - The method delegates the numerical optimization to the external
          `adam_optimizer` function and forwards the provided hyperparameters to it.
        - Convergence and restart semantics depend on the `adam_optimizer` implementation.
        - After optimization, the final noise model is constructed by calling
          `self.loss.x_to_noise_model` with the final averaged parameters.

        Raises:
        ------
        ValueError
            If one or more numeric hyperparameters are out of valid ranges (e.g.,
            non-positive learning rate or non-positive max_iter).
        FileNotFoundError
            If `restart` is True and `restart_file` is provided but does not exist.
        RuntimeError
            If the underlying optimizer reports a failure to converge when such
            failures are surfaced as exceptions.

        Examples:
        --------
        >>> # run with default hyperparameters
        >>> obj.adam_optimize()
        >>> # run with a smaller learning rate and more iterations
        >>> obj.adam_optimize(alpha=0.01, max_iter=1000, h=1e-3)
        """
        self.loss.return_numeric_gradients = True

        self.loss.epsilon = h

        self.loss_history, self.x_history, self.x_avg_history, self.times, self.observable_traj = gradient_descent_optimizer(
            self.loss,
            self.init_x,
            alpha=alpha,
            max_iter=max_iter,
            threshold=threshold,
            max_n_convergence=max_n_convergence,
            tolerance=tolerance,
            restart=restart,
            restart_file=restart_file,
        )

        self.optimal_model = self.loss.x_to_noise_model(self.x_avg_history[-1])


    def cma_optimize(
        self,
        *,
        x_low, 
        x_up, 
        sigma0=0.02, 
        popsize=4, 
        max_iter=500
    ) -> None:

      x_best, loss_best = cma_opt(
          self.loss, 
          self.init_x, 
          x_low=x_low, 
          x_up=x_up, 
          sigma0=sigma0, 
          popsize=popsize, 
          max_iter=max_iter
          )

      self.optimal_model = self.loss.x_to_noise_model(x_best)


    def bayesian_optimize(
        self,
        *,
        x_low,
        x_up,
        n_init=5,
        max_iter=500,
        acq_name="UCB",
        std=0.01,
        beta=2,
        device="cpu"
    ) -> None:

      x_best, loss_best, _, _=bayesian_opt(
          self.loss,
          x_low = x_low,
          x_up = x_up,
          n_init=n_init,
          max_iter=max_iter,
          acq_name=acq_name,
          std=std,
          beta=beta,
          device=device
      )

      self.optimal_model = self.loss.x_to_noise_model(x_best)

    
    def mcmc_optimize(
        self,
        *,
        x_low,
        x_up,
        max_iter=500,
        step_size=0.05, 
        step_rate=0.99, 
        min_step_size=0, 
        temperature=1.0, 
        anneal_rate=0.99,
        patience=100
    ) -> None:

      x_best, loss_best=mcmc_opt(
          self.loss, 
          self.init_x, 
          x_low=x_low, 
          x_up=x_up, 
          max_iter=max_iter, 
          step_size=step_size, 
          step_rate=step_rate, 
          min_step_size=min_step_size, 
          temperature=temperature, 
          anneal_rate=anneal_rate,
          patience=patience
      )

      self.optimal_model = self.loss.x_to_noise_model(x_best)
