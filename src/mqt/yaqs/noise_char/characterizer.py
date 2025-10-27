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

from mqt.yaqs.noise_char.optimization import adam_optimizer

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
    from mqt.yaqs.noise_char.optimization import LossClass
    from mqt.yaqs.noise_char.propagation import PropagatorWithGradients


class Characterizer:
    """Characterizer.

    High-level helper that wraps a trajectory propagator and loss evaluator to
    estimate parameters of a compact noise model via gradient-based optimization.
    This class bundles together:
    - a deep-copied initial CompactNoiseModel (init_guess),
    - a PropagatorWithGradients configured for the simulation and initial state,
    - a LossClass instance that evaluates mismatch to a reference observable
        trajectory and provides conversions between noise models and a flat
        optimization vector,
    - utilities to run an ADAM optimizer and collect optimization history, and
    - reconstruction of an optimal CompactNoiseModel from optimized parameters.
    sim_params : AnalogSimParams
            Simulation parameters passed to PropagatorWithGradients (time step,
            integrator options, truncation tolerances, etc.).
    hamiltonian : MPO
            Many-body Hamiltonian (MPO) describing the coherent dynamics.
    init_guess : CompactNoiseModel
            Initial guess for the compact noise model. The instance stores a deep
            copy of this object and uses it to initialize the optimizer.
    init_state : MPS
            Initial many-body state (MPS) used to generate simulated observable
            trajectories during loss/gradient evaluations.
    ref_traj : list[Observable]
            Reference observable trajectory (list of observables) that the loss
            function will compare simulated trajectories against.
    work_dir : str | Path, optional
            Directory used by the LossClass for output and temporary files. Default
            is the current directory.
    print_to_file : bool, optional
            If True, instructs the LossClass to write detailed output to files in
            work_dir. Default is True.

    Attributes:
    init_guess : CompactNoiseModel
            Deep-copied initial noise model used as a starting point for inference.
    traj_gradients : PropagatorWithGradients
            Simulator capable of producing trajectories and their gradients with
            respect to the compact noise model parameters.
    loss : LossClass
            Callable loss object that compares simulated trajectories to ref_traj,
            computes gradients via traj_gradients and converts between parameter
            vectors and CompactNoiseModel instances.
    init_x : numpy.ndarray
            Flattened optimization-vector representation of init_guess (as returned
            by loss.noise_model_to_x).
    loss_history : list[float]
            Recorded loss values during optimization (set after calling adam_optimize).
    x_history : list[numpy.ndarray]
            Raw parameter vectors visited during optimization (set after calling
            adam_optimize).
    x_avg_history : list[numpy.ndarray]
            Averaged parameter vectors (e.g., ADAM's moving-average or iterate
            averaging) recorded during optimization.
    times : numpy.ndarray
            Time stamps or elapsed times for iterations recorded by the optimizer.
    observable_traj : numpy.ndarray
            Trajectories of observables evaluated for parameters visited during
            optimization.
    optimal_model : CompactNoiseModel
            Reconstructed CompactNoiseModel from the final averaged parameter vector
            (set after calling adam_optimize).

    Methods:
    adam_optimize(*, alpha=0.05, max_iterations=100, threshold=5e-4, max_n_convergence=50,
                                tolerance=1e-8, beta1=0.5, beta2=0.999, epsilon=1e-8, restart=False,
                                restart_file=None)
            Run an ADAM-based optimization that minimizes self.loss w.r.t. the flattened
            noise-parameter vector starting from self.init_x. On completion, populates
            loss_history, x_history, x_avg_history, times, observable_traj and sets
            optimal_model by converting the final averaged vector back into a
            CompactNoiseModel via loss.x_to_noise_model.
            alpha : float
                    Learning rate (step size) for ADAM.
            max_iterations : int
                    Maximum number of optimization iterations.
            threshold : float
                    Convergence threshold on loss improvement to detect stagnation.
            max_n_convergence : int
                    Required number of consecutive iterations satisfying the convergence
                    criterion to stop early.
            tolerance : float
                    Absolute tolerance for parameter updates; used to avoid meaningless
                    tiny steps.
            beta1, beta2 : float
                    ADAM exponential decay rates for first and second moment estimates.
            epsilon : float
                    Small constant for numerical stability in ADAM updates.
            restart : bool
                    If True, attempt to restart optimizer from a saved state.
            restart_file : Path | None
                    Path to saved optimizer state used when restart is True.
                    If numeric hyperparameters are outside valid ranges (for example,
                    If restart is requested but the provided restart_file does not exist.
                    If the underlying optimizer reports a failure that is surfaced as an
                    exception.
    - This class delegates numerical optimization to an external ADAM routine
        (adam_optimizer) and relies on LossClass to provide a compatible callable
        interface (value and gradient) as well as conversion utilities between the
        compact noise model and flat parameter vectors.
    - The optimizer may evaluate many simulator trajectories; ensure PropagatorWithGradients
        and LossClass are configured for acceptable performance and numerical stability.
    - After optimization, optimal_model contains the learned CompactNoiseModel
        reconstructed from the last averaged parameters and can be used for further
        validation or simulation.
    >>> # Typical usage flow (illustrative):
    >>> char = Characterizer(sim_params=sim_params, hamiltonian=H, init_guess=noise0,
    ...                      init_state=psi0, ref_traj=observables, work_dir="out")
    >>> char.adam_optimize(alpha=0.01, max_iterations=500)
    >>> learned_noise = char.optimal_model
    """

    loss: LossClass

    traj_gradients: PropagatorWithGradients

    init_guess: CompactNoiseModel

    init_x: np.ndarray

    loss_history: list[float]

    x_history: list[np.ndarray]

    x_avg_history: list[np.ndarray]

    times: np.ndarray

    observable_traj: np.ndarray

    optimal_model: CompactNoiseModel

    def __init__(
        self,
        *,
        traj_gradients: PropagatorWithGradients,
        init_guess: CompactNoiseModel,
        loss: LossClass,
    ) -> None:
        """Initialize the noise characterizer.

        Parameters
        ----------
        traj_gradients : PropagatorWithGradients
            Propagator capable of producing trajectories and gradients with respect
            to the compact noise-model parameters.
        init_guess : CompactNoiseModel
            Initial guess for the compact noise model. A deep copy of this object
            will be stored on the instance.
        loss : LossClass
            Loss object that compares simulated trajectories to a reference and
            provides utilities to convert between compact noise models and flat
            optimization vectors.

        The constructor stores a deep copy of init_guess, assigns traj_gradients
        and loss, and initializes self.init_x from init_guess.strength_list.
        """
        self.init_guess = copy.deepcopy(init_guess)
        self.traj_gradients = traj_gradients

        self.loss = loss

        self.init_x = self.init_guess.strength_list.copy()

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
        """Run an ADAM-based optimization to fit the noise model parameters.

        This method runs the external `adam_optimizer` routine to minimize the
        callable `self.loss` starting from `self.init_x`. On completion (or
        early termination), several instance attributes are updated with the
        optimizer history and the recovered optimal noise model.
        Parameters.
        ----------
        alpha : float, optional
            The learning rate (step size) used by the ADAM optimizer. Default is 0.05.
        max_iterations : int, optional
            Maximum number of optimization iterations/steps. Default is 100.
        threshold : float, optional
            Convergence threshold on the loss improvement used to detect
            stagnation. Default is 5e-4.
        max_n_convergence : int, optional
            Number of consecutive iterations satisfying the convergence
            criterion required to stop early. Default is 50.
        tolerance : float, optional
            Absolute tolerance for parameter updates (smallest meaningful step).
            Default is 1e-8.
        beta1 : float, optional
            Exponential decay rate for the first moment estimates (ADAM hyperparameter).
            Default is 0.5.
        beta2 : float, optional
            Exponential decay rate for the second moment estimates (ADAM hyperparameter).
            Default is 0.999.
        epsilon : float, optional
            Small constant for numerical stability in ADAM. Default is 1e-8.
        restart : bool, optional
            If True, attempt to restart the optimizer from a previously saved
            state (provided via `restart_file`). Default is False.
        restart_file : Path or None, optional
            Path to a file containing saved optimizer state to restore when
            `restart` is True. If None and `restart` is True, behavior depends on
            the underlying `adam_optimizer` implementation. Default is None.

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
            non-positive learning rate or non-positive max_iterations).
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
        >>> obj.adam_optimize(alpha=0.01, max_iterations=1000)
        """
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

        self.optimal_model = self.loss.x_to_noise_model(self.x_avg_history[-1])
