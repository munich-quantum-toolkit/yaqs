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
from concurrent.futures import CancelledError
from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.yaqs.characterization.noise.backends.gradient_free.cma import cma_opt
from mqt.yaqs.characterization.noise.shared.representation import (
    DEFAULT_LINDBLAD_MAX_QUBITS,
    DEFAULT_VECTOR_MAX_QUBITS,
    NoiseRepresentation,
    ResolvedNoiseRepresentation,
    resolve_noise_representation,
)
from mqt.yaqs.characterization.noise.trajectory_matching.reference import (
    build_simulator,
    build_trajectory_loss,
    resolve_reference_expectations,
)
from mqt.yaqs.characterization.noise.trajectory_matching.results import NoiseCharacterizationResult
from mqt.yaqs.characterization.noise.trajectory_matching.run import run_trajectory_characterization
from mqt.yaqs.core.parallel_utils import ExecutionConfig, MPContext

if TYPE_CHECKING:
    from mqt.yaqs.characterization.noise.shared.loss import TrajectoryLoss
    from mqt.yaqs.characterization.noise.shared.propagation import Propagator
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.data_structures.state import State
    from mqt.yaqs.simulator import Simulator


class NoiseCharacterizer:
    """Entry point for Markovian noise digital-twin workflows.

    **Use:** :meth:`characterize` (fit compact rates from experimental or simulated trajectories)

    **Advanced:** :meth:`from_reference`, :meth:`optimize` (custom optimizer loops)

    Attributes:
        parallel: Whether trajectory simulations run in parallel via a process pool.
        max_workers: Maximum worker processes when ``parallel=True``.
        show_progress: Whether to display a tqdm progress bar.
        representation: ``"density_matrix"`` (Lindblad), ``"vector"`` (MCWF), ``"mps"`` (TJM),
            or ``"auto"``.
        lindblad_max_qubits: Auto cutover to Lindblad master-equation evolution.
        vector_max_qubits: Auto cutover from MCWF to TJM.
        mp_context: Multiprocessing context.
        max_retries: Maximum retry attempts for transient worker errors.
        retry_exceptions: Exception types that trigger a retry.
    """

    def __init__(
        self,
        *,
        parallel: bool = False,
        max_workers: int | None = None,
        show_progress: bool = False,
        representation: NoiseRepresentation = "auto",
        lindblad_max_qubits: int = DEFAULT_LINDBLAD_MAX_QUBITS,
        vector_max_qubits: int = DEFAULT_VECTOR_MAX_QUBITS,
        mp_context: MPContext = "auto",
        max_retries: int = 10,
        retry_exceptions: tuple[type[BaseException], ...] = (CancelledError, TimeoutError, OSError),
        propagator: Propagator | None = None,
        init_guess: CompactNoiseModel | None = None,
        loss: TrajectoryLoss | None = None,
        resolved_representation: ResolvedNoiseRepresentation | None = None,
        simulator: Simulator | None = None,
    ) -> None:
        """Configure execution and representation defaults for noise characterization.

        Args:
            parallel: Whether to parallelize trajectory execution.
            max_workers: Cap on worker processes when ``parallel=True``.
            show_progress: Whether to show tqdm progress bars.
            representation: Forward-model selection (``"auto"`` prefers Lindblad on small chains).
            lindblad_max_qubits: Auto cutover to Lindblad master-equation evolution.
            vector_max_qubits: Auto cutover from MCWF to TJM.
            mp_context: Multiprocessing start method.
            max_retries: Retries for transient worker failures.
            retry_exceptions: Exception types that trigger a worker retry.
            propagator: Optional wired fit propagator (advanced API).
            init_guess: Optional initial compact model (advanced API).
            loss: Optional wired trajectory loss (advanced API).
            resolved_representation: Concrete backend used for wired propagation.
            simulator: Optional simulator used during wired fitting.
        """
        self._execution = ExecutionConfig(
            parallel=parallel,
            max_workers=max_workers,
            show_progress=show_progress,
            mp_context=mp_context,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
        )
        self.representation = representation
        self.lindblad_max_qubits = int(lindblad_max_qubits)
        self.vector_max_qubits = int(vector_max_qubits)
        self.propagator = propagator
        self.init_guess = copy.deepcopy(init_guess) if init_guess is not None else None
        self.loss = loss
        self.init_x = self.init_guess.strength_list.copy() if self.init_guess is not None else None
        self.resolved_representation = resolved_representation
        self.simulator = simulator
        self.result: NoiseCharacterizationResult | None = None

    @property
    def parallel(self) -> bool:
        """Whether parallel trajectory simulation is enabled."""
        return self._execution.parallel

    @property
    def max_workers(self) -> int:
        """Resolved worker-process cap for parallel trajectory jobs."""
        return self._execution.resolved_max_workers()

    @property
    def show_progress(self) -> bool:
        """Whether progress bars are shown during trajectory simulation."""
        return self._execution.show_progress

    @property
    def mp_context(self) -> MPContext:
        """Multiprocessing context used for worker pools."""
        return self._execution.mp_context

    @property
    def max_retries(self) -> int:
        """Maximum retry attempts for transient worker failures."""
        return self._execution.max_retries

    @property
    def retry_exceptions(self) -> tuple[type[BaseException], ...]:
        """Exception types that trigger a worker retry."""
        return self._execution.retry_exceptions

    def _resolved_representation(self, chain_length: int) -> ResolvedNoiseRepresentation:
        """Resolve the forward backend for a chain length under this characterizer's settings.

        Returns:
            Resolved ``"density_matrix"``, ``"vector"``, or ``"mps"``.
        """
        return resolve_noise_representation(
            chain_length,
            self.representation,
            lindblad_max_qubits=self.lindblad_max_qubits,
            vector_max_qubits=self.vector_max_qubits,
        )

    def characterize(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        /,
        *,
        init_state: State,
        init_guess: CompactNoiseModel,
        observables: list[Observable],
        x_low: np.ndarray,
        x_up: np.ndarray,
        reference_model: CompactNoiseModel | None = None,
        ref_expectations: np.ndarray | None = None,
        **optimizer_kwargs: Any,
    ) -> NoiseCharacterizationResult:
        """Fit compact noise strengths by matching observable trajectories.

        Provide exactly one of ``reference_model`` (benchmark shortcut) or
        ``ref_expectations`` (experimental trajectories).

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
            **optimizer_kwargs: Keyword arguments forwarded to the CMA-ES backend.

        Returns:
            Structured optimization result including fitted and reference trajectories.
        """
        self.result = run_trajectory_characterization(
            hamiltonian=hamiltonian,
            sim_params=sim_params,
            init_state=init_state,
            init_guess=init_guess,
            observables=observables,
            x_low=x_low,
            x_up=x_up,
            reference_model=reference_model,
            ref_expectations=ref_expectations,
            execution=self._execution,
            representation=self.representation,
            lindblad_max_qubits=self.lindblad_max_qubits,
            vector_max_qubits=self.vector_max_qubits,
            **optimizer_kwargs,
        )
        self.resolved_representation = self.result.resolved_representation
        return self.result

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
        """Build a wired characterizer from a reference noise model (advanced).

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
            Configured :class:`NoiseCharacterizer` with wired loss and propagator.
        """
        execution = ExecutionConfig(
            parallel=parallel,
            max_workers=max_workers,
            show_progress=show_progress,
        )
        fit_simulator = simulator or build_simulator(execution)
        ref_array, _, resolved = resolve_reference_expectations(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            init_state=init_state,
            observables=observables,
            reference_model=reference_model,
            ref_expectations=None,
            simulator=fit_simulator,
            representation=representation,
            lindblad_max_qubits=lindblad_max_qubits,
            vector_max_qubits=vector_max_qubits,
        )
        loss, _fit_propagator, resolved = build_trajectory_loss(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            init_state=init_state,
            init_guess=init_guess,
            observables=observables,
            ref_expectations=ref_array,
            simulator=fit_simulator,
            representation=representation,
            lindblad_max_qubits=lindblad_max_qubits,
            vector_max_qubits=vector_max_qubits,
        )
        return cls(
            parallel=parallel,
            max_workers=max_workers,
            show_progress=show_progress,
            representation=representation,
            lindblad_max_qubits=lindblad_max_qubits,
            vector_max_qubits=vector_max_qubits,
            propagator=loss.propagator,
            init_guess=init_guess,
            loss=loss,
            resolved_representation=resolved,
            simulator=fit_simulator,
        )

    def optimize(
        self,
        *,
        x_low: np.ndarray,
        x_up: np.ndarray,
        **kwargs: Any,
    ) -> NoiseCharacterizationResult:
        """Run CMA-ES on a wired characterizer (advanced).

        Args:
            x_low: Lower parameter bounds.
            x_up: Upper parameter bounds.
            **kwargs: Keyword arguments forwarded to
                :func:`~mqt.yaqs.characterization.noise.backends.gradient_free.cma.cma_opt`.

        Returns:
            Structured optimization result.

        Raises:
            RuntimeError: If the characterizer was not wired via :meth:`from_reference`.
        """
        if self.loss is None or self.init_x is None or self.init_guess is None:
            msg = "optimize() requires a wired characterizer from from_reference()."
            raise RuntimeError(msg)

        x_best, best_loss, loss_history, parameter_history = cma_opt(
            self.loss,
            self.init_x,
            x_low=x_low,
            x_up=x_up,
            **kwargs,
        )

        optimal_model = self.loss.x_to_noise_model(x_best)
        self.propagator = self.loss.propagator
        self.propagator.run(optimal_model)
        fit_traj = np.asarray(self.propagator.obs_array, dtype=float)

        self.result = NoiseCharacterizationResult(
            optimal_model=optimal_model,
            best_loss=float(best_loss),
            best_parameters=np.asarray(x_best, dtype=float),
            loss_history=loss_history,
            parameter_history=parameter_history,
            ref_traj=np.asarray(self.loss.ref_traj_array, dtype=float),
            fit_traj=fit_traj,
            times=np.asarray(self.propagator.times, dtype=float),
            resolved_representation=self.resolved_representation,
        )
        return self.result


__all__ = ["NoiseCharacterizer"]
