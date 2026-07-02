# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing entry point for Markovian noise-parameter characterization."""

# ruff: noqa: ANN401 -- optimizer kwargs forwarded to CMA-ES

from __future__ import annotations

from concurrent.futures import CancelledError
from typing import TYPE_CHECKING, Any

from mqt.yaqs.characterization.noise.shared.representation import (
    DEFAULT_LINDBLAD_MAX_QUBITS,
    DEFAULT_VECTOR_MAX_QUBITS,
    NoiseRepresentation,
    ResolvedNoiseRepresentation,
)
from mqt.yaqs.characterization.noise.trajectory_matching.run import run_trajectory_characterization
from mqt.yaqs.core.parallel_utils import ExecutionConfig, MPContext

if TYPE_CHECKING:
    import numpy as np

    from mqt.yaqs.characterization.noise.trajectory_matching.results import NoiseCharacterizationResult
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
    from mqt.yaqs.core.data_structures.state import State


class NoiseCharacterizer:
    """Entry point for Markovian noise digital-twin workflows.

    **Use:** :meth:`characterize` (fit noise rates from experimental or simulated trajectories)

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
        self.resolved_representation: ResolvedNoiseRepresentation | None = None
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

    def characterize(
        self,
        hamiltonian: Hamiltonian,
        sim_params: AnalogSimParams,
        /,
        *,
        init_state: State,
        init_guess: NoiseModel,
        observables: list[Observable],
        x_low: np.ndarray,
        x_up: np.ndarray,
        reference_model: NoiseModel | None = None,
        ref_expectations: np.ndarray | None = None,
        **optimizer_kwargs: Any,
    ) -> NoiseCharacterizationResult:
        """Fit noise strengths by matching observable trajectories.

        Provide exactly one of ``reference_model`` (benchmark shortcut) or
        ``ref_expectations`` (experimental trajectories).

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


__all__ = ["NoiseCharacterizer"]
