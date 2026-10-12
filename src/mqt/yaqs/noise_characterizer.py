# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""User-facing entry point for Markovian noise-parameter characterization."""

# ruff:file-ignore[any-type] -- optimizer kwargs forwarded to CMA-ES

from __future__ import annotations

from concurrent.futures import CancelledError
from typing import TYPE_CHECKING, Any

from mqt.yaqs.characterization.noise.shared.representation import (
    DEFAULT_LINDBLAD_MAX_QUBITS,
    DEFAULT_VECTOR_MAX_QUBITS,
    NoiseRepresentation,
)
from mqt.yaqs.core.parallel_utils import ExecutionConfig, MPContext

from .core._validation import validate_choice, validate_integer

if TYPE_CHECKING:
    import numpy as np

    from mqt.yaqs.characterization.noise.optimization.results import NoiseCharacterizationResult
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.observable import Observable
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from mqt.yaqs.core.data_structures.state import State


_NOISE_REPRESENTATIONS: tuple[NoiseRepresentation, ...] = ("density_matrix", "vector", "mps", "auto")


class NoiseCharacterizer:
    """Fit Markovian noise rates to measured or simulated dynamics.

    :meth:`characterize` matches observable trajectories with a numerical forward model.

    Attributes:
        result: Most recent fit, or ``None`` before the first :meth:`characterize` call.
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
            parallel: Boolean that enables parallel trajectory execution.
            max_workers: Positive worker-process cap, or ``None`` for the default.
            show_progress: Boolean that controls tqdm progress bars.
            representation: ``"density_matrix"`` (Lindblad), ``"vector"`` (MCWF),
                ``"mps"`` (TJM), or ``"auto"`` using the qubit cutovers below.
            lindblad_max_qubits: Non-negative auto cutover to Lindblad evolution.
            vector_max_qubits: Non-negative auto cutover from MCWF to TJM.
            mp_context: ``"auto"``, ``"fork"``, or ``"spawn"`` multiprocessing start method.
            max_retries: Non-negative retry count for transient worker failures.
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
        self.representation = validate_choice(
            representation,
            name="representation",
            allowed=_NOISE_REPRESENTATIONS,
        )
        self.lindblad_max_qubits = validate_integer(
            lindblad_max_qubits,
            name="lindblad_max_qubits",
            minimum=0,
        )
        self.vector_max_qubits = validate_integer(vector_max_qubits, name="vector_max_qubits", minimum=0)
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

        Supply exactly one reference: a noise model to simulate or measured expectation values.

        Args:
            hamiltonian: System Hamiltonian.
            sim_params: Analog simulation settings.
            init_state: Initial state.
            init_guess: Initial noise model defining the one- or two-site jump operators to fit.
            observables: Fitting observables, in the order of the reference rows.
            x_low: Lower parameter bounds.
            x_up: Upper parameter bounds.
            reference_model: Noise model used to generate reference trajectories.
            ref_expectations: Measured trajectories with shape ``(n_obs, n_times)``.
            **optimizer_kwargs: ``max_iter`` (positive integer, default 500) limits
                CMA-ES generations. ``popsize`` (integer >= 2, default 4) sets candidates
                per generation; ``sigma0`` (positive float, default 0.01) sets the
                initial search scale. ``seed`` (non-negative integer or ``None``)
                seeds CMA-ES, separately from trajectory sampling. A single parameter
                with finite bounds uses a deterministic scalar search, where
                ``max_iter`` limits search evaluations and the other settings have
                no effect. Initial-model and final fitted-trajectory evaluations are
                outside this budget. SciPy can use two evaluations when ``max_iter=1``.

        Returns:
            Fitted noise model, optimization diagnostics, and fitted and reference trajectories.

        Raises:
            ValueError: If neither or both reference inputs are supplied.
        """
        if (reference_model is None) == (ref_expectations is None):
            msg = "Specify exactly one of reference_model= or ref_expectations=."
            raise ValueError(msg)

        # CMA imports its optional plotting support. Keep that path out of a
        # normal top-level YAQS import, which must work without Matplotlib.
        from mqt.yaqs.characterization.noise.optimization.run import (  # ruff: ignore[import-outside-top-level]
            run_optimization_characterization,
        )

        self.result = run_optimization_characterization(
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
        return self.result


__all__ = ["NoiseCharacterizer"]
