# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Characterization entry point for YAQS."""

from __future__ import annotations

import importlib
from concurrent.futures import CancelledError
from typing import TYPE_CHECKING, Literal

from mqt.yaqs.characterization.memory.combs.tomography import DenseComb, MPOComb, construct_process_tensor
from mqt.yaqs.core.parallel_utils import ExecutionConfig, MPContext

if TYPE_CHECKING:
    from typing import TypeAlias

    import numpy as np
    from torch.utils.data import TensorDataset

    from mqt.yaqs.characterization.memory.combs.surrogates.model import TransformerComb
    from mqt.yaqs.characterization.memory.combs.tomography.basis import TomographyBasis
    from mqt.yaqs.core.data_structures.mpo import MPO
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

    Comb: TypeAlias = DenseComb | MPOComb | TransformerComb

_LAZY_EXPORTS = {
    "TransformerComb": ("mqt.yaqs.characterization.memory.combs.surrogates.model", "TransformerComb"),
    "create_surrogate": ("mqt.yaqs.characterization.memory.combs.surrogates.workflow", "create_surrogate"),
    "generate_data": ("mqt.yaqs.characterization.memory.combs.surrogates.workflow", "generate_data"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_path), attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


class Characterizer:
    """Public entry point for process-tensor characterization workflows.

    A :class:`Characterizer` owns the execution-side configuration: how sequence
    rollouts are parallelized, how many workers to use, whether to display progress
    bars, which multiprocessing context to use, and the retry policy for transient
    worker errors. Physics inputs (Hamiltonian, simulation parameters, schedules)
    are passed per call to the characterization methods.

    Multiple method calls share the same configuration. Each call constructs its
    own short-lived process pool when ``parallel=True``.
    """

    def __init__(
        self,
        *,
        parallel: bool = True,
        max_workers: int | None = None,
        show_progress: bool = True,
        mp_context: MPContext = "auto",
        max_retries: int = 10,
        retry_exceptions: tuple[type[BaseException], ...] = (CancelledError, TimeoutError, OSError),
    ) -> None:
        """Initialize the characterizer with execution-side configuration."""
        self._execution = ExecutionConfig(
            parallel=parallel,
            max_workers=max_workers,
            show_progress=show_progress,
            mp_context=mp_context,
            max_retries=max_retries,
            retry_exceptions=retry_exceptions,
        )

    @property
    def parallel(self) -> bool:
        return self._execution.parallel

    @property
    def max_workers(self) -> int:
        return self._execution.resolved_max_workers()

    @property
    def show_progress(self) -> bool:
        return self._execution.show_progress

    @property
    def mp_context(self) -> MPContext:
        return self._execution.mp_context

    @property
    def max_retries(self) -> int:
        return self._execution.max_retries

    @property
    def retry_exceptions(self) -> tuple[type[BaseException], ...]:
        return self._execution.retry_exceptions

    def construct_process_tensor(
        self,
        operator: MPO,
        sim_params: AnalogSimParams,
        timesteps: list[float] | None = None,
        *,
        noise_model: NoiseModel | None = None,
        num_trajectories: int = 100,
        basis: TomographyBasis = "tetrahedral",
        basis_seed: int | None = None,
        return_type: Literal["dense", "mpo"] = "dense",
        check: bool = True,
        atol: float = 1e-8,
        compress_every: int = 100,
        tol: float = 1e-12,
        max_bond_dim: int | None = None,
        n_sweeps: int = 2,
    ) -> DenseComb | MPOComb:
        """Construct a process tensor via exhaustive discrete-basis tomography."""
        return construct_process_tensor(
            operator,
            sim_params,
            timesteps,
            noise_model=noise_model,
            num_trajectories=num_trajectories,
            basis=basis,
            basis_seed=basis_seed,
            return_type=return_type,
            check=check,
            atol=atol,
            compress_every=compress_every,
            tol=tol,
            max_bond_dim=max_bond_dim,
            n_sweeps=n_sweeps,
            _execution=self._execution,
        )

    def generate_data(
        self,
        operator: MPO,
        sim_params: AnalogSimParams,
        *,
        k: int,
        n: int,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        timesteps: list[float] | None = None,
        init_mode: str = "eigenstate",
        solver: str | None = None,
    ) -> TensorDataset:
        """Generate surrogate training data by sampling interventions and simulating rollouts."""
        from mqt.yaqs.characterization.memory.combs.surrogates.workflow import generate_data as _generate_data

        return _generate_data(
            operator,
            sim_params,
            k=k,
            n=n,
            rng=rng,
            seed=seed,
            timesteps=timesteps,
            init_mode=init_mode,
            solver=solver,
            _execution=self._execution,
        )

    def create_surrogate(
        self,
        operator: MPO,
        sim_params: AnalogSimParams,
        *,
        k: int,
        n: int,
        seed: int | None = None,
        timesteps: list[float] | None = None,
        init_mode: str = "eigenstate",
        model_kwargs: dict | None = None,
        train_kwargs: dict | None = None,
    ) -> TransformerComb:
        """Train a surrogate model end-to-end on sampled rollout data."""
        from mqt.yaqs.characterization.memory.combs.surrogates.workflow import create_surrogate as _create_surrogate

        return _create_surrogate(
            operator,
            sim_params,
            k=k,
            n=n,
            seed=seed,
            timesteps=timesteps,
            init_mode=init_mode,
            model_kwargs=model_kwargs,
            train_kwargs=train_kwargs,
            _execution=self._execution,
        )


__all__ = [
    "Characterizer",
    "DenseComb",
    "MPOComb",
    "TransformerComb",
    "construct_process_tensor",
    "create_surrogate",
    "generate_data",
]
