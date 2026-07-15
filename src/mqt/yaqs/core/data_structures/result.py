# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Result container returned by :meth:`~mqt.yaqs.Simulator.run`.

This module defines :class:`Result`, which holds all outputs produced by a simulation
run. :class:`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams`,
:class:`~mqt.yaqs.core.data_structures.simulation_parameters.StrongSimParams`, and
:class:`~mqt.yaqs.core.data_structures.simulation_parameters.WeakSimParams` remain
read-only configuration; the simulator never mutates the objects passed to
:meth:`~mqt.yaqs.Simulator.run`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .simulation_parameters import AnalogSimParams, StrongSimParams

if TYPE_CHECKING:
    from numpy import complex128, float64
    from numpy.typing import NDArray

    from .noise_model import NoiseModel
    from .simulation_parameters import Observable, WeakSimParams
    from .state import State


def allocate_observable_buffers(
    sim_params: AnalogSimParams | StrongSimParams,
    num_observables: int,
    *,
    num_traj: int,
    num_mid_measurements: int | None = None,
) -> tuple[list[NDArray], list[NDArray], NDArray[float64] | None]:
    """Allocate parallel trajectory and expectation buffers for each observable.

    Args:
        sim_params: Analog or strong simulation parameters (weak sim has no observables).
        num_observables: Number of observables (length of ``result.observables``).
        num_traj: Effective trajectory count for this run.
        num_mid_measurements: Override for strong-sim layer-sampling barrier count.

    Returns:
        tuple[list[NDArray], list[NDArray], NDArray | None]:
            Per-observable trajectory arrays, per-observable expectation buffers,
            and a shared time grid (analog only; ``None`` for circuits).
    """
    trajectories: list[NDArray] = []
    expectation_values: list[NDArray] = []
    times: NDArray[float64] | None = None

    if isinstance(sim_params, AnalogSimParams):
        if sim_params.sample_timesteps:
            times = np.asarray(sim_params.times, dtype=np.float64)
            for _ in range(num_observables):
                trajectories.append(np.empty((num_traj, len(sim_params.times)), dtype=np.float64))
                expectation_values.append(np.empty(len(sim_params.times), dtype=np.float64))
        else:
            times = np.asarray([sim_params.elapsed_time], dtype=np.float64)
            for _ in range(num_observables):
                trajectories.append(np.empty((num_traj, 1), dtype=np.complex128))
                expectation_values.append(np.empty(1, dtype=np.float64))
    elif isinstance(sim_params, StrongSimParams):
        mid = num_mid_measurements if num_mid_measurements is not None else sim_params.num_mid_measurements
        if sim_params.sample_layers:
            for _ in range(num_observables):
                trajectories.append(np.empty((num_traj, mid + 2), dtype=np.complex128))
                expectation_values.append(np.empty(mid + 2, dtype=np.float64))
        else:
            for _ in range(num_observables):
                trajectories.append(np.empty((num_traj, 1), dtype=np.complex128))
                expectation_values.append(np.empty(1, dtype=np.float64))

    return trajectories, expectation_values, times


def allocate_diagnostic_buffers(
    sim_params: AnalogSimParams | StrongSimParams,
    *,
    num_traj: int,
    num_mid_measurements: int | None = None,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Allocate per-trajectory and aggregate buffers for MPS diagnostics.

    Three diagnostics are tracked: runtime contraction cost, maximum bond dimension,
    and total bond dimension. Buffers are shaped ``(3, num_traj, T)`` and ``(3, T)``.

    Args:
        sim_params: Analog or strong simulation parameters.
        num_traj: Effective trajectory count for this run.
        num_mid_measurements: Override for strong-sim layer-sampling barrier count.

    Returns:
        tuple[NDArray, NDArray]: ``(per_traj, aggregate)`` with dtypes ``float64``.
    """
    if isinstance(sim_params, AnalogSimParams):
        num_columns = len(sim_params.times) if sim_params.sample_timesteps else 1
    else:
        mid = num_mid_measurements if num_mid_measurements is not None else sim_params.num_mid_measurements
        num_columns = (mid + 2) if sim_params.sample_layers else 1
    per_traj = np.zeros((3, num_traj, num_columns), dtype=np.float64)
    aggregate = np.zeros((3, num_columns), dtype=np.float64)
    return per_traj, aggregate


def aggregate_diagnostics(per_traj: NDArray[float64]) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
    """Mean over trajectories for each diagnostic row.

    Args:
        per_traj: Buffer shaped ``(3, num_traj, T)``.

    Returns:
        tuple[NDArray, NDArray, NDArray]: ``runtime_cost``, ``max_bond``, ``total_bond`` curves.
    """
    means = np.mean(per_traj, axis=1)
    return means[0], means[1], means[2]


def aggregate_trajectories(result: Result) -> None:
    """Aggregate per-trajectory observable data into ``result.expectation_values``.

    Computes the mean across trajectories (or concatenates Schmidt spectra) for each
    observable index.
    """
    for i, observable in enumerate(result.observables):
        traj = result.trajectories[i]
        if observable.gate.name == "schmidt_spectrum":
            assert isinstance(traj, np.ndarray), "Schmidt spectrum trajectories must be stored in an ndarray"
            all_values = [np.asarray(trajectory).ravel() for trajectory in traj]
            result.expectation_values[i] = np.concatenate(all_values)
        else:
            result.expectation_values[i] = np.mean(traj, axis=0)


def aggregate_counts(result: Result) -> None:
    """Aggregate per-shot measurements into ``result.counts``.

    Sums counts across every non-``None`` entry in ``result.measurements`` so that
    noise-free runs (only index 0 populated), noisy runs (every index populated),
    and any mixed pattern produce a consistent total.
    """
    counts: dict[int, int] = {}
    for measurement in filter(None, result.measurements):
        for key, value in measurement.items():
            counts[key] = counts.get(key, 0) + value
    result.counts = dict(sorted(counts.items()))


@dataclass
class Result:
    """Result of a :meth:`~mqt.yaqs.Simulator.run` call.

    Holds all simulation outputs. :attr:`sim_params` is the read-only configuration
    object the user passed in. :attr:`observables` preserves the user-supplied
    ordering from ``sim_params.observables`` (deep-copied from the configuration);
    :attr:`expectation_values` and
    :attr:`trajectories` hold the corresponding data in lock-step by index.
    For MPS-backed analog and strong-digital runs, :attr:`runtime_cost`,
    :attr:`max_bond`, and :attr:`total_bond` are populated automatically.
    """

    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams
    observables: list[Observable] = field(default_factory=list)
    expectation_values: list[NDArray[float64]] = field(default_factory=list)
    trajectories: list[NDArray] = field(default_factory=list)
    times: NDArray[float64] | None = None
    runtime_cost: NDArray[float64] | None = None
    max_bond: NDArray[float64] | None = None
    total_bond: NDArray[float64] | None = None
    noise_model: NoiseModel | None = None
    output_state: State | None = None
    multi_time_times: NDArray[float64] | None = None
    multi_time_results: NDArray[complex128] | None = None
    measurements: list[dict[int, int] | None] = field(default_factory=list)
    counts: dict[int, int] | None = None
