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

if TYPE_CHECKING:
    from numpy import complex128, float64
    from numpy.typing import NDArray

    from .noise_model import NoiseModel
    from .simulation_parameters import AnalogSimParams, Observable, StrongSimParams, WeakSimParams
    from .state import State


def aggregate_trajectories(result: Result) -> None:
    """Aggregate per-trajectory observable data on ``result.observables``.

    Computes the mean across trajectories (or concatenates Schmidt spectra) and
    stores the result on each observable's ``results`` attribute.
    """
    for observable in result.observables:
        if observable.gate.name == "schmidt_spectrum":
            assert isinstance(observable.trajectories, np.ndarray)
            all_values = [np.asarray(trajectory).ravel() for trajectory in observable.trajectories]
            observable.results = np.concatenate(all_values)
        else:
            assert observable.trajectories is not None
            observable.results = np.mean(observable.trajectories, axis=0)


def aggregate_counts(result: Result) -> None:
    """Aggregate per-shot measurements into ``result.counts``.

    For noise-free weak simulations, ``result.measurements[0]`` holds all shots.
    For noisy runs, counts from every shot dictionary are summed.
    """
    counts: dict[int, int] = {}
    if None in result.measurements:
        assert result.measurements[0] is not None
        counts = dict(result.measurements[0])
    else:
        for measurement in filter(None, result.measurements):
            for key, value in measurement.items():
                counts[key] = counts.get(key, 0) + value
    result.counts = dict(sorted(counts.items()))


@dataclass
class Result:
    """Result of a :meth:`~mqt.yaqs.Simulator.run` call.

    Holds all simulation outputs. :attr:`sim_params` is the read-only configuration
    object the user passed in; observable trajectories and aggregated values live on
    :attr:`observables` (deep-copied from the configuration list).
    """

    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams
    observables: list[Observable] = field(default_factory=list)
    noise_model: NoiseModel | None = None
    output_state: State | None = None
    multi_time_times: NDArray[float64] | None = None
    multi_time_results: NDArray[complex128] | None = None
    measurements: list[dict[int, int] | None] = field(default_factory=list)
    counts: dict[int, int] | None = None
