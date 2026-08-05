# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Result container returned by :meth:`~mqt.yaqs.Simulator.run`.

This module defines :class:`Result`, which holds all outputs produced by a simulation
run. Program results recursively contain one ``Result`` per segment.
:class:`~mqt.yaqs.core.data_structures.simulation_parameters.AnalogSimParams` and
:class:`~mqt.yaqs.core.data_structures.simulation_parameters.DigitalSimParams` remain
read-only configuration; the simulator never mutates the objects passed to
:meth:`~mqt.yaqs.Simulator.run`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np

from .simulation_parameters import AnalogSimParams, DigitalSimParams, Observable

if TYPE_CHECKING:
    from numpy import complex128, float64
    from numpy.typing import NDArray

    from .noise_model import NoiseModel
    from .state import State


def _observable_sites(observable: Observable) -> tuple[int, ...]:
    """Return an observable's target sites in a comparable form."""
    sites = getattr(observable, "sites", None)
    if sites is None:
        return ()
    if isinstance(sites, int):
        return (sites,)
    return tuple(sites)


def _observables_match(left: Observable, right: Observable) -> bool:
    """Return whether two observables have the same measurement semantics."""
    return (
        left.gate.name == right.gate.name
        and _observable_sites(left) == _observable_sites(right)
        and getattr(left.gate, "bitstring", None) == getattr(right.gate, "bitstring", None)
        and np.array_equal(left.gate.matrix, right.gate.matrix)
    )


def allocate_observable_buffers(
    sim_params: AnalogSimParams | DigitalSimParams,
    num_observables: int,
    *,
    num_traj: int,
    num_mid_measurements: int | None = None,
) -> tuple[list[NDArray], list[NDArray], NDArray[float64] | None]:
    """Allocate parallel trajectory and expectation buffers for each observable.

    Args:
        sim_params: Analog or digital simulation parameters.
        num_observables: Number of observables (length of ``result.observables``).
        num_traj: Effective trajectory count for this run.
        num_mid_measurements: Override for digital layer-sampling barrier count.

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
    elif isinstance(sim_params, DigitalSimParams):
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
    sim_params: AnalogSimParams | DigitalSimParams,
    *,
    num_traj: int,
    num_mid_measurements: int | None = None,
) -> tuple[NDArray[float64], NDArray[float64]]:
    """Allocate per-trajectory and aggregate buffers for MPS diagnostics.

    Three diagnostics are tracked: runtime contraction cost, maximum bond dimension,
    and total bond dimension. Buffers are shaped ``(3, num_traj, T)`` and ``(3, T)``.

    Args:
        sim_params: Analog or digital simulation parameters.
        num_traj: Effective trajectory count for this run.
        num_mid_measurements: Override for digital layer-sampling barrier count.

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


def _segment_trace(
    segment: Result,
    observable: Observable,
    *,
    is_program_result: bool,
    position: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Return one segment's trace, or ``None`` for an absent digital observable.

    Raises:
        ValueError: If the segment metadata or stored trace data is invalid.
    """
    segment_label = segment.segment_index if segment.segment_index is not None else position
    is_digital = segment.segment_type == "digital" or (
        not is_program_result and isinstance(segment.sim_params, DigitalSimParams)
    )
    segment_kind = "digital" if is_digital else "analog"
    if is_program_result and segment.segment_type not in {"analog", "digital"}:
        msg = f"Program segment {segment_label} has no valid segment_type."
        raise ValueError(msg)
    if not is_program_result and segment.times is None and not is_digital:
        msg = "Result has no observable trace data."
        raise ValueError(msg)

    matches = [
        index for index, candidate in enumerate(segment.observables) if _observables_match(candidate, observable)
    ]
    if not matches:
        if is_digital:
            return None
        msg = f"Observable is not recorded in analog segment {segment_label}."
        raise ValueError(msg)
    if len(matches) > 1:
        msg = f"Observable is recorded more than once in {segment_kind} segment {segment_label}."
        raise ValueError(msg)

    observable_index = matches[0]
    if observable_index >= len(segment.expectation_values):
        msg = f"Observable in {segment_kind} segment {segment_label} has no expectation values."
        raise ValueError(msg)
    values = np.asarray(segment.expectation_values[observable_index])
    if values.ndim != 1:
        msg = f"Observable in {segment_kind} segment {segment_label} is not a scalar series."
        raise ValueError(msg)

    if is_digital:
        if is_program_result and segment.time_offset is None:
            msg = f"Digital segment {segment_label} has no time_offset."
            raise ValueError(msg)
        offset = segment.time_offset if segment.time_offset is not None else 0.0
        times = np.full(len(values), offset, dtype=np.float64)
    else:
        if segment.times is None:
            msg = f"Analog segment {segment_label} has no time data."
            raise ValueError(msg)
        if is_program_result and segment.time_offset is None:
            msg = f"Analog segment {segment_label} has no time_offset."
            raise ValueError(msg)
        times = np.asarray(segment.times, dtype=np.float64)
        if times.ndim != 1 or len(times) != len(values):
            msg = f"Observable in analog segment {segment_label} is not a scalar time series aligned with times."
            raise ValueError(msg)
        offset = segment.time_offset if segment.time_offset is not None else 0.0
        times = np.asarray(times + offset, dtype=np.float64)

    return times, np.asarray(np.real(values), dtype=np.float64)


@dataclass
class Result:
    """Result of a :meth:`~mqt.yaqs.Simulator.run` call.

    Holds all simulation outputs. For standalone runs, :attr:`sim_params` is the
    read-only configuration object the user passed in. An outer program result uses
    ``sim_params=None`` and stores its ordered segment outputs in
    :attr:`segment_results`. :attr:`observables` preserves the user-supplied
    ordering from ``sim_params.observables`` (deep-copied from the configuration);
    :attr:`expectation_values` and
    :attr:`trajectories` hold the corresponding data in lock-step by index.
    For MPS-backed analog and digital runs with observables, :attr:`runtime_cost`,
    :attr:`max_bond`, and :attr:`total_bond` are populated automatically.
    Nested program results use :attr:`segment_index`, :attr:`segment_type`, and
    :attr:`time_offset` to identify their position and analog-time boundary.
    """

    sim_params: AnalogSimParams | DigitalSimParams | None = None
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
    segment_results: list[Result] = field(default_factory=list)
    segment_index: int | None = None
    segment_type: Literal["analog", "digital"] | None = None
    time_offset: float | None = None

    def observable_trace(self, observable: Observable) -> tuple[NDArray[float64], NDArray[float64]]:
        """Return an observable's expectation values on the physical timeline.

        For a program result, segment outputs are inspected in program order.
        Analog local time grids are shifted by ``time_offset``. Every observation
        from a digital segment is placed at that segment's ``time_offset`` because
        digital operations are instantaneous on the physical program timeline.
        Repeated times are preserved so their array order retains the state order
        within and around digital operations. Digital segments that did not record
        the requested observable are skipped.

        A standalone digital result has no elapsed analog time, so all of its
        observations are returned at time zero. For circuit-depth analysis, use
        the returned values with an application-defined circuit coordinate such as
        ``np.arange(len(values))``.

        Observable matching is structural because simulation results contain
        deep-copied observable metadata. A match requires equal target sites,
        gate name, operator matrix, and PVM bitstring where applicable.

        Args:
            observable: Observable whose aggregated expectation trace to return.

        Returns:
            Newly allocated arrays ``(times, expectation_values)``.

        Raises:
            TypeError: If ``observable`` is not an :class:`Observable`.
            ValueError: If there is no trace data, an analog program interval
                lacks the observable, matching is ambiguous, a program offset is
                missing, or the stored values are not a scalar series aligned
                with its coordinate.
        """
        if not isinstance(observable, Observable):
            msg = f"observable must be Observable, got {type(observable).__name__}."
            raise TypeError(msg)

        is_program_result = bool(self.segment_results)
        segments = self.segment_results if is_program_result else [self]

        time_parts: list[NDArray[np.float64]] = []
        value_parts: list[NDArray[np.float64]] = []
        for position, segment in enumerate(segments):
            trace = _segment_trace(
                segment,
                observable,
                is_program_result=is_program_result,
                position=position,
            )
            if trace is None:
                continue
            times, values = trace
            time_parts.append(times)
            value_parts.append(values)

        if not time_parts:
            msg = "Result has no observable trace data."
            raise ValueError(msg)

        return np.concatenate(time_parts), np.concatenate(value_parts)
