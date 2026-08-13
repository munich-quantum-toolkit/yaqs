# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for private compiled Hamiltonian schedules."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian
from mqt.yaqs.core.data_structures.hamiltonian_schedule import compile_hamiltonian_schedule


def test_compile_hamiltonian_schedule_samples_substep_midpoints() -> None:
    """Schedules are sampled once at every TDVP-substep midpoint."""
    schedule = Mock(side_effect=lambda time: time)
    factory = Mock(side_effect=lambda value: Hamiltonian.pauli(length=1, one_body=[(float(value), "Z")]))
    hamiltonian = Hamiltonian(length=1, parameterized_terms=[(factory, schedule)])
    params = AnalogSimParams(elapsed_time=0.25, dt=0.1, tdvp_sweeps=2)

    compiled = compile_hamiltonian_schedule(hamiltonian, params, physical_dimensions=[2])

    np.testing.assert_allclose(
        [substep.midpoint for interval in compiled.intervals for substep in interval.substeps],
        [0.025, 0.075, 0.125, 0.175, 0.2125, 0.2375],
    )
    np.testing.assert_allclose([interval.duration for interval in compiled.intervals], [0.1, 0.1, 0.05])
    assert schedule.call_count == 6
    assert factory.call_count == 1


def test_compile_hamiltonian_schedule_rejects_static_hamiltonian() -> None:
    """Schedule compilation is reserved for parameterized Hamiltonians."""
    with pytest.raises(ValueError, match="requires a parameterized Hamiltonian"):
        compile_hamiltonian_schedule(
            Hamiltonian.pauli(length=1, one_body=[(1.0, "Z")]),
            AnalogSimParams(elapsed_time=0.1, dt=0.1),
            physical_dimensions=[2],
        )


def test_parameterized_schedule_validates_once_and_caches_repeated_values() -> None:
    """Constant parameters validate once and resolve once per worker cache."""
    factory = Mock(side_effect=lambda value: Hamiltonian.pauli(length=1, one_body=[(float(value), "Z")]))
    hamiltonian = Hamiltonian(length=1, parameterized_terms=[(factory, lambda _time: 0.5)])
    params = AnalogSimParams(elapsed_time=0.3, dt=0.1)
    compiled = compile_hamiltonian_schedule(hamiltonian, params, physical_dimensions=[2])
    assert factory.call_count == 1

    first = compiled.resolve(compiled.intervals[0].substeps[0])
    second = compiled.resolve(compiled.intervals[1].substeps[0])
    assert first is second
    assert factory.call_count == 2


def test_parameterized_schedule_validates_state_dimensions() -> None:
    """Resolved factory outputs must match the execution layout."""
    hamiltonian = Hamiltonian(
        length=1,
        parameterized_terms=[(lambda _value: Hamiltonian.pauli(length=1, one_body=[(1.0, "Z")]), lambda _t: 0.0)],
    )
    with pytest.raises(ValueError, match=r"physical legs .* expected \(3, 3\)"):
        compile_hamiltonian_schedule(
            hamiltonian,
            AnalogSimParams(elapsed_time=0.1, dt=0.1),
            physical_dimensions=[3],
        )


def test_array_parameters_bypass_worker_cache() -> None:
    """Unhashable NumPy parameter bundles are resolved for every substep."""
    factory = Mock(side_effect=lambda value: Hamiltonian.pauli(length=1, one_body=[(float(value[0]), "Z")]))
    hamiltonian = Hamiltonian(
        length=1,
        parameterized_terms=[(factory, lambda time: np.asarray([time]))],
    )
    compiled = compile_hamiltonian_schedule(
        hamiltonian,
        AnalogSimParams(elapsed_time=0.2, dt=0.1),
        physical_dimensions=[2],
    )
    substeps = [interval.substeps[0] for interval in compiled.intervals]

    first = compiled.resolve(substeps[0])
    second = compiled.resolve(substeps[1])

    assert first is not second
    assert factory.call_count == 3


def test_hamiltonian_schedule_evicts_oldest_cached_mpo() -> None:
    """Resolving a ninth distinct value evicts and later recomputes the oldest MPO."""
    factory = Mock(side_effect=lambda value: Hamiltonian.pauli(length=1, one_body=[(float(value), "Z")]))
    hamiltonian = Hamiltonian(length=1, parameterized_terms=[(factory, lambda time: time)])
    compiled = compile_hamiltonian_schedule(
        hamiltonian,
        AnalogSimParams(elapsed_time=0.9, dt=0.1),
        physical_dimensions=[2],
    )
    substeps = [interval.substeps[0] for interval in compiled.intervals]

    for substep in substeps:
        compiled.resolve(substep)
    calls_after_nine_values = factory.call_count
    compiled.resolve(substeps[0])

    assert calls_after_nine_values == 10
    assert factory.call_count == 11
