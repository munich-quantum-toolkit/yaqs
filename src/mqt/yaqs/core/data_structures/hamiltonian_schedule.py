# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Private compiled schedules for time-dependent Hamiltonians."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .hamiltonian import Hamiltonian

if TYPE_CHECKING:
    from .mpo import MPO
    from .simulation_parameters import AnalogSimParams

_TermFactory = Callable[[object], "Hamiltonian | MPO"]
_CACHE_SIZE = 8


@dataclass(frozen=True)
class HamiltonianSubstep:
    """One midpoint-resolved TDVP substep specification."""

    midpoint: float
    duration: float
    parameters: tuple[object, ...]


@dataclass(frozen=True)
class HamiltonianInterval:
    """One physical sampling interval and its unitary substeps."""

    start: float
    end: float
    duration: float
    substeps: tuple[HamiltonianSubstep, ...]


@dataclass
class HamiltonianSchedule:
    """Compact sampled parameters plus a bounded worker-local MPO cache."""

    length: int
    factories: tuple[_TermFactory, ...]
    intervals: tuple[HamiltonianInterval, ...]
    _cache: OrderedDict[tuple[object, ...], MPO] = field(default_factory=OrderedDict, init=False, repr=False)

    @staticmethod
    def _cache_key(parameters: Sequence[object]) -> tuple[object, ...] | None:
        """Return a hashable cache key, or ``None`` for unhashable parameters."""
        key = tuple(parameters)
        try:
            hash(key)
        except (TypeError, ValueError):
            return None
        return key

    def resolve(self, substep: HamiltonianSubstep) -> MPO:
        """Resolve one substep, reusing a small least-recently-used cache.

        Returns:
            The static MPO for the substep's parameter tuple.
        """
        key = self._cache_key(substep.parameters)
        if key is not None and key in self._cache:
            mpo = self._cache.pop(key)
            self._cache[key] = mpo
            return mpo

        mpo = Hamiltonian._resolve_factories(  # ruff: ignore[private-member-access]
            self.factories,
            substep.parameters,
            length=self.length,
        )
        if key is not None:
            self._cache[key] = mpo
            if len(self._cache) > _CACHE_SIZE:
                self._cache.popitem(last=False)
        return mpo


def _validate_physical_dimensions(mpo: MPO, physical_dimensions: Sequence[int]) -> None:
    """Validate resolved MPO physical legs against an execution state.

    Raises:
        ValueError: If any site has incompatible physical legs.
    """
    for site, (tensor, dimension) in enumerate(zip(mpo.tensors, physical_dimensions, strict=True)):
        if tensor.shape[:2] != (dimension, dimension):
            msg = (
                f"Parameterized Hamiltonian MPO site {site} has physical legs {tensor.shape[:2]}, "
                f"expected ({dimension}, {dimension})."
            )
            raise ValueError(msg)


def compile_hamiltonian_schedule(
    hamiltonian: Hamiltonian,
    sim_params: AnalogSimParams,
    *,
    physical_dimensions: Sequence[int],
) -> HamiltonianSchedule:
    """Sample schedules and validate every distinct resolved factory input.

    Returns:
        Compact interval schedule containing parameters but no retained MPOs.

    Raises:
        ValueError: If ``hamiltonian`` is static or an output is incompatible
            with the execution state.
    """
    if not hamiltonian.is_parameterized:
        msg = "compile_hamiltonian_schedule requires a parameterized Hamiltonian."
        raise ValueError(msg)
    terms = hamiltonian._parameterized_terms  # ruff: ignore[private-member-access]
    assert terms is not None
    factories = tuple(factory for factory, _schedule in terms)
    interval_specs: list[HamiltonianInterval] = []
    validated: set[tuple[object, ...]] = set()

    for start_raw, end_raw in zip(sim_params.times[:-1], sim_params.times[1:], strict=True):
        start = float(start_raw)
        end = float(end_raw)
        duration = end - start
        substep_duration = duration / sim_params.tdvp_sweeps
        substeps: list[HamiltonianSubstep] = []
        for sweep_index in range(sim_params.tdvp_sweeps):
            midpoint = start + (sweep_index + 0.5) * substep_duration
            parameters = hamiltonian._parameters_at(midpoint)  # ruff: ignore[private-member-access]
            substep = HamiltonianSubstep(midpoint, substep_duration, parameters)
            substeps.append(substep)
            key = HamiltonianSchedule._cache_key(parameters)  # ruff: ignore[private-member-access]
            if key is None or key not in validated:
                mpo = Hamiltonian._resolve_factories(factories, parameters, length=hamiltonian.length)  # ruff: ignore[private-member-access]
                _validate_physical_dimensions(mpo, physical_dimensions)
                if key is not None:
                    validated.add(key)
        interval_specs.append(HamiltonianInterval(start, end, duration, tuple(substeps)))

    return HamiltonianSchedule(hamiltonian.length, factories, tuple(interval_specs))
