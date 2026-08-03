# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Composable analog and digital simulation program specifications."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import KW_ONLY, dataclass

from qiskit.circuit import QuantumCircuit

from .hamiltonian import Hamiltonian
from .noise_model import NoiseModel
from .simulation_parameters import AnalogSimParams, DigitalSimParams

__all__ = ["AnalogSegment", "DigitalSegment", "SimulationProgram"]


@dataclass(frozen=True)
class AnalogSegment:
    """Specification of one static analog part of a simulation program.

    Args:
        hamiltonian: Hamiltonian evolved during this segment.
        sim_params: Analog simulation parameters for the segment. If omitted,
            program execution supplies its documented internal defaults.
        noise_model: Optional segment noise model. Noisy program execution is
            introduced by Feature A2.

    Raises:
        TypeError: If an argument does not have the corresponding analog type.
    """

    hamiltonian: Hamiltonian
    _: KW_ONLY
    sim_params: AnalogSimParams | None = None
    noise_model: NoiseModel | None = None

    def __post_init__(self) -> None:
        """Validate the closed analog segment specification."""
        if not isinstance(self.hamiltonian, Hamiltonian):
            msg = f"hamiltonian must be Hamiltonian, got {type(self.hamiltonian).__name__}."
            raise TypeError(msg)
        if self.sim_params is not None and not isinstance(self.sim_params, AnalogSimParams):
            msg = f"sim_params must be AnalogSimParams or None, got {type(self.sim_params).__name__}."
            raise TypeError(msg)
        if self.noise_model is not None and not isinstance(self.noise_model, NoiseModel):
            msg = f"noise_model must be NoiseModel or None, got {type(self.noise_model).__name__}."
            raise TypeError(msg)


@dataclass(frozen=True)
class DigitalSegment:
    """Specification of one digital part of a simulation program.

    Args:
        circuit: Qiskit circuit executed during this segment.
        sim_params: Digital simulation parameters for the segment. If omitted,
            program execution supplies its documented internal defaults.
        noise_model: Optional segment noise model. Noisy program execution is
            introduced by Feature A2.

    Raises:
        TypeError: If an argument does not have the corresponding digital type.
    """

    circuit: QuantumCircuit
    _: KW_ONLY
    sim_params: DigitalSimParams | None = None
    noise_model: NoiseModel | None = None

    def __post_init__(self) -> None:
        """Validate the closed digital segment specification."""
        if not isinstance(self.circuit, QuantumCircuit):
            msg = f"circuit must be QuantumCircuit, got {type(self.circuit).__name__}."
            raise TypeError(msg)
        if self.sim_params is not None and not isinstance(self.sim_params, DigitalSimParams):
            msg = f"sim_params must be DigitalSimParams or None, got {type(self.sim_params).__name__}."
            raise TypeError(msg)
        if self.noise_model is not None and not isinstance(self.noise_model, NoiseModel):
            msg = f"noise_model must be NoiseModel or None, got {type(self.noise_model).__name__}."
            raise TypeError(msg)


@dataclass(frozen=True, init=False)
class SimulationProgram:
    """Ordered analog and digital segments that share one evolving state.

    The supplied iterable is copied into a tuple. Its order therefore cannot be
    changed through the caller's original collection after construction.

    Args:
        segments: Non-empty iterable of analog and digital segment specifications.
        get_state: Whether program execution should retain the final state in the
            outer :class:`~mqt.yaqs.Result`.

    Raises:
        TypeError: If ``segments`` is not iterable, contains an unsupported item,
            or ``get_state`` is not a Boolean.
        ValueError: If ``segments`` is empty.
    """

    segments: tuple[AnalogSegment | DigitalSegment, ...]
    get_state: bool

    def __init__(
        self,
        segments: Iterable[AnalogSegment | DigitalSegment],
        *,
        get_state: bool = False,
    ) -> None:
        """Initialize and validate an immutable ordered program."""
        if isinstance(segments, (str, bytes)):
            msg = "segments must be an iterable of AnalogSegment or DigitalSegment."
            raise TypeError(msg)
        try:
            normalized_segments = tuple(segments)
        except TypeError as error:
            msg = "segments must be an iterable of AnalogSegment or DigitalSegment."
            raise TypeError(msg) from error

        if not normalized_segments:
            msg = "SimulationProgram requires at least one segment."
            raise ValueError(msg)
        for index, segment in enumerate(normalized_segments):
            if not isinstance(segment, (AnalogSegment, DigitalSegment)):
                msg = (
                    f"segments[{index}] must be AnalogSegment or DigitalSegment, "
                    f"got {type(segment).__name__}."
                )
                raise TypeError(msg)
        if not isinstance(get_state, bool):
            msg = f"get_state must be bool, got {type(get_state).__name__}."
            raise TypeError(msg)

        object.__setattr__(self, "segments", normalized_segments)
        object.__setattr__(self, "get_state", get_state)

    def __iter__(self) -> Iterator[AnalogSegment | DigitalSegment]:
        """Iterate over segments in program order."""
        return iter(self.segments)

    def __len__(self) -> int:
        """Return the number of segments in the program."""
        return len(self.segments)
