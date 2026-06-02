# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Process-pool workers and picklable task payloads for equivalence checking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.libraries.gate_library import BaseGate


@dataclass(frozen=True)
class SerializedGate:
    """Picklable gate data for parallel workers."""

    name: str
    interaction: int
    sites: tuple[int, ...]
    matrix: NDArray[np.complex128]
    tensor: NDArray[np.complex128] | None


@dataclass(frozen=True)
class MpoPairUpdateTask:
    """Task payload for one checkerboard MPO pair update."""

    site: int
    tensor_n: NDArray[np.complex128]
    tensor_n1: NDArray[np.complex128]
    gates1: tuple[SerializedGate, ...]
    gates2: tuple[SerializedGate, ...]
    threshold: float
    apply_conjugate_on_second: bool


@dataclass(frozen=True)
class MpoPairUpdateResult:
    """Result of one MPO pair update."""

    site: int
    tensor_n: NDArray[np.complex128]
    tensor_n1: NDArray[np.complex128]


def serialize_gates(gates: list[BaseGate]) -> tuple[SerializedGate, ...]:
    """Convert gate objects to picklable payloads."""
    serialized: list[SerializedGate] = []
    for gate in gates:
        tensor = None
        if gate.interaction == 2:
            tensor = np.asarray(gate.tensor, dtype=np.complex128)
        serialized.append(
            SerializedGate(
                name=gate.name,
                interaction=gate.interaction,
                sites=tuple(gate.sites),
                matrix=np.asarray(gate.matrix, dtype=np.complex128),
                tensor=tensor,
            )
        )
    return tuple(serialized)


def _gate_from_serialized(spec: SerializedGate) -> BaseGate:
    from ...core.libraries.gate_library import BaseGate

    gate = BaseGate(spec.matrix.copy())
    gate.name = spec.name
    gate.interaction = spec.interaction
    if spec.tensor is not None:
        gate.tensor = spec.tensor.copy()
    gate.set_sites(*spec.sites)
    return gate


def mpo_pair_update_worker(task: MpoPairUpdateTask) -> MpoPairUpdateResult:
    """Compute updated MPO site tensors for one qubit pair (process-pool entry point)."""
    from .mpo_utils import compute_pair_update

    gates1 = [_gate_from_serialized(spec) for spec in task.gates1]
    gates2 = [_gate_from_serialized(spec) for spec in task.gates2]
    qubits = [task.site, task.site + 1]
    tensor_n, tensor_n1 = compute_pair_update(
        task.tensor_n,
        task.tensor_n1,
        gates1,
        gates2,
        task.threshold,
        qubits,
        apply_conjugate_on_second=task.apply_conjugate_on_second,
    )
    return MpoPairUpdateResult(site=task.site, tensor_n=tensor_n, tensor_n1=tensor_n1)
