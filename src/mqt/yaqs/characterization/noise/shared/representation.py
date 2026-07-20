# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Forward-model representation selection for Markovian noise characterization."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.state import State
    from mqt.yaqs.core.data_structures.state_utils import Representation

NoiseRepresentation = Literal["density_matrix", "vector", "mps", "auto"]

DEFAULT_LINDBLAD_MAX_QUBITS = 8
DEFAULT_VECTOR_MAX_QUBITS = 10


def resolve_noise_representation(
    chain_length: int,
    representation: NoiseRepresentation,
    *,
    lindblad_max_qubits: int = DEFAULT_LINDBLAD_MAX_QUBITS,
    vector_max_qubits: int = DEFAULT_VECTOR_MAX_QUBITS,
) -> Representation:
    """Resolve the Simulator forward backend for noise characterization.

    ``auto`` prefers deterministic Lindblad on small chains, then MCWF, then TJM.

    Args:
        chain_length: Number of qubits in the Hamiltonian chain.
        representation: User representation selection.
        lindblad_max_qubits: Inclusive upper qubit count for ``auto`` → ``density_matrix``.
        vector_max_qubits: Inclusive upper qubit count for ``auto`` → ``vector``.

    Returns:
        Resolved ``"density_matrix"``, ``"vector"``, or ``"mps"``.

    Raises:
        ValueError: If ``representation`` is invalid.
    """
    rep = str(representation).strip().lower()
    n_sites = int(chain_length)
    if rep == "density_matrix":
        return "density_matrix"
    if rep == "vector":
        return "vector"
    if rep == "mps":
        return "mps"
    if rep == "auto":
        if n_sites <= int(lindblad_max_qubits):
            return "density_matrix"
        if n_sites <= int(vector_max_qubits):
            return "vector"
        return "mps"
    msg = f"representation must be 'density_matrix', 'vector', 'mps', or 'auto', got {representation!r}."
    raise ValueError(msg)


def prepare_state_for_representation(
    init_state: State,
    representation: Representation,
) -> State:
    """Return a deep copy of ``init_state`` encoded for the requested representation.

    Args:
        init_state: Initial state supplied by the user.
        representation: Resolved forward-model representation.

    Returns:
        Prepared :class:`~mqt.yaqs.core.data_structures.state.State`.
    """
    prepared = copy.deepcopy(init_state)
    prepared.ensure_encoded(representation)
    prepared.representation = representation
    return prepared
