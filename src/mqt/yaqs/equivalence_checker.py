# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Circuit equivalence checker with MPO and dense matrix backends.

This module provides :class:`EquivalenceChecker` for comparing two quantum circuits.
The scalable MPO algorithm is the primary backend; a dense tensorized matrix backend is
available for very small circuits. With ``representation="auto"``, circuits with at most
:data:`DEFAULT_MATRIX_MAX_QUBITS` qubits use the matrix backend and larger circuits use MPO.
Pass ``representation="mpo"`` explicitly for production workloads.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal, cast

from qiskit.converters import circuit_to_dag

from .core.data_structures.mpo import MPO
from .digital.utils.matrix_utils import check_equivalence_matrix
from .digital.utils.mpo_utils import iterate

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

__all__ = ["DEFAULT_MATRIX_MAX_QUBITS", "EquivalenceChecker", "EquivalenceRepresentation"]

EquivalenceRepresentation = Literal["auto", "matrix", "mpo"]
DEFAULT_MATRIX_MAX_QUBITS = 7


def _validate_representation(representation: str) -> EquivalenceRepresentation:
    allowed = ("auto", "matrix", "mpo")
    if representation not in allowed:
        msg = f"representation must be one of {allowed!r}, got {representation!r}."
        raise ValueError(msg)
    return cast("EquivalenceRepresentation", representation)


def _validate_matrix_max_qubits(matrix_max_qubits: int) -> int:
    if isinstance(matrix_max_qubits, bool) or not isinstance(matrix_max_qubits, int):
        msg = f"matrix_max_qubits must be int, got {type(matrix_max_qubits).__name__}."
        raise TypeError(msg)
    if matrix_max_qubits < 0:
        msg = f"matrix_max_qubits must be non-negative, got {matrix_max_qubits}."
        raise ValueError(msg)
    return matrix_max_qubits


class EquivalenceChecker:
    """Public entry point for circuit equivalence checking.

    The MPO backend is the primary, scalable method; the matrix backend is intended for
    very small qubits counts. Owns numerical thresholds and backend selection. The two
    circuits to compare are passed per call to :meth:`check`.

    Attributes:
        threshold: Singular-value truncation threshold used during SVD in the MPO update.
        fidelity: Fidelity threshold for deciding whether the composed operator is identity-like.
        representation: Backend selection (``"auto"``, ``"matrix"``, or ``"mpo"``).
        matrix_max_qubits: Qubit count cutover for ``representation="auto"``.
    """

    def __init__(
        self,
        *,
        threshold: float = 1e-13,
        fidelity: float = 1 - 1e-13,
        representation: EquivalenceRepresentation = "auto",
        matrix_max_qubits: int = DEFAULT_MATRIX_MAX_QUBITS,
    ) -> None:
        """Initialize the checker with numerical thresholds and backend options.

        Args:
            threshold: SVD truncation threshold in the MPO update (default ``1e-13``).
            fidelity: Minimum fidelity to treat the composed operator as identity (default ``1 - 1e-13``).
            representation: ``"auto"`` picks matrix for ``num_qubits <= matrix_max_qubits``, else MPO;
                ``"matrix"`` or ``"mpo"`` force that backend.
            matrix_max_qubits: Cutover for ``representation="auto"`` (default ``7``).
        """
        self.threshold = threshold
        self.fidelity = fidelity
        self.representation = _validate_representation(representation)
        self.matrix_max_qubits = _validate_matrix_max_qubits(matrix_max_qubits)

    def _resolve_representation(self, num_qubits: int) -> Literal["matrix", "mpo"]:
        if self.representation == "matrix":
            return "matrix"
        if self.representation == "mpo":
            return "mpo"
        return "matrix" if num_qubits <= self.matrix_max_qubits else "mpo"

    def check(
        self,
        circuit1: QuantumCircuit,
        circuit2: QuantumCircuit,
    ) -> dict[str, bool | float | str]:
        """Check whether two quantum circuits are equivalent.

        If the circuits differ only up to global phase and numerical error, the composed
        operator ``U2† U1`` approximates the identity.

        Args:
            circuit1: First quantum circuit.
            circuit2: Second quantum circuit (must have the same number of qubits).

        Returns:
            dict[str, bool | float | str]: ``equivalent`` (bool), ``elapsed_time`` (float, seconds),
            and ``representation`` (``"matrix"`` or ``"mpo"``) indicating the backend used.

        Raises:
            ValueError: If the circuits have different numbers of qubits.
        """
        if circuit1.num_qubits != circuit2.num_qubits:
            msg = "Circuits must have the same number of qubits."
            raise ValueError(msg)

        backend = self._resolve_representation(circuit1.num_qubits)
        start_time = time.time()

        if backend == "matrix":
            equivalent = check_equivalence_matrix(circuit1, circuit2, fidelity=self.fidelity)
        else:
            mpo = MPO.identity(circuit1.num_qubits)
            circuit1_dag = circuit_to_dag(circuit1)
            circuit2_dag = circuit_to_dag(circuit2)
            iterate(mpo, circuit1_dag, circuit2_dag, self.threshold)
            equivalent = mpo.check_if_identity(self.fidelity)

        return {
            "equivalent": equivalent,
            "elapsed_time": time.time() - start_time,
            "representation": backend,
        }
