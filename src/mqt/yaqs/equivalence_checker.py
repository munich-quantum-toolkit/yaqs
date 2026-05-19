# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""MPO-based equivalence checker for quantum circuits.

This module provides :class:`EquivalenceChecker` for comparing two quantum circuits
using an MPO-based algorithm. Circuits are converted to DAGs and an MPO representation
is updated iteratively to determine whether the composed operator approximates the
identity within a configurable fidelity threshold.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from qiskit.converters import circuit_to_dag

from .core.data_structures.mpo import MPO
from .digital.utils.mpo_utils import iterate

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

__all__ = ["EquivalenceChecker"]


class EquivalenceChecker:
    """Public entry point for MPO-based circuit equivalence checking.

    Owns the numerical tuning (SVD truncation threshold and fidelity threshold).
    The two circuits to compare are passed per call to :meth:`check`.

    Attributes:
        threshold: Singular-value truncation threshold used during SVD in the MPO update.
        fidelity: Fidelity threshold for deciding whether the final MPO is identity-like.
    """

    def __init__(
        self,
        *,
        threshold: float = 1e-13,
        fidelity: float = 1 - 1e-13,
    ) -> None:
        """Initialize the checker with numerical thresholds.

        Args:
            threshold: SVD truncation threshold in the MPO update (default ``1e-13``).
            fidelity: Minimum fidelity to treat the final MPO as identity (default ``1 - 1e-13``).
        """
        self.threshold = threshold
        self.fidelity = fidelity

    def check(
        self,
        circuit1: QuantumCircuit,
        circuit2: QuantumCircuit,
    ) -> dict[str, bool | float]:
        """Check whether two quantum circuits are equivalent.

        Converts both circuits to DAGs and applies an iterative MPO update. If the
        circuits differ only up to global phase and numerical error, the final MPO
        approximates the identity.

        Args:
            circuit1: First quantum circuit.
            circuit2: Second quantum circuit (must have the same number of qubits).

        Returns:
            dict[str, bool | float]: ``equivalent`` (bool) and ``elapsed_time`` (float, seconds).
        """
        assert circuit1.num_qubits == circuit2.num_qubits, "Circuits must have the same number of qubits."

        start_time = time.time()
        mpo = MPO()
        mpo.identity(circuit1.num_qubits)

        circuit1_dag = circuit_to_dag(circuit1)
        circuit2_dag = circuit_to_dag(circuit2)

        iterate(mpo, circuit1_dag, circuit2_dag, self.threshold)

        return {"equivalent": mpo.check_if_identity(self.fidelity), "elapsed_time": time.time() - start_time}
