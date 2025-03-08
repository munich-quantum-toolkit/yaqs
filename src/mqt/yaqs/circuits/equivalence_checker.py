# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from qiskit.converters import circuit_to_dag

from ..core.data_structures.networks import MPO
from .utils.mpo_utils import iterate

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit


def run(
    circuit1: QuantumCircuit, circuit2: QuantumCircuit, threshold: float = 1e-13, fidelity: float = 1 - 1e-13
) -> dict[str, bool | float]:
    """Check the equivalence of two quantum circuits using an MPO-based algorithm.

    This function converts two quantum circuits into their DAG representations and then applies an iterative
    MPO update procedure to compare the two circuits. An identity MPO is initialized for the number of qubits,
    and the gate layers from the two circuits are applied to update the MPO. If the circuits are equivalent,
    the final MPO will approximate the identity operation. Equivalence is determined by checking whether the
    MPO fidelity exceeds the provided fidelity threshold.

    Args:
        circuit1 (QuantumCircuit): The first quantum circuit.
        circuit2 (QuantumCircuit): The second quantum circuit.
        threshold (float, optional): The singular value truncation threshold used during SVD decomposition
            in the MPO update process (default: 1e-13).
        fidelity (float, optional): The fidelity threshold for determining circuit equivalence
            (default: 1 - 1e-13).

    Returns:
        dict[str, bool | float]: A dictionary containing:
            'equivalent' (bool): True if the circuits are equivalent (i.e., the final MPO approximates the identity
                                  within the specified fidelity), False otherwise.
            'elapsed_time' (float): The total runtime (in seconds) of the equivalence check.
    """
    assert circuit1.num_qubits == circuit2.num_qubits, "Circuits must have the same number of qubits."

    start_time = time.time()
    mpo = MPO()
    mpo.init_identity(circuit1.num_qubits)

    circuit1_dag = circuit_to_dag(circuit1)
    circuit2_dag = circuit_to_dag(circuit2)

    iterate(mpo, circuit1_dag, circuit2_dag, threshold)

    return {"equivalent": mpo.check_if_identity(fidelity), "elapsed_time": time.time() - start_time}
