# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import copy
import time

import numpy as np
from helper import state_vector_simulator, tdvp_simulator, tebd_simulator
from qiskit.circuit import QuantumCircuit


def create_long_range_circuit(num_qubits):
    # Create a circuit with the specified number of qubits
    qc = QuantumCircuit(num_qubits)
    # for i in range(num_qubits):
    #     qc.rz(0.01, i)
    # qc.rxx(0.01, 0, num_qubits - 1)
    for i in range(num_qubits):
        qc.rz(0.01, i)
    for i in range(num_qubits // 2):
        # theta = np.random.uniform(0, 2 * np.pi)
        qc.rxx(0.1, i, num_qubits - 1 - i)
        qc.barrier()
    # qc.measure_all()
    # print(qc)
    return qc


def test_rxx() -> None:
    threshold = 0
    max_bond = 2**16

    pad = 2
    for num_qubits in range(4, 21):
        # circ = create_heisenberg_circuit(num_qubits, 1, 1, 1, 0.1, 0.1, 10)
        # circ = create_heisenberg_circuit(num_qubits, 1, 1, 1, 0.1, 0.1, 10)
        circ = create_long_range_circuit(num_qubits)
        circ2 = copy.deepcopy(circ)
        circ2.save_statevector()
        exact_result = state_vector_simulator(circ2)
        time.time()
        _mps, tdvp_result = tdvp_simulator(circ, max_bond, threshold, pad)

        time.time()
        tebd_result = tebd_simulator(circ2, max_bond, threshold)
        np.abs(1 - np.abs(np.vdot(exact_result, tebd_result)) ** 2)
        np.abs(1 - np.abs(np.vdot(exact_result, tdvp_result)) ** 2)


if __name__ == "__main__":
    test_rxx()
