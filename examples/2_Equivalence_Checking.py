# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Example: Quantum Circuit Equivalence Checking with YAQS.

This script demonstrates how to create a TwoLocal quantum circuit, assign random parameters,
measure the circuit, and then transpile it into a different basis. The original and transpiled
circuits are then compared for equivalence using the YAQS equivalence checker.

Usage:
    Run this module as a script to execute the equivalence check.
"""

from __future__ import annotations

import copy

import numpy as np
import qiskit.compiler
from qiskit.circuit.library.n_local import TwoLocal

from mqt.yaqs.circuits import equivalence_checker

if __name__ == "__main__":
    # Define the number of qubits and circuit depth.
    num_qubits = 5
    depth = num_qubits

    # Create a TwoLocal circuit and decompose it.
    twolocal = TwoLocal(num_qubits, ["rx"], ["rzz"], entanglement="linear", reps=depth).decompose()
    num_pars = len(twolocal.parameters)
    # Assign random parameters uniformly in [-pi, pi].
    values = np.random.uniform(low=-np.pi, high=np.pi, size=num_pars)
    circuit = copy.deepcopy(twolocal).assign_parameters(values)
    circuit.measure_all()

    # Transpile the circuit to a new basis.
    basis_gates = ["cz", "rz", "sx", "x", "id"]
    transpiled_circuit = qiskit.compiler.transpile(circuit, basis_gates=basis_gates, optimization_level=1)

    # Define parameters for equivalence checking.
    threshold = 1e-6
    fidelity = 1 - 1e-13
    result = equivalence_checker.run(circuit, transpiled_circuit, threshold, fidelity)
