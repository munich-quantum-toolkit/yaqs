# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Example: Benchmarker for Circuit TJM.

This example demonstrates how to benchmark a quantum circuit using the YAQS benchmarker.
In this example, an Ising circuit is generated using create_Ising_circuit, and then the
benchmarker is used to evaluate its performance. The benchmark output is displayed in a
"dot" style visualization.

Usage:
    Run this script directly to execute the benchmark.
"""

from __future__ import annotations

from mqt.yaqs.circuits import benchmarker
from mqt.yaqs.core.libraries.circuit_library import create_Ising_circuit

if __name__ == "__main__":
    num_qubits = 10
    J = 1
    g = 0.5
    demo_circuit = create_Ising_circuit(num_qubits, J, g, dt=0.1, timesteps=10)
    benchmarker.run(demo_circuit, style="dots")
