# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from mqt.yaqs.core.libraries.circuit_library import create_Ising_circuit
from mqt.yaqs.circuits import benchmarker

if __name__ == "__main__":
    num_qubits = 10
    model = {"name": "Ising", "L": num_qubits, "J": 1, "g": 0.5}
    demo_circuit = create_Ising_circuit(model, dt=0.1, timesteps=10)
    # Run the benchmark on the demo circuit.
    benchmarker.run(demo_circuit, style="dots")
