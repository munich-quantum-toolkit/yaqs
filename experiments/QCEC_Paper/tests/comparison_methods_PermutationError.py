# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import copy
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import qiskit.circuit
import qiskit.compiler
from qiskit.circuit.library.n_local import TwoLocal
from src.causal_algorithm import run

from mqt import qcec

max_qubits = 128
threshold = 1e-1
fidelity = 1 - 1e-13
errors = 10
starting_gates = ["h", "x", "cx", "cz", "swap", "id", "rz", "rx", "ry", "rxx", "ryy", "rzz"]

# Original
# basis_gates = ['h', 'x', 'cx', 'rz', 'id']
# IBM Heron
basis_gates = ["cz", "rz", "sx", "x", "id"]
# Quantinuum H1-1, H1-2
# basis_gates = ['rx', 'ry', 'rz', 'rzz']

cutoff = 1e6
calculate_TN = False
calculate_DD = True
calculate_ZX = False
calculate = [calculate_TN, calculate_DD, calculate_ZX]
assert sum(calculate) == 1


# TN
# x_list = range(2, 34, 2) # 1 errors
# x_list = range(2, 20, 2) # 5 errors
x_list = range(2, 18, 2)  # 10 errors

# DD
x_list = range(2, 9)

samples = 10
runs = {"method": "TN", "N": x_list, "t": []}
for _sample in range(samples):
    TN_times = []
    DD_times = []
    ZX_times = []
    for num_qubits in x_list:
        depth = num_qubits
        circuit = qiskit.circuit.QuantumCircuit(num_qubits)
        twolocal = TwoLocal(num_qubits, ["rx"], ["rzz"], entanglement="linear", reps=depth).decompose()
        num_pars = len(twolocal.parameters)
        values = np.random.uniform(low=-np.pi, high=np.pi, size=num_pars)
        circuit = copy.deepcopy(twolocal).assign_parameters(values)

        transpiled_circuit = qiskit.compiler.transpile(circuit, basis_gates=basis_gates, optimization_level=1)
        for _ in range(errors):
            qubits = range(num_qubits)
            qubit0 = np.random.choice(qubits)
            swap_circuit = qiskit.QuantumCircuit(num_qubits)
            if qubit0 == num_qubits - 1:
                swap_circuit.swap(qubit0 - 1, qubit0)
            else:
                swap_circuit.swap(qubit0, qubit0 + 1)
            transpiled_circuit = swap_circuit.compose(transpiled_circuit)

        if calculate_TN:
            start_time = time.time()
            result = run(copy.deepcopy(circuit), copy.deepcopy(transpiled_circuit), threshold, fidelity)
            end_time = time.time()

            TN_time = end_time - start_time
        else:
            TN_time = None

        if calculate_DD:
            start_time = time.time()
            ecm = qcec.EquivalenceCheckingManager(circ1=circuit, circ2=transpiled_circuit)
            ecm.set_zx_checker(False)
            ecm.set_parallel(False)
            ecm.set_simulation_checker(False)
            ecm.set_timeout(120)
            ecm.run()
            result = qcec.verify(
                circuit,
                transpiled_circuit,
                fuse_single_qubit_gates=False,
                run_simulation_checker=False,
                run_alternating_checker=True,
                run_zx_checker=False,
            )
            end_time = time.time()
            DD_time = end_time - start_time
            if ecm.get_results().equivalence == "no_information":
                DD_time = 3600
            if DD_time > cutoff:
                calculate_DD = False
        else:
            DD_time = None

        if calculate_ZX:
            start_time = time.time()
            result = qcec.verify(
                circuit,
                transpiled_circuit,
                fuse_single_qubit_gates=False,
                run_simulation_checker=False,
                run_alternating_checker=False,
                run_construction_checker=False,
                run_zx_checker=True,
            )
            end_time = time.time()
            ZX_time = end_time - start_time
            if ZX_time > cutoff:
                calculate_ZX = False
        else:
            ZX_time = None

        TN_times.append(TN_time)

        DD_times.append(DD_time)
        ZX_times.append(ZX_time)

    runs["t"].append(DD_times)
    pickle.dump(runs, open("DD10_permutation.p", "wb"))


plt.title("Verification of VQE Circuit")
plt.plot(x_list, TN_times, label="TN")

plt.plot(x_list, DD_times, label="DD")
plt.plot(x_list, ZX_times, label="ZX")

plt.yscale("log")
plt.ylim(top=cutoff)
plt.xlabel("Qubits")
plt.ylabel("Runtime (s)")
plt.legend()
plt.show()
