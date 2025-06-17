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

from mqt import qcec
from mqt.yaqs.circuits.equivalence_checker import run

if __name__ == "__main__":
    max_qubits = 25
    threshold = 1e-6
    fidelity = 1 - 1e-13
    starting_gates = ["h", "x", "cx", "cz", "swap", "id", "rz", "rx", "ry", "rxx", "ryy", "rzz"]

    # Original
    # basis_gates = ['h', 'x', 'cx', 'rz', 'id']
    # IBM Heron
    basis_gates = ["cz", "rz", "sx", "x", "id"]
    # Quantinuum H1-1, H1-2
    # basis_gates = ['rx', 'ry', 'rz', 'rzz']

    cutoff = 1e6
    calculate_TN = False
    calculate_DD = False
    calculate_ZX = True
    calculate = [calculate_TN, calculate_DD, calculate_ZX]
    assert sum(calculate) == 1

    x_list = range(2, 33, 2)  # TN
    # x_list = range(2, 12) # DD
    # x_list = range(2, 33, 2) # ZX

    samples = 10
    runs = {"method": "DD", "N": x_list, "t": []}
    for _sample in range(samples):
        TN_times = []
        DD_times = []
        ZX_times = []
        for num_qubits in x_list:
            depth = num_qubits
            circuit = qiskit.circuit.QuantumCircuit(num_qubits)
            twolocal = TwoLocal(num_qubits, ["rx"], ["rzz"], entanglement="linear", reps=depth).decompose()
            num_pars = len(twolocal.parameters)
            values = np.random.rand(num_pars)
            circuit = copy.deepcopy(twolocal).assign_parameters(values)
            circuit.measure_all()

            transpiled_circuit = qiskit.compiler.transpile(circuit, basis_gates=basis_gates, optimization_level=1)

            if calculate_TN:
                start_time = time.time()
                result = run(copy.deepcopy(circuit), copy.deepcopy(transpiled_circuit), threshold, fidelity)
                assert result
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
                ecm.set_timeout(10)
                ecm.run()
                result = qcec.verify(
                    circuit,
                    transpiled_circuit,
                    fuse_single_qubit_gates=False,
                    run_simulation_checker=False,
                    run_alternating_checker=True,
                    run_zx_checker=False,
                )
                assert result
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
        pickle.dump(runs, open("DD_Lin.p", "wb"))

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
