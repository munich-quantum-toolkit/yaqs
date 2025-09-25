import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit_aer.noise.errors import PauliLindbladError


import numpy as np
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit, create_heisenberg_circuit
from worker_functions.circuits import build_basis_noncommuting, x_commuting_brickwork, noncommuting_layer
from worker_functions.plotting import plot_series_against_exact, plot_avg_bond_dims, plot_stochastic_variances, pick_subplot_indices
from worker_functions.qiskit_simulators import collect_expectations_and_mps_bond_dims, run_qiskit_exact, run_qiskit_mps, BenchmarkConfig
from worker_functions.mean_error import print_mean_errors_against_exact
from worker_functions.yaqs_simulator import run_yaqs, build_noise_models


# Support running as a script from the repository root
import os
import sys
_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


if __name__ == "__main__":

    L = 8
    num_layers = 10
    noise_strength = 0.01
    num_traj = 1024



    basis_circuit = create_heisenberg_circuit(L, 1, 0.5, 0.5, 0.1, 0.1, 1)
    # basis_circuit = noncommuting_layer(L)
    # basis_circuit = x_commuting_brickwork(L, 1, add_barriers=False)
    # for i in range(L):
    #     basis_circuit.rx(np.pi/2, i)
    #     basis_circuit.barrier()
    # for i in range(L-1):
    #     basis_circuit.rzz(np.pi/2, i, i+1)
    #     basis_circuit.barrier()
    print("circuit initialized")

    

    processes = [
        {"name": "pauli_x", "sites": [i], "strength": noise_strength}
        for i in range(L)] + [{"name": "crosstalk_xx", "sites": [i, i+1], "strength": noise_strength}
        for i in range(L-1) 
        ]
    noise_model_normal, noise_model_projector, noise_model_unitary_2pt, noise_model_unitary_gauss = build_noise_models(processes)

    qiskit_noise_model = QiskitNoiseModel()
    TwoQubit_XX_error = PauliLindbladError([Pauli("IX"), Pauli("XI"), Pauli("XX")], [noise_strength, noise_strength, noise_strength])
    #TwoQubit_YY_error = PauliLindbladError([Pauli("IY"), Pauli("YI"), Pauli("YY")], [noise_strength, noise_strength, noise_strength])
    #TwoQubit_ZZ_error = PauliLindbladError([Pauli("IZ"), Pauli("ZI"), Pauli("ZZ")], [noise_strength, noise_strength, noise_strength])
    for qubit in range(L-1):
        qiskit_noise_model.add_quantum_error(TwoQubit_XX_error, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])
        # qiskit_noise_model.add_quantum_error(TwoQubit_YY_error, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])
        # qiskit_noise_model.add_quantum_error(TwoQubit_ZZ_error, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])

    print("noise models initialized")
    exact = run_qiskit_exact(L, num_layers, basis_circuit, qiskit_noise_model, method="density_matrix")
    print("qiskit exact initialized, starting qiskit mps")
    qiskit_mps_expvals, qiskit_mps_bonds, qiskit_mps_var = run_qiskit_mps(L, num_layers, basis_circuit, qiskit_noise_model, num_traj=num_traj)
    print("qiskit mps initialized, starting yaqs")
    yaqs_results_normal, yaqs_bonds_normal, yaqs_var_normal = run_yaqs(basis_circuit, L, num_layers, noise_model_normal, num_traj=num_traj)
    print("yaqs normal initialized, starting yaqs projector")
    yaqs_results_projector, yaqs_bonds_projector, yaqs_var_projector = run_yaqs(basis_circuit, L, num_layers, noise_model_projector, num_traj=num_traj)
    print("yaqs projector initialized, starting yaqs unitary 2pt")
    yaqs_results_unitary_2pt, yaqs_bonds_unitary_2pt, yaqs_var_unitary_2pt = run_yaqs(basis_circuit, L, num_layers, noise_model_unitary_2pt, num_traj=num_traj)
    print("yaqs unitary 2pt initialized, starting yaqs unitary gauss")
    yaqs_results_unitary_gauss, yaqs_bonds_unitary_gauss, yaqs_var_unitary_gauss = run_yaqs(basis_circuit, L, num_layers, noise_model_unitary_gauss, num_traj=num_traj)


    series_by_label = {
    "standard": yaqs_results_normal,
    "projector": yaqs_results_projector,
    "unitary_2pt": yaqs_results_unitary_2pt,
    "unitary_gauss": yaqs_results_unitary_gauss,
    "qiskit_mps": qiskit_mps_expvals,
    }

    # print variances (lower is better)
    # removed variance-vs-exact; instead use stochastic variances below
    # print mean absolute errors (lower is better)
    mean_errors = print_mean_errors_against_exact(exact, series_by_label)

    # plot results vs exact
    plot_series_against_exact(
        exact,
        series_by_label,
        num_qubits=L,
        num_layers=num_layers,
    )


    plot_stochastic_variances(
        num_layers=num_layers,
        qiskit_var=qiskit_mps_var,
        yaqs_var_by_label={
            "standard": yaqs_var_normal,
            "projector": yaqs_var_projector,
            "unitary_2pt": yaqs_var_unitary_2pt,
            "unitary_gauss": yaqs_var_unitary_gauss,
        },
    )
    plot_avg_bond_dims(
        num_layers=num_layers,
        qiskit_bonds=qiskit_mps_bonds,
        yaqs_bonds_by_label={
            "standard": yaqs_bonds_normal,
            "projector": yaqs_bonds_projector,
            "unitary_2pt": yaqs_bonds_unitary_2pt,
            "unitary_gauss": yaqs_bonds_unitary_gauss,
        },
    )





