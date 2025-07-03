"""
Test file to compare Qiskit noisy quantum circuit simulator with Kraus channel simulator.
Uses dephasing noise models to validate both approaches give consistent results.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer
import matplotlib.pyplot as plt


from mqt.yaqs.noisy_qc_sim.qiskit_noisy_sim import qiskit_noisy_simulator
from mqt.yaqs.noisy_qc_sim.densitymatrix_sim import (
    create_all_zero_density_matrix, 
    evolve_noisy_circuit, 
    circuit_to_unitary_list, 
    z_expectations,
    two_qubit_reverse
)
from mqt.yaqs.noisy_qc_sim.qiskit_noisemodels import qiskit_dephasing_noise, qiskit_bitflip_noise, qiskit_bitflip_noise_2
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.noisy_qc_sim.yaqs_noisemodels import create_yaqs_bitflip_noise, create_yaqs_bitflip_noise_2, create_yaqs_dephasing_noise
from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS



if __name__ == "__main__":

    kraus_results_list = []
    yaqs_results_list = []

    noise_strengths = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    noise_strengths = [0.5]
    

    for noise_strength in noise_strengths:
        print("-"*100)
        print(f"Starting noisy quantum circuit simulator comparison tests for noise strength: {noise_strength}")
        print("-"*100)

        gamma = noise_strength *1.45

        # direct vector matrix simulation
        #########################################################

        pauli_x = np.array([[0, 1], [1, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])

        rho0 = create_all_zero_density_matrix(2)

        CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
        rho1 = CNOT @ rho0 @ CNOT.T

        I_X = np.kron(np.eye(2), pauli_x)
        X_I = np.kron(pauli_x, np.eye(2))
        X_X = np.kron(pauli_x, pauli_x)

        p = noise_strength
        K0 = np.sqrt((1-p)**2) * np.eye(4)
        K1 = np.sqrt(p*(1-p)) * X_I
        K2 = np.sqrt(p*(1-p)) * I_X
        K3 = np.sqrt(p**2) * X_X

        rho_noise = (
            K0 @ rho1 @ K0.conj().T +
            K1 @ rho1 @ K1.conj().T +
            K2 @ rho1 @ K2.conj().T +
            K3 @ rho1 @ K3.conj().T
        )
        
        vector_matrix_sim_expvals = []
        vector_matrix_sim_expvals.append(np.trace(rho_noise @ np.kron(pauli_z, np.eye(2))))
        vector_matrix_sim_expvals.append(np.trace(rho_noise @ np.kron(np.eye(2), pauli_z)))
        vector_matrix_sim_expvals = np.array([[np.real(val) for val in vector_matrix_sim_expvals]])

        kraus_results_list.append(vector_matrix_sim_expvals)
        
        print(f"vector_matrix_sim_expvals: {vector_matrix_sim_expvals}")


        #########################################################

        qc = QuantumCircuit(2)
        qc.cx(0, 1)

        ### 
        # Qiskit simulation
        #########################################################

        noise_model = qiskit_bitflip_noise(num_qubits=2, noise_strengths=[noise_strength])
        qiskit_results = qiskit_noisy_simulator(qc, noise_model, 2, 1)
        print(f"qiskit_results: {qiskit_results}")



        
        processes_yaqs = [{
                "name": "xx",
                "sites": [0, 1],
                    "strength": gamma**2
            },
            {
                "name": "x",
                "sites": [0],
                "strength": (1-gamma)*gamma
            }, 
            {
                "name": "x",
                "sites": [1],
                "strength": (1-gamma)*gamma
            }
            ]


    
        ### 
        # Kraus channel func evolve noisy circuit simulation
        #########################################################

        processes = [{
                "name": "xx",
                "sites": [0, 1],
                "strength": noise_strength**2
            },
            {
                "name": "x",
                "sites": [0],
                "strength": (1-noise_strength)*noise_strength
            }, 
            {
                "name": "x",
                "sites": [1],
                "strength": (1-noise_strength)*noise_strength
            }
            ]
        noise_model = NoiseModel(processes)
        noise_model_yaqs = NoiseModel(processes_yaqs)

        rho0 = create_all_zero_density_matrix(2)

        gate_list = circuit_to_unitary_list(qc)
        
        kraus_results = evolve_noisy_circuit(rho0, gate_list, noise_model, 1)
        
        print(f"evolve_noisy_circuit results: {kraus_results}")


        ### 
        # YAQS simulation
        #########################################################

        sim_params = StrongSimParams(observables=[Observable(gate=Z(), sites=[i]) for i in range(2)], num_traj=1000, max_bond_dim=2, threshold=1e-14, get_state=False)
        
        initial_mps = MPS(2, state = "zeros", pad=2)
        
        simulator.run(initial_mps, qc, sim_params, noise_model_yaqs)
        yaqs_results = []
        for i in range(2):
            yaqs_results.append(sim_params.observables[i].results)

        yaqs_results_list.append(yaqs_results)
        
        print(f"yaqs_results: {yaqs_results}")

    difference = [kraus_results_list[i][0] - yaqs_results_list[i][0] for i in range(len(kraus_results_list))]
    kraus_results_list = [kraus_results_list[i][0] for i in range(len(kraus_results_list))]
    yaqs_results_list = [yaqs_results_list[i][0] for i in range(len(yaqs_results_list))]
    # plot difference between kraus and yaqs results    
    plt.plot(noise_strengths, kraus_results_list, marker="o", label='Kraus')
    plt.plot(noise_strengths, yaqs_results_list, marker="s", label='YAQS')
    plt.plot(noise_strengths, difference, marker="x", linestyle="--", label='Kraus - YAQS')

    plt.xlabel('Noise Strength')
    plt.ylabel('Expectation Value')
    plt.title("Comparison of Kraus and YAQS Simulation")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
        
        
        
        