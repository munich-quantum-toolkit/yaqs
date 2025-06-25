import numpy as np



hadamard_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
pauli_x = np.array([[0, 1], [1, 0]])
pauli_y = np.array([[0, -1j], [1j, 0]])
pauli_z = np.array([[1, 0], [0, -1]])
rho0 = np.array([[1, 0], [0, 0]])



def single_qubit_sim(rho0, noise_strength, gate, kraus_op):
    print('--------------------------------')

    rho1 = gate @ rho0 @ gate.conj().T
    rho1_noisy = rho1 * (1 - noise_strength) + noise_strength * kraus_op @ rho1 @ kraus_op.conj().T
    # print(f"Rho after {gate}:", rho1)
    # print("Rho after noise:", rho1_noisy)
    print(f"Noise strength: {noise_strength}")
    print("Z expectation:", np.trace(pauli_z @ rho1_noisy))
    print("X expectation:", np.trace(pauli_x @ rho1_noisy))
    print("Y expectation:", np.trace(pauli_y @ rho1_noisy))
    print('--------------------------------')
    return 

if __name__ == "__main__":

    single_qubit_sim(rho0, 0, hadamard_gate, pauli_x)
    single_qubit_sim(rho0, 0, hadamard_gate, pauli_y)
    single_qubit_sim(rho0, 0, hadamard_gate, pauli_z)
    single_qubit_sim(rho0, 0.5, hadamard_gate, pauli_x)
    single_qubit_sim(rho0, 0.5, hadamard_gate, pauli_y)
    single_qubit_sim(rho0, 0.5, hadamard_gate, pauli_z)








