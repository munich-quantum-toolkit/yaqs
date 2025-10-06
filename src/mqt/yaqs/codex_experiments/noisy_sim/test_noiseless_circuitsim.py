from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel as qiskit_NoiseModel
from qiskit_aer.primitives import Estimator
from qiskit_aer.noise.errors import PauliError
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer
import numpy as np
from mqt.yaqs. import create_all_zero_density_matrix, evolve_noisy_circuit, circuit_to_unitary_list, z_expectations, get_qiskit_z_expectations



def run_test(test_name, num_qubits, circuit_builder, expected_z_qiskit):
    print(f"\n--- Running Test: {test_name} ({num_qubits} qubits) ---")
    
    qc = QuantumCircuit(num_qubits)
    circuit_builder(qc)
    
    # print("Circuit:")
    # print(qc.draw(output='text'))
    
    rho0 = create_all_zero_density_matrix(num_qubits)
    
    # Run your custom simulator
    rho_final = evolve_noisy_circuit(rho0, circuit_to_unitary_list(qc), None)
    z_vals_custom = z_expectations(rho_final, num_qubits)

    difference = np.abs(z_vals_custom - np.flip(expected_z_qiskit))

    if np.max(difference) < 1e-6:
        print(f"{test_name} Passed: Max difference = {np.max(difference)}")
    else: 
        print(f"{test_name} Failed: Max difference = {np.max(difference)}")

    
    
    print(f"Custom Simulator Z Expectations: {z_vals_custom}")
    print(f"Qiskit Aer Expected Z Expectations: {np.flip(expected_z_qiskit)}")





if __name__ == "__main__":
    # Test Cases

    # 1. Single Qubit - H gate
    n1 = 1
    qc1 = QuantumCircuit(n1)
    qc1.h(0)
    expected_z1 = get_qiskit_z_expectations(qc1, n1) # Should be ~0 for H|0>
    run_test("1-Qubit H Gate", n1, lambda circ: circ.h(0), expected_z1)

    # 2. Single Qubit - X gate
    n2 = 1
    qc2 = QuantumCircuit(n2)
    qc2.x(0)
    expected_z2 = get_qiskit_z_expectations(qc2,n2) # Should be -1 for X|0> = |1>
    run_test("1-Qubit X Gate", n2, lambda circ: circ.x(0), expected_z2)

    # 3. Two Qubits - Bell State (H CNOT)
    n3 = 2
    qc3 = QuantumCircuit(n3)
    qc3.h(0)
    qc3.cx(0, 1)
    expected_z3 = get_qiskit_z_expectations(qc3,n3) # Should be 0, 0 for Bell state
    run_test("2-Qubit Bell State (H CNOT)", n3, lambda circ: (circ.h(0), circ.cx(0, 1)), expected_z3)

    # 4. Two Qubits - X on Qubit 0, Identity on Qubit 1
    n4 = 2
    qc4 = QuantumCircuit(n4)
    qc4.x(0)
    # No op on qubit 1
    expected_z4 = get_qiskit_z_expectations(qc4,n4) # Should be -1 for Q0, 1 for Q1
    run_test("2-Qubit X on Q0", n4, lambda circ: circ.x(0), expected_z4)

    # 5. Two Qubits - CX with reversed order (1,0) - this tests two_qubit_reverse
    # Note: This test requires `two_qubit_reverse` to correctly handle the matrix
    # AND `evolve_noisy_circuit` to correctly embed it.
    n5 = 2
    qc5 = QuantumCircuit(n5)
    qc5.h(0)
    qc5.cx(1, 0) # control 1, target 0
    expected_z5 = get_qiskit_z_expectations(qc5,n5)
    run_test("2-Qubit CX(1,0) (reversed)", n5, lambda circ: (circ.h(0), circ.cx(1,0)), expected_z5)

    # 6. Three Qubits - GHZ State preparation (H CX CX)
    n6 = 3
    qc6 = QuantumCircuit(n6)
    qc6.h(0)
    qc6.cx(0, 1)
    qc6.cx(1, 2) # This is where the `evolve_noisy_circuit`'s 2-qubit `U` construction for non-adjacent original sites
                # would be problematic if `gate.sites` were [0,2] but here it's [1,2] following [0,1].
                # It relies on the current `U` construction handling sequential adjacent CX gates.
    expected_z6 = get_qiskit_z_expectations(qc6,n6) # Should be 0,0,0 for GHZ state
    run_test("3-Qubit GHZ State", n6, lambda circ: (circ.h(0), circ.cx(0,1), circ.cx(1,2)), expected_z6)

    # 7. Three Qubits - All Z gates
    n7 = 3
    qc7 = QuantumCircuit(n7)
    qc7.z(0)
    qc7.z(1)
    qc7.z(2)
    expected_z7 = get_qiskit_z_expectations(qc7,n7) # Should be 1,1,1 for Z|0>=|0>
    run_test("3-Qubit All Z Gates", n7, lambda circ: (circ.z(0), circ.z(1), circ.z(2)), expected_z7)

    # 8. Custom Circuit from previous run
    n_custom = 2
    qc_custom = QuantumCircuit(n_custom)
    qc_custom.h(0)
    qc_custom.cx(0, 1)
    qc_custom.y(1)
    qc_custom.cx(1,0)
    qc_custom.z(0)
    expected_z_custom = get_qiskit_z_expectations(qc_custom,n_custom)
    run_test("Custom Circuit (Original Example)", n_custom, 
             lambda circ: (circ.h(0), circ.cx(0,1), circ.y(1), circ.cx(1,0), circ.z(0)), 
             expected_z_custom)
    
    # 9. 2-Qubit one X Gate
    n9 = 2
    qc9 = QuantumCircuit(n9)
    qc9.x(0)
    expected_z9 = get_qiskit_z_expectations(qc9,n9) # Should be 1,1,1 for Z|0>=|0>
    run_test("2-Qubit one X Gate", n9, lambda circ: (circ.x(0)), expected_z9)

    
    # ===== NEW COMPLEX TEST CASES =====

    # 10. Three Qubits - Mixed Pauli Gates with Hadamard
    n10 = 3
    qc10 = QuantumCircuit(n10)
    qc10.h(0)
    qc10.x(1)
    qc10.y(2)
    qc10.cx(0, 1)
    qc10.cz(1, 2)
    expected_z10 = get_qiskit_z_expectations(qc10, n10)
    run_test("3-Qubit Mixed Pauli + Entangling", n10, 
             lambda circ: (circ.h(0), circ.x(1), circ.y(2), circ.cx(0,1), circ.cz(1,2)), 
             expected_z10)

    # 11. Four Qubits - Linear Chain Entanglement
    n11 = 4
    qc11 = QuantumCircuit(n11)
    qc11.h(0)
    qc11.cx(0, 1)
    qc11.cx(1, 2)
    qc11.cx(2, 3)
    qc11.rz(np.pi/4, 1)
    qc11.ry(np.pi/3, 2)
    expected_z11 = get_qiskit_z_expectations(qc11, n11)
    run_test("4-Qubit Linear Chain + Rotations", n11, 
             lambda circ: (circ.h(0), circ.cx(0,1), circ.cx(1,2), circ.cx(2,3), 
                          circ.rz(np.pi/4, 1), circ.ry(np.pi/3, 2)), 
             expected_z11)

    # 12. Three Qubits - W State Preparation
    n12 = 3
    qc12 = QuantumCircuit(n12)
    qc12.ry(np.arccos(np.sqrt(2/3)), 0)
    qc12.cz(0, 1)
    qc12.cx(1, 2)
    qc12.x(0)
    expected_z12 = get_qiskit_z_expectations(qc12, n12)
    run_test("3-Qubit W State", n12, 
             lambda circ: (circ.ry(np.arccos(np.sqrt(2/3)), 0), circ.cz(0,1), 
                          circ.cx(1,2), circ.x(0)), 
             expected_z12)

    # 13. Four Qubits - Star Connectivity
    n13 = 4
    qc13 = QuantumCircuit(n13)
    qc13.h(0)  # Center qubit
    qc13.cx(0, 1)
    qc13.cx(1, 2)
    qc13.cx(2, 3)
    qc13.rz(np.pi/6, 0)
    qc13.rx(np.pi/4, 1)
    qc13.ry(np.pi/8, 2)
    expected_z13 = get_qiskit_z_expectations(qc13, n13)
    run_test("4-Qubit Star Topology + Rotations", n13, 
             lambda circ: (circ.h(0), circ.cx(0,1), circ.cx(1,2), circ.cx(2,3), 
                          circ.rz(np.pi/6, 0), circ.rx(np.pi/4, 1), circ.ry(np.pi/8, 2)), 
             expected_z13)

    # 14. Five Qubits - Complex Entangling Circuit
    n14 = 5
    qc14 = QuantumCircuit(n14)
    qc14.h(0)
    qc14.h(2)
    qc14.cx(0, 1)
    qc14.cx(2, 3)
    qc14.cx(3, 4)
    qc14.cz(3, 4)
    qc14.cz(4, 3)
    qc14.h(1)
    qc14.y(3)
    expected_z14 = get_qiskit_z_expectations(qc14, n14)
    run_test("5-Qubit Complex Entangling", n14, 
             lambda circ: (circ.h(0), circ.h(2), circ.cx(0,1), circ.cx(2,3), 
                          circ.cx(3,4), circ.cz(3,4), circ.cz(4,3), circ.h(1), circ.y(3)), 
             expected_z14)

    # 15. Three Qubits - Quantum Fourier Transform
    n15 = 3
    qc15 = QuantumCircuit(n15)
    # Prepare initial state
    qc15.x(0)
    qc15.h(1)
    # Apply QFT
    qc15.h(0)
    qc15.h(1)
    qc15.h(2)
    # Swap qubits
    qc15.swap(1, 2)
    expected_z15 = get_qiskit_z_expectations(qc15, n15)
    run_test("3-Qubit QFT Circuit", n15, 
             lambda circ: (circ.x(0), circ.h(1), circ.h(0), circ.h(1), 
                          circ.h(2), circ.swap(1, 2)), 
             expected_z15)

    # 16. Four Qubits - Toffoli Chain
    n16 = 4
    qc16 = QuantumCircuit(n16)
    qc16.h(0)
    qc16.h(1)

    qc16.rz(np.pi/3, 2)
    expected_z16 = get_qiskit_z_expectations(qc16, n16)
    run_test("4-Qubit Toffoli Chain", n16, 
             lambda circ: (circ.h(0), circ.h(1),  circ.rz(np.pi/3, 2)), 
             expected_z16)

    # 17. Six Qubits - Ring Topology with Barriers
    n17 = 6
    qc17 = QuantumCircuit(n17)
    # Create ring of entanglement
    for i in range(6):
        qc17.h(i)
    for i in range(5):
        qc17.cx(i, (i+1))
    # Add some single qubit rotations
    qc17.rz(np.pi/6, 0)
    qc17.ry(np.pi/4, 2)
    qc17.rx(np.pi/8, 4)
    # More entangling
    qc17.cx(2, 3)
    qc17.cx(3, 4)
    qc17.cx(4, 5)
    expected_z17 = get_qiskit_z_expectations(qc17, n17)
    run_test("6-Qubit Ring Topology", n17, 
             lambda circ: (
                 [circ.h(i) for i in range(5)] + 
                 [circ.cx(i, (i+1)) for i in range(5)] +
                 [circ.rz(np.pi/6, 0), circ.ry(np.pi/4, 2), circ.rx(np.pi/8, 4)] +
                 [circ.cx(2, 3), circ.cx(3, 4), circ.cx(4, 5)]
             ), 
             expected_z17)

    # 18. Five Qubits - Random Quantum Circuit
    n18 = 5
    qc18 = QuantumCircuit(n18)
    # Layer 1
    qc18.ry(np.pi/7, 0)
    qc18.rz(np.pi/5, 1)
    qc18.rx(np.pi/3, 2)
    qc18.h(3)
    qc18.h(4)
    # Layer 2 - entangling
    qc18.cx(1, 2)
    qc18.cz(2, 3)
    qc18.cz(3, 4)
    # Layer 3
    qc18.y(0)
    qc18.ry(np.pi/4, 1)
    qc18.rz(np.pi/6, 3)
    # Layer 4 - more entangling
    qc18.cx(3, 2)

    expected_z18 = get_qiskit_z_expectations(qc18, n18)
    run_test("5-Qubit Random Circuit", n18, 
             lambda circ: (
                 circ.ry(np.pi/7, 0), circ.rz(np.pi/5, 1), circ.rx(np.pi/3, 2), 
                 circ.h(3), circ.h(4), circ.cx(1, 2), circ.cz(2, 3), circ.cz(3, 4),
                 circ.y(0), circ.ry(np.pi/4, 1), circ.rz(np.pi/6, 3),
                 circ.cx(3, 2)
             ), 
             expected_z18)

    # 19. Six Qubits - Quantum Approximate Optimization Algorithm (QAOA) inspired
    n19 = 6
    qc19 = QuantumCircuit(n19)
    # Initial state preparation
    for i in range(6):
        qc19.h(i)
    # Problem Hamiltonian layer
    gamma = np.pi/8
    for i in range(5):
        qc19.cx(i, i+1)
        qc19.rz(2*gamma, i+1)
        qc19.cx(i, i+1)
    # Mixer Hamiltonian layer
    beta = np.pi/6
    for i in range(6):
        qc19.rx(2*beta, i)
    # Second layer
    for i in [0, 2, 4]:
        if i+1 < 6:
            qc19.cx(i, i+1)
            qc19.rz(2*gamma, i+1)
            qc19.cx(i, i+1)
    expected_z19 = get_qiskit_z_expectations(qc19, n19)
    run_test("6-Qubit QAOA-inspired", n19, 
             lambda circ: (
                 [circ.h(i) for i in range(6)] +
                 [(circ.cx(i, i+1), circ.rz(2*np.pi/8, i+1), circ.cx(i, i+1)) for i in range(5)] +
                 [circ.rx(2*np.pi/6, i) for i in range(6)] +
                 [(circ.cx(i, i+1), circ.rz(2*np.pi/8, i+1), circ.cx(i, i+1)) for i in [0, 2, 4] if i+1 < 6]
             ), 
             expected_z19)

    print("\n--- All tests completed ---")