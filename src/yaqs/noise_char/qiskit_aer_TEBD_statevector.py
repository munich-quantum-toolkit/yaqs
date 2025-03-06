import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import qiskit.quantum_info as qi



'''This file contains 

-TEBD simulation of quantum circuits (including hard/soft truncation)

-statevector simulation of quantum circuits

both methods are implemented using qiskit-aer simulator.'''

def expectation_value(rho, op):
    """Compute the expectation value of observable op given a density matrix rho."""
    return np.real(np.trace(rho.data @ op))





def TEBD_evolve(initial_state, circuit, observables, max_bond=8, threshold=1e-10):
    """
    Evolves an initial state through a circuit using TEBD and snapshots.
    
    Parameters:
      initial_state (array-like): The statevector for the initial state.
      circuit (QuantumCircuit): A QuantumCircuit object (without snapshots).
      observables (dict): A dictionary of observables (name: 2x2 numpy array).
      
      
    Returns:
      snapshot_labels (list): List of snapshot labels corresponding to time steps.
      expectations (dict): Dictionary mapping snapshot labels to a dictionary of
                           qubit expectation values for each observable.
                           Format: { snapshot_label: { qubit_index: {obs_name: value, ...}, ... }, ... }
    """
    n = circuit.num_qubits
    # Create a new circuit that will include initialization and snapshots.
    evolved_circuit = QuantumCircuit(n)
    snapshot_labels = []
    
    # If an initial state is provided, initialize the qubits.
    if initial_state is not None:
        evolved_circuit.initialize(initial_state, range(n))
    # Snapshot the initial state.
    label = "t0"
    evolved_circuit.save_statevector(label=label)
    snapshot_labels.append(label)
    
    # Append every instruction from the input circuit and insert a snapshot after each.
    for i, (instr, qargs, cargs) in enumerate(circuit.data):
        evolved_circuit.append(instr, qargs, cargs)
        label = f"t{i+1}"
        evolved_circuit.save_statevector(label=label)
        snapshot_labels.append(label)
    
    # Transpile and simulate the circuit with the desired backend.
    simulator = AerSimulator(method='matrix_product_state', matrix_product_state_max_bond_dimension = max_bond, matrix_product_state_truncation_threshold = threshold)
    evolved_circuit = transpile(evolved_circuit, simulator)
    result = simulator.run(evolved_circuit).result()
    states_data = result.data(0)  # Dictionary mapping snapshot labels to statevectors.
    
    # Compute expectation values.
    expectations = {}
    for label in snapshot_labels:
        state = states_data[label]
        rho = qi.DensityMatrix(state)
        expectations[label] = {}
        for qubit in range(n):
            # Trace out all qubits except the current one.
            traced_out = [i for i in range(n) if i != qubit]
            rho_reduced = qi.partial_trace(rho, traced_out)
            expectations[label][qubit] = {}
            for obs_name, op in observables.items():
                expectations[label][qubit][obs_name] = expectation_value(rho_reduced, op)
    return snapshot_labels, expectations





def statevector_evolve(initial_state, circuit, observables):
    """
    Evolves an initial state through a circuit using TEBD and snapshots.
    
    Parameters:
      initial_state (array-like): The statevector for the initial state.
      circuit (QuantumCircuit): A QuantumCircuit object (without snapshots).
      observables (dict): A dictionary of observables (name: 2x2 numpy array).
      simulator_method (str): The simulation method; default is 'matrix_product_state'.
      
    Returns:
      snapshot_labels (list): List of snapshot labels corresponding to time steps.
      expectations (dict): Dictionary mapping snapshot labels to a dictionary of
                           qubit expectation values for each observable.
                           Format: { snapshot_label: { qubit_index: {obs_name: value, ...}, ... }, ... }
    """
    n = circuit.num_qubits
    # Create a new circuit that will include initialization and snapshots.
    evolved_circuit = QuantumCircuit(n)
    snapshot_labels = []
    
    # If an initial state is provided, initialize the qubits.
    if initial_state is not None:
        evolved_circuit.initialize(initial_state, range(n))
    # Snapshot the initial state.
    label = "t0"
    evolved_circuit.save_statevector(label=label)
    snapshot_labels.append(label)
    
    # Append every instruction from the input circuit and insert a snapshot after each.
    for i, (instr, qargs, cargs) in enumerate(circuit.data):
        evolved_circuit.append(instr, qargs, cargs)
        label = f"t{i+1}"
        evolved_circuit.save_statevector(label=label)
        snapshot_labels.append(label)
    
    # Transpile and simulate the circuit with the desired backend.
    simulator = AerSimulator(method='statevector')
    evolved_circuit = transpile(evolved_circuit, simulator)
    result = simulator.run(evolved_circuit).result()
    states_data = result.data(0)  # Dictionary mapping snapshot labels to statevectors.
    
    # Compute expectation values.
    expectations = {}
    for label in snapshot_labels:
        state = states_data[label]
        rho = qi.DensityMatrix(state)
        expectations[label] = {}
        for qubit in range(n):
            # Trace out all qubits except the current one.
            traced_out = [i for i in range(n) if i != qubit]
            rho_reduced = qi.partial_trace(rho, traced_out)
            expectations[label][qubit] = {}
            for obs_name, op in observables.items():
                expectations[label][qubit][obs_name] = expectation_value(rho_reduced, op)
    return snapshot_labels, expectations

if __name__ == '__main__':
    # Define Pauli matrices as observables.
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    observables = {'X': X, 'Y': Y, 'Z': Z}

    # Define the initial state for 2 qubits: |00>
    initial_state = [1, 0, 0, 0]

    # Create an example circuit (without measurements).
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.ry(0.5, 0)
    circuit.rx(0.7, 1)
    circuit.cz(0, 1)

    # Evolve the circuit using TEBD_evolve with the MPS backend.
    snapshot_labels, expectations = TEBD_evolve(initial_state, circuit, observables, 4)

    # Print the computed expectation values per timestep.
    print("Expectation values per timestep:")
    for label in snapshot_labels:
        print(f"Time step '{label}':")
        for qubit in expectations[label]:
            print(f"  Qubit {qubit}: {expectations[label][qubit]}")
    
    # Plot the expectation values.
    steps = np.arange(len(snapshot_labels))
    plt.figure(figsize=(10, 6))
    num_qubits = circuit.num_qubits
    for qubit in range(num_qubits):
        for obs_name in observables.keys():
            values = [expectations[label][qubit][obs_name] for label in snapshot_labels]
            plt.plot(steps, values, marker='o', label=f"Qubit {qubit} {obs_name}")
    plt.xticks(steps, snapshot_labels)
    plt.xlabel("Time step")
    plt.ylabel("Expectation value")
    plt.title("Expectation values of observables for all qubits (MPS Backend)")
    plt.legend()
    plt.tight_layout()
    plt.show()

