"""
Debug test file to analyze differences between Qiskit and Kraus channel simulation.
Provides detailed analysis of noise application and intermediate results.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer

from .noisy_QC_simulator_qiskit import simulate_noisy_layers_with_estimator
from .krauschannel_simulation import (
    create_all_zero_density_matrix, 
    evolve_noisy_circuit, 
    circuit_to_unitary_list, 
    z_expectations
)
from .qiskit_noisemodels import qiskit_dephasing_noise
from mqt.yaqs.core.libraries.noise_library import NoiseLibrary


class KrausCompatibleNoiseModel:
    """A noise model compatible with the KrausChannel function."""
    
    def __init__(self, num_qubits: int, noise_strengths: list):
        self.num_qubits = num_qubits
        single_qubit_strength = noise_strengths[0]
        pair_qubit_strength = noise_strengths[1] if len(noise_strengths) > 1 else single_qubit_strength
        
        # Create site-indexed structure expected by KrausChannel
        self.processes = [[] for _ in range(num_qubits)]
        self.strengths = [[] for _ in range(num_qubits)]
        
        # Add single qubit dephasing
        for qubit in range(num_qubits):
            self.processes[qubit].append("dephasing")
            self.strengths[qubit].append(single_qubit_strength)


def debug_noise_application():
    """Debug the noise application process step by step."""
    print("="*60)
    print("DEBUGGING NOISE APPLICATION")
    print("="*60)
    
    # Simple test case: single qubit H gate
    num_qubits = 1
    noise_strength = 0.1
    
    # Create circuit
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    
    print(f"Circuit: {qc}")
    print(f"Number of qubits: {num_qubits}")
    print(f"Noise strength: {noise_strength}")
    
    # Create noise models
    qiskit_noise_model = qiskit_dephasing_noise(num_qubits, [noise_strength])
    kraus_noise_model = KrausCompatibleNoiseModel(num_qubits, [noise_strength])
    
    print(f"\nQiskit noise model: {qiskit_noise_model}")
    print(f"Kraus noise model processes: {kraus_noise_model.processes}")
    print(f"Kraus noise model strengths: {kraus_noise_model.strengths}")
    
    # Test Qiskit simulation
    print("\n" + "-"*40)
    print("QISKIT SIMULATION")
    print("-"*40)
    
    qiskit_results = simulate_noisy_layers_with_estimator(qc, qiskit_noise_model, 2)
    print(f"Qiskit results: {qiskit_results}")
    
    # Test Kraus simulation step by step
    print("\n" + "-"*40)
    print("KRAUS SIMULATION STEP BY STEP")
    print("-"*40)
    
    rho0 = create_all_zero_density_matrix(num_qubits)
    print(f"Initial density matrix:\n{rho0}")
    
    gate_list = circuit_to_unitary_list(qc)
    print(f"Gate list: {gate_list}")
    
    # Apply first layer
    rho_after_first = evolve_noisy_circuit(rho0, gate_list, kraus_noise_model)
    print(f"Density matrix after first layer:\n{rho_after_first}")
    z_first = z_expectations(rho_after_first, num_qubits)
    print(f"Z expectation after first layer: {z_first}")
    
    # Apply second layer (compose circuit with itself)
    cumulative_circuit = qc.compose(qc)
    gate_list_cumulative = circuit_to_unitary_list(cumulative_circuit)
    print(f"Cumulative gate list: {gate_list_cumulative}")
    
    rho_after_second = evolve_noisy_circuit(rho0, gate_list_cumulative, kraus_noise_model)
    print(f"Density matrix after second layer:\n{rho_after_second}")
    z_second = z_expectations(rho_after_second, num_qubits)
    print(f"Z expectation after second layer: {z_second}")
    
    # Compare results
    print("\n" + "-"*40)
    print("COMPARISON")
    print("-"*40)
    print(f"Qiskit results: {qiskit_results}")
    print(f"Kraus results: [{z_first[0]}, {z_second[0]}]")
    print(f"Differences: {np.abs(qiskit_results.flatten() - np.array([z_first[0], z_second[0]]))}")


def debug_noiseless_comparison():
    """Compare noiseless simulations to establish baseline."""
    print("\n" + "="*60)
    print("NOISELESS COMPARISON")
    print("="*60)
    
    # Simple test case: single qubit H gate, no noise
    num_qubits = 1
    
    # Create circuit
    qc = QuantumCircuit(num_qubits)
    qc.h(0)
    
    print(f"Circuit: {qc}")
    
    # Test Qiskit simulation (no noise)
    print("\n" + "-"*40)
    print("QISKIT NOISELESS SIMULATION")
    print("-"*40)
    
    # Create empty noise model for Qiskit
    from qiskit_aer.noise import NoiseModel
    empty_noise_model = NoiseModel()
    
    qiskit_results = simulate_noisy_layers_with_estimator(qc, empty_noise_model, 2)
    print(f"Qiskit noiseless results: {qiskit_results}")
    
    # Test Kraus simulation (no noise)
    print("\n" + "-"*40)
    print("KRAUS NOISELESS SIMULATION")
    print("-"*40)
    
    rho0 = create_all_zero_density_matrix(num_qubits)
    gate_list = circuit_to_unitary_list(qc)
    
    # First layer
    rho_after_first = evolve_noisy_circuit(rho0, gate_list, None)
    z_first = z_expectations(rho_after_first, num_qubits)
    
    # Second layer
    cumulative_circuit = qc.compose(qc)
    gate_list_cumulative = circuit_to_unitary_list(cumulative_circuit)
    rho_after_second = evolve_noisy_circuit(rho0, gate_list_cumulative, None)
    z_second = z_expectations(rho_after_second, num_qubits)
    
    print(f"Kraus noiseless results: [{z_first[0]}, {z_second[0]}]")
    
    # Compare
    print("\n" + "-"*40)
    print("NOISELESS COMPARISON")
    print("-"*40)
    print(f"Qiskit noiseless: {qiskit_results.flatten()}")
    print(f"Kraus noiseless: [{z_first[0]}, {z_second[0]}]")
    print(f"Differences: {np.abs(qiskit_results.flatten() - np.array([z_first[0], z_second[0]]))}")


def debug_dephasing_operator():
    """Debug the dephasing operator implementation."""
    print("\n" + "="*60)
    print("DEPHASING OPERATOR DEBUG")
    print("="*60)
    
    # Check YAQS dephasing operator
    yaqs_dephasing = NoiseLibrary.dephasing().matrix
    print(f"YAQS dephasing operator:\n{yaqs_dephasing}")
    
    # Check Qiskit dephasing (Z operator)
    qiskit_z = np.array([[1, 0], [0, -1]])
    print(f"Qiskit Z operator (dephasing):\n{qiskit_z}")
    
    # Compare
    print(f"Are they the same? {np.allclose(yaqs_dephasing, qiskit_z)}")
    
    # Test applying dephasing to |+> state
    plus_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    plus_dm = np.outer(plus_state, plus_state.conj())
    print(f"|+><+| density matrix:\n{plus_dm}")
    
    # Apply YAQS dephasing
    yaqs_result = yaqs_dephasing @ plus_dm @ yaqs_dephasing.conj().T
    print(f"After YAQS dephasing:\n{yaqs_result}")
    
    # Apply Qiskit dephasing
    qiskit_result = qiskit_z @ plus_dm @ qiskit_z.conj().T
    print(f"After Qiskit dephasing:\n{qiskit_result}")
    
    print(f"Results match? {np.allclose(yaqs_result, qiskit_result)}")


if __name__ == "__main__":
    print("Starting debug analysis of noisy quantum circuit simulation...")
    
    debug_dephasing_operator()
    debug_noiseless_comparison()
    debug_noise_application()
    
    print("\n" + "="*60)
    print("DEBUG ANALYSIS COMPLETED")
    print("="*60) 