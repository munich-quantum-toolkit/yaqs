"""
Test file to compare Qiskit noisy quantum circuit simulator with Kraus channel simulator.
Uses dephasing noise models to validate both approaches give consistent results.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import Aer



from mqt.yaqs.noisy_qc_sim.qiskit_noisy_sim import qiskit_noisy_simulator
from mqt.yaqs.noisy_qc_sim.densitymatrix_sim import (
    create_all_zero_density_matrix, 
    evolve_noisy_circuit, 
    circuit_to_unitary_list, 
    z_expectations,
    two_qubit_reverse
)
from mqt.yaqs.noisy_qc_sim.qiskit_noisemodels import qiskit_dephasing_noise
from mqt.yaqs.core.data_structures.noise_model import NoiseModel


def create_yaqs_dephasing_noise(num_qubits: int, noise_strengths: list) -> NoiseModel:
    """Create a YAQS noise model with dephasing noise for single qubits and qubit pairs."""
    single_qubit_strength = noise_strengths[0]
    pair_qubit_strength = noise_strengths[1] if len(noise_strengths) > 1 else single_qubit_strength
    
    processes = []
    
    # Single qubit dephasing
    for qubit in range(num_qubits):
        processes.append({
            "name": "dephasing",
            "sites": [qubit],
            "strength": single_qubit_strength
        })
    
    # Two qubit ZZ dephasing (equivalent to double_dephasing in NoiseLibrary)
    for qubit in range(num_qubits - 1):
        processes.append({
            "name": "double_dephasing",
            "sites": [qubit, qubit + 1],
            "strength": pair_qubit_strength
        })
    
    return NoiseModel(processes)


def run_noisy_comparison_test(test_name: str, num_qubits: int, circuit_builder, 
                             noise_strengths: list, num_layers: int = 1, tolerance: float = 1e-3):
    """Run a comparison test between Qiskit and Kraus channel simulators."""
    print(f"\n--- Running Noisy Test: {test_name} ({num_qubits} qubits) ---")
    print(f"Noise strengths: {noise_strengths}")
    
    # Create the base circuit
    qc = QuantumCircuit(num_qubits)
    circuit_builder(qc)
    
    print("Circuit:")
    print(qc.draw(output='text'))
    
    # Create noise models
    qiskit_noise_model = qiskit_dephasing_noise(num_qubits, noise_strengths)
    yaqs_noise_model = create_yaqs_dephasing_noise(num_qubits, noise_strengths)
    
    # Test Qiskit noiseless first
    print("Running Qiskit noiseless simulation...")
    qiskit_noiseless_results = qiskit_noisy_simulator(qc, None, num_qubits, num_layers)
    
    # Run Qiskit simulation with noise
    print("Running Qiskit simulation with noise...")
    qiskit_results = qiskit_noisy_simulator(qc, qiskit_noise_model, num_qubits, num_layers)
    
    # Run Kraus channel simulation
    print("Running Kraus channel simulation...")
    rho0 = create_all_zero_density_matrix(num_qubits)
    print('rho shape', rho0.shape)
    gate_list = circuit_to_unitary_list(qc)
    
    kraus_results = evolve_noisy_circuit(rho0, gate_list, yaqs_noise_model, num_layers)
    
    # Compare results
    print(f"\nResults comparison:")
    print(f"Qiskit noiseless: {qiskit_noiseless_results}")
    print(f"Qiskit with noise: {qiskit_results}")
    print(f"Kraus with noise: {kraus_results}")
    
    # Check if noise is actually being applied
    qiskit_noise_diff = np.abs(np.array(qiskit_results) - np.array(qiskit_noiseless_results))
    print(f"Qiskit noise effect: {qiskit_noise_diff}")
    
    # Compare Qiskit vs Kraus
    difference = np.abs(np.flip(qiskit_results) - kraus_results)
    max_diff = np.max(difference)
    
    print(f"Qiskit vs Kraus max difference: {max_diff:.6f}")
    print(f"Tolerance: {tolerance}")
    
    if max_diff < tolerance:
        print(f"✅ {test_name} PASSED: Max difference = {max_diff:.6f}")
        return True
    else:
        print(f"❌ {test_name} FAILED: Max difference = {max_diff:.6f}")
        print("\nDetailed comparison:")
        print(f"  Qiskit: {qiskit_results}")
        print(f"  Kraus:  {kraus_results}")
        print(f"  Diff:   {difference}")
        return False


def test_single_qubit_noise():
    """Test single qubit circuits with dephasing noise."""
    print("\n" + "="*60)
    print("SINGLE QUBIT NOISE TESTS")
    print("="*60)
    
    # Test 1: Single qubit H gate with dephasing
    run_noisy_comparison_test(
        "1-Qubit H Gate with Dephasing",
        num_qubits=1,
        circuit_builder=lambda circ: circ.h(0),
        noise_strengths=[0.1],
        num_layers=1
    )
    
    # Test 2: Single qubit X gate with dephasing
    run_noisy_comparison_test(
        "1-Qubit X Gate with Dephasing",
        num_qubits=1,
        circuit_builder=lambda circ: circ.x(0),
        noise_strengths=[0.05],
        num_layers=3
    )


def test_two_qubit_noise():
    """Test two qubit circuits with dephasing noise."""
    print("\n" + "="*60)
    print("TWO QUBIT NOISE TESTS")
    print("="*60)
    
    # Test 1: Bell state with dephasing
    def bell_circuit(circ):
        circ.h(0)
        circ.cx(0, 1)
    
    run_noisy_comparison_test(
        "2-Qubit Bell State with Dephasing",
        num_qubits=2,
        circuit_builder=bell_circuit,
        noise_strengths=[0.1, 0.1],  # Single qubit, two qubit
        num_layers=3
    )
    
    # Test 2: Simple two qubit circuit
    def xh_circuit(circ):
        circ.x(0)
        circ.h(1)
    
    run_noisy_comparison_test(
        "2-Qubit X+H with Dephasing",
        num_qubits=2,
        circuit_builder=xh_circuit,
        noise_strengths=[0.08, 0.12],
        num_layers=3
    )


def test_three_qubit_noise():
    """Test three qubit circuits with dephasing noise."""
    print("\n" + "="*60)
    print("THREE QUBIT NOISE TESTS")
    print("="*60)
    
    # Test 1: GHZ state with dephasing
    run_noisy_comparison_test(
        "3-Qubit GHZ State with Dephasing",
        num_qubits=3,
        circuit_builder=lambda circ: (circ.h(0), circ.cx(0, 1), circ.cx(1, 2)),
        noise_strengths=[0.03, 0.06],
        num_layers=3
    )
    
    # Test 2: Mixed circuit
    run_noisy_comparison_test(
        "3-Qubit Mixed Circuit with Dephasing",
        num_qubits=3,
        circuit_builder=lambda circ: (circ.h(0), circ.x(1), circ.cx(0, 1), circ.cx(1, 2)),
        noise_strengths=[0.04, 0.08],
        num_layers=3
    )


def test_varying_noise_strengths():
    """Test with different noise strengths to validate scaling."""
    print("\n" + "="*60)
    print("VARYING NOISE STRENGTH TESTS")
    print("="*60)
    
    # Test with weak noise
    run_noisy_comparison_test(
        "Weak Dephasing Noise",
        num_qubits=2,
        circuit_builder=lambda circ: (circ.h(0), circ.cx(0, 1)),
        noise_strengths=[0.01, 0.02],
        num_layers=3,
        tolerance=1e-4
    )
    
    # Test with strong noise
    run_noisy_comparison_test(
        "Strong Dephasing Noise",
        num_qubits=2,
        circuit_builder=lambda circ: (circ.h(0), circ.cx(0, 1)),
        noise_strengths=[0.2, 0.3],
        num_layers=3,
        tolerance=1e-2  # Relaxed tolerance for strong noise
    )


def test_multiple_layers():
    """Test circuits with multiple layers to see noise accumulation."""
    print("\n" + "="*60)
    print("MULTIPLE LAYERS TESTS")
    print("="*60)
    
    # Test with more layers
    run_noisy_comparison_test(
        "Multiple Layers with Dephasing",
        num_qubits=2,
        circuit_builder=lambda circ: (circ.h(0), circ.cx(0, 1)),
        noise_strengths=[0.05, 0.1],
        num_layers=5,
        tolerance=1e-3
    )


def debug_noiseless_results():
    """Debug the noiseless results to understand the discrepancy."""
    print("\n" + "="*60)
    print("DEBUGGING NOISELESS RESULTS")
    print("="*60)
    
    # Create a simple circuit: X on qubit 0, H on qubit 1
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    
    print("Circuit:")
    print(qc.draw(output='text'))
    
    # Test Qiskit noiseless
    print("\nQiskit noiseless results:")
    qiskit_results = qiskit_noisy_simulator(qc, None, 2, 1)
    print(f"Z expectations: {qiskit_results}")
    
    # Test Kraus noiseless
    print("\nKraus noiseless results:")
    rho0 = create_all_zero_density_matrix(2)
    gate_list = circuit_to_unitary_list(qc)
    print(f"Gate list: {gate_list}")
    
    # Apply gates without noise
    rho = rho0.copy()
    for gate in gate_list:
        # Apply unitary without noise
        if len(gate.sites) == 1:
            U = np.eye(1)
            for i in range(2):
                U = np.kron(U, gate.matrix if i == gate.sites[0] else np.eye(2))
        elif len(gate.sites) == 2:
            # Handle two-qubit gate
            idx0, idx1 = gate.sites[0], gate.sites[1]
            if idx0 > idx1:
                idx0, idx1 = idx1, idx0
                gate.matrix = two_qubit_reverse(gate.matrix)
            U = np.eye(1)
            i = 0
            while i < 2:
                if i == idx0:
                    U = np.kron(U, gate.matrix)
                    i += 2
                else:
                    U = np.kron(U, np.eye(2))
                    i += 1
        
        rho = U @ rho @ U.conj().T
        print(f"After gate {gate.sites}: rho shape {rho.shape}")
    
    kraus_results = z_expectations(rho, 2)
    print(f"Z expectations: {kraus_results}")
    
    # Theoretical expectation
    print("\nTheoretical expectation:")
    print("Starting from |00⟩:")
    print("1. X on qubit 0: |00⟩ → |10⟩")
    print("2. H on qubit 1: |10⟩ → |1⟩ ⊗ (|0⟩ + |1⟩)/√2 = (|10⟩ + |11⟩)/√2")
    print("Z expectations: qubit 0 = -1, qubit 1 = 0")
    print(f"Theoretical: [-1, 0]")
    print(f"Kraus: {kraus_results}")
    print(f"Qiskit: {qiskit_results}")


if __name__ == "__main__":
    print("Starting noisy quantum circuit simulator comparison tests...")
    print("Comparing Qiskit Estimator vs Kraus Channel simulation")
    
    # Debug noiseless results first
    debug_noiseless_results()
    
    # Run all test categories
    test_single_qubit_noise()
    test_two_qubit_noise()
    test_three_qubit_noise()
    test_varying_noise_strengths()
    test_multiple_layers()
    
    # print("\n" + "="*60)
    # print("ALL TESTS COMPLETED")
    # print("="*60) 