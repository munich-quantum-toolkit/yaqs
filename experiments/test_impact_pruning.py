import numpy as np
import time
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_noisy_state_preparation_batch
)
from mqt.yaqs.optimization.parameterized_circuit import (
    ParameterizedCircuit,
    ParameterizedGate,
    create_brickwall_matrix_product_disentangler_parameterized_circuit,
)

def get_disordered_tfim_ground_state(num_qubits: int, seed: int):
    rng = np.random.RandomState(seed)
    dim = 2**num_qubits
    H = np.zeros((dim, dim), dtype=np.complex128)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    
    J = rng.uniform(0.8, 1.2, size=num_qubits - 1)
    h = rng.uniform(0.8, 1.2, size=num_qubits)
    
    for i in range(num_qubits - 1):
        op = [I]*num_qubits
        op[i] = Z
        op[i+1] = Z
        term = op[0]
        for j in range(1, num_qubits):
            term = np.kron(term, op[j])
        H -= J[i] * term
        
    for i in range(num_qubits):
        op = [I]*num_qubits
        op[i] = X
        term = op[0]
        for j in range(1, num_qubits):
            term = np.kron(term, op[j])
        H -= h[i] * term
        
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvectors[:, 0]

def evaluate_fidelity(circuit, theta, statevec):
    noise_none = NoiseModel([])
    tjm_none = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    options = KrotovOptions(max_iterations=0)
    res = train_krotov_noisy_state_preparation_batch(
        circuit, statevec, noise_none, tjm_none, initial_theta=theta, options=options
    )
    return res.trace['fidelity'][-1]

def impact_aware_prune(circuit: ParameterizedCircuit, theta: np.ndarray, statevec: np.ndarray, keep_k: int):
    print("Computing gradients via Parameter Shift Rule...", flush=True)
    start = time.time()
    gradients = np.zeros(circuit.num_params)
    base_fidelity = evaluate_fidelity(circuit, theta, statevec)
    
    for gate in circuit.gates:
        if gate.is_trainable:
            i = gate.param_index
            if gradients[i] != 0: continue # Already computed
            
            theta_plus = theta.copy()
            theta_plus[i] += np.pi / 2
            fid_plus = evaluate_fidelity(circuit, theta_plus, statevec)
            
            theta_minus = theta.copy()
            theta_minus[i] -= np.pi / 2
            fid_minus = evaluate_fidelity(circuit, theta_minus, statevec)
            
            gradients[i] = 0.5 * (fid_plus - fid_minus)
    print(f"Computed {circuit.num_params} gradients in {time.time()-start:.1f}s", flush=True)

    impacts = []
    for gate in circuit.gates:
        if gate.is_trainable:
            i = gate.param_index
            # Taylor-expanded impact: |theta * gradient|
            impact = abs(theta[i] * gradients[i])
            impacts.append((impact, i))
            
    impacts.sort(reverse=True, key=lambda x: x[0])
    keep_indices = set([x[1] for x in impacts[:keep_k]])

    new_gates = []
    new_theta = []
    old_to_new_index = {}
    for gate in circuit.gates:
        if gate.is_trainable:
            if gate.param_index not in keep_indices:
                continue
            if gate.param_index not in old_to_new_index:
                old_to_new_index[gate.param_index] = len(new_theta)
                new_theta.append(theta[gate.param_index])
            new_gate = ParameterizedGate(
                name=gate.name, sites=gate.sites, param_index=old_to_new_index[gate.param_index],
                angle_scale=gate.angle_scale, angle_offset=gate.angle_offset,
                data_map=gate.data_map, fixed_params=gate.fixed_params
            )
            new_gates.append(new_gate)
        else:
            new_gates.append(gate)
    pruned_circuit = ParameterizedCircuit(
        num_qubits=circuit.num_qubits, gates=new_gates, num_params=len(new_theta)
    )
    return pruned_circuit, np.array(new_theta, dtype=np.float64)

if __name__ == "__main__":
    num_qubits = 8
    seed = 400
    statevec = get_disordered_tfim_ground_state(num_qubits, seed=seed)
    circuit = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, 4, initial_single_qubit_layer=True)
    rng = np.random.RandomState(42)
    theta = rng.randn(circuit.num_params) * 0.1
    print(f"Initial params: {circuit.num_params}")
    pruned_circuit, new_theta = impact_aware_prune(circuit, theta, statevec, 50)
    print(f"Pruned params: {pruned_circuit.num_params}")
