import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath("."))

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.optimization.krotov import (
    KrotovOptions,
    KrotovTJMOptions,
    train_krotov_noisy_state_preparation_batch,
    noisy_state_preparation_metrics
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

def prune_circuit_top_k(circuit: ParameterizedCircuit, theta: np.ndarray, keep_k: int):
    gate_angles = []
    for gate in circuit.gates:
        if gate.is_trainable:
            angle = circuit.angle(gate, theta, None)
            angle_mod = angle % (2 * np.pi)
            if angle_mod < 0:
                angle_mod += 2 * np.pi
            dist_to_0 = min(angle_mod, 2 * np.pi - angle_mod)
            gate_angles.append(dist_to_0)
            
    gate_angles.sort(reverse=True)
    if keep_k < len(gate_angles):
        threshold = gate_angles[keep_k - 1]
    else:
        threshold = 0.0

    new_gates = []
    new_theta = []
    old_to_new_index = {}
    for gate in circuit.gates:
        if gate.is_trainable:
            angle = circuit.angle(gate, theta, None)
            angle_mod = angle % (2 * np.pi)
            if angle_mod < 0:
                angle_mod += 2 * np.pi
            dist_to_0 = min(angle_mod, 2 * np.pi - angle_mod)
            if dist_to_0 < threshold:
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

def build_hardware_noise_model(hardware: str, num_qubits: int):
    processes = []
    if hardware == "IBM":
        p_1q = 3e-4 / 3.0  
        p_2q = 3e-3 / 2.0  
        for i in range(num_qubits):
            for pauli in ["pauli_x", "pauli_y", "pauli_z"]:
                processes.append({"name": pauli, "sites": [i], "strength": p_1q})
        for i in range(num_qubits - 1):
            for crosstalk in ["crosstalk_xx", "crosstalk_zz"]:
                processes.append({"name": crosstalk, "sites": [i, i+1], "strength": p_2q})
    return NoiseModel(processes)

def run_rigorous_benchmark():
    num_qubits = 8
    state_seeds = [100, 200, 300, 400, 500]
    
    noise_ibm = build_hardware_noise_model("IBM", num_qubits)
    noise_none = NoiseModel([])
    
    tjm_noiseless = KrotovTJMOptions(num_trajectories=1, use_crn=False)
    tjm_train = KrotovTJMOptions(num_trajectories=3, trajectory_update="cross", use_crn=True)
    
    # NEW RIGOROUS EVALUATION PARAMETERS
    NUM_EVAL_TRAJECTORIES = 500
    tjm_test = KrotovTJMOptions(num_trajectories=NUM_EVAL_TRAJECTORIES, use_crn=False)
    
    results = []
    
    for state_idx, seed in enumerate(state_seeds):
        print(f"\n==================================================")
        print(f"STATE {state_idx+1}/5: Disordered TFIM (Seed={seed})")
        print(f"==================================================", flush=True)
        
        statevec = get_disordered_tfim_ground_state(num_qubits, seed=seed)
        
        # -------------------------------------------------------------
        # METHOD 1: STANDARD VQA (Depth 4)
        # -------------------------------------------------------------
        print("\n>>> Method 1: Standard VQA (Depth 4, 276 Params) <<<", flush=True)
        circuit_vqa = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, 4, initial_single_qubit_layer=True)
        rng = np.random.RandomState(seed * 10)
        theta_vqa = rng.randn(circuit_vqa.num_params) * 0.1
        
        options_vqa = KrotovOptions(max_iterations=300, batch_step_size=0.2, batch_schedule="exp", batch_decay=0.01, seed=seed)
        res_vqa = train_krotov_noisy_state_preparation_batch(
            circuit_vqa, statevec, noise_ibm, tjm_train, initial_theta=theta_vqa, options=options_vqa
        )
        final_theta_vqa = res_vqa.theta
        
        print(f"  [Eval] Calculating True Hardware Fidelity for VQA...", flush=True)
        _, fid_vqa_noisy, _ = noisy_state_preparation_metrics(circuit_vqa, final_theta_vqa, statevec, noise_ibm, tjm_test)
        _, fid_vqa_noiseless, _ = noisy_state_preparation_metrics(circuit_vqa, final_theta_vqa, statevec, noise_none, tjm_noiseless)
        print(f"    Train: {res_vqa.trace['fidelity'][-1]:.4f} | Test (N={NUM_EVAL_TRAJECTORIES}): {fid_vqa_noisy:.4f} | Noiseless: {fid_vqa_noiseless:.4f}", flush=True)
        
        
        # -------------------------------------------------------------
        # METHOD 2: BOTTOM-UP ADAPT
        # -------------------------------------------------------------
        print("\n>>> Method 2: Bottom-Up ADAPT (Growth to Depth 4) <<<", flush=True)
        depths = [1, 2, 3, 4]
        current_theta_adapt = None
        
        for d in depths:
            circuit_adapt = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, d, initial_single_qubit_layer=True)
            if current_theta_adapt is None:
                rng = np.random.RandomState(seed * 20)
                initial_theta = rng.randn(circuit_adapt.num_params) * 0.05
            else:
                initial_theta = np.zeros(circuit_adapt.num_params)
                initial_theta[:len(current_theta_adapt)] = current_theta_adapt
                rng = np.random.RandomState(seed * 20 + d)
                initial_theta[len(current_theta_adapt):] = rng.randn(circuit_adapt.num_params - len(current_theta_adapt)) * 0.001
                
            options_adapt = KrotovOptions(max_iterations=100, batch_step_size=1.0, batch_schedule="constant", seed=seed*30)
            res_adapt_step = train_krotov_noisy_state_preparation_batch(
                circuit_adapt, statevec, noise_none, tjm_noiseless, initial_theta=initial_theta, options=options_adapt
            )
            current_theta_adapt = res_adapt_step.theta
            
        # Fine-tune ADAPT under noise
        options_adapt_finetune = KrotovOptions(max_iterations=200, batch_step_size=0.2, batch_schedule="exp", batch_decay=0.01, seed=seed*40)
        res_adapt_finetune = train_krotov_noisy_state_preparation_batch(
            circuit_adapt, statevec, noise_ibm, tjm_train, initial_theta=current_theta_adapt, options=options_adapt_finetune
        )
        final_theta_adapt = res_adapt_finetune.theta
        
        print(f"  [Eval] Calculating True Hardware Fidelity for ADAPT...", flush=True)
        _, fid_adapt_noisy, _ = noisy_state_preparation_metrics(circuit_adapt, final_theta_adapt, statevec, noise_ibm, tjm_test)
        _, fid_adapt_noiseless, _ = noisy_state_preparation_metrics(circuit_adapt, final_theta_adapt, statevec, noise_none, tjm_noiseless)
        print(f"    Train: {res_adapt_finetune.trace['fidelity'][-1]:.4f} | Test (N={NUM_EVAL_TRAJECTORIES}): {fid_adapt_noisy:.4f} | Noiseless: {fid_adapt_noiseless:.4f}", flush=True)
        
        
        # -------------------------------------------------------------
        # METHOD 3: OUR TOP-DOWN PRUNING KROTOV
        # -------------------------------------------------------------
        print("\n>>> Method 3: Top-Down Pruning Krotov (Depth 16 -> 150 params) <<<", flush=True)
        noise_strength_pre = 0.0005
        processes_pre = [{"name": "pauli_x", "sites": [i], "strength": noise_strength_pre} for i in range(num_qubits)]
        noise_pretrain = NoiseModel(processes_pre)
        tjm_pretrain = KrotovTJMOptions(num_trajectories=1, trajectory_update="cross", use_crn=True)
        
        pretrain_depths = [4, 8, 12, 16]
        step_sizes = [10.0, 2.0, 0.5, 0.1]
        current_theta_td = None
        final_pretrain_circuit = None
        
        for i, d in enumerate(pretrain_depths):
            circuit_td = create_brickwall_matrix_product_disentangler_parameterized_circuit(num_qubits, d, initial_single_qubit_layer=True)
            final_pretrain_circuit = circuit_td
            
            if current_theta_td is None:
                rng = np.random.RandomState(seed * 50)
                initial_theta = rng.randn(circuit_td.num_params) * 0.05
            else:
                initial_theta = np.zeros(circuit_td.num_params)
                initial_theta[:len(current_theta_td)] = current_theta_td
                rng = np.random.RandomState(seed * 50 + d)
                initial_theta[len(current_theta_td):] = rng.randn(circuit_td.num_params - len(current_theta_td)) * 0.001
                
            options_td_pre = KrotovOptions(
                max_iterations=250, batch_step_size=step_sizes[i], batch_schedule="exp", batch_decay=0.01, seed=seed*60+i
            )
            res_td_pre = train_krotov_noisy_state_preparation_batch(
                circuit_td, statevec, noise_pretrain, tjm_pretrain, initial_theta=initial_theta, options=options_td_pre
            )
            current_theta_td = res_td_pre.theta
            
        # Pruning
        target_k = 150
        pruned_circuit, theta_pruned = prune_circuit_top_k(final_pretrain_circuit, current_theta_td, keep_k=target_k)
        
        # Fine-tuning
        options_td_finetune = KrotovOptions(
            max_iterations=300, batch_step_size=0.2, batch_schedule="exp", batch_decay=0.01, seed=seed*70
        )
        res_td_finetune = train_krotov_noisy_state_preparation_batch(
            pruned_circuit, statevec, noise_ibm, tjm_train, initial_theta=theta_pruned, options=options_td_finetune
        )
        final_theta_td = res_td_finetune.theta
        
        print(f"  [Eval] Calculating True Hardware Fidelity for Top-Down...", flush=True)
        _, fid_td_noisy, _ = noisy_state_preparation_metrics(pruned_circuit, final_theta_td, statevec, noise_ibm, tjm_test)
        _, fid_td_noiseless, _ = noisy_state_preparation_metrics(pruned_circuit, final_theta_td, statevec, noise_none, tjm_noiseless)
        print(f"    Train: {res_td_finetune.trace['fidelity'][-1]:.4f} | Test (N={NUM_EVAL_TRAJECTORIES}): {fid_td_noisy:.4f} | Noiseless: {fid_td_noiseless:.4f}", flush=True)
        
        # -------------------------------------------------------------
        # SAVE RESULTS
        # -------------------------------------------------------------
        results.append({
            "State_Seed": seed,
            "Method": "Standard VQA",
            "Train_Fidelity": res_vqa.trace['fidelity'][-1],
            "Noiseless_Fidelity": fid_vqa_noiseless,
            "True_Hardware_Fidelity": fid_vqa_noisy
        })
        results.append({
            "State_Seed": seed,
            "Method": "ADAPT-VQE",
            "Train_Fidelity": res_adapt_finetune.trace['fidelity'][-1],
            "Noiseless_Fidelity": fid_adapt_noiseless,
            "True_Hardware_Fidelity": fid_adapt_noisy
        })
        results.append({
            "State_Seed": seed,
            "Method": "Top-Down Krotov",
            "Train_Fidelity": res_td_finetune.trace['fidelity'][-1],
            "Noiseless_Fidelity": fid_td_noiseless,
            "True_Hardware_Fidelity": fid_td_noisy
        })
        
        # Save dynamically so we don't lose data if it crashes
        df = pd.DataFrame(results)
        os.makedirs("experiments/results", exist_ok=True)
        df.to_csv("experiments/results/rigorous_benchmark_5states.csv", index=False)
        print(f"--> Saved intermediate results to rigorous_benchmark_5states.csv", flush=True)

    # -------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------
    print("\n[Final] Creating plots...", flush=True)
    plt.figure(figsize=(12, 7))
    
    seeds = [100, 200, 300, 400, 500]
    x = np.arange(len(seeds))
    width = 0.25
    
    vqa_noisy = df[df["Method"] == "Standard VQA"]["True_Hardware_Fidelity"].values
    adapt_noisy = df[df["Method"] == "ADAPT-VQE"]["True_Hardware_Fidelity"].values
    td_noisy = df[df["Method"] == "Top-Down Krotov"]["True_Hardware_Fidelity"].values
    
    vqa_noiseless = df[df["Method"] == "Standard VQA"]["Noiseless_Fidelity"].values
    adapt_noiseless = df[df["Method"] == "ADAPT-VQE"]["Noiseless_Fidelity"].values
    td_noiseless = df[df["Method"] == "Top-Down Krotov"]["Noiseless_Fidelity"].values
    
    # Plot true hardware fidelities as solid bars
    plt.bar(x - width, vqa_noisy, width, label='Standard VQA (True Hardware Fidelity)', color='#e74c3c')
    plt.bar(x, adapt_noisy, width, label='ADAPT-VQE (True Hardware Fidelity)', color='#f39c12')
    plt.bar(x + width, td_noisy, width, label='Top-Down Krotov (True Hardware Fidelity)', color='#2ecc71')
    
    # Plot noiseless fidelities as translucent bars behind them (showing the coherent gap)
    plt.bar(x - width, vqa_noiseless, width, alpha=0.3, color='#e74c3c', edgecolor='black', hatch='//')
    plt.bar(x, adapt_noiseless, width, alpha=0.3, color='#f39c12', edgecolor='black', hatch='//')
    plt.bar(x + width, td_noiseless, width, alpha=0.3, color='#2ecc71', edgecolor='black', hatch='//')
    
    # Add a dummy bar just for legend of noiseless
    plt.bar([0], [0], width=0, color='grey', alpha=0.3, edgecolor='black', hatch='//', label='Algorithmic Limit (Noiseless)')
    
    plt.axhline(y=0.80, color='k', linestyle='--', linewidth=2, label='80% SOTA Target')
    
    plt.xlabel("Disordered TFIM Target State (Seed)", fontsize=12)
    plt.ylabel("Fidelity", fontsize=12)
    plt.title("Rigorous Benchmark: True Hardware Fidelity vs Algorithmic Limits (N=500)", fontsize=14, pad=15)
    plt.xticks(x, [f"Seed={s}" for s in seeds])
    plt.ylim(0, 1.0)
    
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.tight_layout()
    
    plt.savefig("experiments/results/rigorous_benchmark_5states.png", dpi=300, bbox_inches="tight")
    print("Saved plot to experiments/results/rigorous_benchmark_5states.png", flush=True)

if __name__ == "__main__":
    run_rigorous_benchmark()
