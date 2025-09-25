import numpy as np
import argparse

from qiskit import QuantumCircuit


from worker_functions.qiskit_noisy_sim import qiskit_noisy_simulator
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import  Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPS


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run YAQS vs Qiskit comparison.")
    parser.add_argument("--with-qiskit", action="store_true", help="Run Qiskit baseline simulations as well.")
    parser.add_argument("--plot", action="store_true", help="Create plots (imports matplotlib lazily).")
    args = parser.parse_args()

    num_qubits = 6
    J = 1.0
    g = 0.5
    timestepsize = 0.1
    num_layers = 15
    noise_factor = 0.1
    num_traj = 1024



    # complete circuit with layersampling
    basis_circuit=create_ising_circuit(num_qubits, J, g, timestepsize, 1, periodic=False)
    complete_qc = QuantumCircuit(num_qubits)
    for i in range(num_layers):
        complete_qc.compose(basis_circuit, qubits=range(num_qubits), inplace=True)
        if i < num_layers - 1:
            complete_qc.barrier(label = "SAMPLE_OBSERVABLES")
    # complete_qc.draw(output="mpl")
    # plt.show()
    # basis_circuit.draw(output="mpl")

    complete_observables = [Observable(Z(), i) for i in range(num_qubits)]
    complete_sim_params = StrongSimParams(complete_observables, num_traj = num_traj, sample_layers = True)
    complete_state = MPS(num_qubits, state = "zeros", pad=2)
    complete_noise_model = NoiseModel([{"name": "pauli_x", "sites": [i], "strength": noise_factor} for i in range(num_qubits)] + [{"name": "crosstalk_xx", "sites": [i, i+1], "strength": noise_factor} for i in range(num_qubits-1)] + [{"name": "crosstalk_yy", "sites": [i, i+1], "strength": noise_factor} for i in range(num_qubits-1)] + [{"name": "pauli_y", "sites": [i], "strength": noise_factor} for i in range(num_qubits)])
    simulator.run(complete_state, complete_qc, complete_sim_params, complete_noise_model, parallel = True)


    from qiskit_aer.noise.errors import PauliLindbladError
    from qiskit.quantum_info import Pauli
    from qiskit_aer.noise import NoiseModel as QiskitNoiseModel

    # qiskit simulation (MPS backend)
    generators_two_qubit = [Pauli("IX"), Pauli("XI"), Pauli("XX"), Pauli("IY"), Pauli("YI"), Pauli("YY")] 
    SPLM_error_two_qubit = PauliLindbladError(generators_two_qubit, [noise_factor, noise_factor, noise_factor, noise_factor, noise_factor, noise_factor])
    noise_model = QiskitNoiseModel()
    for qubit in range(num_qubits - 1):
        noise_model.add_quantum_error(SPLM_error_two_qubit, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])
    mps_qiskit_results_list = [[1.0] for _ in range(num_qubits)]
    for i in range(num_layers):
        qc = create_ising_circuit(num_qubits, J, g, timestepsize, i + 1, periodic=False)
        qiskit_results = np.real(np.asarray(qiskit_noisy_simulator(qc, noise_model, num_qubits, 1, method="matrix_product_state"))).flatten()
        for qubit_idx in range(num_qubits):
            mps_qiskit_results_list[qubit_idx].append(float(qiskit_results[qubit_idx]))

    # qiskit simulation (density-matrix backend)
    generators_two_qubit = [Pauli("IX"), Pauli("XI"), Pauli("XX"), Pauli("IY"), Pauli("YI"), Pauli("YY")] 
    SPLM_error_two_qubit = PauliLindbladError(generators_two_qubit, [noise_factor, noise_factor, noise_factor, noise_factor, noise_factor, noise_factor])
    noise_model = QiskitNoiseModel()
    for qubit in range(num_qubits - 1):
        noise_model.add_quantum_error(SPLM_error_two_qubit, ["cx", "cz", "swap", "rxx", "ryy", "rzz", "rzx"], [qubit, qubit + 1])
    dm_qiskit_results_list = [[1.0] for _ in range(num_qubits)]
    for i in range(num_layers):
        qc = create_ising_circuit(num_qubits, J, g, timestepsize, i + 1, periodic=False)
        qiskit_results = np.real(np.asarray(qiskit_noisy_simulator(qc, noise_model, num_qubits, 1, method="density_matrix"))).flatten()
        for qubit_idx in range(num_qubits):
            dm_qiskit_results_list[qubit_idx].append(float(qiskit_results[qubit_idx]))


    import matplotlib.pyplot as plt

    t = np.arange(0, num_layers)
    plt.figure(figsize=(10, 5))
    for i in range(num_qubits):
        if args.with_qiskit:
            plt.plot(t, mps_qiskit_results_list[i][:num_layers], label=f"qiskit mps qubit {i}", marker="o", linestyle="solid")
            plt.plot(t, dm_qiskit_results_list[i][:num_layers], label=f"Exact qubit {i}", marker="x", linestyle="solid")
            plt.plot(t, np.abs(np.asarray(dm_qiskit_results_list[i][:num_layers]) - np.asarray(mps_qiskit_results_list[i][:num_layers])), label=f"DIFF: qiskit MPS to exact qubit {i} ", linestyle="dashed")
        plt.plot(t, complete_sim_params.observables[i].results.real[:num_layers], label=f"YAQS qubit {i}", linestyle="dashed")
        if args.with_qiskit:
            plt.plot(t, np.abs(np.asarray(complete_sim_params.observables[i].results.real[:num_layers]) - np.asarray(dm_qiskit_results_list[i][:num_layers])), label=f"DIFF: YAQS to exact qubit {i}", linestyle="solid")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    # # ==== Build data matrices (use exactly what you've computed) ====
    # # Each list currently starts with a baseline 1.0 at index 0. Your plots use [:num_layers].
    # mps_all = np.stack([np.asarray(mps_qiskit_results_list[q][:num_layers]) for q in range(num_qubits)])  # (Q, L)
    # dm_all  = np.stack([np.asarray(dm_qiskit_results_list[q][:num_layers])  for q in range(num_qubits)])
    # yaqs_all = np.stack([np.asarray(complete_sim_params.observables[q].results.real[:num_layers]) for q in range(num_qubits)])

    # # For error metrics we usually exclude the t=0 baseline; keep both if you want.
    # mps  = mps_all[:, 1:]   # shape: (Q, L-1)
    # dm   = dm_all[:, 1:]
    # yaqs = yaqs_all[:, 1:]
    # layers = np.arange(1, num_layers)  # 1..num_layers-1

    # # ==== Errors vs exact (density-matrix) ====
    # err_mps  = np.abs(mps  - dm)   # |MPS - Exact|
    # err_yaqs = np.abs(yaqs - dm)   # |YAQS - Exact|

    # # ==== Simple variance/summary metrics ====
    # # Variance of errors across layers for each qubit
    # var_mps_per_qubit  = np.var(err_mps,  axis=1, ddof=1)
    # var_yaqs_per_qubit = np.var(err_yaqs, axis=1, ddof=1)

    # # RMSE across layers per qubit
    # rmse_mps  = np.sqrt(np.mean((mps  - dm)**2, axis=1))
    # rmse_yaqs = np.sqrt(np.mean((yaqs - dm)**2, axis=1))

    # print("=== Error variance across layers (per qubit) ===")
    # print(f"MPS  mean var: {var_mps_per_qubit.mean():.3e} | median: {np.median(var_mps_per_qubit):.3e}")
    # print(f"YAQS mean var: {var_yaqs_per_qubit.mean():.3e} | median: {np.median(var_yaqs_per_qubit):.3e}")

    # print("=== Overall RMSE across layers (averaged over qubits) ===")
    # print(f"MPS  RMSE (mean over qubits): {rmse_mps.mean():.3e}")
    # print(f"YAQS RMSE (mean over qubits): {rmse_yaqs.mean():.3e}")

    # # ==== (1) Heatmaps: |error| over qubits × layers ====
    # fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    # im0 = axs[0].imshow(err_mps, aspect='auto', origin='lower', cmap='viridis')
    # axs[0].set_title("|MPS − Exact|")
    # axs[0].set_xlabel("Layer")
    # axs[0].set_ylabel("Qubit")
    # axs[0].set_xticks(np.arange(0, len(layers), max(1, len(layers)//10)))
    # axs[0].set_xticklabels(layers[axs[0].get_xticks().astype(int)])
    # fig.colorbar(im0, ax=axs[0])

    # im1 = axs[1].imshow(err_yaqs, aspect='auto', origin='lower', cmap='viridis')
    # axs[1].set_title("|YAQS − Exact|")
    # axs[1].set_xlabel("Layer")
    # axs[1].set_xticks(np.arange(0, len(layers), max(1, len(layers)//10)))
    # axs[1].set_xticklabels(layers[axs[1].get_xticks().astype(int)])
    # fig.colorbar(im1, ax=axs[1])

    # fig.suptitle("Absolute error heatmaps (per qubit × layer)")
    # plt.tight_layout()
    # plt.show()

    # # ==== (2) Fan charts: median error per layer with 10–90% band across qubits ====
    # def fan(ax, err, label, color=None):
    #     med = np.median(err, axis=0)
    #     p10 = np.percentile(err, 10, axis=0)
    #     p90 = np.percentile(err, 90, axis=0)
    #     ax.plot(layers, med, label=f"{label} median")
    #     ax.fill_between(layers, p10, p90, alpha=0.25, label=f"{label} 10–90%")

    # fig, ax = plt.subplots(figsize=(10, 4))
    # fan(ax, err_mps,  "MPS")
    # fan(ax, err_yaqs, "YAQS")
    # ax.set_xlabel("Layer")
    # ax.set_ylabel("|Δ⟨Z⟩| vs exact")
    # ax.set_title("Per-layer error (median across qubits, with spread)")
    # ax.grid(True, linestyle="--", alpha=0.6)
    # ax.legend()
    # plt.tight_layout()
    # plt.show()

    # # ==== (3) Per-qubit RMSE bars (across layers) ====
    # x = np.arange(num_qubits)
    # width = 0.38
    # plt.figure(figsize=(10, 4))
    # plt.bar(x - width/2, rmse_mps,  width, label="MPS")
    # plt.bar(x + width/2, rmse_yaqs, width, label="YAQS")
    # plt.xlabel("Qubit")
    # plt.ylabel("RMSE across layers")
    # plt.title("Per-qubit RMSE vs exact")
    # plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()






