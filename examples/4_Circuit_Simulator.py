import numpy as np
import qiskit.circuit
import matplotlib.pyplot as plt

from yaqs.core.data_structures.networks import MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import Observable, StrongSimParams
from yaqs.core.libraries.circuit_library import create_Ising_circuit

from yaqs import Simulator

# Define the circuit
num_qubits = 10
depth = num_qubits
circuit = qiskit.circuit.QuantumCircuit(num_qubits)

model = {'name': 'Ising', 'L': num_qubits, 'J': 1, 'g': 0.5}
circuit = create_Ising_circuit(model, dt=0.1, timesteps=10)
circuit.measure_all()

# Define the initial state
state = MPS(num_qubits, state='zeros')

# Define the simulation parameters
N = 100
max_bond_dim = 4
threshold = 1e-6
window_size = 0
measurements = [Observable('z', site) for site in range(num_qubits)]

gammas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
if __name__ == "__main__":
    heatmap = np.empty((num_qubits, len(gammas)))

    for j, gamma in enumerate(gammas):
        # Define the noise model
        sim_params = StrongSimParams(measurements, N, max_bond_dim, threshold, window_size)
        noise_model = NoiseModel(['relaxation'], [gamma])
        Simulator.run(state, circuit, sim_params, noise_model)
        for i, observable in enumerate(sim_params.observables):
            heatmap[i, j] = observable.results

    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(heatmap, aspect='auto', vmin=0.5, vmax=1)
    ax.set_ylabel('Site')
    ax.set_xticks(np.arange(len(gammas)))
    formatted_gammas = [f"$10^{{{int(np.log10(g))}}}$" for g in gammas]
    ax.set_xticklabels(formatted_gammas)

    fig.subplots_adjust(top=0.95, right=0.88)
    cbar_ax = fig.add_axes([0.9, 0.11, 0.025, 0.8])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_title('$\\langle Z \\rangle$')

    plt.show()
