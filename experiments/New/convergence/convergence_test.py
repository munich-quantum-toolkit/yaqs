import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pickle
# import qutip as qt

from yaqs.data_structures.MPO import MPO
from yaqs.data_structures.MPS import MPS
from yaqs.data_structures.noise_model import NoiseModel
from yaqs.data_structures.simulation_parameters import Observable, SimulationParams
from yaqs.methods.TJM import TJM


# Define the system Hamiltonian
L = 10
d = 2
J = 1
g = 0.5
H_0 = MPO()
H_0.init_Ising(L, d, J, g)

# Define the initial state
state = MPS(L, state='zeros')

# Define the noise model
gamma_relaxation = 0.1
gamma_dephasing = 0.1
noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma_relaxation, gamma_dephasing])



fig, ax = plt.subplots(2, 1)
if __name__ == "__main__":
    # Define the simulation parameters
    T = 1
    dt = 0.01
    sample_timesteps = False
    N = 10000
    threshold = 1e-6
    max_bond_dim = 4
    order = 2
    measurements = [Observable('x', site) for site in range(L)]
    sim_params = SimulationParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)
    TJM(state, H_0, noise_model, sim_params)

    filename = f"TJM_Convergence_dt001.pickle"
    with open(filename, 'wb') as f:
        pickle.dump({
            'sim_params': sim_params,
        }, f)

    # heatmap = []
    # for observable in sim_params.observables:
    #     heatmap.append(observable.results)

    # im = ax[0].imshow(heatmap, aspect='auto', extent=[0, T, L, 0])
    # ax[0].set_ylabel('Site')

    # # Centers site ticks
    # ax[0].set_yticks([x-0.5 for x in list(range(1,L+1))], range(1,L+1))
    # cbar = plt.colorbar(im, ax=ax[0])
    # cbar.ax.set_title('$\\langle X \\rangle$')
    #########################################

    ######### QuTip Exact Solver ############
    # # Time vector
    # t = np.arange(0, sim_params.T + sim_params.dt, sim_params.dt)

    # # Define Pauli matrices
    # sx = qt.sigmax()
    # sy = qt.sigmay()
    # sz = qt.sigmaz()

    # # Construct the Ising Hamiltonian
    # H = 0
    # for i in range(L-1):
    #     H += J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
    # for i in range(L):
    #     H += g * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])

    # # Construct collapse operators
    # c_ops = []

    # # Dephasing operators
    # for i in range(L):
    #     c_ops.append(np.sqrt(gamma_dephasing) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))

    # # Relaxation operators
    # for i in range(L):
    #     c_ops.append(np.sqrt(gamma_relaxation) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))

    # # Initial state
    # psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

    # # Define measurement operators
    # sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    # # Exact Lindblad solution
    # result_lindblad = qt.mesolve(H, psi0, t, c_ops, sx_list, progress_bar=True)
    # filename = f"QuTip_exact_convergence.pickle"
    # with open(filename, 'wb') as f:
    #     pickle.dump({
    #         'sim_params': sim_params,
    #         'observables': result_lindblad.expect,
    #     }, f)
    # heatmap2 = []
    # for site in range(len(sx_list)):
    #         heatmap2.append(result_lindblad.expect[site])

    # # Error heatmap
    # heatmap = np.array(heatmap)
    # heatmap2 = np.array(heatmap2)
    # im2 = ax[1].imshow(np.abs(heatmap2-heatmap), cmap='Reds', aspect='auto', extent=[0, T, L, 0], norm=LogNorm())
    # ax[1].set_yticks([x-0.5 for x in list(range(1,L+1))], range(1,L+1))

    # cbar = plt.colorbar(im2, ax=ax[1])
    # cbar.ax.set_title('Error')
    # ax[1].set_xlabel('t')
    # ax[1].set_ylabel('Site')
    # plt.show()
