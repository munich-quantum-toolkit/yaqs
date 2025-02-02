import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from yaqs.general.data_structures.MPO import MPO
from yaqs.general.data_structures.MPS import MPS
from yaqs.general.data_structures.noise_model import NoiseModel
from yaqs.general.data_structures.simulation_parameters import Observable, PhysicsSimParams
from yaqs.physics.methods import TJM


# Define the system Hamiltonian
L = 3
d = 2
J = 1
g = 0.5
H_0 = MPO()
H_0.init_Ising(L,d,J,g)

# Define the initial state
state = MPS(L, state='zeros')

# Define the noise model
gamma = 0.1
noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma, gamma])

# Define the simulation parameters
T = 5
dt = 0.1
sample_timesteps = True
N = 100
max_bond_dim = 4
threshold = 1e-6
order = 2

# measurements = []
# for i in range(L):
#     measurements.append(Observable('x',i))
#     measurements.append(Observable('y',i))
#     measurements.append(Observable('z',i))
measurements = [Observable('x', site) for site in range(L)] + [Observable('y', site) for site in range(L)] + [Observable('z', site) for site in range(L)]
sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)


if __name__ == "__main__":
    ########## TJM Example #################
    TJM.run(state, H_0, sim_params, noise_model)



    #########################################

    # ######### QuTip Exact Solver ############
    # Time vector
    t = np.arange(0, sim_params.T + sim_params.dt, sim_params.dt)

    # Define Pauli matrices
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # Construct the Ising Hamiltonian
    H = 0
    for i in range(L-1):
        H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
    for i in range(L):
        H += -g * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])




    # Construct collapse operators
    c_ops = []

    # Relaxation operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))

    # Dephasing operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))

    # Initial state
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

 
    sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    sy_list = [qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    # obs_list = []
    # for i in range(len(sx_list)):
    #     obs_list.append(sx_list[i])
    #     obs_list.append(sy_list[i])
    #     obs_list.append(sz_list[i])
    obs_list = sx_list + sy_list + sz_list

    # Exact Lindblad solution
    result_lindblad = qt.mesolve(H, psi0, t, c_ops, obs_list, progress_bar=True)



    # Plotting the difference between QuTiP and TJM results
    plt.figure(figsize=(10, 8))

    t = np.arange(0, sim_params.T + sim_params.dt, sim_params.dt)

    tjm_results = []
    for observable in sim_params.observables:
        tjm_results.append(observable.results)

    qutip_results = []
    for i in range(len(obs_list)):
        qutip_results.append(result_lindblad.expect[i]) 

    # Plot differences with proper labels
    for j in range(len(obs_list)):
        #plt.plot(t, qutip_results[j], label=f'Qutip Obs {j}')
        difference = qutip_results[j] - tjm_results[j]  # Compute difference
        # Get the observable name and site from the original observables list
        observable = sim_params.observables[j]
        label = f'difference:{observable.name}{observable.site}'  # Format: e.g., "x0", "y1", "z2"
        plt.plot(t, difference, linestyle='-', label=label)  # Plot difference with proper label
        #plt.plot(t,tjm_results[j], linestyle='--',label=f'TJM Obs {j}')

    plt.xlabel('Time')
    plt.ylabel('Difference (QuTiP - TJM)')
    plt.title('Difference Between QuTiP and TJM Expectation Values')
    plt.axhline(0, color='black', linestyle='dotted')  # Reference line at 0
    plt.legend()
    plt.grid()
    plt.show()
