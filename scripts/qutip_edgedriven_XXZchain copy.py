import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

import concurrent.futures
import os
import pickle
from functools import partial

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs import simulator


# Build Hamiltonian
def qt_build_XXZ_operator(delta, L):
    """
    Constructs the XXZ Hamiltonian for a chain of L spins according to PhysRevLett.107.137201.
    """
    sp = qt.sigmap()
    sm = qt.sigmam()
    sz = qt.sigmaz()
    I = qt.qeye(2)
    
    H = 0
    for j in range(L - 1):
        sp_sm = qt.tensor([sp if n == j else sm if n == j + 1 else I for n in range(L)])
        sm_sp = qt.tensor([sm if n == j else sp if n == j + 1 else I for n in range(L)])
        sz_sz = qt.tensor([sz if n == j or n == j + 1 else I for n in range(L)])
        H += 2 * (sp_sm + sm_sp) + delta * sz_sz
    return H

def qt_build_lindblad_operators(L, epsilon):
    """
    Constructs the Lindblad operators for the XXZ chain.
    L: Number of spins
    epsilon: Coupling strength
    """
    sp = qt.sigmap()
    sm = qt.sigmam()
    I = qt.qeye(2)

    c_ops = []
    
    c_ops.append(np.sqrt(epsilon) * qt.tensor([sm] + [I] * (L - 1)))
    c_ops.append(np.sqrt(epsilon) * qt.tensor([I] * (L - 1) + [sp]))

    return c_ops


# analytical steady state for the XXZ chain, epsilon >> 
def steady_state(L):
    """
    Computes the exact steady state for the XXZ chain.
    """
    sz_exact = [np.cos(np.pi * (j-1) / (L - 1)) for j in range(1,L+1)]
    return np.array(sz_exact)


if __name__ == "__main__":

    # make directory for output
    output_dir = "tjm_trajectories"
    os.makedirs(output_dir, exist_ok=True)

    # Define parameters
    L = 100
    J_x = 1
    J_y = 1
    J_z = 1 # 0.25
    g = 0
    delta = 1
    factor = 5
    epsilon = factor*2*np.pi  # coupling strength (ε)


    T = 100000
    dt = 0.1
    t = np.arange(0, T + dt, dt)



    # H_qt = qt_build_XXZ_operator(delta, L)
    # c_ops = []
    # c_ops = qt_build_lindblad_operators(L, epsilon)


    # # psi0 = qt.rand_ket(2**L) # random initial state

    # zero = qt.basis(2, 0)  # single-qubit |0⟩
    # one = qt.basis(2, 1) # single-qubit |1⟩


    # # psi0 = qt.tensor([zero] * L)  # all zero state

    # # wall state: half |0⟩ and half |1⟩
    # half = L // 2
    # state_list = [zero]*half + [one]*(L - half)
    # psi0 = qt.tensor(state_list)

    # psi0.dims = [[2]*L, [1]]  # Matches the L-qubit tensor structure

    # sx = qt.sigmax()
    # sy = qt.sigmay()
    # sz = qt.sigmaz()
    # print(sz)



    # sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    # # Lindblad solution
    # result_lindblad = qt.mesolve(H_qt, psi0, t, c_ops, sz_list, progress_bar=True)




    # analytical steady state
    steadystate_exact = steady_state(L)


    # TJM setup
    H_0 = MPO()
    H_0.init_heisenberg(L, J_x, J_y, J_z, g)

    # Define the initial state
    state = MPS(L, state='wall')

    # Define the simulation parameters
    sample_timesteps = True
    N = 100
    batchsize = 100
    max_bond_dim = 4
    threshold = 1e-6
    order = 1
    measurements = [Observable(Z(), site) for site in range(L)]
    gamma =  epsilon*2  # coupling strength (γ)

    noise_model = NoiseModel(
    [['excitation']] + [['excitation'] for _ in range(L - 2)] + [['relaxation']],
    [[gamma]] + [[0] for _ in range(L - 2)] + [[gamma]]
)
 
    measurements = [Observable(Z(), site) for site in range(L)]
    for batch in range(N// batchsize):
        sim_params = PhysicsSimParams(measurements, T, dt, batchsize, max_bond_dim, threshold, order, sample_timesteps=sample_timesteps)
        simulator.run(state, H_0, sim_params, noise_model)
        filename = os.path.join(output_dir, f"tjm_results_L{L}_T{T}_factor{5}_10trajectories_pack{batch}.pkl")
        with open(filename, 'wb') as f:
                pickle.dump({
        'sim_params': sim_params,
    }, f)


    # plot the results
    plt.figure(figsize=(10, 6))
    plt.title('TJM Simulation Results')
    plt.xlabel('Time')
    plt.ylabel('Observable Expectation Values')
    for i, observable in enumerate(sim_params.observables):
        plt.plot(t, observable.results, label=f'⟨{observable.gate.name} {observable.site}⟩ TJM', color='orange')
        plt.axhline(y=steadystate_exact[i], linestyle='--', color='gray', label=f'⟨Z_{i}⟩ (exact steady state)' if i == 0 else None)
        # plt.plot(t, result_lindblad.expect[i], label=f'⟨Z_{i}⟩ QT', color='blue')
        # difference = result_lindblad.expect[i] - observable.results
        # plt.plot(t, difference, label=f'Difference ⟨Z_{i}⟩ - ⟨Z_{i}⟩_exact', linestyle='--', color='red')
    plt.legend()
    plt.grid()
    plt.show()










