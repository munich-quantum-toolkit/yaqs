import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams

from mqt.yaqs.simulator import _run_physics
from dataclasses import dataclass
from mqt.yaqs.core.libraries.gate_library import *





@dataclass
class SimulationParameters:
    T: float = 1
    dt: float = 0.1
    L: int = 2
    J: float = 1
    g: float = 0.5
    gamma_rel: float = 0.1
    gamma_deph: float = 0.1




def qutip_traj(sim_params_class: SimulationParameters):

    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph


    t = np.arange(0, T + dt, dt) 

    '''QUTIP Initialization + Simulation'''

#region

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
        c_ops.append(np.sqrt(gamma_rel) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))

    # Dephasing operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma_deph) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))

    # Initial state
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

    # Define measurement operators
    sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    sy_list = [qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    obs_list = sx_list # + sy_list + sz_list

    # Exact Lindblad solution
    result_lindblad = qt.mesolve(H, psi0, t, c_ops, obs_list, progress_bar=True)
    real_exp_vals = []
    for site in range(len(obs_list)):
        real_exp_vals.append(result_lindblad.expect[site])

    
    return t, real_exp_vals



def tjm(sim_params_class: SimulationParameters, N=3000):

    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph


    t = np.arange(0, T + dt, dt) 


    # Define the system Hamiltonian
    d = 2
    H_0 = MPO()
    H_0.init_ising(L, J, g)
    # Define the initial state
    state = MPS(L, state='zeros')

    # Define the noise model
    # gamma_relaxation = noise_params[0]
    # gamma_dephasing = noise_params[1]
    noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma_rel, gamma_deph])

    sample_timesteps = True
    # N = 10
    threshold = 1e-6
    max_bond_dim = 4
    order = 2
    measurements = [Observable(X(), site) for site in range(L)]  #+ [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]

    sim_params = PhysicsSimParams(measurements, T, dt, N, max_bond_dim, threshold, order)
    _run_physics(state, H_0, sim_params, noise_model, parallel = True)

    tjm_exp_vals = []
    for observable in sim_params.observables:
        tjm_exp_vals.append(observable.results)
        # print(f"Observable at site {observable.site}: {observable.results}")
    # print(tjm_exp_vals)


    return t, tjm_exp_vals


if __name__ == "__main__":




    params_default = SimulationParameters()

    params_default.T = 15

    print(params_default.T)
    print(params_default.dt)
    print(params_default.L)
    print(params_default.J)
    print(params_default.g)
    print(params_default.gamma_rel)
    print(params_default.gamma_deph)


    ## Run both simulations with the same set of parameters
    t, qutip_results=qutip_traj(params_default)

    t_traj, tjm_results=tjm(params_default)





    # L = 5
    # T = 5
    # dt = 0.1
    # J = 1
    # g = 0.5
    # gamma = 0.1



    # sample_timesteps = True
    # N = 1000
    # threshold = 1e-6
    # max_bond_dim = 4
    # order = 2
    # measurements = [Observable('x', site) for site in range(L)] # + [Observable('y', site) for site in range(L)] + [Observable('z', site) for site in range(L)]

    # sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)




    # t = np.arange(0, sim_params.T + sim_params.dt, sim_params.dt)

    # # Define Pauli matrices
    # sx = qt.sigmax()
    # sy = qt.sigmay()
    # sz = qt.sigmaz()

    # # Construct the Ising Hamiltonian
    # H = 0
    # for i in range(L-1):
    #     H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
    # for i in range(L):
    #     H += -g * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])

    # # Construct collapse operators
    # c_ops = []

    # # Relaxation operators
    # for i in range(L):
    #     c_ops.append(np.sqrt(gamma) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))

    # # Dephasing operators
    # for i in range(L):
    #     c_ops.append(np.sqrt(gamma) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))

    # # Initial state
    # psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

    # # Define measurement operators
    # sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    # # Exact Lindblad solution
    # result_lindblad = qt.mesolve(H, psi0, t, c_ops, sx_list, progress_bar=True)
    # qutip_results = []
    # for site in range(len(sx_list)):
    #     qutip_results.append(result_lindblad.expect[site])

    # H_0 = MPO()
    # H_0.init_Ising(L, 2, J, g)

    # # Define the initial state
    # state = MPS(L, state='zeros')

    # # Define the noise model
    # noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma, gamma])

    # Simulator.run(state, H_0, sim_params, noise_model)

    # tjm_results = []
    # for observable in sim_params.observables:
    #     tjm_results.append(observable.results)




    




    # Plot results
    plt.figure(figsize=(10,8))
    for i in range(len(tjm_results)):
        plt.plot(t, qutip_results[i], label=f'exp val qutip obs {i}')
        plt.plot(t, tjm_results[i], label=f'exp val tjm obs {i}')
        # plt.plot(t, qutip_results[i]-tjm_results[i], label = f'observable {i}')
    plt.xlabel('times')
    plt.ylabel('expectation value')
    plt.legend()
    plt.show()