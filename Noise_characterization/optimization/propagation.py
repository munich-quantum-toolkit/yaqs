import numpy as np
import qutip as qt

from yaqs.core.data_structures.networks import MPO, MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from yaqs import Simulator
from dataclasses import dataclass





@dataclass
class SimulationParameters:
    T: float = 1
    dt: float = 0.1
    L: int = 4
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
    gammas = []

    # Relaxation operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma_rel) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))
        gammas.append(gamma_rel)

    # Dephasing operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma_deph) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))
        gammas.append(gamma_deph)

    # Initial state
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

    # Define measurement operators
    sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    sy_list = [qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    obs_list = sx_list  + sy_list + sz_list


    # Create new set of observables by multiplying every operator in obs_list with every operator in c_ops
    A_kn_list= []
    for i,c_op in enumerate(c_ops):
        for obs in obs_list:
            A_kn_list.append(  (1/gammas[i]) * (c_op.dag()*obs*c_op  -  0.5*obs*c_op.dag()*c_op  -  0.5*c_op.dag()*c_op*obs)   )



    new_obs_list = obs_list + A_kn_list




    n_obs= len(obs_list)
    n_jump= len(c_ops)


    # Exact Lindblad solution
    result_lindblad = qt.mesolve(H, psi0, t, c_ops, new_obs_list, progress_bar=True)

    exp_vals = []
    for i in range(len(new_obs_list)):
        exp_vals.append(result_lindblad.expect[i])
    

    # Separate original and new expectation values
    original_exp_vals = exp_vals[:n_obs]
    new_exp_vals = exp_vals[n_obs:]

    # Reshape new_exp_vals to be a list of lists with dimensions n_jump times n_obs
    A_kn_exp_vals = [new_exp_vals[i * n_obs:(i + 1) * n_obs] for i in range(n_jump)]
    
    # Compute the integral of the new expectation values to obtain the derivatives
    d_On_d_gk = [ [trapezoidal(A_kn_exp_vals[i][j],t)  for j in range(n_obs)] for i in range(n_jump) ]


    return t, original_exp_vals, d_On_d_gk
    



def tjm(sim_params_class: SimulationParameters, N=1000):

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
    H_0.init_Ising(L, d, J, g)
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
    measurements = [Observable('x', site) for site in range(L)]  + [Observable('y', site) for site in range(L)] + [Observable('z', site) for site in range(L)]

    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)
    Simulator.run(state, H_0, sim_params, noise_model)

    tjm_exp_vals = []
    for observable in sim_params.observables:
        tjm_exp_vals.append(observable.results)
        # print(f"Observable at site {observable.site}: {observable.results}")
    # print(tjm_exp_vals)


    return t, tjm_exp_vals

