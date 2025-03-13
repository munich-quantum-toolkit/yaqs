import numpy as np
import qutip as qt

import scipy as sp

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from mqt.yaqs import simulator
from dataclasses import dataclass

from yaqs.noise_char.optimization import trapezoidal

import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT
import scikit_tt.solvers.ode as ode
import scikit_tt



@dataclass
class SimulationParameters:
    T: float = 1
    dt: float = 0.1
    L: int = 4
    J: float = 1
    g: float = 0.5
    gamma_rel: float = 0.1
    gamma_deph: float = 0.1
    observables = ['x','y','z']

    # For scikit_tt
    N:int = 100
    rank: int= 8



    
def qutip_traj(sim_params_class: SimulationParameters):

    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph


    t = np.arange(0, T + dt, dt) 

    n_t = len(t)

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

    #c_ops = [rel0, rel1, rel2,... rel(L-1), deph0, deph1,..., deph(L-1)]

    # Initial state
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])



    # Create obs_list based on the observables in sim_params_class.observables
    obs_list = []


    for obs_type in sim_params_class.observables:
        if obs_type.lower() == 'x':
            # For each site, create the measurement operator for 'x'
            obs_list.extend([qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])
        elif obs_type.lower() == 'y':
            obs_list.extend([qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])
        elif obs_type.lower() == 'z':
            obs_list.extend([qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])





    jump_site_list = [ qt.destroy(2)  ,  sz]

    obs_site_list = [sx, sy, sz]


    A_kn_site_list = []


    n_jump_site = len(jump_site_list)
    n_obs_site = len(obs_site_list)


    for lk in jump_site_list:
        for on in obs_site_list:
            for k in range(L):
                A_kn_site_list.append( qt.tensor([  lk.dag()*on*lk  -  0.5*on*lk.dag()*lk  -  0.5*lk.dag()*lk*on   if n == k else qt.qeye(2) for n in range(L)]) )



    new_obs_list = obs_list + A_kn_site_list

    n_obs= len(obs_list)
    n_jump= len(c_ops)

        # Exact Lindblad solution
    result_lindblad = qt.mesolve(H, psi0, t, c_ops, new_obs_list, progress_bar=True)

    exp_vals = []
    for i in range(len(new_obs_list)):
        exp_vals.append(result_lindblad.expect[i])



    # Separate original and new expectation values from result_lindblad.
    n_obs = len(obs_list)  # number of measurement operators (should be L * n_types)
    original_exp_vals = exp_vals[:n_obs]
    new_exp_vals = exp_vals[n_obs:]  # these correspond to the A_kn operators



    # Compute the integral of the new expectation values to obtain the derivatives
    d_On_d_gk = [ trapezoidal(new_exp_vals[i],t)  for i in range(len(A_kn_site_list)) ]

    d_On_d_gk = np.array(d_On_d_gk).reshape(n_jump_site, n_obs_site, L, n_t)
    original_exp_vals = np.array(original_exp_vals).reshape(n_obs_site, L, n_t)

    # return t, original_exp_vals, d_On_d_gk, A_kn_exp_vals
    return t, original_exp_vals, d_On_d_gk
    








def tjm_traj(sim_params_class: SimulationParameters):

    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph
    N=sim_params_class.N



    t = np.arange(0, T + dt, dt) 
    n_t = len(t)


    # Define the system Hamiltonian
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
    obs_list = [Observable('x', site) for site in range(L)]  + [Observable('y', site) for site in range(L)] + [Observable('z', site) for site in range(L)]



    jump_site_list = [ 'rel'  ,  'deph']

    obs_site_list = ['x', 'y', 'z']


    A_kn_site_list = []


    n_jump_site = len(jump_site_list)
    n_obs_site = len(obs_site_list)


    for lk in jump_site_list:
        for on in obs_site_list:
            for k in range(L):
                A_kn_site_list.append(  Observable( lk+'_'+on, k) )



    new_obs_list = obs_list + A_kn_site_list




    sim_params = PhysicsSimParams(new_obs_list, T, dt, N, max_bond_dim, threshold, order, sample_timesteps=True)
    simulator.run(state, H_0, sim_params, noise_model)

    exp_vals = []
    for observable in sim_params.observables:
        exp_vals.append(observable.results)




    # Separate original and new expectation values from result_lindblad.
    n_obs = len(obs_list)  # number of measurement operators (should be L * n_types)
    original_exp_vals = exp_vals[:n_obs]
    new_exp_vals = exp_vals[n_obs:]  # these correspond to the A_kn operators



    # Compute the integral of the new expectation values to obtain the derivatives
    d_On_d_gk = [ trapezoidal(new_exp_vals[i],t)  for i in range(len(A_kn_site_list)) ]

    d_On_d_gk = np.array(d_On_d_gk).reshape(n_jump_site, n_obs_site, L, n_t)
    original_exp_vals = np.array(original_exp_vals).reshape(n_obs_site, L, n_t)



    return t, original_exp_vals, d_On_d_gk




def construct_Ank(O_list, L_list):
    """Construct observable array (same for all sites)"""
    # define necessary tensors for compact construction
    L = np.asarray(L_list)
    L_dag = np.conjugate(L).transpose([0,2,1])
    L_prod = np.einsum('ijk,ikl->ijl', L_dag, L)
    O = np.asarray(O_list)
    I = np.eye(2)
    # define A_nk
    A_nk = np.einsum('ijk,ilm->ijklm', L_dag, L) - 0.5*np.einsum('ij,klm->kijlm',I,L_prod) - 0.5*np.einsum('ijk,lm->ijklm',L_prod,I)
    A_nk = np.einsum('ijklm,nkl->ijnm', A_nk, O)
    return A_nk


def evaluate_Ank(A_nk, state):
    """Compute expected values of A_nk for each site (assuming that state is right-orthonormal)"""
    n_jumps = A_nk.shape[0]
    n_obs = A_nk.shape[2]
    n_sites = state.order
    A = np.zeros([n_sites, n_jumps, n_sites, n_obs], dtype=complex)

    A[0,:,0,:] = np.einsum('ijk, mjnl, ilk -> mn', np.conjugate(state.cores[0][:,:,0,:]), A_nk, state.cores[0][:,:,0,:])
    
    for i in range(state.order-1):
        # apply SVD
        [u, s, v] = sp.linalg.svd(state.cores[i].reshape(state.ranks[i]*state.row_dims[i]*state.col_dims[i],state.ranks[i + 1]), full_matrices=False, overwrite_a=True, check_finite=False)
        # define updated rank and core
        state.ranks[i + 1] = u.shape[1]
        state.cores[i] = u.reshape(state.ranks[i], state.row_dims[i], state.col_dims[i], state.ranks[i + 1])
        # shift non-orthonormal part to next core
        state.cores[i + 1] = np.tensordot(np.diag(s).dot(v), state.cores[i + 1], axes=(1, 0))

        A[i+1,:,i+1,:] = np.einsum('ijk, mjnl, ilk -> mn', np.conjugate(state.cores[i+1][:,:,0,:]), A_nk, state.cores[i+1][:,:,0,:])

    return A.transpose([1,0,3,2]).reshape([n_sites*n_jumps, n_sites*n_obs])






def scikit_tt_traj(sim_params_class: SimulationParameters):


    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph

    rank = sim_params_class.rank
    N = sim_params_class.N


    t = np.arange(0, T + dt, dt) 
    timesteps=len(t)-1


    # Parameters
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    L_1 = np.array([[0, 1], [0, 0]])
    L_2 = np.array([[1, 0], [0, -1]])
    O_list = [X, Y, Z]
    L_list = [L_1, L_2]

    # A_nk = construct_Ank(O_list, L_list)

    
    cores = [None] * L
    cores[0] = tt.build_core([[-g * X, - J * Z, I]])
    for i in range(1, L - 1):
        cores[i] = tt.build_core([[I, 0, 0], [Z, 0, 0], [-g * X, - J * Z, I]])
    cores[-1] = tt.build_core([I, Z, -g*X])
    hamiltonian = TT(cores)# jump operators and parameters

    jump_operator_list = [[L_1, L_2] for _ in range(L)]
    jump_parameter_list = [[np.sqrt(gamma_rel), np.sqrt(gamma_deph)] for _ in range(L)]


    obs_list=[]

    for pauli in O_list:
        for j in range(L):
            obs= tt.eye(dims=[2]*L)
            obs.cores[j]=np.zeros([1,2,2,1], dtype=complex)
            obs.cores[j][0,:,:,0]=pauli
            obs_list.append(obs)

    

    n_obs= len(obs_list)
    n_jump= 2*L



    exp_vals = np.zeros([n_obs,timesteps+1])

    A_nk=construct_Ank(O_list, L_list)


    A_kn_numpy=np.zeros([n_jump, n_obs, timesteps+1],dtype=complex)

    for k in range(N):
        initial_state = tt.unit([2] * L, [0] * L)
        for i in range(rank - 1):
            initial_state += tt.unit([2] * L, [0] * L)
        initial_state = initial_state.ortho()
        initial_state = (1 / initial_state.norm()) * initial_state

        
        for j in range(n_obs):
            exp_vals[j,0] += initial_state.transpose(conjugate=True)@obs_list[j]@initial_state
        
        A_kn_numpy[:,:,0]+= evaluate_Ank(A_nk, initial_state)
        
        
        
        for i in range(timesteps):
            initial_state = ode.tjm(hamiltonian, jump_operator_list, jump_parameter_list, initial_state, dt, 1)[-1]

            for j in range(n_obs):                
                exp_vals[j,i+1] += initial_state.transpose(conjugate=True)@obs_list[j]@initial_state

            A_kn_numpy[:,:,i+1] += evaluate_Ank(A_nk, initial_state)
            


    exp_vals = (1/N)*exp_vals
    
    A_kn_numpy = (1/N)*A_kn_numpy
    


    ## The .real part is added as a workaround 
    d_On_d_gk = [ [trapezoidal(A_kn_numpy[i,j].real,t)  for j in range(n_obs)] for i in range(n_jump) ]




    return t, exp_vals, d_On_d_gk