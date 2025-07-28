import numpy as np

import qutip as qt

import scipy as sp

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable,AnalogSimParams
from mqt.yaqs import simulator
from dataclasses import dataclass

from mqt.yaqs.noise_char.optimization import trapezoidal


from mqt.yaqs.core.libraries.gate_library import *

import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT
import scikit_tt.solvers.ode as ode
import scikit_tt
import re


class SimulationParameters:
    """
    A class to encapsulate simulation parameters for open quantum system simulations.
    Attributes:
        T (float): Total simulation time. Default is 5.
        dt (float): Time step for the simulation. Default is 0.1.
        J (float): Coupling constant for the Ising Hamiltonian. Default is 1.
        g (float): Transverse field strength. Default is 0.5.
        observables (list): List of observables to measure (e.g., ['x', 'y', 'z']).
        threshold (float): Threshold for truncation in tensor network simulations. Default is 1e-6.
        max_bond_dim (int): Maximum bond dimension for tensor network simulations. Default is 4.
        order (int): Order of the integration method. Default is 2.
        N (int): Number of trajectories or samples for stochastic simulations. Default is 100.
        rank (int): Rank for tensor train decompositions. Default is 8.
        req_cpus (int): Number of CPUs requested for parallel simulations. Default is 1.
        scikit_tt_solver (dict): Dictionary specifying the solver and method for scikit_tt simulations.
    Args:
        L (int): Number of sites in the system.
        gamma_rel (list or float): Relaxation rates for each site, or a single float for all sites.
        gamma_deph (list or float): Dephasing rates for each site, or a single float for all sites.
    Methods:
        set_gammas(gamma_rel, gamma_deph):
            Sets the relaxation and dephasing rates for the system. Accepts either a list of length L or a single float.
        set_solver(solver='tdvp1', local_solver='krylov_5'):
            Sets the solver and local solver method for scikit_tt simulations.
            Args:
                solver (str): 'tdvp1' or 'tdvp2'.
                local_solver (str): 'krylov_<number>' or 'exact'.
    """
    T: float = 5
    dt: float = 0.1
    J: float = 1
    g: float = 0.5

    observables = ['x','y','z']

    threshold: float = 1e-6
    max_bond_dim: int = 4
    order: int = 2

    # For scikit_tt
    N:int = 100
    rank: int= 8

    req_cpus: int = 1


    scikit_tt_solver: dict = {"solver": 'tdvp1', "method": 'krylov', "dimension": 5}


    def __init__(self, L : int, gamma_rel : list | float, gamma_deph : list | float):

        self.L = L

        self.set_gammas(gamma_rel, gamma_deph)

        

    def set_gammas(self, gamma_rel : list | float, gamma_deph : list | float):
        """
        Set the relaxation (gamma_rel) and dephasing (gamma_deph) rates for the system.
        Parameters
        ----------
        gamma_rel : list or float
            Relaxation rates. If a float is provided, the same value is used for all sites (length L).
            If a list is provided, it must have length L.
        gamma_deph : list or float
            Dephasing rates. If a float is provided, the same value is used for all sites (length L).
            If a list is provided, it must have length L.
        Raises
        ------
        ValueError
            If gamma_rel or gamma_deph is a list and its length does not match L.
        Notes
        -----
        This method sets the attributes `gamma_rel` and `gamma_deph` as lists of length L.
        """


        if isinstance(gamma_rel, list) and len(gamma_rel) != self.L:
            raise ValueError("gamma_rel must be a list of length L.")
        if isinstance(gamma_deph, list) and len(gamma_deph) != self.L:
            raise ValueError("gamma_deph must be a list of length L.")

        if isinstance(gamma_rel, float): 
            self.gamma_rel = [gamma_rel] * self.L
        else:
            self.gamma_rel = list(gamma_rel)

        if isinstance(gamma_deph, float):
            self.gamma_deph = [gamma_deph] * self.L
        else:
            self.gamma_deph = list(gamma_deph)

    def set_solver(self, solver: str = 'tdvp1', local_solver: str = 'krylov_5'):
        """
        Set the solver configuration for the propagation algorithm.
        Parameters
        ----------
        solver : str, optional
            The global solver to use. Must be either 'tdvp1' or 'tdvp2'. Default is 'tdvp1'.
        local_solver : str, optional
            The local solver method. Must be either 'exact' or match the pattern 'krylov_<number>',
            where <number> is a positive integer (e.g., 'krylov_5'). Default is 'krylov_5'.
        Raises
        ------
        ValueError
            If `solver` is not 'tdvp1' or 'tdvp2'.
            If `local_solver` does not match 'exact' or the pattern 'krylov_<number>'.
            If the number in 'krylov_<number>' is not a positive integer.
        Notes
        -----
        Updates the `scikit_tt_solver` dictionary with the selected solver and method.
        """

        if solver not in ('tdvp1', 'tdvp2'):
            raise ValueError("solver can be only 'tdvp1' or 'tdvp2'")
        self.scikit_tt_solver["solver"] = solver

        if not re.match(r'^krylov_\d+$', local_solver) and not local_solver == 'exact':
            raise ValueError("local_solver must match the pattern 'krylov_<number>' or be 'exact'")
        
        if local_solver == 'exact':
            self.scikit_tt_solver["method"] = local_solver

        if local_solver.startswith('krylov_'):
            self.scikit_tt_solver["method"] = 'krylov'
            self.scikit_tt_solver["dimension"] = int(local_solver.split('_')[-1])
            if self.scikit_tt_solver["dimension"] < 1:
                raise ValueError("local_solver must be a positive integer when using 'krylov_<number>' format")


    



    
def qutip_traj(sim_params_class: SimulationParameters):
    """
    Simulates the time evolution of an open quantum system using the Lindblad master equation with QuTiP.
    This function constructs the system Hamiltonian and collapse operators for a spin chain with relaxation and dephasing noise,
    initializes the system state, and computes the expectation values of specified observables and their derivatives with respect
    to the noise parameters over time.
    Parameters
    ----------
    sim_params_class : SimulationParameters
        An instance of SimulationParameters containing simulation parameters:
            - T (float): Total simulation time.
            - dt (float): Time step.
            - L (int): Number of sites (spins) in the chain.
            - J (float): Ising coupling strength.
            - g (float): Transverse field strength.
            - gamma_rel (array-like): Relaxation rates for each site.
            - gamma_deph (array-like): Dephasing rates for each site.
            - observables (list of str): List of observables to measure ('x', 'y', 'z').
    Returns
    -------
    t : numpy.ndarray
        Array of time points at which the system was evolved.
    original_exp_vals : numpy.ndarray
        Expectation values of the specified observables at each site and time, shape (n_obs_site, L, n_t).
    d_On_d_gk : numpy.ndarray
        Derivatives of the observables with respect to the noise parameters, shape (n_jump_site, n_obs_site, L, n_t).
    avg_min_max_traj_time : list
        Placeholder list [None, None, None] for compatibility with other interfaces.
    Notes
    -----
    - The function uses QuTiP for quantum object and solver operations.
    - The system is initialized in the ground state |0>^{⊗L}.
    - The Hamiltonian is an Ising model with a transverse field.
    - Collapse operators are constructed for both relaxation and dephasing noise.
    - The function computes both the expectation values of observables and their derivatives with respect to noise parameters
      using the Lindblad master equation.
    """


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
        c_ops.append(np.sqrt(gamma_rel[i]) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))
        gammas.append(gamma_rel)

    # Dephasing operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma_deph[i]) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))
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


    avg_min_max_traj_time = [None, None, None]


    return t, original_exp_vals, d_On_d_gk, avg_min_max_traj_time
    






# from memory_profiler import profile

# @profile
def tjm_traj(sim_params_class: SimulationParameters):
    """
    Simulates the time evolution of an open quantum system using the Lindblad master equation with TJM.
    This function constructs the system Hamiltonian and collapse operators for a spin chain with relaxation and dephasing noise,
    initializes the system state, and computes the expectation values of specified observables and their derivatives with respect
    to the noise parameters over time.
    Parameters
    ----------
    sim_params_class : SimulationParameters
        An instance of SimulationParameters containing simulation parameters:
            - T (float): Total simulation time.
            - dt (float): Time step.
            - L (int): Number of sites (spins) in the chain.
            - J (float): Ising coupling strength.
            - g (float): Transverse field strength.
            - gamma_rel (array-like): Relaxation rates for each site.
            - gamma_deph (array-like): Dephasing rates for each site.
            - observables (list of str): List of observables to measure ('x', 'y', 'z').
    Returns
    -------
    t : numpy.ndarray
        Array of time points at which the system was evolved.
    original_exp_vals : numpy.ndarray
        Expectation values of the specified observables at each site and time, shape (n_obs_site, L, n_t).
    d_On_d_gk : numpy.ndarray
        Derivatives of the observables with respect to the noise parameters, shape (n_jump_site, n_obs_site, L, n_t).
    avg_min_max_traj_time : list
        Placeholder list [None, None, None] for compatibility with other interfaces.
    Notes
    -----
    - The function uses QuTiP for quantum object and solver operations.
    - The system is initialized in the ground state |0>^{⊗L}.
    - The Hamiltonian is an Ising model with a transverse field.
    - Collapse operators are constructed for both relaxation and dephasing noise.
    - The function computes both the expectation values of observables and their derivatives with respect to noise parameters
      using the Lindblad master equation.
    """


    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph
    N=sim_params_class.N


    threshold = sim_params_class.threshold
    max_bond_dim = sim_params_class.max_bond_dim
    order = sim_params_class.order



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


    noise_model = NoiseModel([{"name": "relaxation", "sites": [i], "strength": gamma_rel[i]} for i in range(L)] + [{"name": "dephasing", "sites": [i], "strength": gamma_deph[i]} for i in range(L)])


    sample_timesteps = True

    
    obs_list = [Observable(X(), site) for site in range(L)]  + [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]



    jump_site_list = [ Destroy()  ,  Z()]

    obs_site_list = [X(), Y(), Z()]


    A_kn_site_list = []


    n_jump_site = len(jump_site_list)
    n_obs_site = len(obs_site_list)


    for lk in jump_site_list:
        for on in obs_site_list:
            for k in range(L):
                A_kn_site_list.append(  Observable( lk.dag()*on*lk  -  0.5*on*lk.dag()*lk  -  0.5*lk.dag()*lk*on , k) )



    new_obs_list = obs_list + A_kn_site_list




    sim_params = AnalogSimParams(new_obs_list, T, dt, N, max_bond_dim, threshold, order, sample_timesteps=True)
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


    avg_min_max_traj_time = [None, None, None]  # Placeholder for average, min, and max trajectory time


    return t, original_exp_vals, d_On_d_gk, avg_min_max_traj_time











def construct_Akn(O_list, L_list):
    """
    This function computes the A_nk tensor based on lists of operator matrices O_list and L_list,
    typically representing system and Lindblad operators, respectively. The construction involves
    tensor contractions and combinations relevant for Lindblad master equation propagation.
    Args:
        O_list (array-like): List or array of operator matrices O_k (shape: [n_obs]).
        L_list (array-like): List or array of Lindblad operators L_n (shape: [n_jump]).
    Returns:
        np.ndarray: The constructed A_nk tensor with shape (n_jump, n_obs, d, d).
    """
    
    # define necessary tensors for compact construction
    L = np.asarray(L_list)
    L_dag = np.conjugate(L).transpose([0,2,1])
    L_prod = np.einsum('ijk,ikl->ijl', L_dag, L)
    O = np.asarray(O_list)
    I = np.eye(2)
    # define A_nk
    A_kn = np.einsum('ijk,ilm->ijklm', L_dag, L) - 0.5*np.einsum('ij,klm->kijlm',I,L_prod) - 0.5*np.einsum('ijk,lm->ijklm',L_prod,I)
    A_kn = np.einsum('ijklm,nkl->ijnm', A_kn, O)
    return A_kn


def evaluate_Akn(A_kn, state):
    """
    Evaluates the observable A_kn for a given quantum state represented in tensor train (TT) or matrix product state (MPS) format.
    This function computes the expectation values of the observable A_kn at each site of the quantum state by contracting the observable with the state tensors. It also applies singular value decomposition (SVD) to orthonormalize the state cores and updates the ranks accordingly.
    Parameters
    ----------
    A_kn : np.ndarray
        A 3D complex-valued array of shape (n_jumps, n_obs, n_sites) representing the observable(s) to be evaluated.
    state : object
        An object representing the quantum state in TT/MPS format. It must have the following attributes:
            - cores: list of np.ndarray
                The TT/MPS cores of the state.
            - order: int
                The number of sites (length of the TT/MPS chain).
            - ranks: list of int
                The TT/MPS ranks.
            - row_dims: list of int
                The physical dimensions of the rows for each core.
            - col_dims: list of int
                The physical dimensions of the columns for each core.
    Returns
    -------
    A : np.ndarray
        A complex-valued array of shape (n_jumps, n_obs, n_sites) containing the evaluated observable values at each site.
    Notes
    -----
    - The function modifies the `state` object in-place by updating its cores and ranks.
    - The contraction is performed using Einstein summation (`np.einsum`).
    - SVD is used to maintain orthonormality of the TT/MPS cores during the evaluation.
    """
    
    n_jumps = A_kn.shape[0]
    n_obs = A_kn.shape[2]
    n_sites = state.order
    A = np.zeros([n_jumps, n_obs, n_sites], dtype=complex)

    A[:,:,0] = np.einsum('ijk, mjnl, ilk -> mn', np.conjugate(state.cores[0][:,:,0,:]), A_kn, state.cores[0][:,:,0,:])
    
    for i in range(state.order-1):
        # apply SVD
        [u, s, v] = sp.linalg.svd(state.cores[i].reshape(state.ranks[i]*state.row_dims[i]*state.col_dims[i],state.ranks[i + 1]), full_matrices=False, overwrite_a=True, check_finite=False)
        # define updated rank and core
        state.ranks[i + 1] = u.shape[1]
        state.cores[i] = u.reshape(state.ranks[i], state.row_dims[i], state.col_dims[i], state.ranks[i + 1])
        # shift non-orthonormal part to next core
        state.cores[i + 1] = np.tensordot(np.diag(s).dot(v), state.cores[i + 1], axes=(1, 0))

        A[:,:,i+1] = np.einsum('ijk, mjnl, ilk -> mn', np.conjugate(state.cores[i+1][:,:,0,:]), A_kn, state.cores[i+1][:,:,0,:])

    return A


import multiprocessing
import os
import time


def process_k(k, L, rank, n_obs_site, n_jump_site, timesteps, dt, hamiltonian, jump_operator_list, jump_parameter_list, obs_list, A_kn, scikit_tt_solver):
    """
    Simulates the time evolution of a single open quantum system trajectory and computes expectation values and A_kn results, so they can be averaged later.
    Args:
        k (int): Index or identifier for the current trajectory or process.
        L (int): Number of sites in the quantum system.
        rank (int): Rank of the initial state tensor.
        n_obs_site (int): Number of observable sites.
        n_jump_site (int): Number of jump operator sites.
        timesteps (int): Number of time steps for the simulation.
        dt (float): Time step size.
        hamiltonian: Hamiltonian operator for the system (in TT format).
        jump_operator_list (list): List of jump operators (in TT format).
        jump_parameter_list (list): List of parameters for each jump operator.
        obs_list (list): List of observable operators (in TT format).
        A_kn: Function or object used to evaluate A_kn for a given state.
        scikit_tt_solver: ODE solver to use for time evolution (compatible with scikit-tt).
    Returns:
        exp_result (np.ndarray): Array of shape (len(obs_list), timesteps+1) containing the expectation values of observables at each time step.
        A_kn_result (np.ndarray): Array of shape (n_jump_site, n_obs_site, L, timesteps+1) containing evaluated A_kn results at each time step.
        traj_time (float): Total time taken to compute the trajectory (in seconds).
    """

    start_time = time.time()

    initial_state = tt.unit([2] * L, [0] * L)
    for i in range(rank - 1):
        initial_state += tt.unit([2] * L, [0] * L)
    initial_state = initial_state.ortho()
    initial_state = (1 / initial_state.norm()) * initial_state


    n_obs=len(obs_list)

    n_t=timesteps+1

    A_kn_result = np.zeros([n_jump_site, n_obs_site, L, n_t],dtype=complex)
    exp_result = np.zeros([len(obs_list),n_t])

    
    for j in range(n_obs):
        exp_result[j,0] = np.real(initial_state.transpose(conjugate=True)@obs_list[j]@initial_state)

    A_kn_result[:,:,:,0] = evaluate_Akn(A_kn, initial_state)
    
    
    
    for i in range(timesteps):
        initial_state = ode.tjm(hamiltonian, jump_operator_list, jump_parameter_list, initial_state, dt, 1, solver=scikit_tt_solver)[-1]

        for j in range(n_obs):                
            exp_result[j,i+1] = np.real(initial_state.transpose(conjugate=True)@obs_list[j]@initial_state)

        A_kn_result[:,:,:,i+1] = evaluate_Akn(A_kn, initial_state)

    
    end_time = time.time()

    traj_time = end_time - start_time

    return exp_result, A_kn_result, traj_time



def scikit_tt_traj(sim_params_class: SimulationParameters):
    """
    Simulates the time evolution of an open quantum system using the Lindblad master equation with Scikit_tt.
    This function constructs the system Hamiltonian and collapse operators for a spin chain with relaxation and dephasing noise,
    initializes the system state, and computes the expectation values of specified observables and their derivatives with respect
    to the noise parameters over time.
    Parameters
    ----------
    sim_params_class : SimulationParameters
        An instance of SimulationParameters containing simulation parameters:
            - T (float): Total simulation time.
            - dt (float): Time step.
            - L (int): Number of sites (spins) in the chain.
            - J (float): Ising coupling strength.
            - g (float): Transverse field strength.
            - gamma_rel (array-like): Relaxation rates for each site.
            - gamma_deph (array-like): Dephasing rates for each site.
            - observables (list of str): List of observables to measure ('x', 'y', 'z').
    Returns
    -------
    t : numpy.ndarray
        Array of time points at which the system was evolved.
    original_exp_vals : numpy.ndarray
        Expectation values of the specified observables at each site and time, shape (n_obs_site, L, n_t).
    d_On_d_gk : numpy.ndarray
        Derivatives of the observables with respect to the noise parameters, shape (n_jump_site, n_obs_site, L, n_t).
    avg_min_max_traj_time : list
        Placeholder list [None, None, None] for compatibility with other interfaces.
    Notes
    -----
    - The function uses QuTiP for quantum object and solver operations.
    - The system is initialized in the ground state |0>^{⊗L}.
    - The Hamiltonian is an Ising model with a transverse field.
    - Collapse operators are constructed for both relaxation and dephasing noise.
    - The function computes both the expectation values of observables and their derivatives with respect to noise parameters
      using the Lindblad master equation.
    """


    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph

    rank = sim_params_class.rank
    N = sim_params_class.N

    req_cpus = sim_params_class.req_cpus

    scikit_tt_solver = sim_params_class.scikit_tt_solver


    t = np.arange(0, T + dt, dt) 
    n_t = len(t)
    timesteps=n_t-1


    # Parameters
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    L_1 = np.array([[0, 1], [0, 0]])
    L_2 = np.array([[1, 0], [0, -1]])
    O_list = [X, Y, Z]
    L_list = [L_1, L_2]


    
    cores = [None] * L
    cores[0] = tt.build_core([[-g * X, - J * Z, I]])
    for i in range(1, L - 1):
        cores[i] = tt.build_core([[I, 0, 0], [Z, 0, 0], [-g * X, - J * Z, I]])
    cores[-1] = tt.build_core([I, Z, -g*X])
    hamiltonian = TT(cores)# jump operators and parameters

    jump_operator_list = [[L_1, L_2] for _ in range(L)]
    jump_parameter_list = [[np.sqrt(gamma_rel[i]), np.sqrt(gamma_deph[i])] for i in range(L)]


    obs_list=[]

    for pauli in O_list:
       for j in range(L):
           obs= tt.eye(dims=[2]*L)
           obs.cores[j]=np.zeros([1,2,2,1], dtype=complex)
           obs.cores[j][0,:,:,0]=pauli
           obs_list.append(obs)

    

    n_obs_site= len(O_list) ## Number of observables per site. Should be 3
    n_jump_site= len(L_list)  ## Number of jump operators per site. Should be 2


    n_obs=len(obs_list)   ## Total number of observable. Should be n_obs_site*L 


    exp_vals = np.zeros([n_obs,n_t])

    A_kn=construct_Akn(O_list, L_list)


    A_kn_numpy=np.zeros([n_jump_site, n_obs_site, L, n_t],dtype=complex)



    avail_num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", multiprocessing.cpu_count())) 


    if req_cpus > avail_num_cpus:
        nthreads= max(1, avail_num_cpus - 1)
        print(f"Requested {req_cpus} CPUs, but only {avail_num_cpus} are available. Using {avail_num_cpus} CPUs.")
    else:
        nthreads = max(1, req_cpus - 1)



    args_list = [
    (k,  L, rank, n_obs_site, n_jump_site, timesteps, dt, hamiltonian, jump_operator_list, jump_parameter_list, obs_list, A_kn, scikit_tt_solver)
    for k in range(N) ]
    

    with multiprocessing.Pool(processes=nthreads) as pool:
        results = pool.starmap(process_k, args_list)


    exp_vals = np.sum([res[0] for res in results], axis=0)/N
    A_kn_numpy = np.sum([res[1] for res in results], axis=0)/N


    ## The .real part is added as a workaround 
    d_On_d_gk = [ [[trapezoidal(A_kn_numpy[i,j,k].real,t) for k in range(L)] for j in range(n_obs_site)] for i in range(n_jump_site)  ]



    exp_vals = np.array(exp_vals).reshape(n_obs_site, L, n_t)


    traj_time_list = np.array([res[2] for res in results])

    avg_min_max_traj_time = [np.mean(traj_time_list), np.min(traj_time_list), np.max(traj_time_list)]

    return t, exp_vals, d_On_d_gk, avg_min_max_traj_time