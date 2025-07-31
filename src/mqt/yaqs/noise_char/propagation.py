import numpy as np

import qutip as qt

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable,AnalogSimParams
from mqt.yaqs import simulator


from mqt.yaqs.noise_char.optimization import trapezoidal


from mqt.yaqs.core.libraries.gate_library import *


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

    threshold: float = 1e-6
    max_bond_dim: int = 4
    order: int = 2

    # For scikit_tt
    N:int = 100
    rank: int= 8


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


def tjm_traj(sim_params_class: SimulationParameters) -> tuple:
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



