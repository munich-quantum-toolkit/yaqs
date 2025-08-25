# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Performs the simulation of the Ising model and returns expectations values and  A_kn trahectories."""

from __future__ import annotations

import numpy as np

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Destroy, X, Y, Z
from mqt.yaqs.noise_char.optimization import trapezoidal


class SimulationParameters:
    """A class to encapsulate simulation parameters for open quantum system simulations.

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
    N: int = 100
    rank: int = 8

    def __init__(self, sites: int, gamma_rel: list[float] | float, gamma_deph: list[float] | float) -> None:
        """Defines the system  with the number of sites and noise parameters.

        Parameters
        ----------
        sites : int
            The number of sites in the system
        gamma_rel : list[float] | float | np.ndarray
            Relaxation rates. If a float is provided, the same value is used for all sites (length L).
            If a list or array is provided, it must have length L.
        gamma_deph : list[float] | float | np.ndarray
            Dephasing rates. If a float is provided, the same value is used for all sites (length L).
            If a list or array is provided, it must have length L.
        """
        self.L = sites

        self.set_gammas(gamma_rel, gamma_deph)

    def set_gammas(
        self, gamma_rel: np.ndarray | list[float] | float, gamma_deph: np.ndarray | list[float] | float
    ) -> None:
        """Set the relaxation (gamma_rel) and dephasing (gamma_deph) rates for the system.

        Args:
            gamma_rel (list[float] | float | np.ndarray): Relaxation rates. If a float is
                provided, the same value is used for all sites (length L). If a list or array
                is provided, it must have length L.
            gamma_deph (list[float] | float | np.ndarray): Dephasing rates. If a float is
                provided, the same value is used for all sites (length L). If a list or array
                is provided, it must have length L.

        Raises:
            ValueError: If ``gamma_rel`` is a list or array and its length does not match ``L``.
            ValueError: If ``gamma_deph`` is a list or array and its length does not match ``L``.

        Notes:
            This method sets the attributes ``gamma_rel`` and ``gamma_deph`` as lists of
            length L.
        """
        if (isinstance(gamma_rel, (list, np.ndarray))) and len(gamma_rel) != self.L:
            msg = "gamma_rel must be a list of length L."
            raise ValueError(msg)
        if (isinstance(gamma_deph, (list, np.ndarray))) and len(gamma_deph) != self.L:
            msg = "gamma_deph must be a list of length L."
            raise ValueError(msg)

        if isinstance(gamma_rel, float):
            self.gamma_rel = [gamma_rel] * self.L
        else:
            self.gamma_rel = list(gamma_rel)

        if isinstance(gamma_deph, float):
            self.gamma_deph = [gamma_deph] * self.L
        else:
            self.gamma_deph = list(gamma_deph)






class Propagator:

    """A class to encapsulate the propagator for the Ising model with noise.

    This class provides methods to set the Hamiltonian, noise model, loss function,
    and simulation parameters for the Ising model with noise.

    Attributes:
        sim_params (AnalogSimParams): Simulation parameters.
        hamiltonian (MPO): The Hamiltonian of the system.
        noise_model (NoiseModel): The noise model of the system.
        loss_function (LossClass): The loss function for optimization.
    """

    def __init__(self,*,sim_params: AnalogSimParams, hamiltonian: MPO, noise_model: NoiseModel, obs_list: list[Observable], init_state: MPS) -> None:
        self.sim_params: AnalogSimParams | None = None
        self.hamiltonian: MPO | None = None
        self.noise_model: NoiseModel | None = None
        self.obs_list: list[Observable] | None = obs_list
        self.init_state: MPS | None = None


        




    def __call__(self, noise_model: NoiseModel):


        n_t = len(self.sim_params.times)  ## number of time steps
        sites = self.hamiltonian.length  ## number of sites in the chain



        jump_site_list = [Destroy(), Z()]

        obs_site_list = [X(), Y(), Z()]

        a_kn_site_list: list[Observable] = []

        n_jump_site = len(jump_site_list)
        n_obs_site = len(obs_site_list)

        for lk in jump_site_list:
            for on in obs_site_list:
                a_kn_site_list.extend(
                    Observable(lk.dag() * on * lk - 0.5 * on * lk.dag() * lk - 0.5 * lk.dag() * lk * on, k)
                    for k in range(sites)
                )

        new_obs_list = self.obs_list + a_kn_site_list

        simulator.run(self.initial_state, self.hamiltonian, self.sim_params, noise_model)

        exp_vals = [observable.results for observable in self.sim_params.observables]

        # Separate original and new expectation values from result_lindblad.
        n_obs = len(self.obs_list)  # number of measurement operators (should be sites * n_types)
        original_exp_vals = exp_vals[:n_obs]
        new_exp_vals = exp_vals[n_obs:]  # these correspond to the A_kn operators
        assert all(v is not None for v in new_exp_vals)

        # Compute the integral of the new expectation values to obtain the derivatives
        d_on_d_gk = [trapezoidal(new_exp_vals[i], t) for i in range(len(a_kn_site_list))]

        d_on_d_gk = np.array(d_on_d_gk).reshape(n_jump_site, n_obs_site, sites, n_t)
        original_exp_vals = np.array(original_exp_vals).reshape(n_obs_site, sites, n_t)

        avg_min_max_traj_time = [None, None, None]  # Placeholder for average, min, and max trajectory time

        return self.sim_params.times, original_exp_vals, d_on_d_gk, avg_min_max_traj_time









def tjm_traj(sim_params_class: SimulationParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[None]]:
    """Simulates the time evolution of an open quantum system using the Lindblad master equation with TJM.

    This function constructs the system Hamiltonian and collapse operators for a spin chain with
    relaxation and dephasing noise, initializes the system state, and computes the expectation values
    of specified observables and their derivatives with respect to the noise parameters over time.
    Parameters.
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

    Returns:
    -------
    t : numpy.ndarray
        Array of time points at which the system was evolved.
    original_exp_vals : numpy.ndarray
        Expectation values of the specified observables at each site and time, shape (n_obs_site, L, n_t).
    d_on_d_gk : numpy.ndarray
        Derivatives of the observables with respect to the noise parameters, shape (n_jump_site, n_obs_site, L, n_t).
    avg_min_max_traj_time : list
        Placeholder list [None, None, None] for compatibility with other interfaces.

    Notes:
    -----
    - The function uses QuTiP for quantum object and solver operations.
    - The system is initialized in the ground state |0>^{⊗L}.
    - The Hamiltonian is an Ising model with a transverse field.
    - Collapse operators are constructed for both relaxation and dephasing noise.
    - The function computes both the expectation values of observables
    and their derivatives with respect to noise parameters
      using the Lindblad master equation.
    """
    sim_time = sim_params_class.T
    dt = sim_params_class.dt
    sites = sim_params_class.L
    coupl = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph
    ntraj = sim_params_class.N

    threshold = sim_params_class.threshold
    max_bond_dim = sim_params_class.max_bond_dim
    order = sim_params_class.order

    t = np.arange(0, sim_time + dt, dt)
    n_t = len(t)

    # Define the system Hamiltonian
    h_0 = MPO()

    h_0.init_ising(sites, coupl, g)
    # Define the initial state
    state = MPS(sites, state="zeros")

    noise_model = NoiseModel(
        [{"name": "lowering", "sites": [i], "strength": gamma_rel[i]} for i in range(sites)]
        + [{"name": "pauli_z", "sites": [i], "strength": gamma_deph[i]} for i in range(sites)]
    )

    obs_list = (
        [Observable(X(), site) for site in range(sites)]
        + [Observable(Y(), site) for site in range(sites)]
        + [Observable(Z(), site) for site in range(sites)]
    )

    jump_site_list = [Destroy(), Z()]

    obs_site_list = [X(), Y(), Z()]

    a_kn_site_list: list[Observable] = []

    n_jump_site = len(jump_site_list)
    n_obs_site = len(obs_site_list)

    for lk in jump_site_list:
        for on in obs_site_list:
            a_kn_site_list.extend(
                Observable(lk.dag() * on * lk - 0.5 * on * lk.dag() * lk - 0.5 * lk.dag() * lk * on, k)
                for k in range(sites)
            )

    new_obs_list = obs_list + a_kn_site_list

    sim_params = AnalogSimParams(
        observables=new_obs_list,
        elapsed_time=sim_time,
        dt=dt,
        num_traj=ntraj,
        max_bond_dim=max_bond_dim,
        threshold=threshold,
        order=order,
        sample_timesteps=True,
    )
    simulator.run(state, h_0, sim_params, noise_model)

    exp_vals = [observable.results for observable in sim_params.observables]

    # Separate original and new expectation values from result_lindblad.
    n_obs = len(obs_list)  # number of measurement operators (should be sites * n_types)
    original_exp_vals = exp_vals[:n_obs]
    new_exp_vals = exp_vals[n_obs:]  # these correspond to the A_kn operators
    assert all(v is not None for v in new_exp_vals)

    # Compute the integral of the new expectation values to obtain the derivatives
    d_on_d_gk = [trapezoidal(new_exp_vals[i], t) for i in range(len(a_kn_site_list))]

    d_on_d_gk = np.array(d_on_d_gk).reshape(n_jump_site, n_obs_site, sites, n_t)
    original_exp_vals = np.array(original_exp_vals).reshape(n_obs_site, sites, n_t)

    avg_min_max_traj_time = [None, None, None]  # Placeholder for average, min, and max trajectory time

    return t, original_exp_vals, d_on_d_gk, avg_min_max_traj_time
