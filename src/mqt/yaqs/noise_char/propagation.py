# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Performs the simulation of the Ising model and returns expectations values and  A_kn trahectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Destroy, GateLibrary, X, Y, Z
from mqt.yaqs.noise_char.optimization import trapezoidal

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

    def __init__(
        self,
        *,
        sim_params: AnalogSimParams,
        hamiltonian: MPO,
        noise_model: NoiseModel,
        obs_list: list[Observable],
        init_state: MPS,
    ) -> None:
        self.sim_params: AnalogSimParams = sim_params
        self.hamiltonian: MPO = hamiltonian
        self.noise_model: NoiseModel = noise_model
        self.obs_list: list[Observable] = obs_list
        self.init_state: MPS = init_state

        self.n_t = len(self.sim_params.times)  # number of time steps

        self.sites = self.hamiltonian.length  # number of sites in the chain

        all_obs_sites = [
            site for obs in obs_list for site in (obs.sites if isinstance(obs.sites, list) else [obs.sites])
        ]

        if max(all_obs_sites) >= self.sites:
            msg = "Observable site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)

        all_jump_sites = [site for proc in self.noise_model.processes for site in proc["sites"]]

        if max(all_jump_sites) >= self.sites:
            msg = "Noise process site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)

        self.obs_mat = self.make_observable_matrix()

        self.jump_mat = self.make_jump_matrix()

    def make_observable_matrix(self) -> NDArray[np.object_]:
        """Returns the observable matrix for the current observables.

        Returns:
            np.ndarray: The observable matrix. The shape is (n_obs_site, sites).
            Entries are zero if the corresponding observable is not measured for that site.
        """
        obs_site_set = list({obs.gate for obs in self.obs_list})

        self.n_obs_site = len(obs_site_set)

        obs_matrix = np.zeros((self.n_obs_site, self.sites), dtype=object)

        for obs in self.obs_list:
            if isinstance(obs.sites, int):
                site_list = [obs.sites]
            elif isinstance(obs.sites, list):
                site_list = obs.sites

            for site in site_list:
                gate = obs.gate

                obs_idx = obs_site_set.index(gate)
                obs_matrix[obs_idx, site] = gate

        return obs_matrix

    def make_jump_matrix(self) -> NDArray[np.object_]:
        jump_site_list = list({getattr(GateLibrary, proc["name"]) for proc in self.noise_model.processes})

        self.n_jump_site = len(jump_site_list)

        jump_matrix = np.zeros((self.n_jump_site, self.sites), dtype=object)

        for proc in self.noise_model.processes:
            for site in proc["sites"]:
                gate = getattr(GateLibrary, proc["name"])

                jump_idx = jump_site_list.index(gate)

                jump_matrix[jump_idx, site] = gate

        return jump_matrix

    def __call__(self, noise_model: NoiseModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a_kn_site_list: list[Observable] = []

        for i in range(self.n_jump_site):
            for j in range(self.n_obs_site):
                a_kn_site_list.extend(
                    Observable(
                        self.jump_mat[i, k].dag() * self.obs_mat[j, k] * self.jump_mat[i, k]
                        - 0.5 * self.obs_mat[j, k] * self.jump_mat[i, k].dag() * self.jump_mat[i, k]
                        - 0.5 * self.jump_mat[i, k].dag() * self.jump_mat[i, k] * self.obs_mat[j, k],
                        k,
                    )
                    for k in range(self.sites)
                )

        self.obs_list + a_kn_site_list

        simulator.run(self.init_state, self.hamiltonian, self.sim_params, noise_model)

        exp_vals = [observable.results for observable in self.sim_params.observables]

        # Separate original and new expectation values from result_lindblad.
        n_obs = len(self.obs_list)  # number of measurement operators (should be sites * n_types)
        original_exp_vals = exp_vals[:n_obs]
        new_exp_vals = exp_vals[n_obs:]  # these correspond to the A_kn operators
        assert all(v is not None for v in new_exp_vals)

        # Compute the integral of the new expectation values to obtain the derivatives
        d_on_d_gk = [trapezoidal(new_exp_vals[i], self.sim_params.times) for i in range(len(a_kn_site_list))]

        d_on_d_gk = np.array(d_on_d_gk).reshape(self.n_jump_site, self.n_obs_site, self.sites, self.n_t)
        original_exp_vals = np.array(original_exp_vals).reshape(self.n_obs_site, self.sites, self.n_t)

        return self.sim_params.times, original_exp_vals, d_on_d_gk

