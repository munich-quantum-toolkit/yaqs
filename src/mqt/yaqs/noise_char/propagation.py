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
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import GateLibrary, Zero
from mqt.yaqs.noise_char.optimization import trapezoidal
import copy


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.networks import MPO, MPS
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel



def noise_model_to_operator_list(noise_model: NoiseModel) -> list[Observable]:
    """Converts a noise model to a list of observables.

    Args:
        noise_model (NoiseModel): The noise model to convert.

    Returns:
        list[Observable]: A list of observables corresponding to the noise processes in the noise model.
    """
    noise_list: list[Observable] = []

    for proc in noise_model.processes:
        gate = getattr(GateLibrary, proc["name"])
        for site in proc["sites"]:
            noise_list.append(Observable(gate, site))
    return noise_list



def flatten_noise_model(self, noise_model: NoiseModel) -> NoiseModel:
    """Serializes the noise model.

    Args:
        noise_model (NoiseModel): The noise model to serialize.

    Returns:
        NoiseModel: The serialized noise model.
    """

    noise_list=[]

    index_list=[]

    for i, proc in enumerate(noise_model.processes):
        for site in proc["sites"]:
            noise_list.append({"name": proc["name"], "sites": [site], "strength": proc["strength"]})
            index_list.append(i)

    return noise_list, index_list






    return copy.deepcopy(noise_model)



class PropagatorWithGradients:
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
        init_state: MPS,
    ) -> None:
        self.sim_params: AnalogSimParams = sim_params
        self.hamiltonian: MPO = hamiltonian
        self.input_noise_model: NoiseModel = noise_model
        self.init_state: MPS = init_state


        self.flat_noise_model, self.index_list = flatten_noise_model(self.input_noise_model)
        self.noise_list: list[Observable] = noise_model_to_operator_list(self.flat_noise_model)

        self.n_jump=len(self.noise_list)  # number of jump operators

        if max([ proc["sites"][0]  for proc in self.flat_noise_model]) >= self.sites:
            msg = "Noise site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)

        self.n_t = len(self.sim_params.times)  # number of time steps

        self.sites = self.hamiltonian.length  # number of sites in the chain

        self.set_observables=False
    

    def set_observable_list(self, obs_list: list[Observable]) -> None:
        self.obs_list=copy.deepcopy(obs_list)

        all_obs_sites = [
            site for obs in obs_list for site in (obs.sites if isinstance(obs.sites, list) else [obs.sites])
        ]

        if max(all_obs_sites) >= self.sites:
            msg = "Observable site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)
        

        self.n_obs = len(obs_list)  # number of measurement operators   

        self.set_observables=True


    def set_time(self, *, elapsed_time: float, dt: float = 0.1):

        self.sim_params.elapsed_time = elapsed_time
        self.sim_params.dt = dt
        self.sim_params.times = np.arange(0, elapsed_time + dt, dt)
        self.n_t = len(self.sim_params.times)

        print(f"Simulation time set to {elapsed_time} with time step {dt}. Number of time steps: {self.n_t}")



    def __call__(self, noise_model: NoiseModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        if self.set_observables==False:
            msg = "Observable list not set. Please use the set_observable_list method to set the observables."
            raise ValueError(msg)
        


        for i, proc in enumerate(noise_model.processes):
            for j, site in enumerate(proc["sites"]):
                if (
                    proc["name"] != self.input_noise_model.processes[i]["name"]
                    or site != self.input_noise_model.processes[i]["sites"][j]
                ):
                    msg = "Noise model processes or sites do not match the initialized noise model."
                    raise ValueError(msg)

        a_kn_site_list: list[Observable] = []


        for lk in self.noise_list:
            for on in self.obs_list:
                if lk.sites == on.sites:
                    a_kn_site_list.append(lk.dag()*on*lk - 0.5*on*lk.dag()*lk - 0.5*lk.dag()*lk*on)
                    


        new_obs_list=self.obs_list + a_kn_site_list

        new_sim_params = AnalogSimParams(observables=new_obs_list, elapsed_time=self.sim_params.elapsed_time, 
                                         dt=self.sim_params.dt, num_traj=self.sim_params.num_traj, 
                                         max_bond_dim=self.sim_params.max_bond_dim, threshold=self.sim_params.threshold, 
                                         order=self.sim_params.order, sample_timesteps=True)
        simulator.run(self.init_state, self.hamiltonian, new_sim_params, noise_model)


        # Separate original and new expectation values from result_lindblad.
        original_exp_vals = new_sim_params.observables[:self.n_obs]

        d_on_d_gk_list = new_sim_params.observables[self.n_obs:]  # these correspond to the A_kn operators

        for obs in d_on_d_gk_list:
            obs.results = trapezoidal(obs.results, self.sim_params.times)


        zero_obs=Observable(Zero(),0)
        zero_obs.results = np.zeros(self.n_t)

        d_on_d_gk = np.zeros((self.n_jump, self.n_obs), dtype=object)


        count = 0
        for i, lk in enumerate(self.noise_list):
            for j, on in enumerate(self.obs_list):
                if lk.sites == on.sites:
                    d_on_d_gk[i, j] = d_on_d_gk_list[count]
                    count += 1
                else:
                    d_on_d_gk[i, j] = zero_obs
        
        
        obs_array = np.array([obs.results for obs in original_exp_vals])

        d_on_d_gk_array = np.array([[d_on_d_gk[i, j].results for j in range(self.n_obs)] for i in range(self.n_jump)])

        return self.sim_params.times, obs_array, d_on_d_gk_array


