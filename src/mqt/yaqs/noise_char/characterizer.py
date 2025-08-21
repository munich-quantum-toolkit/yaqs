# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains the optimization routines for noise characterization."""

from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.noise_char.optimization import LossClass





class Characterizer:


    sim_params: AnalogSimParams

    hamiltonian: MPO

    noise_model: NoiseModel

    loss_function: LossClass


    




    def set_hamiltonian():


    
    def set_noise_model():



    def set_loss_function(*, ):
        loss_function = LossFunction(
            observables=observables,
            target_values=target_values,
            weights=weights,
            noise_model=noise_model,
        )


    def set_propagator():
    

    def set_sim_params(*, ):
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


    def set_adam_optimizer():

    


    def run():
