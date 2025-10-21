# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the propagation module's noise characterization functionality."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.noise_char import propagation
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z

from mqt.yaqs.core.data_structures.networks import MPO, MPS



class Parameters:
    def __init__(self) -> None:
        self.sites = 1
        self.sim_time = 0.6
        self.dt = 0.2
        self.order = 1
        self.threshold = 1e-4
        self.ntraj = 1
        self.max_bond_dim = 4
        self.j = 1
        self.g = 0.5


        self.times = np.arange(0, self.sim_time + self.dt, self.dt)

        self.n_obs = self.sites * 3  # x, y, z for each site
        self.n_jump = self.sites * 2  # lowering and pauli_z for each site
        self.n_t = len(self.times)

        self.gamma_rel = 0.1
        self.gamma_deph = 0.15


        self.d = 2



def create_propagator_instance(test: Parameters) -> tuple[MPO, MPS, list[Observable], AnalogSimParams, CompactNoiseModel, propagation.PropagatorWithGradients]:
    """Helper function to create Hamiltonian, observable list, and noise model for tests."""
    h_0 = MPO()
    h_0.init_ising(test.sites, test.j, test.g)


    # Define the initial state
    init_state = MPS(test.sites, state='zeros')


    obs_list = [Observable(X(), site) for site in range(test.sites)]  + [Observable(Y(), site) for site in range(test.sites)] + [Observable(Z(), site) for site in range(test.sites)]


    sim_params = AnalogSimParams(observables=obs_list, elapsed_time=test.sim_time, dt=test.dt, num_traj=test.ntraj, max_bond_dim=test.max_bond_dim, threshold=test.threshold, order=test.order, sample_timesteps=True)



    ref_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(test.sites)], "strength": test.gamma_rel}] + [{"name": "pauli_z", "sites": [i for i in range(test.sites)], "strength": test.gamma_deph}])


    propagator = propagation.PropagatorWithGradients(
            sim_params=sim_params,
            hamiltonian=h_0,
            compact_noise_model=ref_noise_model,
            init_state=init_state
        )


    return h_0, init_state, obs_list, sim_params, ref_noise_model, propagator


def test_propagatorwithgradients_runs() -> None:
    """Test that `propagation.tjm_traj` executes correctly and returns expected output shapes.

    This test verifies that:
    - The function can be called with a valid `SimulationParameters` instance.
    - The returned values `t`, `original_exp_vals`, and `d_on_d_gk` are NumPy arrays.
    - The shapes of the outputs match the expected dimensions based on simulation parameters.
    - The average minimum and maximum trajectory time is returned as a list of None values.
    """
    # Prepare SimulationParameters
    test = Parameters()



    _, _, obs_list, _, ref_noise_model, propagator = create_propagator_instance(test)
    
    propagator.set_observable_list(obs_list)
    propagator.run(ref_noise_model)



    assert isinstance(propagator.times, np.ndarray)
    assert isinstance(propagator.obs_array, np.ndarray)
    assert isinstance(propagator.d_on_d_gk_array, np.ndarray)



    assert propagator.times.shape == (test.n_t,)
    assert propagator.obs_array.shape == (test.n_obs, test.n_t)
    assert propagator.d_on_d_gk_array.shape == (test.n_jump, test.n_obs, test.n_t)




def test_raises_num_sites_hamiltonian() -> None:
    """Test that `PropagatorWithGradients` raises a ValueError when the number of sites in the Hamiltonian does not match the initial state.

    This test verifies that:
    - A ValueError is raised when the Hamiltonian's number of sites differs from that of the initial state.
    - The error message contains the expected text indicating the mismatch.
    """
    test = Parameters()


    h_0, init_state, _, sim_params, _, _ = create_propagator_instance(test)
    


    ref_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(test.sites+1)], "strength": test.gamma_rel}] + [{"name": "pauli_z", "sites": [i for i in range(test.sites)], "strength": test.gamma_deph}])


    with pytest.raises(ValueError, match="Noise site index exceeds number of sites in the Hamiltonian."):
        propagator = propagation.PropagatorWithGradients(
            sim_params=sim_params,
            hamiltonian=h_0,
            compact_noise_model=ref_noise_model,
            init_state=init_state
        )

    ref_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(test.sites)], "strength": test.gamma_rel}] + [{"name": "pauli_z", "sites": [i for i in range(test.sites)], "strength": test.gamma_deph}])

    obj_list = [Observable(X(), site) for site in range(test.sites)]  + [Observable(Y(), site) for site in range(test.sites)] + [Observable(Z(), site) for site in range(test.sites+1)] 


    with pytest.raises(ValueError, match="Observable site index exceeds number of sites in the Hamiltonian."):
        propagator = propagation.PropagatorWithGradients(
            sim_params=sim_params,
            hamiltonian=h_0,
            compact_noise_model=ref_noise_model,
            init_state=init_state
        )

        propagator.set_observable_list(obj_list)
    