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


def test_propagatorwithgradients_runs() -> None:
    """Test that `propagation.tjm_traj` executes correctly and returns expected output shapes.

    This test verifies that:
    - The function can be called with a valid `SimulationParameters` instance.
    - The returned values `t`, `original_exp_vals`, and `d_on_d_gk` are NumPy arrays.
    - The shapes of the outputs match the expected dimensions based on simulation parameters.
    - The average minimum and maximum trajectory time is returned as a list of None values.
    """
    # Prepare SimulationParameters
    j=1
    g=0.5

    sim_time = 1
    dt=0.5
    ntraj=2
    max_bond_dim=4
    threshold=1e-4
    order=1


    nsites = 2



    h_0 = MPO()
    h_0.init_ising(nsites, j, g)


    # Define the initial state
    init_state = MPS(nsites, state='zeros')


    obs_list = [Observable(X(), site) for site in range(nsites)]  + [Observable(Y(), site) for site in range(nsites)] + [Observable(Z(), site) for site in range(nsites)]


    sim_params = AnalogSimParams(observables=obs_list, elapsed_time=sim_time, dt=dt, num_traj=ntraj, max_bond_dim=max_bond_dim, threshold=threshold, order=order, sample_timesteps=True)


    ref_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(nsites)], "strength": 0.1}])



    propagator = propagation.PropagatorWithGradients(
            sim_params=sim_params,
            hamiltonian=h_0,
            compact_noise_model=ref_noise_model,
            init_state=init_state
        )
    
    propagator.set_observable_list(obs_list)
    propagator.run(ref_noise_model)



    assert isinstance(propagator.times, np.ndarray)
    assert isinstance(propagator.obs_array, np.ndarray)
    assert isinstance(propagator.d_on_d_gk_array, np.ndarray)

    n_obs = 3*nsites
    n_jump = nsites
    n_t = int(sim_time / dt) + 1

    assert propagator.times.shape == (n_t,)
    assert propagator.obs_array.shape == (n_obs, n_t)
    assert propagator.d_on_d_gk_array.shape == (n_jump, n_obs, n_t)

