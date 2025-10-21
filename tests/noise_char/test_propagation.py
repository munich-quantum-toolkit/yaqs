# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the propagation module's noise characterization functionality."""

from __future__ import annotations

import re

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.noise_char import propagation


class Parameters:
    """Container for default test parameters used in a lightweight open-quantum-system propagation test."""

    def __init__(self) -> None:
        """Initialize default test simulation parameters.

        This constructor sets up a collection of attributes used for running a simple
        open-quantum-system propagation test. Attributes and their meanings:
        - sites (int): Number of sites/spins. Default: 1.
        - sim_time (float): Total simulation time. Default: 0.6.
        - dt (float): Time step for propagation. Default: 0.2.
        - order (int): Integration/order parameter for the propagator. Default: 1.
        - threshold (float): Numerical/truncation tolerance used in algorithms. Default: 1e-4.
        - ntraj (int): Number of trajectories to average over (stochastic methods). Default: 1.
        - max_bond_dim (int): Maximum bond dimension for tensor-network representations. Default: 4.
        - j (float): Coupling constant used in the model Hamiltonian. Default: 1.
        - g (float): Local field (e.g., transverse field) parameter. Default: 0.5.
        - times (np.ndarray): 1-D array of time points computed as np.arange(0, sim_time + dt, dt).
        - n_obs (int): Number of observables (3 per site for Pauli x, y, z). Computed as sites * 3.
        - n_jump (int): Number of jump operators (2 per site, e.g., lowering and Pauli-Z). Computed as sites * 2.
        - n_t (int): Number of time points (len(times)).
        - gamma_rel (float): Relaxation (dissipative) rate. Default: 0.1.
        - gamma_deph (float): Dephasing rate. Default: 0.15.
        - d (int): Local Hilbert-space dimension (e.g., spin-1/2 -> 2). Default: 2.

        Notes:
        - The provided defaults are chosen for lightweight tests and can be modified
            on the instance after construction if different test scenarios are required.
        - The 'times' array explicitly includes the final time by using sim_time + dt
            as the stop value in np.arange.
        """
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


def create_propagator_instance(
    test: Parameters,
) -> tuple[MPO, MPS, list[Observable], AnalogSimParams, CompactNoiseModel, propagation.PropagatorWithGradients]:
    """Create and initialize a PropagatorWithGradients instance.

    It is configured for an analog open quantum system simulation.
    This helper constructs an Ising Hamiltonian (MPO), a zero-filled initial MPS, a list of single-site
    Pauli observables (X, Y, Z for each site), an AnalogSimParams object with sampling enabled, and a
    CompactNoiseModel containing two noise channels ("lowering" and "pauli_z"). It then instantiates a
    propagation.PropagatorWithGradients using those objects, registers the observable list with the
    propagator, and runs a propagation using the reference noise model.
    Parameters
    ----------
    test : Parameters
        A parameter bundle object required to configure the system. Expected attributes:
          - sites (int): number of lattice sites (spins).
          - j (float): Ising coupling strength used to initialize the MPO Hamiltonian.
          - g (float): transverse field strength used to initialize the MPO Hamiltonian.
          - sim_time (float): total simulation elapsed time.
          - dt (float): simulation time step.
          - ntraj (int): number of stochastic trajectories to sample.
          - max_bond_dim (int): maximum MPS/MPO bond dimension for truncation.
          - threshold (float): singular-value threshold for truncation.
          - order (int): Trotter/order parameter for the simulator.
          - gamma_rel (float): strength of the "lowering" (relaxation) noise channel applied to all sites.
          - gamma_deph (float): strength of the "pauli_z" (dephasing) noise channel applied to all sites.

    Returns:
    -------
    tuple[MPO, MPS, list[Observable], AnalogSimParams, CompactNoiseModel, propagation.PropagatorWithGradients]
        A 6-tuple containing, in order:
          - h_0: MPO
              The initialized Ising Hamiltonian MPO for the given system parameters.
          - init_state: MPS
              The initialized many-body state (all zeros).
          - obs_list: list[Observable]
              The list of single-site Observable objects (X, Y, Z for each site).
          - sim_params: AnalogSimParams
              The simulation parameter object used to configure the propagator (with sample_timesteps=True).
          - ref_noise_model: CompactNoiseModel
              The compact noise model containing the "lowering" and "pauli_z" channels applied to all sites.
          - propagator: propagation.PropagatorWithGradients
              The propagator instance after calling set_observable_list(...) and run(ref_noise_model). The
              propagator therefore has performed the configured propagation at least once.
    """
    h_0 = MPO()
    h_0.init_ising(test.sites, test.j, test.g)

    # Define the initial state
    init_state = MPS(test.sites, state="zeros")

    obs_list = (
        [Observable(X(), site) for site in range(test.sites)]
        + [Observable(Y(), site) for site in range(test.sites)]
        + [Observable(Z(), site) for site in range(test.sites)]
    )

    sim_params = AnalogSimParams(
        observables=obs_list,
        elapsed_time=test.sim_time,
        dt=test.dt,
        num_traj=test.ntraj,
        max_bond_dim=test.max_bond_dim,
        threshold=test.threshold,
        order=test.order,
        sample_timesteps=True,
    )

    ref_noise_model = CompactNoiseModel([
        {"name": "lowering", "sites": list(range(test.sites)), "strength": test.gamma_rel},
        {"name": "pauli_z", "sites": list(range(test.sites)), "strength": test.gamma_deph},
    ])

    propagator = propagation.PropagatorWithGradients(
        sim_params=sim_params, hamiltonian=h_0, compact_noise_model=ref_noise_model, init_state=init_state
    )

    propagator.set_observable_list(obs_list)
    propagator.run(ref_noise_model)

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


def test_raises_errors() -> None:
    """Test that `PropagatorWithGradients` raises a ValueError.

    This test verifies that:
    - A ValueError is raised when the Hamiltonian's number of sites differs from that of the initial state.
    - The error message contains the expected text indicating the mismatch.
    """
    test = Parameters()

    h_0, init_state, obs_list, sim_params, ref_noise_model, _ = create_propagator_instance(test)

    # Test that PropagatorWithGradients raises a ValueError when
    # the noise model exceeds the number of sites of the Hamiltonian.
    exceed_ref_noise_model = CompactNoiseModel([
        {"name": "lowering", "sites": list(range(test.sites + 1)), "strength": test.gamma_rel},
        {"name": "pauli_z", "sites": list(range(test.sites)), "strength": test.gamma_deph},
    ])

    msg = "Noise site index exceeds number of sites in the Hamiltonian."
    with pytest.raises(ValueError, match=re.escape(msg)):
        propagator = propagation.PropagatorWithGradients(
            sim_params=sim_params, hamiltonian=h_0, compact_noise_model=exceed_ref_noise_model, init_state=init_state
        )

    # Test that PropagatorWithGradients raises a ValueError when
    # observale list exceeds the number of sites of the Hamiltonian.
    exceed_obs_list = (
        [Observable(X(), site) for site in range(test.sites)]
        + [Observable(Y(), site) for site in range(test.sites)]
        + [Observable(Z(), site) for site in range(test.sites + 1)]
    )
    propagator = propagation.PropagatorWithGradients(
        sim_params=sim_params, hamiltonian=h_0, compact_noise_model=ref_noise_model, init_state=init_state
    )
    msg = "Observable site index exceeds number of sites in the Hamiltonian."
    with pytest.raises(ValueError, match=re.escape(msg)):
        propagator.set_observable_list(exceed_obs_list)

    # Test that PropagatorWithGradients raises a ValueError when
    # observale list is not set.
    propagator = propagation.PropagatorWithGradients(
        sim_params=sim_params, hamiltonian=h_0, compact_noise_model=ref_noise_model, init_state=init_state
    )
    msg = "Observable list not set. Please use the set_observable_list method to set the observables."

    with pytest.raises(ValueError, match=re.escape(msg)):
        propagator.run(ref_noise_model)

    # Test that PropagatorWithGradients raises a ValueError when
    # the provided noise model does not match the initialized noise model.
    wrong_ref_noise_model = CompactNoiseModel([
        {"name": "lowering", "sites": list(range(test.sites)), "strength": test.gamma_rel},
        {"name": "pauli_x", "sites": list(range(test.sites)), "strength": test.gamma_deph},
    ])

    propagator = propagation.PropagatorWithGradients(
        sim_params=sim_params, hamiltonian=h_0, compact_noise_model=ref_noise_model, init_state=init_state
    )
    propagator.set_observable_list(obs_list)

    msg = "Noise model processes or sites do not match the initialized noise model."
    with pytest.raises(ValueError, match=re.escape(msg)):
        propagator.run(wrong_ref_noise_model)
