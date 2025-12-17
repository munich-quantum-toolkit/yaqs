# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Unit tests for the characterizer module's noise characterization functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.noise_char import propagation
from mqt.yaqs.noise_char.characterizer import Characterizer
from mqt.yaqs.noise_char.loss import LossClass

if TYPE_CHECKING:
    from pathlib import Path

    from mqt.yaqs.noise_char.propagation import Propagator


class Parameters:
    """Container for default test parameters used in a lightweight open-quantum-system propagation test."""

    def __init__(self) -> None:
        """Initialize default parameters for the test optimization.

        This constructor sets up default simulation and model parameters used by the
        test case, including system size, time discretization, numerical tolerances,
        noise rates, and derived quantities (time grid and counts of observables/jumps).

        Parameters
        ----------
        None

        Attributes:
        ----------
        sites : int
            Number of sites in the system.
        sim_time : float
            Total simulation time.
        dt : float
            Time step size.
        order : int
            Numerical order (e.g., integration or algorithm order).
        threshold : float
            Convergence or truncation threshold (tolerance).
        ntraj : int
            Number of trajectories (for stochastic/unraveled simulations).
        max_bond_dim : int
            Maximum bond dimension for tensor network methods.
        j : float
            Hamiltonian coupling constant (e.g., exchange or hopping strength).
        g : float
            System parameter (e.g., field or coupling strength).

        times : numpy.ndarray
            1D array of time points from 0 to ``sim_time`` inclusive with spacing ``dt``.
        n_obs : int
            Number of observables tracked (here 3 per site: x, y, z).
        n_jump : int
            Number of jump operators per site (here lowering and pauli_z per site).
        n_t : int
            Number of time points (length of ``times``).

        gamma_rel : float
            Relaxation (dissipative) rate.
        gamma_deph : float
            Dephasing rate.

        d : int
            Local Hilbert space dimension (e.g., spin-1/2 -> 2).
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


def create_instances(
    test: Parameters, tmp_path: Path
) -> tuple[MPO, MPS, list[Observable], AnalogSimParams, CompactNoiseModel, Propagator, LossClass]:
    """Create and initialize objects required for an analog open-quantum-system simulation.

    This helper constructs an Ising Hamiltonian MPO, an initial MPS, a list of single-site
    observables (X, Y, Z for each site), simulation parameters, a compact noise model
    (with lowering and dephasing channels), and a Propagator instance.
    The propagator is configured with the observables and then executed (propagator.run)
    with the constructed noise model before being returned.
    Parameters.
    ----------
    test : Parameters
        Test parameter object providing simulation settings. Expected attributes:
        - sites (int): number of lattice sites
        - j (float): Ising interaction strength
        - g (float): transverse field strength
        - sim_time (float): total simulation time
        - dt (float): integration timestep
        - ntraj (int): number of trajectories / stochastic samples
        - max_bond_dim (int): maximum bond dimension for tensor-network objects
        - threshold (float): truncation threshold for tensor-network operations
        - order (int): integrator order (if applicable)
        - gamma_rel (float): relaxation (lowering) noise strength
        - gamma_deph (float): dephasing (Pauli-Z) noise strength
    tmp_path : Path
        Temporary directory path for any required output (not used in current implementation).

    Returns:
    -------
    tuple[MPO, MPS, list[Observable], AnalogSimParams, CompactNoiseModel, Propagator]
        A 6-tuple containing:
        - h_0 (MPO): Ising Hamiltonian MPO initialized with (sites, j, g).
        - init_state (MPS): initial MPS in the "zeros" product state.
        - obs_list (list[Observable]): list of single-site X, Y and Z observables for all sites.
        - sim_params (AnalogSimParams): simulation parameters assembled from `test` and `obs_list`.
        - ref_noise_model (CompactNoiseModel): compact noise model with "lowering" and "pauli_z"
          channels applied to all sites, using strengths `gamma_rel` and `gamma_deph`.
        - propagator (Propagator): propagator configured with the Hamiltonian,
          noise model and initial state; observables are set and the propagation has been run
          with `ref_noise_model` prior to return.

    Notes:
    -----
    - The function has the side effect of running the propagator (propagator.run(...)),
      so returned `propagator` already contains results of the propagation.
    - `tmp_path` is present to match common pytest fixture signatures but is not used
      in the current implementation; future versions may use it for temporary output.

    Examples:
    --------
    >>> # Typical usage in a test:
    >>> h_0, init_state, obs_list, sim_params, ref_noise_model, propagator = create_instances(tmp_path, test)
    >>> # After this call, `propagator` has been run and the constructed objects are available for assertions.
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

    propagator = propagation.Propagator(
        sim_params=sim_params, hamiltonian=h_0, compact_noise_model=ref_noise_model, init_state=init_state
    )

    propagator.set_observable_list(obs_list)
    propagator.run(ref_noise_model)

    loss = LossClass(ref_traj=propagator.obs_traj, traj_gradients=propagator, working_dir=tmp_path, print_to_file=False)

    return h_0, init_state, obs_list, sim_params, ref_noise_model, propagator, loss


def test_characterizer_init(tmp_path: Path) -> None:
    """Test that Characterizer initializes correctly with given parameters."""
    test = Parameters()

    _h_0, _init_state, _obs_list, _sim_params, ref_noise_model, propagator, loss = create_instances(test, tmp_path)

    characterizer = Characterizer(
        traj_gradients=propagator,
        init_guess=ref_noise_model,
        loss=loss,
    )

    assert isinstance(characterizer.init_guess, CompactNoiseModel)
    assert isinstance(characterizer.traj_gradients, propagation.Propagator)
    assert isinstance(characterizer.loss, LossClass)
    assert isinstance(characterizer.init_x, np.ndarray)

    characterizer.adam_optimize(max_iterations=1)

    assert isinstance(characterizer.optimal_model, CompactNoiseModel)
