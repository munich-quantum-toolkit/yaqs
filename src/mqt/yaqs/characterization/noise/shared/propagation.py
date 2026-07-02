# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Forward-model propagation for Markovian noise characterization."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.simulator import Simulator

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
    from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import Observable
    from mqt.yaqs.core.data_structures.state import State


class Propagator:
    """Run Lindblad simulations and collect observable trajectories."""

    def __init__(
        self,
        *,
        sim_params: AnalogSimParams,
        hamiltonian: Hamiltonian,
        compact_noise_model: CompactNoiseModel,
        init_state: State,
        simulator: Simulator | None = None,
    ) -> None:
        """Store simulation inputs and validate site indices.

        Args:
            sim_params: Base analog simulation parameters (observables may be empty).
            hamiltonian: System Hamiltonian.
            compact_noise_model: Compact noise model whose topology is fixed during fitting.
            init_state: Initial state for propagation.
            simulator: Optional :class:`~mqt.yaqs.Simulator` instance.

        Raises:
            ValueError: If a noise site index exceeds the Hamiltonian length.
        """
        self.sim_params = copy.deepcopy(sim_params)
        self.hamiltonian = copy.deepcopy(hamiltonian)
        self.compact_noise_model = copy.deepcopy(compact_noise_model)
        self.init_state = copy.deepcopy(init_state)
        self._simulator = simulator or Simulator(show_progress=False)

        self.expanded_noise_model = copy.deepcopy(self.compact_noise_model.expanded_noise_model)
        self.sites = self.hamiltonian.length
        self.obs_list: list[Observable] = []
        self.n_obs = 0
        self.set_observables = False
        self.times = np.asarray(self.sim_params.times, dtype=float)
        self.obs_array = np.empty((0, len(self.times)))

        if self.expanded_noise_model.processes:
            max_site = max(max(proc["sites"]) for proc in self.expanded_noise_model.processes)
            if max_site >= self.sites:
                msg = "Noise site index exceeds number of sites in the Hamiltonian."
                raise ValueError(msg)

    def set_observable_list(self, obs_list: list[Observable]) -> None:
        """Register observables whose trajectories will be simulated.

        Args:
            obs_list: Observables to track during propagation.

        Raises:
            ValueError: If any observable references an out-of-range site.
        """
        if not obs_list:
            msg = "Observable list must not be empty."
            raise ValueError(msg)

        self.obs_list = copy.deepcopy(obs_list)
        all_obs_sites = [
            site for obs in obs_list for site in (obs.sites if isinstance(obs.sites, list) else [obs.sites])
        ]
        if max(all_obs_sites) >= self.sites:
            msg = "Observable site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)

        self.n_obs = len(self.obs_list)
        self.set_observables = True

    def run(self, noise_model: CompactNoiseModel) -> None:
        """Propagate under the supplied compact noise strengths.

        Args:
            noise_model: Candidate compact noise model with updated strengths.

        Raises:
            ValueError: If observables were not set or the topology changed.
        """
        if not self.set_observables:
            msg = "Observable list not set. Call set_observable_list first."
            raise ValueError(msg)

        for i, proc in enumerate(noise_model.compact_processes):
            ref = self.compact_noise_model.compact_processes[i]
            if proc["name"] != ref["name"] or proc["sites"] != ref["sites"]:
                msg = "Noise model topology does not match the initialized compact model."
                raise ValueError(msg)

        run_params = AnalogSimParams(
            observables=self.obs_list,
            elapsed_time=self.sim_params.elapsed_time,
            dt=self.sim_params.dt,
            num_traj=self.sim_params.num_traj,
            max_bond_dim=self.sim_params.max_bond_dim,
            svd_threshold=self.sim_params.svd_threshold,
            order=self.sim_params.order,
            sample_timesteps=True,
            random_seed=self.sim_params.random_seed,
        )
        result = self._simulator.run(
            self.init_state,
            self.hamiltonian,
            run_params,
            noise_model.expanded_noise_model,
        )
        self.times = np.asarray(run_params.times, dtype=float)
        self.obs_array = np.asarray(result.expectation_values, dtype=float)
