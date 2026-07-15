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
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import Observable
    from mqt.yaqs.core.data_structures.state import State


def _propagation_run_params(
    base: AnalogSimParams,
    observables: list[Observable],
) -> AnalogSimParams:
    """Clone ``base`` simulation parameters with a new observable list.

    Args:
        base: Template simulation parameters.
        observables: Observables to attach to the cloned parameters.

    Returns:
        Fresh :class:`AnalogSimParams` for a single propagation call.
    """
    return AnalogSimParams(
        observables=observables,
        elapsed_time=base.elapsed_time,
        dt=base.dt,
        num_traj=base.num_traj,
        max_bond_dim=base.max_bond_dim,
        trunc_mode=base.trunc_mode,
        svd_threshold=base.svd_threshold,
        krylov_tol=base.krylov_tol,
        order=base.order,
        preset=base.preset,
        sample_timesteps=base.sample_timesteps,
        evolution_mode=base.evolution_mode,
        get_state=base.get_state,
        random_seed=base.random_seed,
        multi_time_observables=base.multi_time_observables,
        tdvp_sweeps=base.tdvp_sweeps,
        tdvp_mode=base.tdvp_mode,
    )


class Propagator:
    """Run Lindblad simulations and collect observable trajectories."""

    def __init__(
        self,
        *,
        sim_params: AnalogSimParams,
        hamiltonian: Hamiltonian,
        noise_model: NoiseModel,
        init_state: State,
        simulator: Simulator | None = None,
    ) -> None:
        """Store simulation inputs and validate site indices.

        Args:
            sim_params: Base analog simulation parameters (observables may be empty).
            hamiltonian: System Hamiltonian.
            noise_model: Noise model whose topology is fixed during fitting.
            init_state: Initial state for propagation (already encoded for the target backend).
            simulator: Optional :class:`~mqt.yaqs.Simulator` instance.

        Raises:
            ValueError: If a noise site index exceeds the Hamiltonian length.
        """
        self.sim_params = sim_params
        self.hamiltonian = hamiltonian
        self.noise_model = copy.deepcopy(noise_model)
        self.init_state = init_state
        self.representation = init_state.representation
        self._simulator = simulator or Simulator(show_progress=False)

        self.sites = self.hamiltonian.length
        self.obs_list: list[Observable] = []
        self.set_observables = False
        self.times = np.asarray(self.sim_params.times, dtype=float)
        self.obs_array = np.empty((0, len(self.times)))

        if self.noise_model.processes:
            max_site = max(max(proc["sites"]) for proc in self.noise_model.processes)
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

        self.obs_list = list(obs_list)
        all_obs_sites = [
            site for obs in obs_list for site in (obs.sites if isinstance(obs.sites, list) else [obs.sites])
        ]
        if max(all_obs_sites) >= self.sites:
            msg = "Observable site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)

        self.set_observables = True

    def run(self, noise_model: NoiseModel) -> None:
        """Propagate under the supplied noise strengths.

        For ``density_matrix`` states the Simulator uses deterministic Lindblad evolution and
        ignores ``num_traj``. Stochastic MCWF/TJM paths average over ``num_traj`` trajectories.

        Args:
            noise_model: Candidate noise model with updated strengths.

        Raises:
            ValueError: If observables were not set or the topology changed.
        """
        if not self.set_observables:
            msg = "Observable list not set. Call set_observable_list first."
            raise ValueError(msg)

        if len(noise_model.processes) != len(self.noise_model.processes):
            msg = "Noise model topology does not match the initialized model."
            raise ValueError(msg)

        for i, proc in enumerate(noise_model.processes):
            ref = self.noise_model.processes[i]
            if proc["name"] != ref["name"] or list(proc["sites"]) != list(ref["sites"]):
                msg = "Noise model topology does not match the initialized model."
                raise ValueError(msg)

        run_params = _propagation_run_params(self.sim_params, self.obs_list)
        result = self._simulator.run(
            self.init_state,
            self.hamiltonian,
            run_params,
            noise_model,
        )
        self.times = np.asarray(run_params.times, dtype=float)
        self.obs_array = np.asarray(result.expectation_values, dtype=float)
