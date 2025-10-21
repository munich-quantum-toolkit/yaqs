# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Performs the simulation of the Ising model and returns expectations values and  A_kn trahectories."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import GateLibrary, Zero
from mqt.yaqs.noise_char.optimization import trapezoidal

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.networks import MPO, MPS
    from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel, NoiseModel


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
        noise_list.extend(Observable(gate(), site) for site in proc["sites"])
    return noise_list


class PropagatorWithGradients:
    r"""High-level propagator that runs an MPS-based Lindblad simulation.

    The class wraps simulator inputs, performs
    consistency checks between noise models and the Hamiltonian, augments the
    observable set with Lindblad-derived A_kn operators (sensitivities of
    expectation values w.r.t. jump rates), runs the underlying simulator, and
    post-processes simulator outputs into convenient arrays for analysis.

    Attributes:
    obs_list : list[Observable]
        (Set after set_observable_list) Deep copy of user-provided observables.
    n_obs : int
        (Set after set_observable_list) Number of observables.
    times : array-like
        Time grid used by the most recent run (copied from sim_params.times).
    obs_traj : list[Observable]
        Observables returned by the simulator corresponding to the original
        user-requested observables (populated by run).
    obs_array : numpy.ndarray
        Array of observable trajectories with shape (n_obs, n_timesteps).
    d_on_d_gk : numpy.ndarray
        Object-array of Observable entries (shape [n_jump, n_obs]) corresponding
        to A_kn-like operators (or zero placeholders) computed by the simulator
        and integrated in time.
    d_on_d_gk_array : numpy.ndarray
        Numeric array of integrated A_kn trajectories (shape [n_jump, n_obs]).
    Other internal fields may be set during execution (e.g., temporary lists and
    simulator-specific containers).
    Public methods
    set_observable_list(obs_list: list[Observable]) -> None
        Store and validate a deep copy of obs_list. Validates that every site index
        referenced by the observables is within the range [0, sites-1]. Sets
        n_obs and flips set_observables to True. Raises ValueError for empty lists
        or out-of-range site indices.
    run(noise_model: CompactNoiseModel) -> None
        Execute the propagation. Requires that set_observables has been called.
        Validates that the provided compact noise_model matches the one used to
        construct this propagator (same process names and site assignments).
        Constructs A_kn-like observables for each matching jump/operator pair,
        appends them to the observable list, builds a new AnalogSimParams instance
        for the simulator, and invokes the underlying simulator with the expanded
        noise model. Post-processes results by trapezoidally integrating A_kn
        trajectories, arranging them into object and numeric arrays (d_on_d_gk and
        d_on_d_gk_array) and extracting obs_traj and obs_array for the original
        observables.
        - During initialization: if any site index in compact_noise_model.expanded_noise_model
          exceeds the number of sites in the Hamiltonian.
        - set_observable_list: if obs_list is empty or contains observables that
          reference out-of-range site indices.
        - run: if observables have not been set (set_observables is False) or if the
          provided noise_model does not match the initialized compact_noise_model
          in process names or site indices.
    - All constructor inputs are deep-copied to avoid accidental external mutation.
    - The class expects external types (AnalogSimParams, MPO, MPS, CompactNoiseModel,
      Observable) to expose particular attributes (for example, `times`, `length`,
      `expanded_noise_model`, `compact_processes`, `gate`, `sites`, and `results`).
    - The A_kn operators constructed in run follow the Lindblad derivative form:
      L_k^\dagger O L_k - 0.5 {L_k^\dagger L_k, O}, computed only for observables
      that act on the same site(s) as the corresponding jump operator.
    - The user-facing numeric arrays (obs_array and d_on_d_gk_array) are convenient
      summaries for optimization or analysis tasks (e.g., gradient-based fitting of
      jump rates).
    """

    def __init__(
        self,
        *,
        sim_params: AnalogSimParams,
        hamiltonian: MPO,
        compact_noise_model: CompactNoiseModel,
        init_state: MPS,
    ) -> None:
        """Initialize a Propagation object for simulating open quantum system dynamics.

        This constructor deep-copies the provided inputs and derives internal
        structures needed for propagation of an MPS under a Hamiltonian with
        a compact noise model.
        Parameters.
        ----------
        sim_params : AnalogSimParams
            Simulation parameters container. A deep copy is stored as
            self.sim_params. It is expected to provide a sequence/array
            `times` used to determine the number of time steps.
        hamiltonian : MPO
            Matrix product operator representing the Hamiltonian. A deep copy
            is stored as self.hamiltonian. The MPO must expose a `length`
            attribute indicating the number of sites.
        compact_noise_model : CompactNoiseModel
            Compact representation of the noise model. A deep copy is stored
            as self.compact_noise_model. Its `expanded_noise_model` attribute
            is deep-copied to self.expanded_noise_model and converted into a
            list of jump operators.
        init_state : MPS
            Initial many-body quantum state as a matrix product state.
            A deep copy is stored as self.init_state.
        Attributes set
        --------------
        sim_params : AnalogSimParams
            Deep copy of the provided simulation parameters.
        hamiltonian : MPO
            Deep copy of the provided Hamiltonian MPO.
        compact_noise_model : CompactNoiseModel
            Deep copy of the provided compact noise model.
        init_state : MPS
            Deep copy of the provided initial state.
        expanded_noise_model
            Deep copy of compact_noise_model.expanded_noise_model.
        noise_list : list[Observable]
            List of noise (jump) operators produced by converting the expanded
            noise model via noise_model_to_operator_list.
        n_jump : int
            Number of jump operators (len(self.noise_list)).
        n_t : int
            Number of time steps (len(self.sim_params.times)).
        sites : int
            Number of sites in the chain (self.hamiltonian.length).
        set_observables : bool
            Flag indicating whether observables have been set (initialized to False).

        Raises:
        ------
        ValueError: If any site index referenced in expanded_noise_model.processes is
            greater than or equal to the number of sites in the Hamiltonian,
            a ValueError is raised with the message
            "Noise site index exceeds number of sites in the Hamiltonian."

        Notes:
        -----
        - All provided inputs are deep-copied to avoid accidental external mutation.
        - This method performs basic consistency checking between the noise
          model and the Hamiltonian site count.
        """
        self.sim_params: AnalogSimParams = copy.deepcopy(sim_params)
        self.hamiltonian: MPO = copy.deepcopy(hamiltonian)
        self.compact_noise_model: CompactNoiseModel = copy.deepcopy(compact_noise_model)
        self.init_state: MPS = copy.deepcopy(init_state)

        self.expanded_noise_model = copy.deepcopy(self.compact_noise_model.expanded_noise_model)

        self.noise_list: list[Observable] = noise_model_to_operator_list(self.expanded_noise_model)

        self.n_jump: int = len(self.noise_list)  # number of jump operators

        self.n_t: int = len(self.sim_params.times)  # number of time steps

        self.sites: int = self.hamiltonian.length  # number of sites in the chain

        self.set_observables: bool = False

        if max(proc["sites"][0] for proc in self.expanded_noise_model.processes) >= self.sites:
            msg = "Noise site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)

    def set_observable_list(self, obs_list: list[Observable]) -> None:
        """Set the list of observables to be used for propagation.

        This method stores a deep copy of the provided observable list on the instance,
        validates that all referenced site indices lie within the allowed range of the
        Hamiltonian, and updates bookkeeping attributes.

        Args:
            obs_list (list[Observable]): Sequence of Observable objects. Each Observable
                must expose a `sites` attribute that is either an int (single site) or
                a list of ints (multiple sites).
        Side effects:
            - self.obs_list is set to a deep copy of obs_list.
            - self.n_obs is set to the number of observables (len(self.obs_list)).
            - self.set_observables is set to True.

        Raises:
            ValueError: If any site index in the observables is greater than or equal
                to self.sites (i.e., outside the range of available sites).
            ValueError: If obs_list is empty (which makes site-index validation via max()
                impossible) or if observables do not provide valid site information.
        """
        self.obs_list = copy.deepcopy(obs_list)

        all_obs_sites = [
            site for obs in obs_list for site in (obs.sites if isinstance(obs.sites, list) else [obs.sites])
        ]

        if max(all_obs_sites) >= self.sites:
            msg = "Observable site index exceeds number of sites in the Hamiltonian."
            raise ValueError(msg)

        self.n_obs = len(self.obs_list)  # number of measurement operators

        self.set_observables = True

    def run(self, noise_model: CompactNoiseModel) -> None:
        """Run the propagation routine with augmented Lindblad-derived operators.

        Parameters
        ----------
        noise_model : CompactNoiseModel
            The compact representation of the noise model to use for propagation.
            The method verifies that the list of compact processes and their sites
            in `noise_model` match the model used to initialize this propagator
            (self.compact_noise_model). The expanded form of this model is passed
            to the underlying simulator.

        Side effects / State changes
        ----------------------------
        On successful completion, several attributes of self are set or updated:
        - self.obs_traj : list[Observable]
            The list of original observables (with their computed time trajectories)
            extracted from the simulator results.
        - self.d_on_d_gk : numpy.ndarray of shape (n_jump, n_obs) with Observable entries
            A matrix of the A_kn-like operators (or zero placeholders) corresponding
            to each jump operator / observable pair; entries are Observable objects
            whose .results have been integrated (trapezoidally) over time.
        - self.d_on_d_gk_array : numpy.ndarray
            2D array of numeric trajectories corresponding to d_on_d_gk (shape
            [n_jump, n_obs, n_timesteps]).
        - self.obs_array : numpy.ndarray
            2D array of numeric trajectories for the original observables
            (shape [n_obs, n_timesteps]).
        - self.times : array-like
            Time grid used by the simulation (copied from self.sim_params.times).

        Raises:
            ValueError: If the observable list has not been initialized (self.set_observables is False).
            ValueError: If any process name or site in the provided noise_model does not match
              the corresponding entry in self.compact_noise_model.

        Notes:
        -----
        - The purpose of the added A_kn observables is to provide sensitivity-like
          quantities (derivatives of observable expectations with respect to
          jump rates) that are computed by the same underlying simulator and then
          post-processed into arrays suitable for analysis or parameter updates.
        """
        if not self.set_observables:
            msg = "Observable list not set. Please use the set_observable_list method to set the observables."
            raise ValueError(msg)

        for i, proc in enumerate(noise_model.compact_processes):
            for j, site in enumerate(proc["sites"]):
                if (
                    proc["name"] != self.compact_noise_model.compact_processes[i]["name"]
                    or site != self.compact_noise_model.compact_processes[i]["sites"][j]
                ):
                    msg = "Noise model processes or sites do not match the initialized noise model."
                    raise ValueError(msg)

        a_kn_site_list: list[Observable] = []

        for lk in self.noise_list:
            a_kn_site_list.extend(
                Observable(
                    lk.gate.dag() * on.gate * lk.gate
                    - 0.5 * on.gate * lk.gate.dag() * lk.gate
                    - 0.5 * lk.gate.dag() * lk.gate * on.gate,
                    lk.sites,
                )
                for on in self.obs_list
                if lk.sites == on.sites
            )

        new_obs_list = self.obs_list + a_kn_site_list

        new_sim_params = AnalogSimParams(
            observables=new_obs_list,
            elapsed_time=self.sim_params.elapsed_time,
            dt=self.sim_params.dt,
            num_traj=self.sim_params.num_traj,
            max_bond_dim=self.sim_params.max_bond_dim,
            threshold=self.sim_params.threshold,
            order=self.sim_params.order,
            sample_timesteps=True,
        )

        simulator.run(self.init_state, self.hamiltonian, new_sim_params, noise_model.expanded_noise_model)

        # Separate original and new expectation values from result_lindblad.
        self.obs_traj = new_sim_params.observables[: self.n_obs]

        d_on_d_gk_list = new_sim_params.observables[self.n_obs :]  # these correspond to the A_kn operators

        for obs in d_on_d_gk_list:
            obs.results = trapezoidal(obs.results, self.sim_params.times)

        zero_obs = Observable(Zero(), 0)
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

        self.times = self.sim_params.times

        self.obs_array = np.array([obs.results for obs in self.obs_traj])

        self.d_on_d_gk_array = np.array([
            [d_on_d_gk[i, j].results for j in range(self.n_obs)]
            for i in range(self.n_jump)
])
