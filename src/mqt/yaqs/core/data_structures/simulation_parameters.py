# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Simulation Parameters for each type of simulation allowed in YAQS.

This module provides classes for representing observables and simulation parameters
for quantum simulations. It defines the Observable class for measurement, as well as
the PhysicsSimParams, WeakSimParams, and StrongSimParams classes for configuring simulation
runs. These classes encapsulate settings such as simulation time, time steps, bond dimension limits,
thresholds, and window sizes, and they include methods for aggregating simulation results.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from ..libraries.observables_library import ObservablesLibrary

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.networks import MPS


class TensorEvolMode(Enum):
    """Enumerates the different modes of tensor evolution in the simulation."""

    TDVP = "tdvp"
    BUG = "bug"


class Observable:
    """Observable class.

    A class to represent an observable in a quantum simulation.

    Attributes:
    ----------
    name : str
        The name of the observable, which must be a valid attribute in the GateLibrary.
    site : int
        The site (or qubit) on which the observable is measured.
    results : NDArray[np.float64] | None
        The results of the simulation, initialized to None.
    trajectories : NDArray[np.float64] | None
        The trajectories of the simulation, initialized to None.

    Methods:
    -------
    __init__(name: str, site: int) -> None
        Initializes the Observable with a name and site, and checks if the name is valid in the GateLibrary.
    initialize(sim_params: PhysicsSimParams | StrongSimParams | WeakSimParams) -> None
        Initializes the results and trajectories arrays based on the type of simulation parameters provided.
    """

    def __init__(self, name: str, site: int) -> None:
        """Initializes an Observable instance.

        Parameters
        ----------
        name : str
            The name of the observable. Must correspond to a valid gate or operator in GateLibrary.
        site : int
            The qubit or site index on which this observable is measured.

        Raises:
        ------
        AssertionError
            If the provided `name` is not a valid attribute in the GateLibrary.
        """
        assert name in ObservablesLibrary
        self.name = name
        self.site = site
        self.results: NDArray[np.float64] | None = None
        self.trajectories: NDArray[np.float64] | None = None

    def initialize(self, sim_params: PhysicsSimParams | StrongSimParams | WeakSimParams) -> None:
        """Observable initialization before simulation.

        Initialize the observables based on the type of simulation.
        Parameters:
        sim_params (PhysicsSimParams | StrongSimParams | WeakSimParams): The simulation parameters object
        which can be of type PhysicsSimParams, StrongSimParams, or WeakSimParams.
        """
        if isinstance(sim_params, PhysicsSimParams):
            if sim_params.sample_timesteps:
                self.trajectories = np.empty((sim_params.num_traj, len(sim_params.times)), dtype=np.float64)
                self.times = sim_params.times
            else:
                self.trajectories = np.empty((sim_params.num_traj, 1), dtype=np.float64)
                self.times = sim_params.elapsed_time
            self.results = np.empty(len(sim_params.times), dtype=np.float64)
        elif isinstance(sim_params, WeakSimParams):
            self.trajectories = np.empty((sim_params.shots, 1), dtype=np.complex128)
            self.results = np.empty(1, dtype=np.float64)
        elif isinstance(sim_params, StrongSimParams):
            self.trajectories = np.empty((sim_params.num_traj, 1), dtype=np.complex128)
            self.results = np.empty(1, dtype=np.float64)


class PhysicsSimParams:
    """Hamiltonian Simulation Parameters.

    A class to represent the parameters for a physics simulation.

    Attributes:
    -----------
    observables : list[Observable]
        A list of observables to be tracked during the simulation.
    sorted_observables : list[Observable]
        A list of observables sorted by site and name.
    elapsed_time : float
        The total time for the simulation.
    dt : float, optional
        The time step for the simulation (default is 0.1).
    times : numpy.ndarray
        An array of time points from 0 to T with step dt.
    sample_timesteps : bool, optional
        A flag to indicate whether to sample timesteps (default is True).
    num_traj : int, optional
        The number of samples to be taken (default is 1000).
    max_bond_dim : int, optional
        The maximum bond dimension (default is 2).
    threshold : float, optional
        The threshold value for the simulation (default is 1e-6).
    order : int, optional
        The order of the simulation (default is 1).
    get_state: bool, optional
        If True, output MPS is returned.

    Methods:
    --------
    aggregate_trajectories() -> None:
        Aggregates the trajectories of the observables by computing their mean.
    """

    output_state: MPS | None = None

    def __init__(
        self,
        observables: list[Observable],
        elapsed_time: float,
        dt: float = 0.1,
        num_traj: int = 1000,
        max_bond_dim: int = 2,
        threshold: float = 1e-6,
        order: int = 1,
        *,
        sample_timesteps: bool = True,
        tensorevol_mode: TensorEvolMode = TensorEvolMode.TDVP,
        get_state: bool = False,
    ) -> None:
        """Physics simulation parameters initialization.

        Initializes parameters for a physics-based quantum simulation.

        Parameters
        ----------
        observables : list[Observable]
            List of observables to measure during the simulation.
        elapsed_time : float
            Total simulation time.
        dt : float, optional
            Time step interval, by default 0.1.
        num_traj : int, optional
            Number of simulation samples, by default 1000.
        max_bond_dim : int, optional
            Maximum bond dimension allowed, by default 2.
        threshold : float, optional
            Threshold for simulation accuracy, by default 1e-6.
        order : int, optional
            Order of approximation or numerical scheme, by default 1.
        sample_timesteps : bool, optional
            Flag indicating whether to sample at intermediate time steps, by default True.
        tensorevol_mode : TensorEvolMode, optional
            Mode of tensor evolution in the simulation, by default TensorEvolMode.TDVP.
        get_state : bool, optional
            If True, output MPS is returned.
        """
        self.observables = observables
        self.sorted_observables = sorted(observables, key=lambda obs: (obs.site, obs.name))
        self.elapsed_time = elapsed_time
        self.dt = dt
        self.times = np.arange(0, elapsed_time + dt, dt)
        self.sample_timesteps = sample_timesteps
        self.num_traj = num_traj
        self.max_bond_dim = max_bond_dim
        self.threshold = threshold
        self.order = order
        self.tensorevol_mode = tensorevol_mode
        self.get_state = get_state

    def aggregate_trajectories(self) -> None:
        """Aggregates trajectories for result.

        Aggregates the trajectories of each observable by computing the mean
        across all trajectories and storing the result in the observable's results.
        This method iterates over all observables and updates their results
        attribute with the mean value of their trajectories along the specified axis.
        """
        for observable in self.observables:
            observable.results = np.mean(observable.trajectories, axis=0)


class WeakSimParams:
    """A class to represent the parameters for a weak simulation.

    Attributes:
    -----------
    dt : int
        A placeholder property for code compatibility.
    num_traj : int
        A placeholder property for code compatibility.
    shots : int
        The number of shots for the simulation.
    max_bond_dim : int
        The maximum bond dimension for the simulation.
    threshold : float
        The threshold value for the simulation.
    window_size : int | None
        The window size for the simulation.
    get_state:
        If True, output MPS is returned.

    Methods:
    --------
    __init__(shots: int, max_bond_dim: int = 2, threshold: float = 1e-6, window_size: int | None = None) -> None
        Initializes the WeakSimParams with the given parameters.
    aggregate_measurements() -> None
        Aggregates the measurements from the simulation.
    """

    # Properties set as placeholders for code compatibility
    dt = 1
    num_traj = 0
    output_state: MPS | None = None

    def __init__(
        self,
        shots: int,
        max_bond_dim: int = 2,
        threshold: float = 1e-6,
        window_size: int | None = 0,
        *,
        get_state: bool = False,
    ) -> None:
        """Weak circuit simulation initialization.

        Initializes parameters for a weak circuit simulation.

        Parameters
        ----------
        shots : int
            Number of measurement shots to simulate.
        max_bond_dim : int, optional
            Maximum bond dimension for simulation, by default 2.
        threshold : float, optional
            Accuracy threshold for truncating tensors, by default 1e-6.
        window_size : int or None, optional
            Window size for the simulation algorithm, by default None.
        get_state:
            If True, output MPS is returned.
        """
        self.measurements: list[dict[int, int] | None] = [None] * shots
        self.shots = shots
        self.max_bond_dim = max_bond_dim
        self.threshold = threshold
        self.window_size = window_size
        self.get_state = get_state

    def aggregate_measurements(self) -> None:
        """Aggregates shots into final result.

        Aggregates measurement results from multiple simulations.
        This method processes the `measurements` attribute, which is a list of dictionaries
        containing measurement results. If the first element of `measurements` is `None`,
        it assumes a noise-free simulation and directly uses the first element as the results.
        Otherwise, it aggregates the results from all non-None dictionaries in the list.
        The aggregated results are stored in the `results` attribute, which is a dictionary
        mapping measurement outcomes to their respective counts. The results are sorted
        by the measurement outcomes.
        """
        self.results: dict[int, int] = {}
        # Noise-free simulation stores shots in first element
        if None in self.measurements:
            assert self.measurements[0] is not None
            self.results = self.measurements[0]
            self.results = dict(sorted(self.results.items()))

        else:
            for d in filter(None, self.measurements):
                for key, value in d.items():
                    self.results[key] = self.results.get(key, 0) + value
            self.results = dict(sorted(self.results.items()))


class StrongSimParams:
    """Strong Circuit Simulation Parameters.

    A class to represent the parameters for a strong simulation.

    Attributes:
    -----------
    dt : int
        A placeholder property for code compatibility.
    output_state: MPS
        Output state following simulation if get_state is True
    observables : list[Observable]
        A list of observables to be tracked during the simulation.
    sorted_observables : list[Observable]
        A list of observables sorted by site and name.
    num_traj : int
        The number of trajectories to simulate. Default is 1000.
    max_bond_dim : int
        The maximum bond dimension for the simulation. Default is 2.
    threshold : float
        The threshold value for the simulation. Default is 1e-6.
    window_size : int or None
        The size of the window for the simulation. Default is None.
    get_state:
        If True, output MPS is returned.

    Methods:
    --------
    __init__(self, observables: list[Observable], num_traj: int = 1000, max_bond_dim: int = 2,
             threshold: float = 1e-6, window_size: int | None = None, get_state: bool = False) -> None:
        Initializes the StrongSimParams with the given parameters.
    aggregate_trajectories(self) -> None:
        Aggregates the trajectories of the observables by computing the mean across all trajectories.
    """

    # Properties set as placeholders for code compatibility
    dt = 1
    output_state: MPS | None = None

    def __init__(
        self,
        observables: list[Observable],
        num_traj: int = 1000,
        max_bond_dim: int = 2,
        threshold: float = 1e-6,
        window_size: int | None = 0,
        *,
        get_state: bool = False,
    ) -> None:
        """Strong circuit simulation parameters initialization.

        Initializes parameters for a strong quantum circuit simulation.

        Parameters
        ----------
        observables : list[Observable]
            List of observables to measure during simulation.
        num_traj : int, optional
            Number of trajectories to simulate, by default 1000.
        max_bond_dim : int, optional
            Maximum bond dimension allowed in simulation, by default 2.
        threshold : float, optional
            Threshold for simulation accuracy, by default 1e-6.
        window_size : int or None, optional
            Window size for simulation, by default None.
        get_state:
            If True, output MPS is returned.
        """
        self.observables = observables
        self.sorted_observables = sorted(observables, key=lambda obs: (obs.site, obs.name))
        self.num_traj = num_traj
        self.max_bond_dim = max_bond_dim
        self.threshold = threshold
        self.window_size = window_size
        self.get_state = get_state

    def aggregate_trajectories(self) -> None:
        """Aggregate trajectories for result.

        Aggregates the trajectories of each observable by computing the mean across all trajectories.
        This method iterates over all observables and replaces their `results` attribute with the mean
        of their `trajectories` along the first axis.
        """
        for observable in self.observables:
            observable.results = np.mean(observable.trajectories, axis=0)
