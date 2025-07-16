# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""High-level simulator module for using YAQS.

This module implements the common simulation routine for both circuit-based and Hamiltonian (analog) simulations.
It provides functions to run simulation trajectories in parallel using an MPS representation of the quantum state.
Depending on the type of simulation parameters provided (WeakSimParams, StrongSimParams, or AnalogSimParams),
the simulation is dispatched to the appropriate backend:
  - For circuit simulations, a QuantumCircuit is used and processed via the _run_circuit function.
  - For analog simulations, an MPO is used to represent the Hamiltonian and processed via the _run_analog function.

The module supports both strong and weak simulation schemes, including functionality for:
  - Initializing the state (MPS) to a canonical form (B normalized).
  - Running trajectories with noise (using a provided NoiseModel) and aggregating results.
  - Parallel execution of trajectories using a ProcessPoolExecutor with progress reporting via tqdm.

All simulation results (e.g., observables, measurements) are aggregated and returned as part of the simulation process.
"""

from __future__ import annotations

import concurrent.futures
import copy
import multiprocessing
from typing import TYPE_CHECKING

from qiskit.circuit import QuantumCircuit
from tqdm import tqdm

from .analog.analog_tjm import analog_tjm_1, analog_tjm_2
from .core.data_structures.networks import MPO
from .core.data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams
from .digital.digital_tjm import digital_tjm

if TYPE_CHECKING:
    from .core.data_structures.networks import MPS
    from .core.data_structures.noise_model import NoiseModel


import os


def available_cpus() -> int:
    """Determine the number of available CPU cores for parallel execution.

    This function checks if the SLURM_CPUS_ON_NODE environment variable is set (indicating a SLURM-managed cluster job).
    If so, it returns the number of CPUs specified by SLURM. Otherwise, it returns the total number of CPUs available
    on the machine as reported by multiprocessing.cpu_count().

    Returns:
        int: The number of available CPU cores for parallel execution.
    """
    slurm_cpus = int(os.environ["SLURM_CPUS_ON_NODE"]) if "SLURM_CPUS_ON_NODE" in os.environ else None
    machine_cpus = multiprocessing.cpu_count()

    if slurm_cpus is None:
        return machine_cpus
    return slurm_cpus


def _run_strong_sim(
    initial_state: MPS,
    operator: QuantumCircuit,
    sim_params: StrongSimParams,
    noise_model: NoiseModel | None,
    *,
    parallel: bool,
) -> None:
    """Run strong simulation trajectories for a quantum circuit using a strong simulation scheme.

    This function executes circuit-based simulation trajectories using the 'digital_tjm' backend.
    If the noise model is absent or its strengths are all zero, only a single trajectory is executed.
    For each observable in sim_params.sorted_observables, the function initializes the observable,
    runs the simulation trajectories (in parallel if specified), and aggregates the results.

    Args:
        initial_state (MPS): The initial system state as an MPS.
        operator (QuantumCircuit): The quantum circuit representing the operation to simulate.
        sim_params (StrongSimParams): Simulation parameters for strong simulation,
                                      including the number of trajectories (num_traj),
                                      time step (dt), and sorted observables.
        noise_model (NoiseModel | None): The noise model applied during simulation.
        parallel (bool): Flag indicating whether to run trajectories in parallel.
    """
    backend = digital_tjm

    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        sim_params.num_traj = 1
    else:
        assert not sim_params.get_state, "Cannot return state in noisy circuit simulation due to stochastics."

    for observable in sim_params.sorted_observables:
        observable.initialize(sim_params)

    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)]
    if parallel and sim_params.num_traj > 1:
        max_workers = max(1, available_cpus() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(backend, arg): arg[0] for arg in args}
            with tqdm(total=sim_params.num_traj, desc="Running trajectories", ncols=80) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    result = future.result()
                    for obs_index, observable in enumerate(sim_params.sorted_observables):
                        assert observable.trajectories is not None, "Trajectories should have been initialized"
                        observable.trajectories[i] = result[obs_index]
                    pbar.update(1)
    else:
        for i, arg in enumerate(args):
            result = backend(arg)
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                assert observable.trajectories is not None, "Trajectories should have been initialized"
                observable.trajectories[i] = result[obs_index]
    sim_params.aggregate_trajectories()


def _run_weak_sim(
    initial_state: MPS,
    operator: QuantumCircuit,
    sim_params: WeakSimParams,
    noise_model: NoiseModel | None,
    *,
    parallel: bool,
) -> None:
    """Run weak simulation trajectories for a quantum circuit using a weak simulation scheme.

    This function executes circuit-based simulation trajectories using the 'digital_tjm' backend,
    adjusted for weak simulation parameters. If the noise model is absent or its strengths are all zero,
    only a single trajectory is executed; otherwise, sim_params.num_traj is set to sim_params.shots and then shots
    is set to 1. The trajectories are then executed (in parallel if specified) and the measurement results
    are aggregated.

    Args:
        initial_state (MPS): The initial system state as an MPS.
        operator (QuantumCircuit): The quantum circuit representing the operation to simulate.
        sim_params (WeakSimParams): Simulation parameters for weak simulation,
                                    including shot count and sorted observables.
        noise_model (NoiseModel | None): The noise model applied during simulation.
        parallel (bool): Flag indicating whether to run trajectories in parallel.


    """
    backend = digital_tjm

    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        sim_params.num_traj = 1
    else:
        sim_params.num_traj = sim_params.shots
        sim_params.shots = 1
        assert not sim_params.get_state, "Cannot return state in noisy circuit simulation due to stochastics."

    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)]
    if parallel and sim_params.num_traj > 1:
        max_workers = max(1, available_cpus() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(backend, arg): arg[0] for arg in args}
            with tqdm(total=sim_params.num_traj, desc="Running trajectories", ncols=80) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    result = future.result()
                    sim_params.measurements[i] = result
                    pbar.update(1)
    else:
        for i, arg in enumerate(args):
            result = backend(arg)
            sim_params.measurements[i] = result
    sim_params.aggregate_measurements()


def _run_circuit(
    initial_state: MPS,
    operator: QuantumCircuit,
    sim_params: WeakSimParams | StrongSimParams,
    noise_model: NoiseModel | None,
    *,
    parallel: bool,
) -> None:
    """Run circuit-based simulation trajectories.

    This function validates that the number of qubits in the quantum circuit matches the length of the MPS,
    reverses the bit order of the circuit, and dispatches the simulation to the appropriate backend based on
    whether the simulation parameters indicate strong or weak simulation.

    Args:
        initial_state (MPS): The initial system state as an MPS.
        operator (QuantumCircuit): The quantum circuit to simulate.
        sim_params (WeakSimParams | StrongSimParams): Simulation parameters for circuit simulation.
        noise_model (NoiseModel | None): The noise model applied during simulation.
        parallel (bool): Flag indicating whether to run trajectories in parallel.


    """
    assert initial_state.length == operator.num_qubits, "State and circuit qubit counts do not match."
    operator = copy.deepcopy(operator.reverse_bits())

    if isinstance(sim_params, StrongSimParams):
        _run_strong_sim(initial_state, operator, sim_params, noise_model, parallel=parallel)
    elif isinstance(sim_params, WeakSimParams):
        _run_weak_sim(initial_state, operator, sim_params, noise_model, parallel=parallel)


def _run_analog(
    initial_state: MPS, operator: MPO, sim_params: AnalogSimParams, noise_model: NoiseModel | None, *, parallel: bool
) -> None:
    """Run analog simulation trajectories for Hamiltonian evolution.

    This function selects the appropriate analog simulation backend based on sim_params.order
    (either one-site or two-site evolution) and runs the simulation trajectories for the given Hamiltonian
    (represented as an MPO). The trajectories are executed (in parallel if specified) and the results are aggregated.

    Args:
        initial_state (MPS): The initial system state as an MPS.
        operator (MPO): The Hamiltonian operator represented as an MPO.
        sim_params (AnalogSimParams): Simulation parameters for analog simulation,
                                       including time step and evolution order.
        noise_model (NoiseModel | None): The noise model applied during simulation.
        parallel (bool): Flag indicating whether to run trajectories in parallel.


    """
    backend = analog_tjm_1 if sim_params.order == 1 else analog_tjm_2

    if noise_model is None or all(proc["strength"] == 0 for proc in noise_model.processes):
        sim_params.num_traj = 1
    else:
        assert not sim_params.get_state, "Cannot return state in noisy analog simulation due to stochastics."

    for observable in sim_params.sorted_observables:
        observable.initialize(sim_params)

    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.num_traj)]
    if parallel and sim_params.num_traj > 1:
        max_workers = max(1, available_cpus() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(backend, arg): arg[0] for arg in args}
            with tqdm(total=sim_params.num_traj, desc="Running trajectories", ncols=80) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    result = future.result()
                    for obs_index, observable in enumerate(sim_params.sorted_observables):
                        assert observable.trajectories is not None, "Trajectories should have been initialized"
                        observable.trajectories[i] = result[obs_index]
                    pbar.update(1)
    else:
        for i, arg in enumerate(args):
            result = backend(arg)
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                assert observable.trajectories is not None, "Trajectories should have been initialized"
                observable.trajectories[i] = result[obs_index]
    sim_params.aggregate_trajectories()


def run(
    initial_state: MPS,
    operator: MPO | QuantumCircuit,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    noise_model: NoiseModel | None = None,
    *,
    parallel: bool = True,
) -> None:
    """Execute the common simulation routine for both circuit and Hamiltonian simulations.

    This function first normalizes the initial state (MPS) to B normalization, then dispatches the simulation
    to the appropriate backend based on the type of simulation parameters provided. For circuit-based simulations,
    the operator must be a QuantumCircuit; for Hamiltonian simulations, the operator must be an MPO.

    Args:
        initial_state (MPS): The initial state of the system as an MPS. Must be B normalized.
        operator (MPO | QuantumCircuit): The operator representing the evolution; an MPO for analog simulations
            or a QuantumCircuit for circuit simulations.
        sim_params (AnalogSimParams | StrongSimParams | WeakSimParams): Simulation parameters specifying
                                                                         the simulation mode and settings.
        noise_model (NoiseModel | None): The noise model to apply during simulation.
        parallel (bool, optional): Whether to run trajectories in parallel. Defaults to True.


    """
    # State must start in B normalization
    initial_state.normalize("B")

    if isinstance(sim_params, (StrongSimParams, WeakSimParams)):
        assert isinstance(operator, QuantumCircuit)
        _run_circuit(initial_state, operator, sim_params, noise_model, parallel=parallel)
    elif isinstance(sim_params, AnalogSimParams):
        assert isinstance(operator, MPO)
        _run_analog(initial_state, operator, sim_params, noise_model, parallel=parallel)
