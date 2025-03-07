# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import concurrent.futures
import copy
import multiprocessing
from typing import TYPE_CHECKING

from qiskit.circuit import QuantumCircuit
from tqdm import tqdm

from .core.data_structures.networks import MPO
from .core.data_structures.simulation_parameters import PhysicsSimParams, StrongSimParams, WeakSimParams

if TYPE_CHECKING:
    from .core.data_structures.networks import MPS
    from .core.data_structures.noise_model import NoiseModel


def _run_strong_sim(
    initial_state: MPS,
    operator: QuantumCircuit,
    sim_params: StrongSimParams,
    noise_model: NoiseModel | None,
    parallel: bool,
) -> None:
    from mqt.yaqs.circuits.CircuitTJM import CircuitTJM

    backend = CircuitTJM

    if not noise_model or all(gamma == 0 for gamma in noise_model.strengths):
        sim_params.N = 1

    for observable in sim_params.sorted_observables:
        observable.initialize(sim_params)

    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.N)]
    if parallel and sim_params.N > 1:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(backend, arg): arg[0] for arg in args}
            with tqdm(total=sim_params.N, desc="Running trajectories", ncols=80) as pbar:
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
    parallel: bool,
) -> None:
    from mqt.yaqs.circuits.CircuitTJM import CircuitTJM

    backend = CircuitTJM

    if not noise_model or all(gamma == 0 for gamma in noise_model.strengths):
        sim_params.N = 1
    else:
        sim_params.N = sim_params.shots
        sim_params.shots = 1

    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.N)]
    if parallel and sim_params.N > 1:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(backend, arg): arg[0] for arg in args}
            with tqdm(total=sim_params.N, desc="Running trajectories", ncols=80) as pbar:
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
    parallel: bool,
) -> None:
    assert initial_state.length == operator.num_qubits, "State and circuit qubit counts do not match."
    operator = copy.deepcopy(operator.reverse_bits())

    if isinstance(sim_params, StrongSimParams):
        _run_strong_sim(initial_state, operator, sim_params, noise_model, parallel)
    elif isinstance(sim_params, WeakSimParams):
        _run_weak_sim(initial_state, operator, sim_params, noise_model, parallel)


def _run_physics(
    initial_state: MPS, operator: MPO, sim_params: PhysicsSimParams, noise_model: NoiseModel | None, parallel: bool
) -> None:
    if sim_params.order == 1:
        from mqt.yaqs.physics.PhysicsTJM import PhysicsTJM_1

        backend = PhysicsTJM_1
    else:
        from mqt.yaqs.physics.PhysicsTJM import PhysicsTJM_2

        backend = PhysicsTJM_2

    if not noise_model or all(gamma == 0 for gamma in noise_model.strengths):
        sim_params.N = 1

    for observable in sim_params.sorted_observables:
        observable.initialize(sim_params)

    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.N)]
    if parallel and sim_params.N > 1:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(backend, arg): arg[0] for arg in args}
            with tqdm(total=sim_params.N, desc="Running trajectories", ncols=80) as pbar:
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
    sim_params: PhysicsSimParams | StrongSimParams | WeakSimParams,
    noise_model: NoiseModel | None,
    parallel: bool = True,
) -> None:
    """Common simulation routine used by both circuit and Hamiltonian simulations.
    It normalizes the state, prepares trajectory arguments, runs the trajectories
    in parallel, and aggregates the results.
    """
    # State must start in B normalization
    initial_state.normalize("B")

    if isinstance(sim_params, (StrongSimParams, WeakSimParams)):
        assert isinstance(operator, QuantumCircuit)
        _run_circuit(initial_state, operator, sim_params, noise_model, parallel)
    elif isinstance(sim_params, PhysicsSimParams):
        assert isinstance(operator, MPO)
        _run_physics(initial_state, operator, sim_params, noise_model, parallel)
