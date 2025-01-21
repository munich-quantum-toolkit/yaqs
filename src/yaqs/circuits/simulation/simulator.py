import concurrent.futures
import copy
import multiprocessing
import numpy as np
from qiskit.converters import circuit_to_dag
from tqdm import tqdm

from yaqs.general.data_structures.networks import MPO
from yaqs.general.data_structures.simulation_parameters import WeakSimParams, StrongSimParams
from yaqs.circuits.dag.dag_utils import get_temporal_zone, select_starting_point
from yaqs.circuits.equivalence_checking.mpo_utils import apply_layer, apply_restricted_layer
from yaqs.physics.methods.dynamic_TDVP import dynamic_TDVP
from yaqs.physics.methods.dissipation import apply_dissipation
from yaqs.physics.methods.stochastic_process import stochastic_process
from yaqs.general.operations.operations import measure

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.general.data_structures.networks import MPS
    from qiskit.circuit.quantumcircuit import QuantumCircuit
    from yaqs.general.data_structures.noise_model import NoiseModel


def run_trajectory(args):
    i, initial_state, noise_model, sim_params, circuit = args
    state = copy.deepcopy(initial_state)

    if isinstance(sim_params, StrongSimParams):
        results = np.zeros((len(sim_params.observables), 1))


    dag = circuit_to_dag(circuit)

    # Decides whether to start with even or odd qubits
    first_iterator, second_iterator = select_starting_point(initial_state.length, dag)
    mpo = MPO()
    while dag.op_nodes():
        if not noise_model or all(gamma == 0 for gamma in noise_model.strengths):
            mpo.init_identity(circuit.num_qubits)
            apply_layer(mpo, None, dag, first_iterator, second_iterator, sim_params.threshold)
            dynamic_TDVP(state, mpo, sim_params)
            apply_dissipation(state, noise_model, dt=0)
            state = stochastic_process(state, noise_model, dt=0)
        else:
            for iterator in [first_iterator, second_iterator]:
                mpo.init_identity(circuit.num_qubits)
                # apply_layer(mpo, dag, None, first_iterator, second_iterator, sim_params.threshold)
                apply_restricted_layer(mpo, dag, None, iterator, sim_params.threshold)
                dynamic_TDVP(state, mpo, sim_params)
                apply_dissipation(state, noise_model, dt=1)
                state = stochastic_process(state, noise_model, dt=1)

    if isinstance(sim_params, WeakSimParams):
        if not noise_model or all(gamma == 0 for gamma in noise_model.strengths):
            return measure(state, sim_params.shots)
        else:
            return measure(state, shots=1)
    elif isinstance(sim_params, StrongSimParams):
        for obs_index, observable in enumerate(sim_params.observables):
            results[obs_index, 0] = copy.deepcopy(state).measure(observable)
        return results


def run(initial_state: 'MPS', circuit: 'QuantumCircuit', sim_params, noise_model: 'NoiseModel'=None):
    assert initial_state.length == circuit.num_qubits

    # Guarantee one trajectory if no noise model
    if not noise_model or all(gamma == 0 for gamma in noise_model.strengths):
        sim_params.N = 1
    else:
        if isinstance(sim_params, WeakSimParams):
            # Shots themselves become the trajectories
            sim_params.N = sim_params.shots
            sim_params.shots = 1

    # Reset any previous results
    if isinstance(sim_params, StrongSimParams):
        for observable in sim_params.observables:
            observable.initialize(sim_params)

    # State must start in B form
    initial_state.normalize('B')

    args = [(i, initial_state, noise_model, sim_params, circuit) for i in range(sim_params.N)]
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_trajectory, arg): arg[0] for arg in args}

        with tqdm(total=sim_params.N, desc="Running trajectories", ncols=80) as pbar:
            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                try:
                    result = future.result()
                    if isinstance(sim_params, WeakSimParams):
                        sim_params.measurements[i] = result   
                    elif isinstance(sim_params, StrongSimParams):                     
                        for obs_index, observable in enumerate(sim_params.observables):
                            observable.trajectories[i] = result[obs_index]
                except Exception as e:
                        print(f"\nTrajectory {i} failed with exception: {e}. Retrying...")
                        # Retry could be done here
                finally:
                    pbar.update(1)

    if isinstance(sim_params, WeakSimParams):
        sim_params.aggregate_measurements()
    elif isinstance(sim_params, StrongSimParams):                     
        # Save average value of trajectories
        for observable in sim_params.observables:
            observable.results = np.mean(observable.trajectories, axis=0)