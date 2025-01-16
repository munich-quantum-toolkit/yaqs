import concurrent.futures
import copy
import multiprocessing
import numpy as np
from qiskit.converters import circuit_to_dag
from tqdm import tqdm

from yaqs.general.data_structures.networks import MPO
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
    from yaqs.general.data_structures.simulation_parameters import WeakSimParams
    from yaqs.general.data_structures.noise_model import NoiseModel


def run_trajectory(args):
    i, initial_state, noise_model, sim_params, circuit = args
    state = copy.deepcopy(initial_state)

    # if sim_params.sample_timesteps:
    #     results = np.zeros((len(sim_params.observables), len(sim_params.times)))
    # else:
    # results = np.zeros((len(sim_params.observables), 1))

    # if sim_params.sample_timesteps:
    #     for obs_index, observable in enumerate(sim_params.observables):
    #         results[obs_index, 0] = copy.deepcopy(state).measure(observable)
    # ----------------------- WIP ----------------------------
    # WHILE CIRCUIT:
    # 1. Find temporal zone
    #   a. If no noise, normal temporal zone
    #   b. If noise, up to 2nd qubit gate
    # 2. Create MPO layer
    # 3. TJM with layer

    dag = circuit_to_dag(circuit)
    # N = mpo.length

    # Decides whether to start with even or odd qubits
    first_iterator, second_iterator = select_starting_point(initial_state.length, dag)
    mpo = MPO()
    while dag.op_nodes():
        if not noise_model:
            mpo.init_identity(circuit.num_qubits)
            apply_layer(mpo, dag, None, first_iterator, second_iterator, sim_params.threshold)
            dynamic_TDVP(state, mpo, sim_params)
        else:
            for iterator in [first_iterator, second_iterator]:
                mpo.init_identity(circuit.num_qubits)
                # apply_layer(mpo, dag, None, first_iterator, second_iterator, sim_params.threshold)
                apply_restricted_layer(mpo, dag, None, iterator, sim_params.threshold)
                dynamic_TDVP(state, mpo, sim_params)
                apply_dissipation(state, noise_model, dt=1)
                state = stochastic_process(state, noise_model, dt=1)

    # results = sample_prob_dist(state, sim_params.samples)
    # for obs_index, observable in enumerate(sim_params.observables):
    #     results[obs_index, 0] = copy.deepcopy(state).measure(observable)

    # for j, _ in enumerate(sim_params.times[1:], start=1):
    #     dynamic_TDVP(state, H, sim_params)
    #     # if noise_model:
    #     #     apply_dissipation(state, noise_model, sim_params.dt)
    #     #     state = stochastic_process(state, noise_model, sim_params.dt)
    #     # if sim_params.sample_timesteps:
    #     #     for obs_index, observable in enumerate(sim_params.observables):
    #     #         results[obs_index, j] = copy.deepcopy(state).measure(observable)
    #     if j == len(sim_params.times)-1:
    #         for obs_index, observable in enumerate(sim_params.observables):
    #             results[obs_index, 0] = copy.deepcopy(state).measure(observable)

    if noise_model:
        return measure(state, shots=1)
    else:
        return measure(state, sim_params.shots)


def run(initial_state: 'MPS', circuit: 'QuantumCircuit', sim_params: 'WeakSimParams', noise_model: 'NoiseModel'=None):
    assert initial_state.length == circuit.num_qubits

    # Reset any previous results
    # for observable in sim_params.observables:
    #     observable.initialize(sim_params)

    # Guarantee one trajectory if no noise model
    if not noise_model or all(gamma == 0 for gamma in noise_model.strengths):
        sim_params.N = 1
    else:
        # Shots themselves become the trajectories
        sim_params.N = sim_params.shots
        sim_params.shots = 1

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
                    sim_params.measurements[i] = result                        
                    # for obs_index, observable in enumerate(sim_params.observables):
                    #     observable.trajectories[i] = result[obs_index]
                except Exception as e:
                        print(f"\nTrajectory {i} failed with exception: {e}. Retrying...")
                        # Retry could be done here
                finally:
                    pbar.update(1)

    # Save average value of trajectories
    # for observable in sim_params.observables:
    #     observable.results = np.mean(observable.trajectories, axis=0)