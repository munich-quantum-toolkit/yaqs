import concurrent.futures
import copy
import multiprocessing
import numpy as np
import opt_einsum as oe
from qiskit.converters import circuit_to_dag
from tqdm import tqdm

from yaqs.general.data_structures.networks import MPO
from yaqs.general.data_structures.simulation_parameters import WeakSimParams, StrongSimParams
from yaqs.general.libraries.gate_library import GateLibrary
from yaqs.circuits.dag.dag_utils import get_restricted_temporal_zone, select_starting_point, convert_dag_to_tensor_algorithm
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


def apply_single_qubit_gates(state, tensor_circuit, qubits):
    for gate in tensor_circuit:
        if gate.interaction == 1:
            assert gate.sites[0] in qubits, \
                    "Single-qubit gate must be on one of the sites."
        elif gate.interaction == 2:
            break
        state.tensors[gate.sites[0]] = oe.contract('ab, bcd->acd', gate.tensor, state.tensors[gate.sites[0]])
        tensor_circuit.pop(0)


def construct_generator_MPO(gate, length):
    tensors = []

    site_idx = 0  # Keeps track of the current site being processed



    first_site = min(gate.sites)
    last_site = max(gate.sites)
    if gate.sites[0] < gate.sites[1]:
        first_gen = 0
        second_gen = 1
    else:
        first_gen = 1
        second_gen = 0

    first_site = gate.sites[first_gen]
    last_site = gate.sites[second_gen]
    for site in range(length):
        if site == first_site:
            W = np.zeros((1, 1, 2, 2), dtype=complex)
            W[0, 0] = gate.generator[first_gen]
        elif site == last_site:
            W = np.zeros((1, 1, 2, 2), dtype=complex)
            W[0, 0] = gate.generator[second_gen]
        elif site > first_site and site < last_site:
            W = np.zeros((1, 1, 2, 2), dtype=complex)
            W[0, 0] = np.eye(2)

        tensors.append(W)


    # site_idx = 0  # Keeps track of the current site being processed
    # while site_idx < length:
    #     # Check if this site is part of a two-qubit gate
    #     if two_qubit_gates and site_idx in two_qubit_gates[0].sites:
    #         gate = two_qubit_gates[0]  # The first gate in the list
    #         print(gate)
    #         if site_idx == gate.sites[0]:  # First qubit in the gate
    #             print("0")
    #             W = np.zeros((1, 1, 2, 2), dtype=complex)
    #             W[0, 0] = gate.generator[0]  # Correctly handle θ scaling later

    #         elif site_idx == gate.sites[1]:  # Second qubit in the gate
    #             print("1")
    #             W = np.zeros((1, 1, 2, 2), dtype=complex)
    #             W[0, 0] = gate.generator[1]
    #             two_qubit_gates.pop(0)

    #     else:  # Identity case
    #         print("I")
    #         W = np.zeros((1, 1, 2, 2), dtype=complex)
    #         W[0, 0] = np.eye(2)
    #         # W[0, 1] = np.zeros((2, 2))
    #         # W[1, 0] = np.zeros((2, 2))
    #         # W[1, 1] = np.eye(2)

    #     tensors.append(W)
    #     site_idx += 1  # Move to the next site

    mpo = MPO()
    mpo.init_custom(tensors)
    return mpo


def iterate(state, dag, iterator):
    two_qubit_gates = []
    for n in iterator:
        temporal_zone = get_restricted_temporal_zone(dag, [n, n+1])
        tensor_circuit = convert_dag_to_tensor_algorithm(temporal_zone)
        apply_single_qubit_gates(state, tensor_circuit, [n, n+1])
        if tensor_circuit:
            two_qubit_gates.append(tensor_circuit[-1])
    if two_qubit_gates:
        return two_qubit_gates
    else:
        return None


def run_trajectory(args):
    i, initial_state, noise_model, sim_params, circuit = args
    state = copy.deepcopy(initial_state)

    if isinstance(sim_params, StrongSimParams):
        results = np.zeros((len(sim_params.observables), 1))

    dag = circuit_to_dag(circuit)

    # Decides whether to start with even or odd qubits
    while dag.op_nodes():
        first_iterator, second_iterator = select_starting_point(initial_state.length, dag)
        for iterator in [first_iterator, second_iterator]:
            two_qubit_gates = iterate(state, dag, iterator)
            # mpo = iterate(state, dag, iterator)
            if two_qubit_gates:
                # TODO: Make sure gates are ordered to speed up computation
                for gate in two_qubit_gates:
                    mpo = construct_generator_MPO(gate, state.length)
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