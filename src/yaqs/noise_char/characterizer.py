from __future__ import annotations
import matplotlib.pyplot as plt

import numpy as np
import qutip as qt

from yaqs.core.data_structures.networks import MPO, MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams
from yaqs.core.methods.operations import local_expval

from yaqs.core.libraries.gate_library import GateLibrary



import copy


from yaqs.core.methods.dissipation import apply_dissipation
from yaqs.core.methods.dynamic_TDVP import dynamic_TDVP
from yaqs.core.methods.stochastic_process import stochastic_process



import concurrent.futures
import copy
import multiprocessing
from qiskit.circuit import QuantumCircuit
from tqdm import tqdm

from yaqs.core.data_structures.simulation_parameters import StrongSimParams, WeakSimParams



# Define the system Hamiltonian
L = 4
d = 2
J = 1
g = 0.5
H_0 = MPO()
H_0.init_Ising(L, d, J, g)

# Define the initial state
state = MPS(L, state='zeros')

# Define the noise model
gamma = 0.1
noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma, gamma])

# Define the simulation parameters
T = 2
dt = 0.1
sample_timesteps = True
N = 100
max_bond_dim = 4
threshold = 1e-6
order = 1
measurements = [Observable('x', site) for site in range(L)] + [Observable('y', site) for site in range(L)] + [Observable('z', site) for site in range(L)]
sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)




def run(initial_state: MPS, operator, sim_params, noise_model: NoiseModel=None, parallel: bool=False):
    """
    Common simulation routine used by both circuit and Hamiltonian simulations.
    It normalizes the state, prepares trajectory arguments, runs the trajectories
    in parallel, and aggregates the results.
    """
    if isinstance(operator, QuantumCircuit):
        assert initial_state.length == operator.num_qubits, "State and circuit qubit counts do not match."
        operator = copy.deepcopy(operator.reverse_bits())

    if not noise_model or all(gamma == 0 for gamma in noise_model.strengths):
        sim_params.N = 1
    else:
        if isinstance(sim_params, WeakSimParams):
            sim_params.N = sim_params.shots
            sim_params.shots = 1

    # For Hamiltonian simulations and for circuit simulations with StrongSimParams,
    # initialize observables. For WeakSimParams in the circuit case, no initialization needed.
    if isinstance(operator, MPO) or isinstance(sim_params, StrongSimParams):
    
        for observable in sim_params.sorted_observables:
            observable.initialize(sim_params)

        unique_obs = set(obs.name for obs in sim_params.observables)
    
        # Create a nested dictionary in sim_params to store the A_kn matrices.
        # Outer keys are noise process names, inner keys are observable types.
        sim_params.A_kn = {process: {} for process in noise_model.processes}
        # Loop over noise processes and their jump operators together.
        for process, jump_op in zip(noise_model.processes, noise_model.jump_operators):
            for obs_name in unique_obs:
                observable = next(obs for obs in sim_params.observables if obs.name == obs_name)
                obs_operator = getattr(GateLibrary, obs_name).matrix
                sim_params.A_kn[process][obs_name] = calc_A_kn(jump_op, obs_operator)
        sim_params.expvals_Master = {obs.name: {process: np.zeros((sim_params.N, len(sim_params.times)), dtype=complex) for process in noise_model.processes} for obs in sim_params.sorted_observables}
        print('expvals_Master:', sim_params.expvals_Master)

    # Normalize the state to the B form
    initial_state.normalize('B')

    # Prepare arguments for each trajectory
    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.N)]
    
    if parallel:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(PhysicsTJM_1_analytical_gradient, arg): arg[0] for arg in args}
            with tqdm(total=sim_params.N, desc="Running trajectories", ncols=80) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        result, expvals = future.result()
                        if result is None or expvals is None:
                            print(f"Trajectory {i} returned None.")
                            continue
                        if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
                            sim_params.measurements[i] = result
                        else:
                            for obs_index, observable in enumerate(sim_params.sorted_observables):
                                observable.trajectories[i] = result[obs_index]
                                for process in noise_model.processes:
                                    sim_params.expvals_Master[observable.name][process][i,:] = expvals[observable.name][process][:]
                    except Exception as e:
                        print(f"\nTrajectory {i} failed with exception: {e}.")
                    finally:
                        pbar.update(1)
    else:
        for i, arg in enumerate(args):
            try:
                result, expvals = PhysicsTJM_1_analytical_gradient(arg)
                if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
                    sim_params.measurements[i] = result
                else:
                    for obs_index, observable in enumerate(sim_params.sorted_observables):
                        observable.trajectories[i] = result[obs_index]
                        for process in noise_model.processes:
                            sim_params.expvals_Master[observable.name][process][i,:] = expvals[observable.name][process][:]
                        # here we also have to update sim_params.expvals_Master
            except Exception as e:
                print(f"Trajectory {i} failed with exception: {e}.")
    
    if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
        sim_params.aggregate_measurements()
    else:
        sim_params.aggregate_trajectories()
        # here we have to average the expvals_master over the trajectories
        sim_params.avg_expvals = {obs_name: {process: np.mean(sim_params.expvals_Master[obs_name][process], axis=0) for process in sim_params.expvals_Master[obs_name]} for obs_name in sim_params.expvals_Master}






def calc_A_kn(L,obs):
    A_kn = L.conj().T @ obs @ L - 0.5 * obs @ L.conj().T @ L -0.5 * L.conj().T @ L @ obs
    return A_kn

# def aggregate_expvals(sim_params, noise_processes):
#     for observable in sim_params.observables:
#         for process in noise_processes:
#             attr_name = f"expval_A_kn_{process}"
#             if hasattr(observable, attr_name):
#                 # Compute the mean over trajectories (axis 0)
#                 aggregated = np.mean(getattr(observable, attr_name), axis=0)
#                 setattr(observable, attr_name, aggregated)


def PhysicsTJM_1_analytical_gradient(args):
    i, initial_state, noise_model, sim_params, H = args
    state = copy.deepcopy(initial_state)

    if sim_params.sample_timesteps:
        results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)))
    else:
        results = np.zeros((len(sim_params.sorted_observables), 1))
    

    expvals = {obs.name: {process: np.zeros((len(sim_params.times)), dtype=complex) for process in noise_model.processes} for obs in sim_params.sorted_observables}


    if sim_params.sample_timesteps:
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            results[obs_index, 0] = copy.deepcopy(state).measure(observable)
            for process in noise_model.processes:
                A_kn_op = sim_params.A_kn[process][observable.name]
                measurement_Akn = local_expval(copy.deepcopy(state), A_kn_op, observable.site).real
                #if measurement_Akn > 1e-6:
                #    print('non-zero')
                expvals[observable.name][process][0] = measurement_Akn



    for j, _ in enumerate(sim_params.times[1:], start=1):
        dynamic_TDVP(state, H, sim_params)
        if noise_model:
            apply_dissipation(state, noise_model, sim_params.dt)
            state = stochastic_process(state, noise_model, sim_params.dt)
        if sim_params.sample_timesteps:
            temp_state = copy.deepcopy(state)
            last_site = 0
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                if observable.site > last_site:
                    for site in range(last_site, observable.site):
                        temp_state.shift_orthogonality_center_right(site)
                    last_site = observable.site
                results[obs_index, j] = temp_state.measure(observable)
                for process in noise_model.processes:
                    A_kn_op = sim_params.A_kn[process][observable.name]
                    measurement_Akn = local_expval(copy.deepcopy(state), A_kn_op, observable.site).real
                    # print('measurement_Akn', measurement_Akn)
                    expvals[observable.name][process][j] = measurement_Akn
        elif j == len(sim_params.times)-1:
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                results[obs_index, 0] = copy.deepcopy(state).measure(observable)
                for process in noise_model.processes:

                    A_kn_op = sim_params.A_kn[process][observable.name]
                    measurement_Akn = local_expval(copy.deepcopy(state), A_kn_op, observable.site).real
                    # if measurement_Akn > 1e-6:
                    #'measurement_Akn', measurement_Akn)
                    # Store it in the corresponding numpy array at trajectory index i and time index 0
                    expvals[observable.name][process][0] = measurement_Akn
 
    if results is None or expvals is None:
        raise ValueError("Results or expvals is None.")
    return results, expvals







if __name__ == "__main__":
    ########## TJM Example #################
    run(state, H_0, sim_params, noise_model)

    tjm_results = []
    for observable in sim_params.observables:
        tjm_results.append(observable.results)


    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    # First subplot: Plot the tjm_results
    for i in range(len(tjm_results)):
        ax1.plot(sim_params.times, tjm_results[i], label=f'obs {i}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Expectation Value')
    ax1.set_title('Observables Expectation Values')
    ax1.legend()

    # Second subplot: Plot the averaged A_kn expectation values over trajectories
    for obs_name, process_dict in sim_params.avg_expvals.items():
        for process, avg_exp in process_dict.items():
            ax2.plot(sim_params.times, avg_exp, label=f"{obs_name} ({process})")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("A_kn Expectation Value")
    ax2.set_title("A_kn Averages over Trajectories")
    ax2.legend()


    plt.tight_layout()
    plt.show()