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




def run(initial_state: MPS, operator, sim_params, noise_model: NoiseModel=None, parallel: bool=True):
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

        unique_obs = set(obs.name for obs in sim_params.observables)
    
        # Create a nested dictionary in sim_params to store the A_kn matrices.
        # Outer keys are noise process names, inner keys are observable types.
        sim_params.A_kn = {process: {} for process in noise_model.processes}
        # Loop over noise processes and their jump operators together.
        for process, jump_op in zip(noise_model.processes, noise_model.jump_operators):
            for obs_name in unique_obs:

                observable = next(obs for obs in sim_params.observables if obs.name == obs_name)
                obs_operator = getattr(GateLibrary, obs_name).matrix

                # print('jump_op:', jump_op)
                # print('observable operator:', obs_operator)
                sim_params.A_kn[process][obs_name] = calc_A_kn(jump_op, obs_operator)
                # print('A_kn:',sim_params.A_kn[process][obs_name])


        for observable in sim_params.sorted_observables:
            observable.initialize(sim_params)

            for process in noise_model.processes:
            # Create an attribute name, e.g., A_kn_relaxation for process "relaxation"
                attr_name = f"expval_A_kn_{process}"
            # Initialize with a zero array: dimensions [number of trajectories, number of timesteps]
                setattr(observable, attr_name, np.zeros((sim_params.N, len(sim_params.times)), dtype=complex))



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
                        result = future.result()
                        if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
                            sim_params.measurements[i] = result
                        else:
                            for obs_index, observable in enumerate(sim_params.sorted_observables):
                                observable.trajectories[i] = result[obs_index]
                    except Exception as e:
                        print(f"\nTrajectory {i} failed with exception: {e}.")
                    finally:
                        pbar.update(1)
    else:
        for i, arg in enumerate(args):
            try:
                result = PhysicsTJM_1_analytical_gradient(arg)
                if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
                    sim_params.measurements[i] = result
                else:
                    for obs_index, observable in enumerate(sim_params.sorted_observables):
                        observable.trajectories[i] = result[obs_index]
            except Exception as e:
                print(f"Trajectory {i} failed with exception: {e}.")
    
    if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
        sim_params.aggregate_measurements()
    else:
        sim_params.aggregate_trajectories()
        aggregate_expvals(sim_params, noise_model.processes)





# def run(initial_state: MPS, operator, sim_params, noise_model: NoiseModel=None, parallel: bool=True):
#     """
#     Common simulation routine used by both circuit and Hamiltonian simulations.
#     It normalizes the state, prepares trajectory arguments, runs the trajectories
#     in parallel, and aggregates the results.
#     """
#     if isinstance(operator, QuantumCircuit):
#         assert initial_state.length == operator.num_qubits, "State and circuit qubit counts do not match."
#         operator = copy.deepcopy(operator.reverse_bits())

#     if not noise_model or all(gamma == 0 for gamma in noise_model.strengths):
#         sim_params.N = 1
#     else:
#         if isinstance(sim_params, WeakSimParams):
#             sim_params.N = sim_params.shots
#             sim_params.shots = 1

#     # For Hamiltonian simulations and for circuit simulations with StrongSimParams,
#     # initialize observables. For WeakSimParams in the circuit case, no initialization needed.
#     if isinstance(operator, MPO) or isinstance(sim_params, StrongSimParams):

#         unique_obs = set(obs.name for obs in sim_params.observables)
    
#         # Create a nested dictionary in sim_params to store the A_kn matrices.
#         # Outer keys are noise process names, inner keys are observable types.
#         sim_params.A_kn = {process: {} for process in noise_model.processes}
#         # Loop over noise processes and their jump operators together.
#         for process, jump_op in zip(noise_model.processes, noise_model.jump_operators):
#             for obs_name in unique_obs:

#                 observable = next(obs for obs in sim_params.observables if obs.name == obs_name)
#                 obs_operator = getattr(GateLibrary, obs_name).matrix

#                 # print('jump_op:', jump_op)
#                 # print('observable operator:', obs_operator)
#                 sim_params.A_kn[process][obs_name] = calc_A_kn(jump_op, obs_operator)
#                 # print('A_kn:',sim_params.A_kn[process][obs_name])


#         for observable in sim_params.sorted_observables:
#             observable.initialize(sim_params)

#             for process in noise_model.processes:
#             # Create an attribute name, e.g., A_kn_relaxation for process "relaxation"
#                 attr_name = f"expval_A_kn_{process}"
#             # Initialize with a zero array: dimensions [number of trajectories, number of timesteps]
#                 setattr(observable, attr_name, np.zeros((sim_params.N, len(sim_params.times)), dtype=complex))



#     # Normalize the state to the B form
#     initial_state.normalize('B')

#     # Prepare arguments for each trajectory
#     args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.N)]
    
#     if parallel:
#         max_workers = max(1, multiprocessing.cpu_count() - 1)
#         with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = {executor.submit(PhysicsTJM_1_analytical_gradient, arg): arg[0] for arg in args}
#             with tqdm(total=sim_params.N, desc="Running trajectories", ncols=80) as pbar:
#                 for future in concurrent.futures.as_completed(futures):
#                     i = futures[future]
#                     try:
#                         result = future.result()
#                         if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
#                             sim_params.measurements[i] = result
#                         else:
#                             for obs_index, observable in enumerate(sim_params.sorted_observables):
#                                 observable.trajectories[i] = result[obs_index]
#                                  # Now also store the expval_A_kn values for each process:
#                                 for process in noise_model.processes:
#                                     # The observable attribute expval_A_kn_{process} was initialized as a 2D array.
#                                     # Fill row i with this trajectory’s computed values.
#                                     getattr(observable, f"expval_A_kn_{process}")[i, :] = expvals_A_kn[observable.name][process]
                                    
#                                     # print('inside run expval_A_kn:', getattr(observable, f"expval_A_kn_{process}")[i, :])

#                     except Exception as e:
#                         print(f"\nTrajectory {i} failed with exception: {e}.")
#                     finally:
#                         pbar.update(1)
#     else:
#         for i, arg in enumerate(args):
#             try:
#                 result = PhysicsTJM_1_analytical_gradient(arg)
#                 if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
#                     sim_params.measurements[i] = result
#                 else:
#                     for obs_index, observable in enumerate(sim_params.sorted_observables):
#                         observable.trajectories[i] = result[obs_index]
#             except Exception as e:
#                 print(f"Trajectory {i} failed with exception: {e}.")
    
#     if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
#         sim_params.aggregate_measurements()
#     else:
#         sim_params.aggregate_trajectories()
#         aggregate_expvals(sim_params, noise_model.processes)





def calc_A_kn(L,obs):
    A_kn = L.conj().T @ obs @ L - 0.5 * obs @ L.conj().T @ L -0.5 * L.conj().T @ L @ obs
    return A_kn

def aggregate_expvals(sim_params, noise_processes):
    for observable in sim_params.observables:
        for process in noise_processes:
            attr_name = f"expval_A_kn_{process}"
            if hasattr(observable, attr_name):
                # Compute the mean over trajectories (axis 0)
                aggregated = np.mean(getattr(observable, attr_name), axis=0)
                setattr(observable, attr_name, aggregated)


def PhysicsTJM_1_analytical_gradient(args):
    i, initial_state, noise_model, sim_params, H = args
    state = copy.deepcopy(initial_state)

    if sim_params.sample_timesteps:
        results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)))
    else:
        results = np.zeros((len(sim_params.sorted_observables), 1))

    if sim_params.sample_timesteps:
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            results[obs_index, 0] = copy.deepcopy(state).measure(observable)
            # HERE measure A_kn
            for process in noise_model.processes:
                # Construct the attribute name, e.g., 'expval_A_kn_relaxation'
                attr_name = f"expval_A_kn_{process}"
                # Get the A_kn operator for this observable and process
                A_kn_op = sim_params.A_kn[process][observable.name]
                # Compute the local expectation value for A_kn
                measurement_Akn = local_expval(copy.deepcopy(state), A_kn_op, observable.site).real
                if measurement_Akn > 1e-6:
                    print('non-zero')
                # Store it in the corresponding numpy array at trajectory index i and time index 0
                getattr(observable, attr_name)[i, 0] = measurement_Akn


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
                    # Construct the attribute name, e.g., 'expval_A_kn_relaxsation'
                    attr_name = f"expval_A_kn_{process}"
                    # Get the A_kn operator for this observable and process
                    A_kn_op = sim_params.A_kn[process][observable.name]
                    # Compute the local expectation value for A_kn
                    measurement_Akn = local_expval(copy.deepcopy(state), A_kn_op, observable.site).real
                    if measurement_Akn > 1e-6:
                        print('non-zero:', measurement_Akn)
                    # Store it in the corresponding numpy array at trajectory index i and time index 0
                    getattr(observable, attr_name)[i, j] = measurement_Akn
        elif j == len(sim_params.times)-1:
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                results[obs_index, 0] = copy.deepcopy(state).measure(observable)
                for process in noise_model.processes:
                    # Construct the attribute name, e.g., 'expval_A_kn_relaxation'
                    attr_name = f"expval_A_kn_{process}"
                    # Get the A_kn operator for this observable and process
                    A_kn_op = sim_params.A_kn[process][observable.name]
                    # Compute the local expectation value for A_kn
                    measurement_Akn = local_expval(copy.deepcopy(state), A_kn_op, observable.site).real
                    if measurement_Akn > 1:
                        print('non-zero:', measurement_Akn)
                    # Store it in the corresponding numpy array at trajectory index i and time index 0
                    getattr(observable, attr_name)[i, 0] = measurement_Akn

    return results




# def PhysicsTJM_1_analytical_gradient(args):
#     i, initial_state, noise_model, sim_params, H = args
#     state = copy.deepcopy(initial_state)

#     if sim_params.sample_timesteps:
#         results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)))
#     else:
#         results = np.zeros((len(sim_params.sorted_observables), 1))

#         # Create a local dictionary to hold the A_kn values for this trajectory.
#     expvals_A_kn = {obs.name: {} for obs in sim_params.sorted_observables}
#     for obs in sim_params.sorted_observables:
#         for process in noise_model.processes:
#             expvals_A_kn[obs.name][process] = np.zeros(len(sim_params.times), dtype=complex)


#     if sim_params.sample_timesteps:
#         # Time index 0:
#         for obs_index, observable in enumerate(sim_params.sorted_observables):
#             results[obs_index, 0] = copy.deepcopy(state).measure(observable)
#             for process in noise_model.processes:
#                 A_kn_op = sim_params.A_kn[process][observable.name]
#                 measurement_Akn = local_expval(copy.deepcopy(state), A_kn_op, observable.site).real
#                 # if measurement_Akn > 1e-6:
#                 #     print('non-zero')
#                 # Store it in the corresponding numpy array at trajectory index i and time index 0
#                 expvals_A_kn[observable.name][process][0] = measurement_Akn

#     # Time index > 0:
#     for j, _ in enumerate(sim_params.times[1:], start=1):
#         dynamic_TDVP(state, H, sim_params)
#         if noise_model:
#             apply_dissipation(state, noise_model, sim_params.dt)
#             state = stochastic_process(state, noise_model, sim_params.dt)
#         if sim_params.sample_timesteps:
#             temp_state = copy.deepcopy(state)
#             last_site = 0
#             for obs_index, observable in enumerate(sim_params.sorted_observables):
#                 if observable.site > last_site:
#                     for site in range(last_site, observable.site):
#                         temp_state.shift_orthogonality_center_right(site)
#                     last_site = observable.site
#                 results[obs_index, j] = temp_state.measure(observable)
#                 for process in noise_model.processes:
#                     A_kn_op = sim_params.A_kn[process][observable.name]
#                     # Compute the local expectation value for A_kn
#                     measurement_Akn = local_expval(copy.deepcopy(state), A_kn_op, observable.site).real
#                     # if measurement_Akn > 1e-6:
#                     #     print('non-zero:', measurement_Akn)
#                     # Store it in the corresponding numpy array at trajectory index i and time index 0
#                     expvals_A_kn[observable.name][process][j] = measurement_Akn
#         elif j == len(sim_params.times)-1:
#             for obs_index, observable in enumerate(sim_params.sorted_observables):
#                 results[obs_index, 0] = copy.deepcopy(state).measure(observable)
#                 for process in noise_model.processes:
#                     A_kn_op = sim_params.A_kn[process][observable.name]
#                     measurement_Akn = local_expval(copy.deepcopy(state), A_kn_op, observable.site).real
#                     # if measurement_Akn > 1:
#                     #     print('non-zero:', measurement_Akn)
#                     expvals_A_kn[observable.name][process][0] = measurement_Akn
#     #print('in TJM expvals_A_kn:', expvals_A_kn)
#     # for observable in sim_params.sorted_observables:
#     #     for process in noise_model.processes: 
#     #         for j in sim_params.times:
#     #             if expvals_A_kn[observable.name][process][j] >1e-3:
#     #                 continue
#                     #print(f'in PhysicsTJM, observable {observable.name}, site {observable.site}, process {process}, timestep {j} expval A_kn:', expvals_A_kn[observable.name][process][j])

#     return results, expvals_A_kn





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

    # Second subplot: Plot the A_kn averages over trajectories
    for observable in sim_params.observables:
        for process in noise_model.processes:
            attr_name = f"expval_A_kn_{process}"
            if hasattr(observable, attr_name):
                expval_avg = getattr(observable, attr_name)
                ax2.plot(sim_params.times, expval_avg, label=f"{observable.name} ({process})")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("A_kn Expectation Value")
    ax2.set_title("A_kn Averages over Trajectories")
    ax2.legend()

    plt.tight_layout()
    plt.show()