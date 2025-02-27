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

from yaqs.noise_char.optimization import *
from yaqs.noise_char.propagation import *

from yaqs.core.data_structures.simulation_parameters import StrongSimParams, WeakSimParams






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

    if isinstance(operator, MPO) or isinstance(sim_params, StrongSimParams):
    
        for observable in sim_params.sorted_observables:
            observable.initialize(sim_params)

        # initialize A_kn operators
        unique_obs = set(obs.name for obs in sim_params.observables)
        sim_params.A_kn = {process: {} for process in noise_model.processes}
        for process, jump_op in zip(noise_model.processes, noise_model.jump_operators):
            for obs_name in unique_obs:
                observable = next(obs for obs in sim_params.observables if obs.name == obs_name)
                obs_operator = getattr(GateLibrary, obs_name).matrix
                sim_params.A_kn[process][obs_name] = calc_A_kn(jump_op, obs_operator)

        #initialize Master dictionary to store A_kn exp values
        sim_params.expvals_Master = {(obs.name, obs.site): {process: np.zeros((sim_params.N, len(sim_params.times)), dtype=complex) for process in noise_model.processes} for obs in sim_params.sorted_observables}


   
    initial_state.normalize('B')

    
    args = [(i, initial_state, noise_model, sim_params, operator) for i in range(sim_params.N)]
    
    if parallel:
        max_workers = max(1, multiprocessing.cpu_count() - 1)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(PhysicsTJM_1_analytical_gradient, arg): arg[0] for arg in args}
            with tqdm(total=sim_params.N, desc="Running trajectories", ncols=80) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]
                    try:
                        # output is now tjm average , A_kn expvals
                        result, expvals = future.result()
                        
                        if result is None or expvals is None:
                            print(f"Trajectory {i} returned None.")
                            continue
                        if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
                            sim_params.measurements[i] = result
                        else:
                            for obs_index, observable in enumerate(sim_params.sorted_observables):
                                observable.trajectories[i] = result[obs_index]

                                # put A_kn expectation values into Master dictionary
                                for process in noise_model.processes:
                                    sim_params.expvals_Master[(observable.name, observable.site)][process][i,:] = expvals[(observable.name, observable.site)][process][:]
                    except Exception as e:
                        print(f"\nTrajectory {i} failed with exception: {e}.")
                    finally:
                        pbar.update(1)
    else:
        
        for i, arg in tqdm(enumerate(args), total=sim_params.N, desc="Running trajectories", ncols=80):
   
            try:
                result, expvals = PhysicsTJM_1_analytical_gradient(arg)
                
                if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
                    sim_params.measurements[i] = result
                else:
                    for obs_index, observable in enumerate(sim_params.sorted_observables):
                        observable.trajectories[i] = result[obs_index]

                        # put A_kn expectation values into Master dictionary
                        for process in noise_model.processes:
                            sim_params.expvals_Master[(observable.name, observable.site)][process][i,:] = expvals[(observable.name, observable.site)][process][:]
                        
            except Exception as e:
                print(f"Trajectory {i} failed with exception: {e}.")
    
    if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
        sim_params.aggregate_measurements()
    else:
        sim_params.aggregate_trajectories()
     
        # Optional: check how many A_kn trajectories are equal
        print("Checking for duplicate trajectories in sim_params.expvals_Master:")
        for key, proc_dict in sim_params.expvals_Master.items():
            obs_name, obs_site = key
            for process, arr in proc_dict.items():
                N = arr.shape[0]
    
                unique_rows = np.unique(arr, axis=0)
                num_unique = unique_rows.shape[0]

                print(f"\nObservable: {obs_name} (site {obs_site}), Process: {process}")
                print(f"  Total trajectories: {N}")
                print(f"  Unique trajectories (by np.unique): {num_unique}")


        # Average A_kn means over trajectories and store in extra dictionary
        sim_params.avg_expvals = {key: {process: np.mean(arr, axis=0) for process, arr in proc_dict.items()} for key, proc_dict in sim_params.expvals_Master.items()}




# calculate A_kn operators
def calc_A_kn(L,obs):
    A_kn = L.conj().T @ obs @ L - 0.5 * obs @ L.conj().T @ L -0.5 * L.conj().T @ L @ obs
    return A_kn



def PhysicsTJM_1_analytical_gradient(args):
    i, initial_state, noise_model, sim_params, H = args
    state = copy.deepcopy(initial_state)

    if sim_params.sample_timesteps:
        results = np.zeros((len(sim_params.sorted_observables), len(sim_params.times)))
    else:
        results = np.zeros((len(sim_params.sorted_observables), 1))
    
    # initialize dictionary to store A_kn exp values of a single trajectory
    expvals = {(obs.name, obs.site): {process: np.zeros((len(sim_params.times)), dtype=complex) for process in noise_model.processes} for obs in sim_params.sorted_observables}

    if sim_params.sample_timesteps:
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            results[obs_index, 0] = copy.deepcopy(state).measure(observable)

            # measure A_kn exp values at time 0
            for process in noise_model.processes:
                A_kn_op = sim_params.A_kn[process][observable.name]
                measurement_Akn = local_expval(copy.deepcopy(state), A_kn_op, observable.site).real
                expvals[(observable.name, observable.site)][process][0] = measurement_Akn



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

                #measure A_kn exp values at time j
                for process in noise_model.processes:
                    A_kn_op = sim_params.A_kn[process][observable.name]
                    measurement_Akn = local_expval(temp_state, A_kn_op, observable.site).real
                    expvals[(observable.name, observable.site)][process][j] = measurement_Akn

        elif j == len(sim_params.times)-1:
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                results[obs_index, 0] = copy.deepcopy(state).measure(observable)


                #measure A_kn exp values at time T
                for process in noise_model.processes:
                    A_kn_op = sim_params.A_kn[process][observable.name]
                    measurement_Akn = local_expval(temp_state, A_kn_op, observable.site).real
                    expvals[(observable.name, observable.site)][process][0] = measurement_Akn
                    
    # also return A_kn exp values
    return results, expvals







if __name__ == "__main__":

       
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
    T = 5
    dt = 0.1
    sample_timesteps = True
    N = 100
    max_bond_dim = 4
    threshold = 1e-6
    order = 1
    measurements = [Observable('x', site) for site in range(L)]  + [Observable('y', site) for site in range(L)]  + [Observable('z', site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)


    '''QUTIP calculation of A_kn_exp_vals'''

    qt_params = SimulationParameters()

    qt_params.T = T
    qt_params.dt = dt
    qt_params.L = L
    qt_params.J = J
    qt_params.g = g
    qt_params.gamma_rel = gamma
    qt_params.gamma_deph = gamma
    qt_params.observables = ['x','y', 'z']

    t, qt_ref_traj,dO, qt_A_kn_exp_vals=qutip_traj(qt_params)




    ########## TJM Example #################
    run(state, H_0, sim_params, noise_model)

    tjm_results = []
    for observable in sim_params.observables:
        tjm_results.append(observable.results)




    '''Restructure Qutip A_kn means into same structure as TJM A_kn means:'''

    n_sites = len(qt_A_kn_exp_vals)
    n_types = len(qt_params.observables)
    n_noise = len(noise_model.processes)  
    n_Akn_per_site = n_noise * n_types

    # Create a new dictionary to hold the Qutip data in the same structure as sim_params.avg_expvals.
    qt_avg_dict = {}

    for site in range(n_sites):
        for type_index in range(n_types):
            key = (qt_params.observables[type_index], site)
            qt_avg_dict[key] = {}
            for noise_index in range(n_noise):
                process = noise_model.processes[noise_index]
                # Calculate the index within the sublist for the given noise process and observable type.
                idx = noise_index * n_types + type_index
                qt_avg_dict[key][process] = qt_A_kn_exp_vals[site][idx]

    # Print out the structure for inspection.
    print("Structure of qt_avg_dict:")
    for key, proc_dict in qt_avg_dict.items():
        print(f"Key (Observable, site): {key}")
        for process, arr in proc_dict.items():
            print(f"    Process: {process}, array shape: {np.shape(arr)}")

    '''Structure of TJM and Qutip A_kn means is equal now.'''





    # PLOTTING 1) QUTIP + TJM Simulation 2) TJM A_kn means 3) Qutip A_kn means

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))

    # First subplot: Plot tjm_results
    for i, result in enumerate(tjm_results):
        ax1.plot(sim_params.times, result, label=f'obs {i}')
        ax1.plot(sim_params.times, qt_ref_traj[i], label=f'qt obs{i}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Expectation Value')
    ax1.set_title('Observables Expectation Values')


    # Second subplot: Plot averaged A_kn expectation values over trajectories
    for key, process_dict in sim_params.avg_expvals.items():
        obs_name, obs_site = key
        for process, avg_exp in process_dict.items():
            if process == 'relaxation':
                ax2.plot(sim_params.times, avg_exp, label=f"{obs_name} (site {obs_site}, {process})")
            else: 
                ax2.plot(sim_params.times, avg_exp, linestyle ='--', label=f"{obs_name} (site {obs_site}, {process})")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("A_kn Expectation Value")
    ax2.set_title("A_kn Averages TJM")



    n_sites = len(qt_A_kn_exp_vals)
    n_Akn_per_site = len(qt_A_kn_exp_vals[0])  
    observable_labels = qt_params.observables

    # Third subplot: Plot Qutip A_kn expectation values using the new dictionary format.
    for key, process_dict in qt_avg_dict.items():
        obs_name, obs_site = key
        for process, arr in process_dict.items():
            if process == 'relaxation':
                ax3.plot(t, arr, label=f"{obs_name} (site {obs_site}, {process})")
            else:
                ax3.plot(t, arr, linestyle='--', label=f"{obs_name} (site {obs_site}, {process})")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("A_kn Expectation Value")
    ax3.set_title("A_kn_expvals Qutip")




    # Gather handles and labels for each original subplot
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()

    # Create a new figure with 1 row and 3 columns
    fig_legend, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    # Set a title or label for each legend column if desired:
    axs[0].set_title("Obs Expectation")
    axs[1].set_title("A_kn Averages TJM")
    axs[2].set_title("A_kn_expvals Qutip")

    # Display each legend in its own axis
    axs[0].legend(handles1, labels1, loc='center')
    axs[1].legend(handles2, labels2, loc='center')
    axs[2].legend(handles3, labels3, loc='center')

    # Turn off the axes (no ticks, spines, etc.)
    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()