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

        unique_obs = set(obs.name for obs in sim_params.observables)
    

        sim_params.A_kn = {process: {} for process in noise_model.processes}
        for process, jump_op in zip(noise_model.processes, noise_model.jump_operators):
            for obs_name in unique_obs:
                observable = next(obs for obs in sim_params.observables if obs.name == obs_name)
                obs_operator = getattr(GateLibrary, obs_name).matrix
                sim_params.A_kn[process][obs_name] = calc_A_kn(jump_op, obs_operator)
        

                # sim_params.A_kn[process][obs_name] = np.array([[0, -1j],[1j, 0]], dtype=complex)
        sim_params.expvals_Master = {(obs.name, obs.site): {process: np.zeros((sim_params.N, len(sim_params.times)), dtype=complex) for process in noise_model.processes} for obs in sim_params.sorted_observables}


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
                        for process in noise_model.processes:
                            sim_params.expvals_Master[(observable.name, observable.site)][process][i,:] = expvals[(observable.name, observable.site)][process][:]
                        
            except Exception as e:
                print(f"Trajectory {i} failed with exception: {e}.")
    
    if isinstance(operator, QuantumCircuit) and isinstance(sim_params, WeakSimParams):
        sim_params.aggregate_measurements()
    else:
        sim_params.aggregate_trajectories()
     

        print("Checking for duplicate trajectories in sim_params.expvals_Master:")
        for key, proc_dict in sim_params.expvals_Master.items():
            obs_name, obs_site = key
            for process, arr in proc_dict.items():
                N = arr.shape[0]
                # Get unique rows using np.unique along axis 0.
                unique_rows = np.unique(arr, axis=0)
                num_unique = unique_rows.shape[0]

                print(f"\nObservable: {obs_name} (site {obs_site}), Process: {process}")
                print(f"  Total trajectories: {N}")
                print(f"  Unique trajectories (by np.unique): {num_unique}")

                # # Find and print duplicate trajectory pairs.
                # duplicate_pairs = []
                # for i in range(N):
                #     for j in range(i + 1, N):
                #         if np.allclose(arr[i], arr[j]):
                #             duplicate_pairs.append((i, j))
                # if duplicate_pairs:
                #     print("  Duplicate trajectory pairs found:")
                #     for pair in duplicate_pairs:
                #         print(f"    Trajectory {pair[0]} and Trajectory {pair[1]} are identical.")
                # else:
                #     print("  No duplicate trajectories found.")


        sim_params.avg_expvals = {key: {process: np.mean(arr, axis=0) for process, arr in proc_dict.items()} for key, proc_dict in sim_params.expvals_Master.items()}





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
    

    expvals = {(obs.name, obs.site): {process: np.zeros((len(sim_params.times)), dtype=complex) for process in noise_model.processes} for obs in sim_params.sorted_observables}

    # # Check if expvals in PhysivsTJM_1_analytical_gradient() is initialized correctly
    # for key, proc_dict in expvals.items():
    #     print("Observable (name, site):", key)
    #     for process, arr in proc_dict.items():
    #         print("  Process:", process)
    #         print("    Array shape:", arr.shape)
    #         print("    Values:\n", arr)

    if sim_params.sample_timesteps:
        for obs_index, observable in enumerate(sim_params.sorted_observables):
            results[obs_index, 0] = copy.deepcopy(state).measure(observable)
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
                for process in noise_model.processes:
                    A_kn_op = sim_params.A_kn[process][observable.name]
                    measurement_Akn = local_expval(temp_state, A_kn_op, observable.site).real
                    expvals[(observable.name, observable.site)][process][j] = measurement_Akn
        elif j == len(sim_params.times)-1:
            for obs_index, observable in enumerate(sim_params.sorted_observables):
                results[obs_index, 0] = copy.deepcopy(state).measure(observable)
                for process in noise_model.processes:
                    A_kn_op = sim_params.A_kn[process][observable.name]
                    measurement_Akn = local_expval(temp_state, A_kn_op, observable.site).real
                    expvals[(observable.name, observable.site)][process][0] = measurement_Akn
 
    # if results is None or expvals is None:
    #     raise ValueError("Results or expvals is None.")
    return results, expvals












if __name__ == "__main__":

        # Define the system Hamiltonian
    L = 3
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
    # noise_model = NoiseModel(['relaxation'], [gamma])

    # Define the simulation parameters
    T = 5
    dt = 0.1
    sample_timesteps = True
    N = 100
    max_bond_dim = 4
    threshold = 1e-6
    order = 1
    measurements = [Observable('x', site) for site in range(L)]  + [Observable('y', site) for site in range(L)] + [Observable('z', site) for site in range(L)]
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
    qt_params.observables = ['x','y','z']

    t, qt_ref_traj,dO, qt_A_kn_exp_vals=qutip_traj(qt_params)


    n_jump=len(qt_A_kn_exp_vals)
    n_obs=len(qt_A_kn_exp_vals[0])
    ########## TJM Example #################
    run(state, H_0, sim_params, noise_model)

    tjm_results = []
    for observable in sim_params.observables:
        tjm_results.append(observable.results)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))

    # First subplot: Plot tjm_results
    for i, result in enumerate(tjm_results):
        ax1.plot(sim_params.times, result, label=f'obs {i}')
        ax1.plot(sim_params.times, qt_ref_traj[i], label=f'qt obs{i}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Expectation Value')
    ax1.set_title('Observables Expectation Values')
   # ax1.legend()

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
   # ax2.legend()

    # Assume qt_A_kn_exp_vals has been reshaped into a list of L lists, each with 6 arrays.
    n_sites = len(qt_A_kn_exp_vals)
    n_Akn_per_site = len(qt_A_kn_exp_vals[0])  # should be 6 if ordering is as described
    observable_labels = qt_params.observables

    print('n_sites:', n_sites)
    print(' n_Akn_per_site:',  n_Akn_per_site)




    for site in range(n_sites):
        for idx in range(n_Akn_per_site):
            # Determine observable type: index mod number of types.
            obs_type = observable_labels[idx % len(observable_labels)]
            # Determine jump type: assume first half are relaxation, second half dephasing.
            jump_type = "relaxation" if idx < (n_Akn_per_site // 2) else "dephasing"
            if jump_type == "relaxation":
                print('plot relaxation Qutip')
                print('we plot this now:',qt_A_kn_exp_vals[site][idx])
                    
                ax3.plot(t, qt_A_kn_exp_vals[site][idx],
                        label=f"Site {site}, {obs_type}, {jump_type}")
            else:
                print('plot dephasing Qutip')
                print('we plot this now:',qt_A_kn_exp_vals[site][idx])
                ax3.plot(t, qt_A_kn_exp_vals[site][idx], linestyle='--',
        label=f"Site {site}, {obs_type}, {jump_type}")

    ax3.set_xlabel("Time")
    ax3.set_ylabel("A_kn Expectation Value")
    ax3.set_title("A_kn_expvals Qutip")
    # ax3.legend()

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