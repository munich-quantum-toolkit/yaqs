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


    
def qutip_traj_char(sim_params_class: SimulationParameters):

    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph


    t = np.arange(0, T + dt, dt) 

    '''QUTIP Initialization + Simulation'''

    # Define Pauli matrices
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # Construct the Ising Hamiltonian
    H = 0
    for i in range(L-1):
        H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
    for i in range(L):
        H += -g * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])



    # Construct collapse operators
    c_ops = []
    gammas = []

    # Relaxation operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma_rel) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))
        gammas.append(gamma_rel)

    # Dephasing operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma_deph) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))
        gammas.append(gamma_deph)

    #c_ops = [rel0, rel1, rel2,... rel(L-1), deph0, deph1,..., deph(L-1)]

    # Initial state
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])



    # Create obs_list based on the observables in sim_params_class.observables
    obs_list = []


    for obs_type in sim_params_class.observables:
        if obs_type.lower() == 'x':
            # For each site, create the measurement operator for 'x'
            obs_list.extend([qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])
        elif obs_type.lower() == 'y':
            obs_list.extend([qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])
        elif obs_type.lower() == 'z':
            obs_list.extend([qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])


    # # this is original version of A_kn_list initialization from propagation.py
    # A_kn_list= []
    # for i,c_op in enumerate(c_ops):
    #     for obs in obs_list:
    #         A_kn_list.append(  (1/gammas[i]) * (c_op.dag()*obs*c_op  -  0.5*obs*c_op.dag()*c_op  -  0.5*c_op.dag()*c_op*obs)   )



    A_kn_list = []
    n_types = len(sim_params_class.observables)  # number of observable types
    for site in range(L):
        # For each site, get the two collapse operators and their corresponding gamma values.
        c_op_rel = c_ops[site]             # relaxation collapse operator for this site
        gamma_rel = gammas[site]
        c_op_deph = c_ops[site + L]         # dephasing collapse operator for this site
        gamma_deph = gammas[site + L]

        # For each observable type, get the corresponding operator from obs_list.
        # The operator for the k-th observable type at this site is:
        # obs_list[site + k*L]
        for k in range(n_types):
            obs_current = obs_list[site + k * L]
            A_kn = (1 / gamma_rel) * (c_op_rel.dag() * obs_current * c_op_rel -
                                    0.5 * obs_current * c_op_rel.dag() * c_op_rel -
                                    0.5 * c_op_rel.dag() * c_op_rel * obs_current)
            A_kn_list.append(A_kn)

        for k in range(n_types):
            obs_current = obs_list[site + k * L]
            A_kn = (1 / gamma_deph) * (c_op_deph.dag() * obs_current * c_op_deph -
                                    0.5 * obs_current * c_op_deph.dag() * c_op_deph -
                                    0.5 * c_op_deph.dag() * c_op_deph * obs_current)
            A_kn_list.append(A_kn)

    # Form: A_kn_list = [x0rel0,y0rel0,z0rel0,x0deph0,y0deph0,z0deph0,x1rel1,y1rel1,...,z(L-1)deph(L-1)]


    new_obs_list = obs_list + A_kn_list

    # # Necessary in Original qutip_traj from propagation.py
    # n_obs= len(obs_list)
    # n_jump= len(c_ops)

        # Exact Lindblad solution
    result_lindblad = qt.mesolve(H, psi0, t, c_ops, new_obs_list, progress_bar=True)

    exp_vals = []
    for i in range(len(new_obs_list)):
        exp_vals.append(result_lindblad.expect[i])



    # Separate original and new expectation values from result_lindblad.
    n_obs = len(obs_list)  # number of measurement operators (should be L * n_types)
    original_exp_vals = exp_vals[:n_obs]
    new_exp_vals = exp_vals[n_obs:]  # these correspond to the A_kn operators

    # Determine parameters:
    n_types = len(sim_params_class.observables)    # e.g., 3 for ['x','y','z']
    n_noise = 2  # since you have relaxation and dephasing
    n_Akn_per_site = n_noise * n_types  # e.g., 2*3 = 6

    # new_exp_vals should have a total length of L * n_Akn_per_site.
    # Reshape it into a list of L lists, each containing n_Akn_per_site arrays.
    A_kn_exp_vals = [new_exp_vals[site * n_Akn_per_site : (site + 1) * n_Akn_per_site]
                    for site in range(sim_params_class.L)]

    # Compute the derivative for each A_kn expectation value using trapezoidal integration.
    d_On_d_gk = [
        [trapezoidal(A_kn_exp_vals[site][j], t) for j in range(n_Akn_per_site)]
        for site in range(sim_params_class.L)
    ]

    # # Original reshape from propagation.py: Reshape new_exp_vals to be a list of lists with dimensions n_jump times n_obs
    # A_kn_exp_vals = [new_exp_vals[i * n_obs:(i + 1) * n_obs] for i in range(n_jump)]

    # # Compute the integral of the new expectation values to obtain the derivatives
    # d_On_d_gk = [ [trapezoidal(A_kn_exp_vals[i][j],t)  for j in range(n_obs)] for i in range(n_jump) ]

    # return t, original_exp_vals, d_On_d_gk, A_kn_exp_vals
    return t, original_exp_vals, d_On_d_gk
  



def run_char(initial_state: MPS, operator, sim_params, noise_model: NoiseModel=None, parallel: bool=True):
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
     
        # # Optional: check how many A_kn trajectories are equal
        # print("Checking for duplicate trajectories in sim_params.expvals_Master:")
        # for key, proc_dict in sim_params.expvals_Master.items():
        #     obs_name, obs_site = key
        #     for process, arr in proc_dict.items():
        #         N = arr.shape[0]
    
        #         unique_rows = np.unique(arr, axis=0)
        #         num_unique = unique_rows.shape[0]

        #         print(f"\nObservable: {obs_name} (site {obs_site}), Process: {process}")
        #         print(f"  Total trajectories: {N}")
        #         print(f"  Unique trajectories (by np.unique): {num_unique}")


        # Average A_kn means over trajectories and store in extra dictionary
        sim_params.avg_expvals = {key: {process: np.mean(arr, axis=0) for process, arr in proc_dict.items()} for key, proc_dict in sim_params.expvals_Master.items()}
        n_sites = max(obs.site for obs in sim_params.sorted_observables) + 1

        sim_params.d_On_d_gk = [
    [
        trapezoidal(sim_params.avg_expvals[(obs.name, obs.site)][process], sim_params.times)
        for process in noise_model.processes
        for obs in sim_params.sorted_observables if obs.site == site
    ]
    for site in range(n_sites)
    ]



   


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
    N = 500
    max_bond_dim = 4
    threshold = 1e-6
    order = 1
    measurements = [Observable('x', site) for site in range(L)]  + [Observable('y', site) for site in range(L)]  + [Observable('z', site) for site in range(L)]
    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)


    '''QUTIP calculation'''

    qt_params = SimulationParameters()

    qt_params.T = T
    qt_params.dt = dt
    qt_params.L = L
    qt_params.J = J
    qt_params.g = g
    qt_params.gamma_rel = gamma
    qt_params.gamma_deph = gamma
    qt_params.observables = ['x','y', 'z']

    t, qt_ref_traj,  d_On_d_gk_qt =qutip_traj_char(qt_params)






    ########## TJM Example #################
    run_char(state, H_0, sim_params, noise_model)

    tjm_results = []
    for observable in sim_params.observables:
        tjm_results.append(observable.results)




    # '''Restructure Qutip A_kn means into same structure as TJM A_kn means:'''

    # n_sites = len(qt_A_kn_exp_vals)
    # n_types = len(qt_params.observables)
    # n_noise = len(noise_model.processes)  
    # n_Akn_per_site = n_noise * n_types

    # # Create a new dictionary to hold the Qutip data in the same structure as sim_params.avg_expvals.
    # qt_avg_dict = {}

    # for site in range(n_sites):
    #     for type_index in range(n_types):
    #         key = (qt_params.observables[type_index], site)
    #         qt_avg_dict[key] = {}
    #         for noise_index in range(n_noise):
    #             process = noise_model.processes[noise_index]
    #             # Calculate the index within the sublist for the given noise process and observable type.
    #             idx = noise_index * n_types + type_index
    #             qt_avg_dict[key][process] = qt_A_kn_exp_vals[site][idx]

    # # Print out the structure for inspection.
    # print("Structure of qt_avg_dict:")
    # for key, proc_dict in qt_avg_dict.items():
    #     print(f"Key (Observable, site): {key}")
    #     for process, arr in proc_dict.items():
    #         print(f"    Process: {process}, array shape: {np.shape(arr)}")

    # '''Structure of TJM and Qutip A_kn means is equal now.'''


    # Convert both to numpy arrays
    array1 = np.array(sim_params.d_On_d_gk)
    array2 = np.array(d_On_d_gk_qt)

    # Use np.allclose to compare with a tolerance for floating point differences
    if np.allclose(array1, array2, rtol=1e-2, atol=1e-2):
        print("sim_params.d_On_d_gk and d_On_d_gk_qt are the same!")
    else:
        print("They are different.")

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

    # Loop over sites (assuming L is the number of sites)
    for site in range(L):
        # Convert to NumPy arrays
        arr_tjm = np.array(sim_params.d_On_d_gk[site])  # shape (6, 51)
        arr_qt  = np.array(d_On_d_gk_qt[site])           # shape (6, 51)
        
        # x-axis: 51 points (0 to 50)
        x = np.arange(51)
        
        # Plot each of the 6 trajectories separately
        for obs_idx in range(6):
            ax1.plot(x, arr_tjm[obs_idx, :], marker='o', label=f"Site {site}, Obs {obs_idx}")
            ax2.plot(x, arr_qt[obs_idx, :], marker='o', label=f"Site {site}, Obs {obs_idx}")

    ax1.set_title("TJM d_On_d_gk")
    ax1.set_xlabel("Time index (0-50)")
    ax1.set_ylabel("Integrated Value")
    ax1.legend()

    ax2.set_title("Qutip d_On_d_gk")
    ax2.set_xlabel("Time index (0-50)")
    ax2.set_ylabel("Integrated Value")
    ax2.legend()

    plt.tight_layout()
    plt.show()














    '''PLOTTING 1) QUTIP + TJM Simulation 2) TJM A_kn means 3) Qutip A_kn means'''

    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))

    # # First subplot: Plot tjm_results
    # for i, result in enumerate(tjm_results):
    #     ax1.plot(sim_params.times, result, label=f'obs {i}')
    #     ax1.plot(sim_params.times, qt_ref_traj[i], label=f'qt obs{i}')
    # ax1.set_xlabel('Time')
    # ax1.set_ylabel('Expectation Value')
    # ax1.set_title('Observables Expectation Values')


    # # Second subplot: Plot averaged A_kn expectation values over trajectories
    # for key, process_dict in sim_params.avg_expvals.items():
    #     obs_name, obs_site = key
    #     for process, avg_exp in process_dict.items():
    #         if process == 'relaxation':
    #             ax2.plot(sim_params.times, avg_exp, label=f"{obs_name} (site {obs_site}, {process})")
    #         else: 
    #             ax2.plot(sim_params.times, avg_exp, linestyle ='--', label=f"{obs_name} (site {obs_site}, {process})")
    # ax2.set_xlabel("Time")
    # ax2.set_ylabel("A_kn Expectation Value")
    # ax2.set_title("A_kn Averages TJM")



    # n_sites = len(qt_A_kn_exp_vals)
    # n_Akn_per_site = len(qt_A_kn_exp_vals[0])  
    # observable_labels = qt_params.observables

    # # Third subplot: Plot Qutip A_kn expectation values using the new dictionary format.
    # for key, process_dict in qt_avg_dict.items():
    #     obs_name, obs_site = key
    #     for process, arr in process_dict.items():
    #         if process == 'relaxation':
    #             ax3.plot(t, arr, label=f"{obs_name} (site {obs_site}, {process})")
    #         else:
    #             ax3.plot(t, arr, linestyle='--', label=f"{obs_name} (site {obs_site}, {process})")
    # ax3.set_xlabel("Time")
    # ax3.set_ylabel("A_kn Expectation Value")
    # ax3.set_title("A_kn_expvals Qutip")




    # # Gather handles and labels for each original subplot
    # handles1, labels1 = ax1.get_legend_handles_labels()
    # handles2, labels2 = ax2.get_legend_handles_labels()
    # handles3, labels3 = ax3.get_legend_handles_labels()

    # # Create a new figure with 1 row and 3 columns
    # fig_legend, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    # # Set a title or label for each legend column if desired:
    # axs[0].set_title("Obs Expectation")
    # axs[1].set_title("A_kn Averages TJM")
    # axs[2].set_title("A_kn_expvals Qutip")

    # # Display each legend in its own axis
    # axs[0].legend(handles1, labels1, loc='center')
    # axs[1].legend(handles2, labels2, loc='center')
    # axs[2].legend(handles3, labels3, loc='center')

    # # Turn off the axes (no ticks, spines, etc.)
    # for ax in axs:
    #     ax.axis('off')

    # plt.tight_layout()
    # plt.show()