# import matplotlib.pyplot as plt

# from .initializations import initialize_Ising
# from .Lindblad_functions import *
# from .vector_MCWF import *
# from TN.metrics import fidelity

# TODO: Fix plot titles and labels
def test_N(system, processes, num_trajectories_list, timesteps, T):
    fidelities = []
    rho_SO_full = time_evolution_superoperator(system.density_matrix, system.H_0, system.operators, system.coupling_factors, T, plot_spectrum=False)
    previous_input = ''
    dt = T/timesteps
    for num_trajectories in num_trajectories_list:
        rho_stochastic_result, previous_input = stochastic_unraveling(system, num_trajectories, T, dt, processes, previous_input=previous_input)
        f = fidelity(rho_SO_full, rho_stochastic_result)
        fidelities.append(f)
        print("F:", f)

    plt.title("Fidelity vs. Trajectories, L=%d, T=%f, Timesteps=%d" % (system.L, T, timesteps))
    plt.xlabel('Trajectories')
    plt.ylabel("Fidelity")
    plt.ylim(min(fidelities)-.005, 1.005)
    plt.plot(num_trajectories_list, fidelities, marker='.', linestyle='')
    plt.show()


def test_dt(system, processes, num_trajectories, T, dt_list):
    plt.title("Fidelity vs. Timestep dt, L=%d, T=%f" % (system.L, T))
    plt.xlabel('Trajectories')
    plt.ylabel("Fidelity")

    num_trajectories_list = range(1, num_trajectories, 1)
    rho_SO_full = time_evolution_superoperator(system.density_matrix, system.H_0, system.operators, system.coupling_factors, T, plot_spectrum=False)
    for dt in dt_list:
        print(dt)
        fidelities = []
        previous_input = ''
        for N in num_trajectories_list:
            rho_stochastic_result, previous_input = stochastic_unraveling(system, N, T, dt, processes, previous_input)
            fidelities.append(fidelity(rho_SO_full, rho_stochastic_result))
        plt.plot(range(1, num_trajectories), fidelities, marker='o', linestyle='', label=int(T/dt))

    plt.ylim(min(fidelities)-.005, 1.005)
    plt.legend()
    plt.show()


# TODO: Write this correctly by feeding in parameters
def test_L(SU_system, SO_system, H_s_subsystem, density_matrix_subsystem, num_trajectories, T, dt, L_list):
    plt.title("Fidelity vs. Particle Number L, N=%d, T=%f, dt=%f" % (num_trajectories, T, dt))
    plt.xlabel('Trajectories')
    plt.ylabel("Fidelity")

    psi_local_initial = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    # psi_list = SU_system.psi_list
    # psi_full = SU_system.psi_list

    Lindblad_operator_subsystem = SU_system.operators[0]
    coupling_factor = SU_system.coupling_factors

    global_T1 = 26e-6
    freq = 5
    d = 2
    fidelities = []
    num_trajectories_list = range(1, num_trajectories, 1)
    for L in L_list:
        print(L)
        # # Initialization of effective Hamiltonian
        # H_eff = H_s_subsystem - 1j*coupling_factor[0]/2*(np.conj(Lindblad_operator_subsystem.T) @ Lindblad_operator_subsystem)
        # psi_list = []
        # for i in range(L):
        #     psi_list.append(psi_subsystem)
        # density_matrix = multi_site_tensor_product(density_matrix_subsystem, L)
        # H = create_local_operator_chain(H_s_subsystem, L)
        # Lindblad_operators = create_list_of_local_operators(Lindblad_operator_subsystem, L)
        # coupling_factors = coupling_factor * L

        # SU_system = System(rho=[], psi_list=psi_list, H=H_eff, L=len(psi_list), operators=[Lindblad_operator_subsystem], coupling_factors=coupling_factor)
        # SO_system = System(rho=density_matrix, psi_list=[], H=H, L=len(psi_list), operators=Lindblad_operators, coupling_factors=coupling_factors)
        # Initialize local operators for MCWF
        psi_list = L*[psi_local_initial]
        psi_full = multi_site_tensor_product(psi_local_initial, L)
        density_matrix_local = np.outer(psi_local_initial, np.conj(psi_local_initial))
        density_matrix = multi_site_tensor_product(density_matrix_local, L)

        # For superoperator

        global_relaxation_gamma = 1/np.sqrt(2*global_T1)
        # global_relaxation_gamma = 1/global_T1

        local_relaxation_op = deexcitation_operator(d)
        local_relaxation_gammas = L*[global_relaxation_gamma]

        relaxation_ops = create_list_of_local_operators(local_relaxation_op, L)
        Lindblad_operators = [*relaxation_ops]
        coupling_factors = [*local_relaxation_gammas]

        # Initialization of effective Hamiltonian
        interactions = True
        J = 1
        g = 1
        if not interactions:
            H_s = H_system(freq, d)
        else:
            H_s = initialize_Ising(L, J, g)
        if not interactions:
            H = create_local_operator_chain(H_s, L)
        else:
            H = H_s

        local_gammas = []
        local_gammas.append(global_relaxation_gamma)
        global_ops = []
        for op in relaxation_ops:
            global_ops.append(op)
        # TODO: Expand to different Hamiltonians/different gammas for each site
        H_jump = np.zeros(H.shape)
        # for i, op in enumerate(local_ops):
        #     H_jump += local_gammas[i]*np.conj(op.T) @ op
        for i, op in enumerate(global_ops):
            #TODO: Change iteration over local_gammas
            H_jump += local_gammas[0]*np.conj(op.T) @ op

        H_eff = H - 1j/2*H_jump

        SU_system = System(rho=[], psi_list=psi_full, H=H_eff, L=L, operators=global_ops, coupling_factors=local_gammas)
        SO_system = System(rho=density_matrix, psi_list=[], H=H, L=L, operators=Lindblad_operators, coupling_factors=coupling_factors)
        rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, T, plot_spectrum=False)
        previous_input = ''
        fidelities = []
        for N in num_trajectories_list:
            rho_stochastic_result, previous_input = stochastic_unraveling(SU_system, N, T, dt, processes='', previous_input=previous_input)
            fidelities.append(fidelity(rho_SO_full, rho_stochastic_result))
        # rho_stochastic_result, _ = stochastic_unraveling(SU_system, num_trajectories, T, dt, processes='', previous_input='')
        # fidelities.append(fidelity(rho_SO_full, rho_stochastic_result))
        plt.plot(range(1, num_trajectories), fidelities, marker='o', linestyle='', label=L)

    plt.ylim(min(fidelities)-.005, 1.005)
    # plt.plot(L_list, fidelities, marker='o', linestyle='')
    plt.legend()
    plt.show()

# TODO: This keeps all trajectories which is probably computationally slow, make it true Monte Carlo
def test_linear_T(system, processes, num_trajectories, T, timesteps):
    fidelities = []
    dt = T/timesteps
    rho_at_given_times, times_sampled, noise, vectors = stochastic_unraveling_test_T(system, num_trajectories, T, dt, processes, timesteps)

    rho_SO = []
    for i, t in enumerate(times_sampled):
        rho_SO_full = time_evolution_superoperator(system.density_matrix, system.H_0, system.operators, system.coupling_factors, t, plot_spectrum=False)
        rho_SO.append(rho_SO_full)
        # rho_SO_full = time_evolution_superoperator(system.density_matrix, SO_system.H_0, SO_system.operators, SO_system.coupling_factors, t, plot_spectrum=False)
        f = fidelity(rho_SO_full, rho_at_given_times[i])
        fidelities.append(f)
        print(f)

    plt.title("Fidelity vs. Elapsed time T, L=%d, N=%d, Timesteps=%d" % (system.L, num_trajectories, timesteps))
    plt.xlabel('Elapsed time T')
    plt.ylabel("Fidelity")
    plt.ylim(min(fidelities)-.005, 1.005)
    plt.plot(times_sampled, fidelities, marker='o', linestyle='')

    plt.show()
    return rho_at_given_times, times_sampled, noise, fidelities, vectors, rho_SO

# def test_N_multi_timestep(SU_system, SO_system, processes, num_trajectories_list, T):
#     rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, T, plot_spectrum=False)

#     timesteps_list = [10, 100, 1000]
#     for timesteps in timesteps_list:
#         fidelities = []
#         previous_input = ''
#         dt = T/timesteps
#         for num_trajectories in num_trajectories_list:
#             rho_stochastic_result, previous_input = stochastic_unraveling_test_N(SU_system, num_trajectories, T, dt, processes, previous_input=previous_input)
#             f = fidelity(rho_SO_full, rho_stochastic_result)
#             fidelities.append(f)
#             print(f)
#         plt.plot(num_trajectories_list, fidelities, marker='.', linestyle='', label='Timesteps='+str(timesteps))

#     plt.title("Fidelity vs. N trajectories, L=%d, T=%f" % (SU_system.L, T))
#     plt.xlabel('N trajectories')
#     plt.ylabel("Fidelity")
#     plt.legend()
#     plt.show()





# def test_log_T(SU_system, SO_system, processes, num_trajectories, T, timesteps, samples):

#     fidelities = []
#     rho_at_given_times, times_sampled = stochastic_unraveling_test_log_T_independent_sims(SU_system, num_trajectories, T, processes, samples, timesteps)

#     for i, t in enumerate(times_sampled):
#         rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, t, plot_spectrum=False)
#         f = fidelity(rho_SO_full, rho_at_given_times[i])
#         fidelities.append(f)
#         print(f)

#     plt.title("Fidelity vs. Elapsed time T, L=%d, N=%d, Timesteps=%d, Samples=%d" % (SU_system.L, num_trajectories, timesteps, samples))
#     plt.xlabel('Elapsed time T')
#     plt.ylabel("Fidelity")
#     plt.xscale('log')
#     plt.ylim(min(fidelities)-.005, 1.005)
#     plt.plot(times_sampled, fidelities, marker='o', linestyle='')

#     plt.show()


# def test_log_T_adaptive_timesteps(SU_system, SO_system, processes, num_trajectories, timesteps_per_interval, samples):

#     rho_at_given_times, times_sampled = stochastic_unraveling_test_log_T_dependent_sims(SU_system, num_trajectories, processes, samples, timesteps_per_interval)
#     fidelities = []
#     for i, t in enumerate(times_sampled):
#         rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, t, plot_spectrum=False)
#         # rho_SO_full = np.array([[1, 0], [0, 0]])

#         f = fidelity(rho_SO_full, rho_at_given_times[i])
#         fidelities.append(f)
#         print(f)

#     plt.ylim(min(fidelities)-.005, 1.005)
#     plt.plot(times_sampled, fidelities, marker='', linestyle='-')

#     plt.title("Fidelity vs. Elapsed time T, L=%d, N=%d, Timesteps=%d, Samples=%d" % (SU_system.L, num_trajectories, timesteps_per_interval, samples))
#     plt.xlabel('Elapsed time T')
#     plt.ylabel("Fidelity")
#     plt.xscale('log')
#     plt.show()


# def test_log_T_adaptive_timesteps_multi_N(SU_system, SO_system, processes, timesteps, samples):

#     N_list = [3, 10, 50, 100, 1000]
#     for num_trajectories in N_list:
#         rho_at_given_times, times_sampled = stochastic_unraveling_test_log_T_dependent_sims(SU_system, num_trajectories, processes, samples, timesteps)
#         fidelities = []
#         for i, t in enumerate(times_sampled):
#             rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, t, plot_spectrum=False)
#             f = fidelity(rho_SO_full, rho_at_given_times[i])
#             fidelities.append(f)
#             print(f)

#         plt.plot(times_sampled, fidelities, marker='', linestyle='-', label='N = '+str(num_trajectories))

#     plt.title("Fidelity vs. Elapsed time T, L=%d, Timesteps per Interval=%d, Samples=%d" % (SU_system.L, timesteps, samples))
#     plt.xlabel('Elapsed time T')
#     plt.ylabel("Fidelity")
#     plt.xscale('log')
#     plt.legend()
#     plt.show()


# def test_log_T_adaptive_timesteps_multi_dt(SU_system, SO_system, processes, num_trajectories, samples):

#     timestep_list = [10, 100, 1000] #, 10000]
#     for timesteps in timestep_list:
#         rho_at_given_times, times_sampled = stochastic_unraveling_test_log_T_dependent_sims(SU_system, num_trajectories, processes, samples, timesteps)
#         fidelities = []
#         for i, t in enumerate(times_sampled):
#             rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, t, plot_spectrum=False)

#             f = fidelity(rho_SO_full, rho_at_given_times[i])
#             fidelities.append(f)
#             print(f)

#         plt.plot(times_sampled, fidelities, marker='', linestyle='-', label='Steps = '+str(timesteps))

#     plt.title("Fidelity vs. Elapsed time T, L=%d, N=%d, Samples=%d" % (SU_system.L, num_trajectories, samples))
#     plt.xlabel('Elapsed time T')
#     plt.ylabel("Fidelity")
#     plt.xscale('log')
#     plt.legend()
#     plt.show()


# def test_log_T_adaptive_timesteps_multi_L(SU_system, SO_system, processes, num_trajectories, timesteps, samples):

#     L_list = [1, 2, 3, 4, 5]
#     for L in L_list:
#         d = 2  # d-level system
#         processes = ['relaxation', 'dephasing']
#         interactions = False

#         # Define qubits
#         psi_local_initial = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
#         freq = 5e9
#         freq_list = L*[freq]

#         # Define operators and their gammas
#         global_T1 = 27e-6
#         global_T2star = 39e-6
#         global_temp = 50e-3


#         # Operators used for SU and SO
#         local_ops = []
#         local_gammas = []
#         # Lindblad_operators = []
#         # coupling_factors = []
#         if 'relaxation' in processes:
#             global_relaxation_gamma = 1/np.sqrt(2*global_T1)

#             local_relaxation_op = deexcitation_operator(d)
#             local_relaxation_gammas = L*[global_relaxation_gamma]

#             relaxation_ops = create_list_of_local_operators(local_relaxation_op, L)

#             local_ops.append(local_relaxation_op)
#             local_gammas.append(global_relaxation_gamma)

#             # Lindblad_operators.append(relaxation_ops)
#             # coupling_factors.append(local_relaxation_gammas)

#         if 'thermal' in processes:
#             global_thermal_gamma = 1/np.sqrt(global_temp)
#             local_thermal_op = excitation_operator(d)
#             local_thermal_gammas = L*[global_thermal_gamma]

#             thermal_ops = create_list_of_local_operators(local_thermal_op, L)

#             local_ops.append(local_thermal_op)
#             local_gammas.append(global_thermal_gamma)

#             # Lindblad_operators.append(*thermal_ops)
#             # coupling_factors.append(*local_thermal_gammas)

#         if 'dephasing' in processes:
#             global_dephasing_gamma = 1/np.sqrt(2*global_T2star)

#             local_dephasing_op = sigma_z(d)
#             local_dephasing_gammas = L*[global_dephasing_gamma]

#             dephasing_ops = create_list_of_local_operators(local_dephasing_op, L)

#             local_ops.append(local_dephasing_op)
#             local_gammas.append(global_dephasing_gamma)

#             # Lindblad_operators.append(*dephasing_ops)
#             # coupling_factors.append(*local_dephasing_gammas)

#         # Initialization of effective Hamiltonian
#         if not interactions:
#             H_s = H_system(freq, d)
#         else:
#             H_s = H_system(freq, d)
#         # TODO: Expand to different Hamiltonians/different gammas for each site
#         H_jump = np.zeros(H_s.shape)
#         for i, op in enumerate(local_ops):
#             H_jump += local_gammas[i]*np.conj(op.T) @ op

#         H_eff = H_s - 1j/2*H_jump
#         # Initialize local operators for MCWF
#         psi_list = L*[psi_local_initial]
#         density_matrix_local = np.outer(psi_local_initial, np.conj(psi_local_initial))
#         density_matrix = multi_site_tensor_product(density_matrix_local, L)

#         # For superoperator
#         Lindblad_operators = [*relaxation_ops, *dephasing_ops]
#         coupling_factors = [*local_relaxation_gammas, *local_dephasing_gammas]
#         H = create_local_operator_chain(H_s, L)
#         SU_system = System(rho=[], psi_list=psi_list, H=H_eff, L=len(psi_list), operators=local_ops, coupling_factors=local_gammas)
#         SO_system = System(rho=density_matrix, psi_list=[], H=H, L=len(psi_list), operators=Lindblad_operators, coupling_factors=coupling_factors)

#         rho_at_given_times, times_sampled = stochastic_unraveling_test_log_T_dependent_sims(SU_system, num_trajectories, processes, samples, timesteps)
#         fidelities = []
#         for i, t in enumerate(times_sampled):
#             rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, t, plot_spectrum=False)
#             f = fidelity(rho_SO_full, rho_at_given_times[i])
#             fidelities.append(f)
#             print(f)

#         plt.plot(times_sampled, fidelities, marker='', linestyle='-', label='L = '+str(L))

#     plt.title("Fidelity vs. Elapsed time T, L=%d, N=%d, Timesteps per Interval=%d, Samples=%d" % (SU_system.L, num_trajectories, timesteps, samples))
#     plt.xlabel('Elapsed time T')
#     plt.ylabel("Fidelity")
#     plt.xscale('log')

#     plt.legend()
#     plt.show()

# def test_ratio_T_dt(SU_system, SO_system, num_trajectories, T_list, dt_list):
#     plt.title("Fidelity vs. Ratio T/dt, N=%d, L=%d" % (num_trajectories, SU_system.L))
#     plt.xlabel('Ratio T/dt')
#     plt.ylabel("Fidelity")
#     plt.xscale('log')

#     fidelities = []
#     for T in T_list:
#         fidelities = []
#         x = []
#         for dt in dt_list:
#             if dt > T:
#                 continue
#             print(T, dt)
#             rho_stochastic_result, _ = stochastic_unraveling(SU_system, num_trajectories, T, dt)
#             rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, T, plot_spectrum=False)
#             fidelities.append(fidelity(rho_SO_full, rho_stochastic_result))
#             x.append(T/dt)
#         plt.plot(x, fidelities, marker='o', linestyle='', label='T='+str(T))

#     plt.legend()
#     plt.show()


# def test_ratio_N_L(SU_system, SO_system, H_s_subsystem, density_matrix_subsystem, num_trajectories_list, T, dt, L_list):
#     plt.title("Fidelity vs. Ratio N/L, T=%f, dt=%f" % (T, dt))
#     plt.xlabel('Ratio N/L')
#     plt.ylabel("Fidelity")
#     # plt.xscale('log')
#     psi_subsystem = SU_system.psi_list[0]
#     Lindblad_operator_subsystem = SU_system.operators[0]
#     coupling_factor = SU_system.coupling_factors

#     for L in L_list:
#         ### Expands operators to L-sites using Kronecker product ###
#         psi_list = []
#         for i in range(L):
#             psi_list.append(psi_subsystem)

#         # Initialization of effective Hamiltonian
#         H_eff = H_s_subsystem - 1j*coupling_factor[0]/2*(np.conj(Lindblad_operator_subsystem.T) @ Lindblad_operator_subsystem)
#         density_matrix = multi_site_tensor_product(density_matrix_subsystem, L)
#         H = create_local_operator_chain(H_s_subsystem, L)
#         Lindblad_operators = create_list_of_local_operators(Lindblad_operator_subsystem, L)
#         coupling_factors = coupling_factor * L

#         SU_system = System(rho=[], psi_list=psi_list, H=H_eff, L=len(psi_list), operators=[Lindblad_operator_subsystem], coupling_factors=coupling_factor)
#         SO_system = System(rho=density_matrix, psi_list=[], H=H, L=len(psi_list), operators=Lindblad_operators, coupling_factors=coupling_factors)

#         fidelities = []
#         x = []
#         previous_input = ''
#         rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, T, plot_spectrum=False)
#         for num_trajectories in num_trajectories_list:
#             if num_trajectories/L > 100:
#                 continue
#             rho_stochastic_result, previous_input = stochastic_unraveling(SU_system, num_trajectories, T, dt, previous_input=previous_input)
#             fidelities.append(fidelity(rho_SO_full, rho_stochastic_result))
#             x.append(num_trajectories/L)
#         error = [1-f for f in fidelities]
#         plt.plot(x, error, marker='.', linestyle='', label='L='+str(L))
#         plt.xscale('log')
#     plt.legend()
#     plt.show()


# def test_ratio_L_T(SU_system, SO_system, H_s_subsystem, density_matrix_subsystem, num_trajectories, T_list, dt, L_list):
#     plt.title("Fidelity vs. Ratio L/T, N=%d" % (num_trajectories))
#     plt.xlabel('Ratio L/T')
#     plt.ylabel("Fidelity")
#     plt.xscale('log')

#     psi_subsystem = SU_system.psi_list[0]
#     Lindblad_operator_subsystem = SU_system.operators[0]
#     coupling_factor = SU_system.coupling_factors

#     for L in L_list:
#         ### Expands operators to L-sites using Kronecker product ###
#         psi_list = []
#         for i in range(L):
#             psi_list.append(psi_subsystem)

#         # Initialization of effective Hamiltonian
#         H_eff = H_s_subsystem - 1j*coupling_factor[0]/2*(np.conj(Lindblad_operator_subsystem.T) @ Lindblad_operator_subsystem)
#         density_matrix = multi_site_tensor_product(density_matrix_subsystem, L)
#         H = create_local_operator_chain(H_s_subsystem, L)
#         Lindblad_operators = create_list_of_local_operators(Lindblad_operator_subsystem, L)
#         coupling_factors = coupling_factor * L

#         SU_system = System(rho=[], psi_list=psi_list, H=H_eff, L=len(psi_list), operators=[Lindblad_operator_subsystem], coupling_factors=coupling_factor)
#         SO_system = System(rho=density_matrix, psi_list=[], H=H, L=len(psi_list), operators=Lindblad_operators, coupling_factors=coupling_factors)

#         fidelities = []
#         x = []
#         for T in T_list:
#             print("L, T: ", L, T)
#             rho_stochastic_result, _ = stochastic_unraveling(SU_system, num_trajectories, T, dt)
#             rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, T, plot_spectrum=False)
#             fidelities.append(fidelity(rho_SO_full, rho_stochastic_result))
#             x.append(L/T)
#         plt.plot(x, fidelities, marker='o', linestyle='', label='L='+str(L))

#     plt.legend()
#     plt.show()


# def test_ratio_L_dt(SU_system, SO_system, H_s_subsystem, density_matrix_subsystem, num_trajectories, T, dt_list, L_list):
#     plt.title("Fidelity vs. Ratio L/dt, N=%d" % (num_trajectories))
#     plt.xlabel('Ratio L/dt')
#     plt.ylabel("Fidelity")
#     plt.xscale('log')

#     psi_subsystem = SU_system.psi_list[0]
#     Lindblad_operator_subsystem = SU_system.operators[0]
#     coupling_factor = SU_system.coupling_factors
#     for L in L_list:
#         ### Expands operators to L-sites using Kronecker product ###
#         psi_list = []
#         for i in range(L):
#             psi_list.append(psi_subsystem)

#         # Initialization of effective Hamiltonian
#         H_eff = H_s_subsystem - 1j*coupling_factor[0]/2*(np.conj(Lindblad_operator_subsystem.T) @ Lindblad_operator_subsystem)
#         density_matrix = multi_site_tensor_product(density_matrix_subsystem, L)
#         H = create_local_operator_chain(H_s_subsystem, L)
#         Lindblad_operators = create_list_of_local_operators(Lindblad_operator_subsystem, L)
#         coupling_factors = coupling_factor * L

#         SU_system = System(rho=[], psi_list=psi_list, H=H_eff, L=len(psi_list), operators=[Lindblad_operator_subsystem], coupling_factors=coupling_factor)
#         SO_system = System(rho=density_matrix, psi_list=[], H=H, L=len(psi_list), operators=Lindblad_operators, coupling_factors=coupling_factors)

#         x = []
#         fidelities = []
#         for dt in dt_list:
#             print("L, dt: ", L, dt)
#             rho_stochastic_result, _ = stochastic_unraveling(SU_system, num_trajectories, T, dt)
#             rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, T, plot_spectrum=False)
#             fidelities.append(fidelity(rho_SO_full, rho_stochastic_result))
#             x.append(L/dt)
#         plt.plot(x, fidelities, marker='o', linestyle='', label="L="+str(L))

#     plt.legend()
#     plt.show()


# def test_ratio_N_dt(SU_system, SO_system, num_trajectories_list, T, dt_list):
#     plt.title("Fidelity vs. Ratio N/dt, T=%d" % T)
#     plt.xlabel('Ratio N/dt')
#     plt.ylabel("Fidelity")
#     plt.xscale('log')

#     fidelities = []
#     for dt in dt_list:
#         if dt > T:
#             continue
#         fidelities = []
#         x = []
#         for num_trajectories in num_trajectories_list:
#             print(num_trajectories, dt)
#             rho_stochastic_result, previous_input = stochastic_unraveling(SU_system, num_trajectories, T, dt, previous_input=previous_input)
#             rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, T, plot_spectrum=False)
#             fidelities.append(fidelity(rho_SO_full, rho_stochastic_result))
#             x.append(num_trajectories/dt)
#         plt.plot(x, fidelities, marker='o', linestyle='', label='N='+str(num_trajectories))

#     plt.legend()
#     plt.show()
