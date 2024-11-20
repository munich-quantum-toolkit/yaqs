import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg
import time

from initializations import create_system, create_pauli_x, create_local_operators_list
from superoperator import time_evolution_superoperator
from vector_MCWF import vector_MCWF, vector_MCWF_trajectories
from TN_MCWF import TN_MCWF, TN_MCWF_trajectory_test,  TN_MCWF_variance_test, TN_MCWF_variance_timestep_test, TN_MCWF_trajectory_error_test, TN_MCWF_entropy_test, TN_MCWF_entropy_growth, TN_MCWF_transmon_population2, TN_MCWF_transmon_levels, TN_MCWF_transmon_population
from metrics import fidelity

def method_comparison_expectation_value_1000safety(system, num_trajectories, T, timesteps, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):
    filename = 'localOP_1000sites.csv'
    dt = T/timesteps

    stochastic_exp_values_TN, times_sampled, noise = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)

    # if system.L <= max_state_vector_length and system.calculate_state_vector:
    #     stochastic_exp_values, _, _ = vector_MCWF(system, num_trajectories, T, dt, local_operator, force_noise=False, input_noise_list=noise)
    #     end_time_SV = time.time()
    #     # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN)

    if system.L <= max_exact_length and system.calculate_exact:
        exact_exp_values = []
        exact_times = []
        for i, t in enumerate(times_sampled):
            if len(times_sampled) >= 100:
                if i % 10 == 0:
                    rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
                    exact_exp_value = np.trace(rho_SO_full @ local_operator)
                    exact_exp_values.append(exact_exp_value)
                    exact_times.append(t)
            else:
                rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
                exact_exp_value = np.trace(rho_SO_full @ local_operator)
                exact_exp_values.append(exact_exp_value)
                exact_times.append(t)
        end_time_exact = time.time()
        # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN, "Exact Elapsed:", end_time_exact-end_time_SV)
        plt.plot(exact_times, exact_exp_values, linestyle='solid', label='Exact')

    # if system.L <= max_state_vector_length and system.calculate_state_vector:
    #     plt.plot(times_sampled, stochastic_exp_values, linestyle='dashed', label='Stochastic')
    write_to_csv(filename, ['Times'])
    write_to_csv(filename, times_sampled)
    write_to_csv(filename, ['Values'])
    write_to_csv(filename, stochastic_exp_values_TN)

    plt.plot(times_sampled, stochastic_exp_values_TN)
    plt.title("Expectation Value vs. Elapsed Time, L=%d, N=%d, Timesteps=%d" % (system.L, num_trajectories, timesteps))
    plt.xlabel('tJ')
    plt.ylabel("<$\sigma_x$>")
    # plt.ylim(0, max(stochastic_exp_values_TN)+0.05*max(stochastic_exp_values_TN))
    plt.legend()
    plt.show()

def solve_Lindblad(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, times, local_operator, epsilon_list):
    filename = 'localOP_5sites.csv'
    n = len(epsilon_list)
    colors = plt.cm.jet(np.linspace(0,1,n))
    for i, epsilon in enumerate(epsilon_list):
        model_params['epsilon'] = epsilon
        system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_exact=True, calculate_state_vector=False)
    # stochastic_exp_values_TN, times_sampled, noise = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)

    # if system.L <= max_state_vector_length and system.calculate_state_vector:
    #     stochastic_exp_values, _, _ = vector_MCWF(system, num_trajectories, T, dt, local_operator, force_noise=False, input_noise_list=noise)
    #     end_time_SV = time.time()
        # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN)
    # write_to_csv(filename, ['Times'])
    # write_to_csv(filename, times_sampled)
    # write_to_csv(filename, ['Values'])
    # write_to_csv(filename, stochastic_exp_values_TN)

        exact_exp_values = []
        for t in times:
            rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
            exact_exp_value = np.trace(rho_SO_full @ local_operator)
            exact_exp_values.append(exact_exp_value)
            # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN, "Exact Elapsed:", end_time_exact-end_time_SV)
        if epsilon == 0:
            plt.plot(times, exact_exp_values, linestyle='dashed', color='black') # , label=epsilon)
        else:
            plt.plot(times, exact_exp_values, linestyle='solid', color=colors[i]) # , label=epsilon)

    # write_to_csv(filename, ['Values'])
    # write_to_csv(filename, exact_exp_values)

    # if system.L <= max_state_vector_length and system.calculate_state_vector:
    #     plt.plot(times_sampled, stochastic_exp_values, linestyle='dashed', label='Stochastic')

    # plt.plot(times_sampled, stochastic_exp_values_TN)
    plt.title("Expectation Value vs. Elapsed Time")
    plt.xlabel('t')
    plt.ylabel("<$\sigma_z$>")
    # plt.ylim(0, max(stochastic_exp_values_TN)+0.05*max(stochastic_exp_values_TN))
    plt.legend()
    plt.show()


def method_comparison_expectation_value(system, num_trajectories, T, timesteps, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):
    filename = 'localOP_5sites.csv'
    dt = T/timesteps

    stochastic_exp_values_TN, times_sampled, noise = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)



    if system.L <= max_state_vector_length and system.calculate_state_vector:
            # Create identity matrix
        identity = np.eye(system.d)

        # Initialize the full operator with the first site's operator
        if operator_site == 0:
            full_operator = local_operator
        else:
            full_operator = identity
        L = system.L
        # Build the full operator
        for site in range(1, L):
            if site == operator_site:
                full_operator = np.kron(full_operator, local_operator)
            else:
                full_operator = np.kron(full_operator, identity)
        stochastic_exp_values, _, _ = vector_MCWF(system, num_trajectories, T, dt, full_operator, force_noise=False, input_noise_list=noise)
        end_time_SV = time.time()
        # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN)
    # write_to_csv(filename, ['Times'])
    # write_to_csv(filename, times_sampled)
    # write_to_csv(filename, ['Values'])
    # write_to_csv(filename, stochastic_exp_values_TN)

    if system.L <= max_exact_length and system.calculate_exact:
        exact_exp_values = []
        exact_times = []
        for i, t in enumerate(times_sampled):
            if len(times_sampled) >= 100:
                if i % 10 == 0:
                    rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
                    exact_exp_value = np.trace(rho_SO_full @ full_operator)
                    exact_exp_values.append(exact_exp_value)
                    exact_times.append(t)
            else:
                rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
                exact_exp_value = np.trace(rho_SO_full @ full_operator)
                exact_exp_values.append(exact_exp_value)
                exact_times.append(t)
        end_time_exact = time.time()
        # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN, "Exact Elapsed:", end_time_exact-end_time_SV)
        plt.plot(exact_times, exact_exp_values, linestyle='solid', label='Exact')

    # write_to_csv(filename, ['Values'])
    # write_to_csv(filename, exact_exp_values)

    if system.L <= max_state_vector_length and system.calculate_state_vector:
        plt.plot(times_sampled, stochastic_exp_values, linestyle='dashed', label='Vector')

    plt.plot(times_sampled, stochastic_exp_values_TN, label='TN')
    plt.title("Expectation Value vs. Elapsed Time, L=%d, N=%d, Timesteps=%d" % (system.L, num_trajectories, timesteps))
    plt.xlabel('tJ')
    plt.ylabel("<$\sigma_x$>")
    # plt.ylim(0, max(stochastic_exp_values_TN)+0.05*max(stochastic_exp_values_TN))
    plt.legend()
    plt.show()


def transmon_simulation_levels(model, d, D, L, freq, T1, T2star, temperature, processes, model_params, max_bond_dimension, num_trajectories, T, timesteps, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):
    filename = 'levels_leakage.csv'
    dt = T/timesteps

    for i, d in enumerate([20, 24]):
        D = d
        diag = np.ones(d)
        diag[0] = 0
        diag[1] = 0


        operator = np.diag(diag)
        operator_site = 0

        if L <= max_state_vector_length:
            operators_list = create_local_operators_list(operator, L)
            local_operator = operators_list[operator_site]
        else:
            local_operator = None
        system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, D, calculate_state_vector=False, calculate_exact=False)
        stochastic_exp_values_TN, times_sampled, noise = TN_MCWF_transmon_levels(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)
        
        if i == 0:
            write_to_csv(filename, ['Times'])
            write_to_csv(filename, times_sampled)

        write_to_csv(filename, [d])
        write_to_csv(filename, stochastic_exp_values_TN)




        plt.plot(times_sampled, stochastic_exp_values_TN, label=d)
    plt.title("Population vs. Elapsed Time, N=%d, Timesteps=%d" % (num_trajectories, timesteps))
    plt.xlabel('Time (ns)')
    plt.ylabel("Leakage Qubit 1")
    # plt.ylim(0, max(stochastic_exp_values_TN)+0.05*max(stochastic_exp_values_TN))
    plt.legend()
    plt.show()


def transmon_simulation_population(model, d, D, L, freq, T1, T2star, temperature, processes, model_params, max_bond_dimension, num_trajectories, T, timesteps, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):
    filename = 'transmon_with_noise.csv'
    dt = T/timesteps

    system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, D, calculate_state_vector=False, calculate_exact=False)
    # pop1_0, pop1_1, pop2_0, pop2_1, popR_0, popR_1, times_sampled, noise = TN_MCWF_transmon_population(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)
    pop000, pop001, pop010, pop011, pop100, pop101, pop110, pop111,  times_sampled, noise = TN_MCWF_transmon_population2(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)

    leakage = []
    for i, _ in enumerate(pop000):
        leakage.append(1-pop000[i]-pop001[i]-pop010[i]-pop011[i]-pop100[i]-pop101[i]-pop110[i]-pop111[i])

    plt.plot(times_sampled, pop000, label='000')
    plt.plot(times_sampled, pop001, label='001')
    plt.plot(times_sampled, pop010, label='010')
    plt.plot(times_sampled, pop011, label='011')
    plt.plot(times_sampled, pop100, label='100')
    plt.plot(times_sampled, pop101, label='101')
    plt.plot(times_sampled, pop111, label='111')
    plt.plot(times_sampled, leakage, label='Leakage')

    write_to_csv(filename, ['Times'])
    write_to_csv(filename, times_sampled)

    write_to_csv(filename, ['Prob000'])
    write_to_csv(filename, pop000)
    write_to_csv(filename, ['Prob001'])
    write_to_csv(filename, pop001)
    write_to_csv(filename, ['Prob010'])
    write_to_csv(filename, pop010)
    write_to_csv(filename, ['Prob011'])
    write_to_csv(filename, pop011)
    write_to_csv(filename, ['Prob100'])
    write_to_csv(filename, pop100)
    write_to_csv(filename, ['Prob101'])
    write_to_csv(filename, pop101)
    write_to_csv(filename, ['Prob110'])
    write_to_csv(filename, pop110)
    write_to_csv(filename, ['Prob111'])
    write_to_csv(filename, pop111)
    write_to_csv(filename, ['Leakage'])
    write_to_csv(filename, leakage)

    plt.title("Population vs. Elapsed Time, N=%d, Timesteps=%d" % (num_trajectories, timesteps))
    plt.xlabel('Time (ns)')
    plt.ylabel("Probability")
    # plt.ylim(0, max(stochastic_exp_values_TN)+0.05*max(stochastic_exp_values_TN))
    plt.legend()
    plt.show()


def test_computational_time(model, d, L, freq, T1, T2star, temperature, processes, model_params, max_bond_dimension, num_trajectories, T, timesteps, max_state_vector_length, max_exact_length):
    filename = 'computational_time_globalOp2.csv'

    dt = T/timesteps

    TN_times = {1: [], 2: [], 4: [], 8: [], 16: [], 32: []}
    vector_times = []
    SO_times = []

    cutoff_dict = {'SO': False, 'SV': False, 1: False, 2: False, 4: False, 8: False, 16: False, 32: False}
    time_threshold = 65 # Seconds
    L = 2
    x = []
    while True:
        if cutoff_dict['SO']:
            break

        if all(value == True for value in cutoff_dict.values()):
            break
        else:
            x.append(L)

        operator = create_pauli_x(d)
        operator_site = L//2

        for i, max_bond_dimension in enumerate(TN_times.keys()):
            if cutoff_dict[max_bond_dimension]:
                TN_times[max_bond_dimension].append(np.nan)
                continue

            if not cutoff_dict['SV'] and i == len(TN_times.keys())-1:
                operators_list = create_local_operators_list(operator, L)
                local_operator = operators_list[operator_site]
                system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_state_vector=True, calculate_exact=True)
            else:
                system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_state_vector=False, calculate_exact=False)


            print(L, "TN"+str(max_bond_dimension))
            start_time_TN = time.time()
            stochastic_exp_values_TN, times_sampled, noise = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)
            end_time_TN = time.time()
            TN_times[max_bond_dimension].append(end_time_TN-start_time_TN)
            if end_time_TN-start_time_TN > time_threshold:
                cutoff_dict[max_bond_dimension] = True

        if not cutoff_dict['SV']:
            print(L, "SV")
            start_time_SV = time.time()
            stochastic_exp_values, _, _ = vector_MCWF(system, num_trajectories, T, dt, local_operator, force_noise=False, input_noise_list=noise)
            end_time_SV = time.time()
            vector_times.append(end_time_SV-start_time_SV)
            if end_time_SV-start_time_SV > time_threshold:
                cutoff_dict['SV'] = True
        else:
            vector_times.append(np.nan)

        if not cutoff_dict['SO']:
            exact_exp_values = []
            exact_times = []
            start_time_SO = time.time()
            for i, t in enumerate(times_sampled):
                if t == 0:
                    continue
                rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
                exact_exp_value = np.trace(rho_SO_full @ local_operator)
            end_time_SO = time.time()
            SO_times.append(end_time_SO-start_time_SO)
            if end_time_SO-start_time_SO > time_threshold:
                cutoff_dict['SO'] = True
        else:
            SO_times.append(np.nan)
        print("TN1: ", TN_times[1][-1], "TN2: ", TN_times[2][-1], "TN4: ", TN_times[4][-1], "TN8: ", TN_times[8][-1], "TN16: ", TN_times[16][-1], "TN32: ", TN_times[32][-1], "Vector: ", vector_times[-1], "SO: ", SO_times[-1])
        if L < 20:
            L += 1
        elif L < 50: 
            L += 10
        elif L < 500:
            L += 50
        elif L < 1000:
            L += 100
        elif L < 10000:
            L += 500
        else:
            L += 1000

    write_to_csv(filename, ['Sites'])
    write_to_csv(filename, x)
    write_to_csv(filename, ['SO'])
    write_to_csv(filename, SO_times)
    write_to_csv(filename, ['SV'])
    write_to_csv(filename, vector_times)
    for key in TN_times.keys():
        write_to_csv(filename, [key])
        write_to_csv(filename, TN_times[key])

    plt.plot(x, SO_times, label='Exact', color='tab:blue', linestyle='dotted')
    plt.plot(x, vector_times, label='Vector', color='tab:orange', linestyle='dashed')
    for key in TN_times.keys():
        plt.plot(x, TN_times[key], label='$\chi=$'+str(key), linestyle='solid')
    plt.title("Computational Time vs. Sites, N=%d, Timesteps=%d" % (num_trajectories, timesteps))
    plt.xlabel('Sites (L)')
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()



def test_computational_time_levels(model, d, L, freq, T1, T2star, temperature, processes, model_params, max_bond_dimension, num_trajectories, T, timesteps, max_state_vector_length, max_exact_length):
    dt = T/timesteps

    TN_times = []
    vector_times = []
    SO_times = []



    # SV_cutoff = 10
    # SO_cutoff = 5
    SV_cutoff = 17
    SO_cutoff = 1
    x = []
    for d in range(1, 30):
        operator = create_pauli_x(d)
        operator_site = L//2
        operators_list = create_local_operators_list(operator, L)
        local_operator = operators_list[operator_site]

        if d < SV_cutoff:
            system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_state_vector=True, calculate_exact=True)
        else:
            system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_state_vector=False, calculate_exact=False)

        print(d, "TN")
        start_time_TN = time.time()
        stochastic_exp_values_TN, times_sampled, noise = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)
        end_time_TN = time.time()
        TN_times.append(end_time_TN-start_time_TN)


        if d < SV_cutoff:
            print(d, "SV")
            start_time_SV = time.time()
            stochastic_exp_values, _, _ = vector_MCWF(system, num_trajectories, T, dt, local_operator, force_noise=False, input_noise_list=noise)
            end_time_SV = time.time()
            vector_times.append(end_time_SV-start_time_SV)
        else:
            vector_times.append(np.nan)

        if d < SO_cutoff:
            print(d, "SO")
            exact_exp_values = []
            exact_times = []
            start_time_SO = time.time()
            for i, t in enumerate(times_sampled):
                if t == 0:
                    continue
                rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
                exact_exp_value = np.trace(rho_SO_full @ local_operator)
            end_time_SO = time.time()
            SO_times.append(end_time_SO-start_time_SO)
        else:
            SO_times.append(np.nan)
        x.append(d)
        print("SO: ", SO_times[-1], "Vector: ", vector_times[-1], "TN: ", TN_times[-1])
    plt.plot(x, SO_times, label='Exact', color='tab:blue', linestyle='dotted')
    plt.plot(x, vector_times, label='Vector', color='tab:orange', linestyle='dashed')
    plt.plot(x, TN_times, label='TN', color='tab:green', linestyle='solid')
    plt.title("Computational Time vs. Levels, L=%d, N=%d, Timesteps=%d" % (L, num_trajectories, timesteps))
    plt.xlabel('Levels (d)')
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(x, vector_times, label='Vector', color='tab:orange', linestyle='dashed')
    plt.plot(x, TN_times, label='TN', color='tab:green', linestyle='solid')
    plt.title("Computational Time vs. Levels, L=%d, N=%d, Timesteps=%d" % (L, num_trajectories, timesteps))
    plt.xlabel('Levels (d)')
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()



def test_computational_time_levels_TN_only(model, d, L, freq, T1, T2star, temperature, processes, model_params, max_bond_dimension, num_trajectories, T, timesteps, max_state_vector_length, max_exact_length):
    dt = T/timesteps

    # vector_times = []
    # SO_times = []


    for max_bond_dimension in range(1, 17):
        TN_times = []
        for d in range(2, 17):

            operator = create_pauli_x(d)
            operator_site = L//2
            # operators_list = create_local_operators_list(operator, L)
            # local_operator = operators_list[operator_site]

            system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_state_vector=False, calculate_exact=False)

            print(d, "TN")
            start_time_TN = time.time()
            stochastic_exp_values_TN, times_sampled, noise = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)
            end_time_TN = time.time()
            TN_times.append(end_time_TN-start_time_TN)

            # if d < SV_cutoff:
            #     print(d, "SV")
            #     start_time_SV = time.time()
            #     stochastic_exp_values, _, _ = vector_MCWF(system, num_trajectories, T, dt, local_operator, force_noise=False, input_noise_list=noise)
            #     end_time_SV = time.time()
            #     vector_times.append(end_time_SV-start_time_SV)
            # else:
            #     vector_times.append(np.nan)

            # if d < SO_cutoff:
            #     print(d, "SO")
            #     exact_exp_values = []
            #     exact_times = []
            #     start_time_SO = time.time()
            #     for i, t in enumerate(times_sampled):
            #         if t == 0:
            #             continue
            #         rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
            #         exact_exp_value = np.trace(rho_SO_full @ local_operator)
            #     end_time_SO = time.time()
            #     SO_times.append(end_time_SO-start_time_SO)
            # else:
            #     SO_times.append(np.nan)

        x = range(1, d)
        # plt.plot(x, SO_times, label='Exact', color='tab:blue', linestyle='dotted')
        # plt.plot(x, vector_times, label='Vector', color='tab:orange', linestyle='dashed')
        plt.plot(x, TN_times, label="Max Bond = "+str(max_bond_dimension), linestyle='solid')
    plt.title("Computational Time vs. Levels, N=%d, Timesteps=%d" % (num_trajectories, timesteps))
    plt.xlabel('Levels (d)')
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()



def test_bond_dimension(model, d, L, freq, T1, T2star, temperature, processes, model_params, num_trajectories, T, timesteps, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):
    dt = T/timesteps
    exact_exp_values = []
    exact_times = []

    stochastic_exp_values_TN_bonds = []
    plt.title("Expectation Value vs. Elapsed Time, L=%d, N=%d, Timesteps=%d" % (L, num_trajectories, timesteps))
    plt.xlabel('tJ')
    plt.ylabel("<$\sigma_x$>")


    filename = 'test.csv'
    max_bond_dimension = 1
    if L < 6:
        system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_exact=True, calculate_state_vector=False)
    else:
        system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_exact=False, calculate_state_vector=False)

    stochastic_exp_values_TN, times_sampled, noise = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)

    stochastic_exp_values_TN_bonds.append(stochastic_exp_values_TN)
    plt.plot(times_sampled, stochastic_exp_values_TN, linestyle='dotted', label='$\chi_{max}$ = '+str(max_bond_dimension))

    write_to_csv(filename, ['Times'])
    write_to_csv(filename, times_sampled)
    write_to_csv(filename, [max_bond_dimension])
    write_to_csv(filename, stochastic_exp_values_TN)

    bonds_list = [2, 4] # [2, 4, 8, 16]
    for max_bond_dimension in bonds_list: #range(2, 17, 2):
        if L < 6:
            system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_exact=True, calculate_state_vector=False)
        else:
            system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_exact=False, calculate_state_vector=False)

        stochastic_exp_values_TN, _, _ = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=True, input_noise_list=noise)

        stochastic_exp_values_TN_bonds.append(stochastic_exp_values_TN)
        if max_bond_dimension == bonds_list[-1]:
            plt.plot(times_sampled, stochastic_exp_values_TN, color='black', linestyle='dashed', label='$\chi_{max}$ = '+str(max_bond_dimension))
        else:
            plt.plot(times_sampled, stochastic_exp_values_TN, label='$\chi_{max}$ = '+str(max_bond_dimension))


        write_to_csv(filename, [max_bond_dimension])
        write_to_csv(filename, stochastic_exp_values_TN)

    if L < 6:
        for i, t in enumerate(times_sampled):
            if len(times_sampled) >= 100:
                if i % 10 == 0:
                    rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
                    exact_exp_value = np.trace(rho_SO_full @ local_operator)
                    exact_exp_values.append(exact_exp_value)
                    exact_times.append(t)
            else:
                rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
                exact_exp_value = np.trace(rho_SO_full @ local_operator)
                exact_exp_values.append(exact_exp_value)
                exact_times.append(t)

        plt.plot(exact_times, exact_exp_values, color='black', linestyle='solid', label='Exact')

    # plt.ylim(-2, 2)
    plt.legend()
    plt.show()


def test_trajectories(system, trajectories_list, timesteps, T, local_operator, operator, operator_site):
    dt = T/timesteps
    for i, num_trajectories in enumerate(trajectories_list):
        if i == 0:
            stochastic_exp_values_TN, times_sampled, noise = TN_MCWF_trajectory_test(system, num_trajectories, T, dt, operator, operator_site)
        else:
            stochastic_exp_values_TN, times_sampled, noise = TN_MCWF_trajectory_test(system, num_trajectories, T, dt, operator, operator_site, previous_input=stochastic_exp_values_TN, previous_trajectories=trajectories_list[i-1])

        # stochastic_exp_values_TN, times_sampled, noise = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)

        # exp_value = TN_MCWF_trajectory_test(system, i_trajectory, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None, input_exp_value=exp_value)
        # error = np.abs((exp_value - exact_exp_value)/exact_exp_value)

        plt.title("Expectation Value vs. Trajectories, L=%d, T=%f, Timesteps=%d" % (system.L, T, timesteps))
        plt.xlabel('tJ')
        plt.ylabel("$<\sigma_x>$")
        plt.plot(stochastic_exp_values_TN, label=num_trajectories)

    times = np.arange(0, T+dt, dt)
    exact_exp_values = []
    for t in times:
        rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
        exact_exp_value = np.trace(rho_SO_full @ local_operator)
        exact_exp_values.append(exact_exp_value)
    plt.plot(exact_exp_values, label='Exact')
    plt.legend()
    plt.show()


def write_to_csv(filepath, row):
    with open(filepath, 'a') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(row)


def test_trajectories_fidelities(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, max_num_trajectories, timesteps_list, T, local_operator, operator, operator_site, total_states):
    plt.title("Infidelity vs. Trajectories, L=%d, T=%d, States=%d" % (L, T, total_states))
    plt.xlabel('Trajectories')
    plt.ylabel("Infidelity")
    filename = 'test.csv'
    x = range(1, max_num_trajectories+1)
    write_to_csv(filename, ['Trajectories'])
    write_to_csv(filename, x)


    for timesteps in timesteps_list:
        avg_errors = []
        lowest_errors = [None]*max_num_trajectories
        highest_errors = [None]*max_num_trajectories
        for num_state in range(1, total_states+1):
            system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, initial_state='Random')
            rho_SO_full = time_evolution_superoperator(system, T, plot_spectrum=False)

            exact_exp_value = np.trace(rho_SO_full @ local_operator)

            # write_to_csv(filename, [timesteps])
            dt = T/timesteps
            errors = []
            exp_value = None
            density_matrix = None
            for i_trajectory in range(1, max_num_trajectories+1):
                print("Trajectory", i_trajectory, "of", max_num_trajectories)

                exp_value, density_matrix = vector_MCWF_trajectories(system, i_trajectory, T, dt, local_operator, input_exp_value=exp_value, input_density_matrix=density_matrix)
                # density_matrix = scipy.linalg.expm(-dt*density_matrix)
                # exp_value = TN_MCWF_trajectory_error_test(system, i_trajectory, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None, input_exp_value=exp_value)
                # flat_SO = rho_SO_full.flatten()
                # flat_stoch = density_matrix.flatten()
                error = 1-fidelity(rho_SO_full, density_matrix)
                errors.append(error)
            # print("SO")
            # print(rho_SO_full)
            # print("Stochastic")
            # print(density_matrix)
            if num_state == 1:
                avg_errors = errors
                lowest_errors = avg_errors
                highest_errors = avg_errors
            else:
                avg_errors = [(num_state-1)/num_state*x for x in avg_errors]
                errors = [1/num_state*x for x in errors]
                avg_errors = np.array(avg_errors) + np.array(errors)

            for i, error in enumerate(errors):
                if error < lowest_errors[i]:
                    lowest_errors[i] = error
                if error > highest_errors[i]:
                    highest_errors[i] = error

        if timesteps == 1:
            plt.errorbar(x, avg_errors, yerr=[lowest_errors, highest_errors], label=str(timesteps)+' Timestep')

        else:
            plt.errorbar(x, avg_errors, yerr=[lowest_errors, highest_errors], label=str(timesteps)+' Timesteps')
        write_to_csv(filename, [timesteps])
        write_to_csv(filename, avg_errors)

    expected_error = [1/N for N in range(1, max_num_trajectories+1)]
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(range(1, max_num_trajectories+1), expected_error, label="Expected Error $1/\sqrt{N}$", linestyle='--')
    plt.legend()
    plt.show()


def test_trajectories_error(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, max_num_trajectories, timesteps_list, T, local_operator, operator, operator_site, total_states):


    system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, initial_state='Random')
    plt.title("Error vs. Trajectories, L=%d, T=%d, States=%d" % (system.L, T, total_states))
    plt.xlabel('Trajectories')
    plt.ylabel("$Error$")

    filename = 'test.csv'
    x = range(1, max_num_trajectories+1)
    write_to_csv(filename, ['Trajectories'])
    write_to_csv(filename, x)

    for timesteps in timesteps_list:
        avg_errors = []
        lowest_errors = [None]*max_num_trajectories
        highest_errors = [None]*max_num_trajectories
        for num_state in range(1, total_states+1):
            system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, initial_state='Random')

            rho_SO_full = time_evolution_superoperator(system, T, plot_spectrum=False)
            exact_exp_value = np.trace(rho_SO_full @ local_operator)
            # write_to_csv(filename, [timesteps])
            dt = T/timesteps
            errors = []
            exp_value = None
            density_matrix = None
            for i_trajectory in range(1, max_num_trajectories+1):
                print("Trajectory", i_trajectory, "of", max_num_trajectories)
                exp_value, density_matrix = vector_MCWF_trajectories(system, i_trajectory, T, dt, local_operator, input_exp_value=exp_value, input_density_matrix=density_matrix)
                # exp_value = TN_MCWF_trajectory_error_test(system, i_trajectory, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None, input_exp_value=exp_value)
                error = np.abs((exp_value - exact_exp_value))
                errors.append(error)

            if num_state == 1:
                avg_errors = errors
                lowest_errors = avg_errors
                highest_errors = avg_errors
            else:
                avg_errors = [(num_state-1)/num_state*x for x in avg_errors]
                errors = [1/num_state*x for x in errors]
                avg_errors = np.array(avg_errors) + np.array(errors)

            for i, error in enumerate(errors):
                if error < lowest_errors[i]:
                    lowest_errors[i] = error
                if error > highest_errors[i]:
                    highest_errors[i] = error

        if timesteps == 1:
            plt.errorbar(x, avg_errors, yerr=[lowest_errors, highest_errors], label=str(timesteps)+' Timestep')

        else:
            plt.errorbar(x, avg_errors, yerr=[lowest_errors, highest_errors], label=str(timesteps)+' Timesteps')
        write_to_csv(filename, [timesteps])
        write_to_csv(filename, avg_errors)

    expected_error = [1/N for N in range(1, max_num_trajectories+1)]
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(range(1, max_num_trajectories+1), expected_error, label="Expected Error $1/\sqrt{N}$", linestyle='--')
    plt.legend()
    plt.show()


def test_gamma_error(model, d, L, freq, T1_list, temperature, processes, model_params, num_trajectories, T, timesteps_list, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):
    filename = 'error_multi_gamma.csv'
    max_bond_dimension = 100
    plt.title("Error vs. $\gamma/T$, L=%d, T=%d" % (L, T))
    plt.xlabel('$\gamma/T$')
    plt.ylabel("Error")

    gammas = [1/T1 for T1 in T1_list]
    write_to_csv(filename, ['gammas'])
    write_to_csv(filename, gammas)

    exact_values = []
    for T1 in T1_list:
        print("SO T1", T1)
        T2star = T1
        system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_exact=True, calculate_state_vector=False)
        rho_SO_full = time_evolution_superoperator(system, T, plot_spectrum=False)
        exact_exp_value = np.trace(rho_SO_full @ local_operator)
        exact_values.append(exact_exp_value)
    write_to_csv(filename, ['SO_exact'])
    write_to_csv(filename, exact_values)

    for timesteps in timesteps_list:
        errors = []
        for i, T1 in enumerate(T1_list):
            print("Timesteps:", timesteps, "T1:", T1)
            T2star = T1
            system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_exact=True, calculate_state_vector=False)

            dt = T/timesteps
            stochastic_exp_values_TN, _, _ = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False)
            error = np.abs((stochastic_exp_values_TN[-1] - exact_values[i]))
            errors.append(error)

        write_to_csv(filename, [timesteps])
        write_to_csv(filename, errors)
        if timesteps == 1:
            plt.loglog(gammas, errors, label=str(timesteps)+' Timestep')
        else:
            plt.loglog(gammas, errors, label=str(timesteps)+' Timesteps')

    plt.legend()
    plt.show()


def test_chi_gamma(model, d, L, freq, T1_list, temperature, processes, model_params, num_trajectories, T, timesteps_list, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    max_bond_dimension = 16
    plt.title("Entanglement vs. $\gamma/T$, L=%d, N=%d, T=%d, $\chi=$%d" % (L, num_trajectories, T, max_bond_dimension))
    plt.xlabel('$\gamma/T$')
    plt.ylabel("Entanglement Entropy")
    # ax1.set_ylabel("Entanglement Entropy")
    # ax2.set_ylabel("Bond Dimension")
    gammas = [1/T1 for T1 in T1_list]
    filename = 'chi_gamma.csv'
    write_to_csv(filename, ['gammas'])
    write_to_csv(filename, gammas)

    for timesteps in timesteps_list:
        counters = []
        entropies = []
        dt = T/timesteps
        for T1 in T1_list:
            print(timesteps, T1)
            T2star = T1
            system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_exact=False, calculate_state_vector=False)
            entropy, counter, times, noise_list = TN_MCWF_entropy_test(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)
            entropies.append(entropy)
            counters.append(counter)

        write_to_csv(filename, [timesteps])
        write_to_csv(filename, entropies)

        if timesteps == 1:
            plt.plot(gammas, entropies, label=str(timesteps)+' Timestep')
        else:
            plt.plot(gammas, entropies, label=str(timesteps)+' Timesteps')

    plt.xscale('log')
    plt.legend()
    plt.show()

def test_entanglement_growth_gamma(model, d, L, freq, T1_list, temperature, processes, model_params, num_trajectories, T_list, timesteps, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):
    max_bond_dimension = 16
    plt.title("Entanglement vs. $\gamma T$, L=%d, N=%d, Timesteps=%d, $\chi=$%d" % (L, num_trajectories, timesteps, max_bond_dimension))
    plt.xlabel('$\gamma/T$')
    plt.ylabel("Entanglement Entropy")
    # ax1.set_ylabel("Entanglement Entropy")
    # ax2.set_ylabel("Bond Dimension")
    gammas = [1/T1 for T1 in T1_list]
    filename = 'entanglement_growth.csv'
    # write_to_csv(filename, ['gammas'])
    # write_to_csv(filename, gammas)

    for T in T_list:
        counters = []
        entropies = []
        dt = T/timesteps
        for T1 in T1_list:
            print(T, T1)
            T2star = T1
            system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_exact=False, calculate_state_vector=False)
            entropy, counter, times, noise_list = TN_MCWF_entropy_test(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)
            entropies.append(entropy)
            counters.append(counter)

        # write_to_csv(filename, ['T='+str(T)])
        # write_to_csv(filename, entropies)
        plt.plot(gammas, entropies, label='T='+str(T)+'/J')

    plt.xscale('log')
    plt.legend()
    plt.show()


def test_entanglement_growth(model, d, L, freq, T1_list, temperature, processes, model_params, num_trajectories, T, timesteps, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    max_bond_dimension = 16
    plt.title("Entanglement vs. $\gamma/T$, L=%d, N=%d, T=%d, $\chi=$%d" % (L, num_trajectories, T, max_bond_dimension))
    plt.xlabel('$\gamma/T$')
    plt.ylabel("Entanglement Entropy")
    # ax1.set_ylabel("Entanglement Entropy")
    # ax2.set_ylabel("Bond Dimension")
    gammas = [1/T1 for T1 in T1_list]
    filename = 'chi_gamma.csv'
    # write_to_csv(filename, ['gammas'])
    # write_to_csv(filename, gammas)

    dt = T/timesteps
    for i, T1 in enumerate(T1_list):
        print(timesteps, T1)
        T2star = T1
        system = create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, calculate_exact=False, calculate_state_vector=False)
        exp_values, entropies, times, noise_list = TN_MCWF_entropy_growth(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)

    # write_to_csv(filename, [timesteps])
    # write_to_csv(filename, entropies)

        plt.plot(times, entropies, label='$\gamma = %.2E$' % gammas[i])

        # if timesteps == 1:
        #     ax1.plot(gammas, entropies, label=str(timesteps)+' Timestep')
        # else:
        #     ax1.plot(gammas, entropies, label=str(timesteps)+' Timesteps')
        # ax2.plot(gammas, counters, linestyle='dotted')
    # ax1.set_xscale('log')
    # plt.xscale('log')
    plt.legend()
    plt.show()

# def test_variance(system, num_trajectories, T, timesteps, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):
#     dt = T/timesteps

#     start_time_TN = time.time()
#     stochastic_exp_values_TN, times_sampled, fig, ax = TN_MCWF_variance_test(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)
#     end_time_TN = time.time()

#     if system.L <= max_exact_length and system.calculate_exact:
#         exact_exp_values = []
#         exact_times = []
#         for i, t in enumerate(times_sampled):
#             if len(times_sampled) > 1000:
#                 if i % 100 == 0:
#                     rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
#                     exact_exp_value = np.trace(rho_SO_full @ local_operator)
#                     exact_exp_values.append(exact_exp_value)
#                     exact_times.append(t)
#             else:
#                 rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
#                 exact_exp_value = np.trace(rho_SO_full @ local_operator)
#                 exact_exp_values.append(exact_exp_value)
#                 exact_times.append(t)
#         end_time_exact = time.time()
#         plt.plot(exact_times, exact_exp_values, color='green', linestyle='solid', label='Exact')

#     plt.plot(times_sampled, stochastic_exp_values_TN, color='red', linestyle='solid', label='TN')
#     plt.title("Variance Analysis, L=%d, N=%d, Timesteps=%d" % (system.L, num_trajectories, timesteps))
#     plt.xlabel('tJ')
#     plt.ylabel("<$\sigma_x$>")
#     plt.legend()
#     plt.show()


# def test_variance_timesteps(system, num_trajectories, T, timesteps_list, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):
#     colors = cm.get_cmap('plasma', len(timesteps_list)+1)
#     for i, timesteps in enumerate(timesteps_list):
#         dt = T/timesteps

#         start_time_TN = time.time()
#         if i == 0:
#             stochastic_exp_values_TN, times_sampled, fig, ax = TN_MCWF_variance_timestep_test(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None, color=colors(i/len(timesteps_list)))
#         else:
#             stochastic_exp_values_TN, times_sampled, _, _ = TN_MCWF_variance_timestep_test(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None, fig=fig, ax=ax, color=colors(i/len(timesteps_list)))

#         end_time_TN = time.time()
#         # if system.L <= max_state_vector_length and system.calculate_state_vector:
#         #     stochastic_exp_values, _, _ = vector_MCWF(system, num_trajectories, T, dt, local_operator, force_noise=False, input_noise_list=noise)
#         #     end_time_SV = time.time()
#         #     # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN)

#         if system.L <= max_exact_length and system.calculate_exact and i == len(timesteps_list)-1:
#             exact_exp_values = []
#             exact_times = []
#             for i, t in enumerate(times_sampled):
#                 if len(times_sampled) > 1000:
#                     if i % 100 == 0:
#                         rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
#                         exact_exp_value = np.trace(rho_SO_full @ local_operator)
#                         exact_exp_values.append(exact_exp_value)
#                         exact_times.append(t)
#                 else:
#                     rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
#                     exact_exp_value = np.trace(rho_SO_full @ local_operator)
#                     exact_exp_values.append(exact_exp_value)
#                     exact_times.append(t)
#             end_time_exact = time.time()
#             # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN, "Exact Elapsed:", end_time_exact-end_time_SV)
#             plt.plot(exact_times, exact_exp_values, color='black', linestyle='solid', label='Exact')

#         # if system.L <= max_state_vector_length and system.calculate_state_vector:
#             # plt.plot(times_sampled, stochastic_exp_values, linestyle='dashed', label='Stochastic')
#         # plt.plot(times_sampled, stochastic_exp_values_TN, color='red', linestyle='solid', label='TN')
#     plt.title("Variance vs. Timesteps, L=%d, N=%d" % (system.L, num_trajectories))
#     plt.xlabel('tJ')
#     plt.ylabel("<$\sigma_x$>")
#     plt.legend()
#     plt.show()


# def test_trajectories_error_multiple_T(system, max_num_trajectories, timesteps_list, T_list, local_operator, operator, operator_site):

#     exp_value = None
#     exact_values = []
#     for i, timesteps in enumerate(timesteps_list):
#         final_errors = []
#         for j, T in enumerate(T_list):
#             print(T)
#             if i == 0:
#                 rho_SO_full = time_evolution_superoperator(system, T, plot_spectrum=False)
#                 exact_exp_value = np.trace(rho_SO_full @ local_operator)
#                 exact_values.append(exact_exp_value)
#             errors = []
#             dt = T/timesteps
#             for i_trajectory in range(1, max_num_trajectories+1):
#                 # print("Trajectory", i_trajectory, "of", max_num_trajectories)
#                 exp_value = TN_MCWF_trajectory_error_test(system, i_trajectory, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None, input_exp_value=exp_value)
#                 error = np.abs((exp_value - exact_values[j]))
#                 errors.append(error)
#             final_errors.append(errors[-1])

#         # expected_error = [1/N for N in range(1, max_num_trajectories+1)]
#         # print("Final Error:", errors[-1])

#         plt.plot(T_list, final_errors, label=str(timesteps))
#     # plt.plot(range(1, max_num_trajectories+1), expected_error, label="Expected Error $1/\sqrt{N}$", linestyle='--')
#     plt.title("Absolute Error vs. log(T), L=%d, Trajectories=%d" % (system.L, max_num_trajectories))
#     plt.xlabel(' Log Elapsed Time log(T)')
#     plt.ylabel("$<\sigma_x> - <\sigma_x>_{exact}$")
#     plt.ylim(0, 1)
#     plt.legend()
#     plt.xscale('log')
#     plt.show()


# def find_necessary_trajectories(system, timesteps, T, local_operator, operator, operator_site, error_bound, convergence_criteria, plot=False):
#     rho_SO_full = time_evolution_superoperator(system, T, plot_spectrum=False)
#     exact_exp_value = np.trace(rho_SO_full @ local_operator)

#     dt = T/timesteps
#     errors = []
#     exp_value = None
#     i_trajectory = 1
#     error = np.inf
#     convergence_counter = 0
#     while True:
#         print(T, "Trajectory", i_trajectory, "Error:", error)
#         if i_trajectory == 10000:
#             print("Does not converge")
#             trajectories_needed = np.nan
#             break
#         exp_value = TN_MCWF_trajectory_test(system, i_trajectory, T=T, dt=dt, operator=operator, operator_site=operator_site,  force_noise=False, input_noise_list=None, input_exp_value=exp_value)
#         error = np.abs((exp_value - exact_exp_value)/exact_exp_value)
#         errors.append(error)
#         if error < error_bound:
#             convergence_counter += 1
#             if convergence_counter == 1:
#                 trajectories_needed = i_trajectory
#             elif convergence_counter == convergence_criteria:
#                 break
#         else:
#             convergence_counter = 0

#         i_trajectory += 1

#     print(trajectories_needed, "trajectories needed for error bound of ", error_bound)

#     if plot:
#         expected_error = [1/N for N in range(1, i_trajectory+1)]
#         plt.title("Error vs. Trajectories, L=%d, T=%f, Timesteps=%d" % (system.L, T, timesteps))
#         plt.xlabel('Trajectories')
#         plt.ylabel("Error")
#         plt.plot(range(1, i_trajectory+1), errors, label="Stochastic")
#         plt.plot(range(1, i_trajectory+1), expected_error, label="Expected Error $1/\sqrt{N}$", linestyle='--')
#         plt.legend()
#         plt.show()
#     return trajectories_needed


# def test_timesteps(system, num_trajectories, T, timesteps_list, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):
#     for i, timesteps in enumerate(timesteps_list):
#         dt = T/timesteps

#         start_time_TN = time.time()
#         stochastic_exp_values_TN, times_sampled, noise = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)
#         end_time_TN = time.time()
#         # if system.L <= max_state_vector_length and system.calculate_state_vector:
#         #     stochastic_exp_values, _, _ = vector_MCWF(system, num_trajectories, T, dt, local_operator, force_noise=False, input_noise_list=noise)
#         #     end_time_SV = time.time()
#         #     # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN)

#         # if system.L <= max_state_vector_length and system.calculate_state_vector:
#         #     plt.plot(times_sampled, stochastic_exp_values, linestyle='dashed', label='Stochastic')
#         plt.plot(times_sampled, stochastic_exp_values_TN, linestyle='solid', label=timesteps)

#         if system.L <= max_exact_length and system.calculate_exact and i == len(timesteps_list)-1:
#             exact_exp_values = []
#             exact_times = []
#             for i, t in enumerate(times_sampled):
#                 if len(times_sampled) > 1000:
#                     if i % 100 == 0:
#                         rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
#                         exact_exp_value = np.trace(rho_SO_full @ local_operator)
#                         exact_exp_values.append(exact_exp_value)
#                         exact_times.append(t)
#                 else:
#                     rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
#                     exact_exp_value = np.trace(rho_SO_full @ local_operator)
#                     exact_exp_values.append(exact_exp_value)
#                     exact_times.append(t)
#             end_time_exact = time.time()
#             # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN, "Exact Elapsed:", end_time_exact-end_time_SV)
#             plt.plot(exact_times, exact_exp_values, color='black', linestyle='dashed', label='Exact')
#     plt.title("Timestep Analysis, L=%d, N=%d" % (system.L, num_trajectories))
#     plt.xlabel('tJ')
#     plt.ylabel("<$\sigma_x$>")
#     plt.legend()
#     plt.show()


# def test_timesteps_error(system, num_trajectories, T, timesteps_list, local_operator, operator, operator_site, max_state_vector_length, max_exact_length):
#     for i, timesteps in enumerate(timesteps_list):
#         dt = T/timesteps

#         start_time_TN = time.time()
#         stochastic_exp_values_TN, times_sampled, noise = TN_MCWF(system, num_trajectories=num_trajectories, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None)
#         end_time_TN = time.time()
#         # if system.L <= max_state_vector_length and system.calculate_state_vector:
#         #     stochastic_exp_values, _, _ = vector_MCWF(system, num_trajectories, T, dt, operator, force_noise=False, input_noise_list=noise)
#         #     end_time_SV = time.time()
#         #     # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN)

#         # if system.L <= max_state_vector_length and system.calculate_state_vector:
#         #     plt.plot(times_sampled, stochastic_exp_values, linestyle='dashed', label='Stochastic')
#         # # plt.plot(times_sampled, stochastic_exp_values_TN, linestyle='solid', label=timesteps)

#         if system.L <= max_exact_length and system.calculate_exact:
#             exact_exp_values = []
#             exact_times = []
#             errors = []
#             for i, t in enumerate(times_sampled):
#                 rho_SO_full = time_evolution_superoperator(system, t, plot_spectrum=False)
#                 exact_exp_value = np.trace(rho_SO_full @ local_operator)
#                 exact_exp_values.append(exact_exp_value)
#                 exact_times.append(t)

#                 error = np.abs((stochastic_exp_values_TN[i] - exact_exp_value)/exact_exp_value)
#                 errors.append(error)
#             end_time_exact = time.time()
#             # print("TN Elapsed:", end_time_TN-start_time_TN, "SV Elapsed:", end_time_SV-end_time_TN, "Exact Elapsed:", end_time_exact-end_time_SV)
#             plt.plot(exact_times, errors, label=timesteps)
#     plt.title("Timestep Error Analysis, L=%d, N=%d" % (system.L, num_trajectories))
#     plt.xlabel('tJ')
#     plt.ylabel("Error <$\sigma_x$>")
#     plt.legend()
#     plt.show()