import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.linalg
import time

from .initializations import create_system, create_pauli_x, create_local_operators_list
from .superoperator import time_evolution_superoperator
from .vector_MCWF import vector_MCWF, vector_MCWF_trajectories
from TN.TN_MCWF import TN_MCWF, TN_MCWF_trajectory_test,  TN_MCWF_variance_test, TN_MCWF_variance_timestep_test, TN_MCWF_trajectory_error_test, TN_MCWF_entropy_test, TN_MCWF_entropy_growth, TN_MCWF_transmon_population2, TN_MCWF_transmon_levels, TN_MCWF_transmon_population
from TN.metrics import fidelity


def write_to_csv(filepath, row):
    with open(filepath, 'a') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(row)


def test_trajectory_fidelities_vector(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, max_num_trajectories, timesteps_list, T, local_operator, operator, operator_site, total_states):
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
                # exp_value = TN_MCWF_trajectory_error_test(system, i_trajectory, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None, input_exp_value=exp_value)
                error = 1-fidelity(rho_SO_full, density_matrix)
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




def test_trajectory_fidelities_TN(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, max_num_trajectories, timesteps_list, T, local_operator, operator, operator_site, total_states):
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

                exp_value, density_matrix = TN_MCWF_trajectory_error_test(system, i_trajectory, T=T, dt=dt, operator=operator, operator_site=operator_site, force_noise=False, input_noise_list=None, input_exp_value=exp_value, input_density_matrix=density_matrix)
                error = 1-fidelity(rho_SO_full, density_matrix)
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