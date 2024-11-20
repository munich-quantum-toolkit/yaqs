import copy
import numpy as np
import matplotlib.pyplot as plt
import opt_einsum as oe

from canonical_forms import shift_orthogonality_center_right, site_canonical_form, check_canonical_form, normalize
from MCWF import apply_dissipation, apply_jumps, force_apply_jumps
from metrics import expectation_value, scalar_product
from operations import MPS_to_density_matrix
from TDVP import single_sweep_TDVP
from classes import NoiseModel

import csv


def write_to_csv(filepath, row):
    with open(filepath, 'a') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(row)



# First order Trotter version
def core_algorithm(MPS, system, dt, i_trajectory, j_time, timesteps, force_noise=None, input_noise_list=None, sampling=None):
    MPS_previous_time = copy.deepcopy(MPS)

    stochastic_MPS = apply_dissipation(MPS, dt, system, starting_point='right')
    stochastic_MPS = single_sweep_TDVP(stochastic_MPS, system.H_0_MPO, dt)
    # print(scalar_product(stochastic_MPS, stochastic_MPS))

    # Checks if there are noise processes
    if system.processes:
        if not force_noise:
            stochastic_MPS, site_jumped, jump_type, noise_operator = apply_jumps(MPS_previous_time, stochastic_MPS, dt, system)

        elif force_noise:
            stochastic_MPS = force_apply_jumps(MPS_previous_time, stochastic_MPS, input_noise_list[i_trajectory-1], j_time)
            site_jumped = None
            jump_type = None
            noise_operator = None
    else:
        site_jumped = None
        jump_type = None
        noise_operator = None

    updated_MPS = stochastic_MPS

    return updated_MPS, stochastic_MPS, site_jumped, jump_type, noise_operator




def TN_MCWF(system, num_trajectories, T, dt, operator, operator_site, force_noise=False, input_noise_list={}):
    MPS = copy.deepcopy(system.MPS)

    times = np.arange(0, T+dt, dt)

    exp_values = []
    for i in range(len(times)):
        exp_values.append(None)

    # Site canonical form used to exploit structure for expectation value
    # Local
    # MPS = site_canonical_form(MPS, orthogonality_center=operator_site)
    # exp_values[0] = expectation_value(MPS, operator, MPS, None)
    # MPS = site_canonical_form(MPS, orthogonality_center=0)

    # Global
    MPS = site_canonical_form(MPS, orthogonality_center=0)
    exp_value = expectation_value(MPS, operator, MPS, site=0)
    # for operator_site in range(1, len(MPS)):
    #     MPS = shift_orthogonality_center_right(MPS, current_orthogonality_center=operator_site-1)
    #     exp_value += expectation_value(MPS, operator, MPS, operator_site)
    exp_values[0] = exp_value
    # MPS = site_canonical_form(MPS, orthogonality_center=0)

    output_noise_list = []
    for i_trajectory in range(1, num_trajectories+1, 1):
        noise_times = []
        noise_sites = []
        noise_types = []
        noise_operators = []
        # Local
        # if num_trajectories >= 100 and i_trajectory % 10 == 0:
        #     print("Trajectory", i_trajectory, "of", num_trajectories)
        # elif num_trajectories < 100:
        #     print("Trajectory", i_trajectory, "of", num_trajectories)
        # Global
        print("Trajectory", i_trajectory, "of", num_trajectories)

        trajectory_exp_values = [exp_values[0]]
        stochastic_MPS = copy.deepcopy(system.MPS)
        for j in range(len(times)):
            print("Trajectory", i_trajectory, " Timestep",  j+1, "of", len(times)-1)
            if j == 0:
                continue
            updated_MPS, stochastic_MPS, site_jumped, jump_type, noise_operator = core_algorithm(stochastic_MPS, system, dt, i_trajectory, j, len(times)-1, force_noise, input_noise_list)

            if site_jumped is not None:
                noise_times.append(j)
                noise_sites.append(site_jumped)
                noise_types.append(jump_type)
                noise_operators.append(noise_operator)

            # Makes the operator global for more interesting dynamics
            # Global
            # if system.L >= 50:
            # exp_value = 1
            # exp_value = expectation_value(updated_MPS, operator, updated_MPS, site=0)
            # for operator_site in range(1, len(MPS)):
            #     updated_MPS = shift_orthogonality_center_right(updated_MPS, current_orthogonality_center=operator_site-1)
            #     exp_value += expectation_value(updated_MPS, operator, updated_MPS, operator_site)
            # else:
            # Local
            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=operator_site)
            exp_value = expectation_value(updated_MPS, operator, updated_MPS, operator_site)
            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=0)

            if i_trajectory == 1:
                average_exp_value = exp_value
            else:
                average_exp_value = (i_trajectory-1)/i_trajectory*exp_values[j] + 1/i_trajectory*exp_value

            exp_values[j] = average_exp_value

        noise_model = NoiseModel(noise_times, noise_types, noise_sites, noise_operators)
        output_noise_list.append(noise_model)

    return exp_values, times, output_noise_list


def TN_MCWF_transmon_levels(system, num_trajectories, T, dt, operator, operator_site, force_noise=False, input_noise_list={}):
    MPS = copy.deepcopy(system.MPS)

    times = np.arange(0, T+dt, dt)

    exp_values = []
    for i in range(len(times)):
        exp_values.append(None)

    # Site canonical form used to exploit structure for expectation value
    # Local
    MPS = site_canonical_form(MPS, orthogonality_center=operator_site)
    exp_values[0] = expectation_value(MPS, operator, MPS, operator_site)
    MPS = site_canonical_form(MPS, orthogonality_center=0)

    # Global
    # MPS = site_canonical_form(MPS, orthogonality_center=0)
    # exp_value = expectation_value(MPS, operator, MPS, site=0)
    # for operator_site in range(1, len(MPS)):
    #     MPS = shift_orthogonality_center_right(MPS, current_orthogonality_center=operator_site-1)
    #     exp_value += expectation_value(MPS, operator, MPS, operator_site)
    # exp_values[0] = exp_value
    # MPS = site_canonical_form(MPS, orthogonality_center=0)

    output_noise_list = []
    for i_trajectory in range(1, num_trajectories+1, 1):
        noise_times = []
        noise_sites = []
        noise_types = []
        noise_operators = []
        # Local
        # if num_trajectories >= 100 and i_trajectory % 10 == 0:
        #     print("Trajectory", i_trajectory, "of", num_trajectories)
        # elif num_trajectories < 100:
        #     print("Trajectory", i_trajectory, "of", num_trajectories)
        # Global
        print("Trajectory", i_trajectory, "of", num_trajectories)

        stochastic_MPS = copy.deepcopy(system.MPS)
        for j in range(len(times)):
            if j == 0:
                continue
            updated_MPS, stochastic_MPS, site_jumped, jump_type, noise_operator = core_algorithm(stochastic_MPS, system, dt, i_trajectory, j, len(times)-1, force_noise, input_noise_list)

            if site_jumped is not None:
                noise_times.append(j)
                noise_sites.append(site_jumped)
                noise_types.append(jump_type)
                noise_operators.append(noise_operator)

            # Makes the operator global for more interesting dynamics
            # Global
            # if system.L >= 50:
            # exp_value = 1
            # exp_value = expectation_value(updated_MPS, operator, updated_MPS, site=0)
            # for operator_site in range(1, len(MPS)):
            #     updated_MPS = shift_orthogonality_center_right(updated_MPS, current_orthogonality_center=operator_site-1)
            #     exp_value += expectation_value(updated_MPS, operator, updated_MPS, operator_site)
            # else:
            # Local
            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=operator_site)
            exp_value = expectation_value(updated_MPS, operator, updated_MPS, operator_site)
            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=0)

            if i_trajectory == 1:
                average_exp_value = exp_value
            else:
                average_exp_value = (i_trajectory-1)/i_trajectory*exp_values[j] + 1/i_trajectory*exp_value

            exp_values[j] = average_exp_value

        noise_model = NoiseModel(noise_times, noise_types, noise_sites, noise_operators)
        output_noise_list.append(noise_model)
    return exp_values, times, output_noise_list




def TN_MCWF_transmon_population(system, num_trajectories, T, dt, operator, operator_site, force_noise=False, input_noise_list={}):
    def create_MPO(operators_list):
        MPO = []
        for operator in operators_list:
            W = np.expand_dims(operator, axis=(0, 1))
            MPO.append(W)

    MPS = copy.deepcopy(system.MPS)

    times = np.arange(0, T+dt, dt)

    pop1_0 = []
    pop1_1 = []
    pop2_0 = []
    pop2_1 = []

    popR_0 = []
    popR_1 = []
    for i in range(len(times)):
        pop1_0.append(None)
        pop1_1.append(None)
        pop2_0.append(None)
        pop2_1.append(None)
        popR_0.append(None)
        popR_1.append(None)

    diag = np.zeros(system.d)
    diag[0] = 1
    pop0_op = np.diag(diag)
    diag = np.zeros(system.d)
    diag[1] = 1
    pop1_op = np.diag(diag)

    MPS = site_canonical_form(MPS, orthogonality_center=2)
    pop2_0[0] = expectation_value(MPS, pop0_op, MPS, site=2)
    pop2_1[0] = expectation_value(MPS, pop1_op, MPS, site=2)

    MPS = site_canonical_form(MPS, orthogonality_center=1)
    popR_0[0] = expectation_value(MPS, pop0_op, MPS, site=1)
    popR_1[0] = expectation_value(MPS, pop1_op, MPS, site=1)

    MPS = site_canonical_form(MPS, orthogonality_center=0)
    pop1_0[0] = expectation_value(MPS, pop0_op, MPS, site=0)
    pop1_1[0] = expectation_value(MPS, pop1_op, MPS, site=0)

    output_noise_list = []
    for i_trajectory in range(1, num_trajectories+1, 1):
        noise_times = []
        noise_sites = []
        noise_types = []
        noise_operators = []
        # Local
        # if num_trajectories >= 100 and i_trajectory % 10 == 0:
        #     print("Trajectory", i_trajectory, "of", num_trajectories)
        # elif num_trajectories < 100:
        #     print("Trajectory", i_trajectory, "of", num_trajectories)
        # Global
        print("Trajectory", i_trajectory, "of", num_trajectories)

        stochastic_MPS = copy.deepcopy(system.MPS)
        for j in range(len(times)):
            if j == 0:
                continue
            updated_MPS, stochastic_MPS, site_jumped, jump_type, noise_operator = core_algorithm(stochastic_MPS, system, dt, i_trajectory, j, len(times)-1, force_noise, input_noise_list)

            if site_jumped is not None:
                noise_times.append(j)
                noise_sites.append(site_jumped)
                noise_types.append(jump_type)
                noise_operators.append(noise_operator)

            # Makes the operator global for more interesting dynamics
            # Global
            # if system.L >= 50:
            # exp_value = 1
            # exp_value = expectation_value(updated_MPS, operator, updated_MPS, site=0)
            # for operator_site in range(1, len(MPS)):
            #     updated_MPS = shift_orthogonality_center_right(updated_MPS, current_orthogonality_center=operator_site-1)
            #     exp_value += expectation_value(updated_MPS, operator, updated_MPS, operator_site)
            # else:
            # Local
            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=2)
            pop2_0_value = expectation_value(updated_MPS, pop0_op, updated_MPS, site=2)
            pop2_1_value = expectation_value(updated_MPS, pop1_op, updated_MPS, site=2)

            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=1)
            popR_0_value = expectation_value(updated_MPS, pop0_op, updated_MPS, site=1)
            popR_1_value = expectation_value(updated_MPS, pop1_op, updated_MPS, site=1)

            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=0)
            pop1_0_value = expectation_value(updated_MPS, pop0_op, updated_MPS, site=0)
            pop1_1_value = expectation_value(updated_MPS, pop1_op, updated_MPS, site=0)

            if i_trajectory == 1:
                avg_pop1_0 = pop1_0_value
                avg_pop1_1 = pop1_1_value
                avg_pop2_0 = pop2_0_value
                avg_pop2_1 = pop2_1_value
                avg_popR_0 = popR_0_value
                avg_popR_1 = popR_1_value

            else:
                avg_pop1_0 = (i_trajectory-1)/i_trajectory*pop1_0[j] + 1/i_trajectory*pop1_0_value
                avg_pop1_1 = (i_trajectory-1)/i_trajectory*pop1_1[j] + 1/i_trajectory*pop1_1_value
                avg_pop2_0 = (i_trajectory-1)/i_trajectory*pop2_0[j] + 1/i_trajectory*pop2_0_value
                avg_pop2_1 = (i_trajectory-1)/i_trajectory*pop2_1[j] + 1/i_trajectory*pop2_1_value
                avg_popR_0 = (i_trajectory-1)/i_trajectory*popR_0[j] + 1/i_trajectory*popR_0_value
                avg_popR_1 = (i_trajectory-1)/i_trajectory*popR_1[j] + 1/i_trajectory*popR_1_value

            pop1_0[j] = avg_pop1_0
            pop1_1[j] = avg_pop1_1
            pop2_0[j] = avg_pop2_0
            pop2_1[j] = avg_pop2_1
            popR_0[j] = avg_popR_0
            popR_1[j] = avg_popR_1

        noise_model = NoiseModel(noise_times, noise_types, noise_sites, noise_operators)
        output_noise_list.append(noise_model)
    return pop1_0, pop1_1, pop2_0, pop2_1, popR_0, popR_1, times, output_noise_list


def TN_MCWF_transmon_population2(system, num_trajectories, T, dt, operator, operator_site, force_noise=False, input_noise_list={}):
    def create_MPO(operators_list):
        MPO = []
        for operator in operators_list:
            W = np.expand_dims(operator, axis=(0, 1))
            MPO.append(W)
        return MPO

    MPS = copy.deepcopy(system.MPS)

    times = np.arange(0, T+dt, dt)

    pop000 = []
    pop001 = []
    pop010 = []
    pop011 = []
    pop100 = []
    pop101 = []
    pop110 = []
    pop111 = []

    for i in range(len(times)):
        pop000.append(None)
        pop001.append(None)
        pop010.append(None)
        pop011.append(None)
        pop100.append(None)
        pop101.append(None)
        pop110.append(None)
        pop111.append(None)

    diag = np.zeros(system.d)
    diag[0] = 1
    pop0_op = np.diag(diag)
    diag = np.zeros(system.d)
    diag[1] = 1
    pop1_op = np.diag(diag)

    MPO_000 = create_MPO([pop0_op, pop0_op, pop0_op])
    MPO_001 = create_MPO([pop0_op, pop0_op, pop1_op])
    MPO_010 = create_MPO([pop0_op, pop1_op, pop0_op])
    MPO_011 = create_MPO([pop0_op, pop1_op, pop1_op])
    MPO_100 = create_MPO([pop1_op, pop0_op, pop0_op])
    MPO_101 = create_MPO([pop1_op, pop0_op, pop1_op])
    MPO_110 = create_MPO([pop1_op, pop1_op, pop0_op])
    MPO_111 = create_MPO([pop1_op, pop1_op, pop1_op])

    for tensor in MPO_000:
        print(tensor.shape)
    MPS = site_canonical_form(MPS, orthogonality_center=0)
    pop000[0] = expectation_value(MPS, MPO_000, MPS)
    pop001[0] = expectation_value(MPS, MPO_001, MPS)
    pop010[0] = expectation_value(MPS, MPO_010, MPS)
    pop011[0] = expectation_value(MPS, MPO_011, MPS)
    pop100[0] = expectation_value(MPS, MPO_100, MPS)
    pop101[0] = expectation_value(MPS, MPO_101, MPS)
    pop110[0] = expectation_value(MPS, MPO_110, MPS)
    pop111[0] = expectation_value(MPS, MPO_111, MPS)


    output_noise_list = []
    for i_trajectory in range(1, num_trajectories+1, 1):
        noise_times = []
        noise_sites = []
        noise_types = []
        noise_operators = []
        # Local
        # if num_trajectories >= 100 and i_trajectory % 10 == 0:
        #     print("Trajectory", i_trajectory, "of", num_trajectories)
        # elif num_trajectories < 100:
        #     print("Trajectory", i_trajectory, "of", num_trajectories)
        # Global
        print("Trajectory", i_trajectory, "of", num_trajectories)

        stochastic_MPS = copy.deepcopy(system.MPS)
        for j in range(len(times)):
            if j == 0:
                continue
            updated_MPS, stochastic_MPS, site_jumped, jump_type, noise_operator = core_algorithm(stochastic_MPS, system, dt, i_trajectory, j, len(times)-1, force_noise, input_noise_list)

            if site_jumped is not None:
                noise_times.append(j)
                noise_sites.append(site_jumped)
                noise_types.append(jump_type)
                noise_operators.append(noise_operator)

            # Makes the operator global for more interesting dynamics
            # Global
            # if system.L >= 50:
            # exp_value = 1
            # exp_value = expectation_value(updated_MPS, operator, updated_MPS, site=0)
            # for operator_site in range(1, len(MPS)):
            #     updated_MPS = shift_orthogonality_center_right(updated_MPS, current_orthogonality_center=operator_site-1)
            #     exp_value += expectation_value(updated_MPS, operator, updated_MPS, operator_site)
            # else:
            # Local
            value_000 = expectation_value(updated_MPS, MPO_000, updated_MPS)
            value_001 = expectation_value(updated_MPS, MPO_001, updated_MPS)
            value_010 = expectation_value(updated_MPS, MPO_010, updated_MPS)
            value_011 = expectation_value(updated_MPS, MPO_011, updated_MPS)
            value_100 = expectation_value(updated_MPS, MPO_100, updated_MPS)
            value_101 = expectation_value(updated_MPS, MPO_101, updated_MPS)
            value_110 = expectation_value(updated_MPS, MPO_110, updated_MPS)
            value_111 = expectation_value(updated_MPS, MPO_111, updated_MPS)

            if i_trajectory == 1:
                avg_000 = value_000
                avg_001 = value_001
                avg_010 = value_010
                avg_011 = value_011
                avg_100 = value_100
                avg_101 = value_101
                avg_110 = value_110
                avg_111 = value_111

            else:
                avg_000 = (i_trajectory-1)/i_trajectory*pop000[j] + 1/i_trajectory*value_000
                avg_001 = (i_trajectory-1)/i_trajectory*pop001[j] + 1/i_trajectory*value_001
                avg_010 = (i_trajectory-1)/i_trajectory*pop010[j] + 1/i_trajectory*value_010
                avg_011 = (i_trajectory-1)/i_trajectory*pop011[j] + 1/i_trajectory*value_011
                avg_100 = (i_trajectory-1)/i_trajectory*pop100[j] + 1/i_trajectory*value_100
                avg_101 = (i_trajectory-1)/i_trajectory*pop101[j] + 1/i_trajectory*value_101
                avg_110 = (i_trajectory-1)/i_trajectory*pop110[j] + 1/i_trajectory*value_110
                avg_111 = (i_trajectory-1)/i_trajectory*pop111[j] + 1/i_trajectory*value_111

            pop000[j] = avg_000
            pop001[j] = avg_001
            pop010[j] = avg_010
            pop011[j] = avg_011
            pop100[j] = avg_100
            pop101[j] = avg_101
            pop110[j] = avg_110
            pop111[j] = avg_111

        noise_model = NoiseModel(noise_times, noise_types, noise_sites, noise_operators)
        output_noise_list.append(noise_model)
    return pop000, pop001, pop010, pop011, pop100, pop101, pop110, pop111, times, output_noise_list


def TN_MCWF_entropy_growth(system, num_trajectories, T, dt, operator, operator_site, force_noise=False, input_noise_list={}):
    MPS = copy.deepcopy(system.MPS)

    times = np.arange(0, T+dt, dt)

    exp_values = []
    for i in range(len(times)):
        exp_values.append(None)
    entropies = []
    for i in range(len(times)):
        entropies.append(None)

    # Site canonical form used to exploit structure for expectation value
    MPS = site_canonical_form(MPS, orthogonality_center=operator_site)
    exp_values[0] = expectation_value(MPS, operator, MPS, operator_site)

    output_noise_list = []
    for i_trajectory in range(1, num_trajectories+1, 1):
        noise_times = []
        noise_sites = []
        noise_types = []
        noise_operators = []
        if num_trajectories >= 100 and i_trajectory % 10 == 0:
            print("Trajectory", i_trajectory, "of", num_trajectories)
        elif num_trajectories < 100:
            print("Trajectory", i_trajectory, "of", num_trajectories)

        stochastic_MPS = copy.deepcopy(system.MPS)
        for j in range(len(times)):
            if j == 0:
                continue
            updated_MPS, stochastic_MPS, site_jumped, jump_type, noise_operator = core_algorithm(stochastic_MPS, system, dt, i_trajectory, j, len(times)-1, force_noise, input_noise_list)

            if site_jumped is not None:
                noise_times.append(j)
                noise_sites.append(site_jumped)
                noise_types.append(jump_type)
                noise_operators.append(noise_operator)

            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=operator_site)
            exp_value = expectation_value(updated_MPS, operator, updated_MPS, operator_site)
            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=0)

            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=0)
            tensor1 = updated_MPS[system.L//2-1]
            tensor2 = updated_MPS[system.L//2]

            theta = oe.contract('ijk, jbc->ikcb', tensor1, tensor2)
            theta = np.reshape(theta, (theta.shape[0]*theta.shape[1], theta.shape[2]*theta.shape[3]))
            _, S_vector, _ = np.linalg.svd(theta, full_matrices=0)
            S_vector = [x for x in S_vector if x > np.finfo(float).eps]
            max_value = max(S_vector)
            S_vector = [x for x in S_vector if x > max_value/10]

            S_vector = S_vector/np.linalg.norm(S_vector)

            entropy = 0
            for s in S_vector:
                entropy += -s**2*np.log(s**2)

            if i_trajectory == 1:
                average_exp_value = exp_value
                average_entropy = entropy
            else:
                average_exp_value = (i_trajectory-1)/i_trajectory*exp_values[j] + 1/i_trajectory*exp_value
                average_entropy = (i_trajectory-1)/i_trajectory*entropies[j] + 1/i_trajectory*entropy

            exp_values[j] = average_exp_value
            entropies[j] = average_entropy

        noise_model = NoiseModel(noise_times, noise_types, noise_sites, noise_operators)
        output_noise_list.append(noise_model)
    return exp_values, entropies, times, output_noise_list


def TN_MCWF_bond_dimension_test(system, num_trajectories, T, dt, operator, operator_site, force_noise=False, input_noise_list={}):
    MPS = copy.deepcopy(system.MPS)

    times = np.arange(0, T+dt, dt)

    exp_values = []
    for i in range(len(times)):
        exp_values.append(None)

    # Site canonical form used to exploit structure for expectation value
    MPS = site_canonical_form(MPS, orthogonality_center=operator_site)
    exp_values[0] = expectation_value(MPS, operator, MPS, operator_site)

    output_noise_list = []
    for i_trajectory in range(1, num_trajectories+1, 1):
        noise_times = []
        noise_sites = []
        noise_types = []
        noise_operators = []
        if num_trajectories >= 100 and i_trajectory % 10 == 0:
            print("Trajectory", i_trajectory, "of", num_trajectories)
        elif num_trajectories < 100:
            print("Trajectory", i_trajectory, "of", num_trajectories)

        stochastic_MPS = copy.deepcopy(system.MPS)
        for j in range(len(times)):
            if j == 0:
                continue
            updated_MPS, stochastic_MPS, site_jumped, jump_type, noise_operator = core_algorithm(stochastic_MPS, system, dt, i_trajectory, j, len(times)-1, force_noise, input_noise_list)

            if site_jumped is not None:
                noise_times.append(j)
                noise_sites.append(site_jumped)
                noise_types.append(jump_type)
                noise_operators.append(noise_operator)

            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=operator_site)
            exp_value = expectation_value(updated_MPS, operator, updated_MPS, operator_site)
            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=0)

            if i_trajectory == 1:
                average_exp_value = exp_value
            else:
                average_exp_value = (i_trajectory-1)/i_trajectory*exp_values[j] + 1/i_trajectory*exp_value

            exp_values[j] = average_exp_value

        noise_model = NoiseModel(noise_times, noise_types, noise_sites, noise_operators)
        output_noise_list.append(noise_model)
    return exp_values, times, output_noise_list



def TN_MCWF_entropy_test(system, num_trajectories, T, dt, operator, operator_site, force_noise=False, input_noise_list={}):
    MPS = copy.deepcopy(system.MPS)

    times = np.arange(0, T+dt, dt)

    # exp_values = []
    # for i in range(len(times)):
    #     exp_values.append(None)

    # Site canonical form used to exploit structure for expectation value
    # MPS = site_canonical_form(MPS, orthogonality_center=operator_site)
    # exp_values[0] = expectation_value(MPS, operator, MPS, operator_site)

    output_noise_list = []
    entropies = []
    counters = []
    same_results = 0
    for i_trajectory in range(1, num_trajectories+1, 1):
        noise_times = []
        noise_sites = []
        noise_types = []
        noise_operators = []
        if num_trajectories >= 100 and i_trajectory % 10 == 0:
            print("Trajectory", i_trajectory, "of", num_trajectories)
        elif num_trajectories < 100:
            print("Trajectory", i_trajectory, "of", num_trajectories)

        stochastic_MPS = copy.deepcopy(system.MPS)
        for j in range(len(times)):
            if j == 0:
                continue
            updated_MPS, stochastic_MPS, site_jumped, jump_type, noise_operator = core_algorithm(stochastic_MPS, system, dt, i_trajectory, j, len(times)-1, force_noise, input_noise_list)

            if site_jumped is not None:
                noise_times.append(j)
                noise_sites.append(site_jumped)
                noise_types.append(jump_type)
                noise_operators.append(noise_operator)

        updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=0)
        tensor1 = updated_MPS[system.L//2-1]
        tensor2 = updated_MPS[system.L//2]

        theta = oe.contract('ijk, jbc->ikcb', tensor1, tensor2)
        theta = np.reshape(theta, (theta.shape[0]*theta.shape[1], theta.shape[2]*theta.shape[3]))
        _, S_vector, _ = np.linalg.svd(theta, full_matrices=0)
        S_vector = [x for x in S_vector if x > np.finfo(float).eps]
        max_value = max(S_vector)
        S_vector = [x for x in S_vector if x > max_value/10]

        S_vector = S_vector/np.linalg.norm(S_vector)

        entropy = 0
        counter = len(S_vector)
        old_S_vector = S_vector
        for s in S_vector:
            entropy += -s**2*np.log(s**2)

        if i_trajectory != 1 and np.isclose(old_entropy, entropy):
            same_results += 1
            if same_results == 3:
                break
        else:
            same_results = 0
        old_entropy = entropy
        entropies.append(entropy)
        counters.append(counter)
            # updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=operator_site)
            # exp_value = expectation_value(updated_MPS, operator, updated_MPS, operator_site)
            # updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=0)

            # if i_trajectory == 1:
            #     average_exp_value = exp_value
            # else:
            #     average_exp_value = (i_trajectory-1)/i_trajectory*exp_values[j] + 1/i_trajectory*exp_value

            # exp_values[j] = average_exp_value
        noise_model = NoiseModel(noise_times, noise_types, noise_sites, noise_operators)
        output_noise_list.append(noise_model)

    entropy = np.mean(entropies)
    counter = 2**entropy
    # counter = np.mean(counters)
    return entropy, counter, times, output_noise_list


def TN_MCWF_trajectory_test(system, num_trajectories, T, dt, operator, operator_site, force_noise=False, input_noise_list={}, previous_input='', previous_trajectories=0):
    MPS = copy.deepcopy(system.MPS)

    times = np.arange(0, T+dt, dt)

    if previous_input == '':
        exp_values = []
        for i in range(len(times)):
            exp_values.append(None)

        # Site canonical form used to exploit structure for expectation value
        MPS = site_canonical_form(MPS, orthogonality_center=operator_site)
        exp_values[0] = expectation_value(MPS, operator, MPS, operator_site)
    else:
        exp_values = previous_input

    output_noise_list = []
    for i_trajectory in range(previous_trajectories+1, num_trajectories+1, 1):
        noise_times = []
        noise_sites = []
        noise_types = []
        noise_operators = []
        if num_trajectories >= 100 and i_trajectory % 10 == 0:
            print("Trajectory", i_trajectory, "of", num_trajectories)
        elif num_trajectories < 100:
            print("Trajectory", i_trajectory, "of", num_trajectories)

        updated_MPS = copy.deepcopy(system.MPS)
        for j in range(len(times)):
            if j == 0:
                continue
            updated_MPS, site_jumped, jump_type, noise_operator = core_algorithm(updated_MPS, system, dt, i_trajectory, j, len(times), force_noise, input_noise_list)

            if site_jumped is not None:
                noise_times.append(j)
                noise_sites.append(site_jumped)
                noise_types.append(jump_type)
                noise_operators.append(noise_operator)

            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=operator_site)
            exp_value = expectation_value(updated_MPS, operator, updated_MPS, operator_site)

            if i_trajectory == 1:
                average_exp_value = exp_value
            else:
                average_exp_value = (i_trajectory-1)/i_trajectory*exp_values[j] + 1/i_trajectory*exp_value

            exp_values[j] = average_exp_value

        noise_model = NoiseModel(noise_times, noise_types, noise_sites, noise_operators)
        output_noise_list.append(noise_model)
    return exp_values, times, output_noise_list


def TN_MCWF_variance_test(system, num_trajectories, T, dt, operator, operator_site, force_noise=False, input_noise_list={}):
    fig, ax = plt.subplots()
    MPS = copy.deepcopy(system.MPS)

    times = np.arange(0, T+dt, dt)

    avg_exp_values = []
    for i in range(len(times)):
        avg_exp_values.append(None)
    MPS = site_canonical_form(MPS, orthogonality_center=operator_site)
    avg_exp_values[0] = expectation_value(MPS, operator, MPS,operator_site)
    
    for i_trajectory in range(1, num_trajectories+1, 1):
        if num_trajectories >= 100 and i_trajectory % 10 == 0:
            print("Trajectory", i_trajectory, "of", num_trajectories)
        elif num_trajectories < 100:
            print("Trajectory", i_trajectory, "of", num_trajectories)

        single_trajectory_exp_values = [avg_exp_values[0]]
        updated_MPS = copy.deepcopy(MPS)
        for j in range(len(times)):
            if j == 0:
                continue
            updated_MPS, _, _, _ = core_algorithm(updated_MPS, system, dt, i_trajectory, j, len(times), force_noise, input_noise_list)
            
            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=operator_site)
            exp_value = expectation_value(updated_MPS, operator, updated_MPS, operator_site)
            if i_trajectory == 1:
                average_exp_value = exp_value
            else:
                average_exp_value = (i_trajectory-1)/i_trajectory*avg_exp_values[j] + 1/i_trajectory*exp_value

            avg_exp_values[j] = average_exp_value

            single_trajectory_exp_values.append(exp_value)
        plt.plot(times, single_trajectory_exp_values, linestyle='--', color='0.90')
    return avg_exp_values, times, fig, ax


def TN_MCWF_variance_timestep_test(system, num_trajectories, T, dt, operator, operator_site, force_noise=False, input_noise_list={}, fig=None, ax=None, color='black'):
    if not fig and not ax:
        fig, ax = plt.subplots()
    MPS = copy.deepcopy(system.MPS)

    times = np.arange(0, T+dt, dt)

    avg_exp_values = []
    for i in range(len(times)):
        avg_exp_values.append(None)
    MPS = site_canonical_form(MPS, orthogonality_center=operator_site)
    avg_exp_values[0] = expectation_value(MPS, operator, MPS, operator_site)
    
    for i_trajectory in range(1, num_trajectories+1, 1):
        print("Trajectory", i_trajectory, "of", num_trajectories)
        single_trajectory_exp_values = [avg_exp_values[0]]
        updated_MPS = copy.deepcopy(MPS)
        for j in range(len(times)):
            if j == 0:
                continue
            updated_MPS, _, _, _ = core_algorithm(updated_MPS, system, dt, i_trajectory, j, len(times), force_noise, input_noise_list)

            updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=operator_site)
            exp_value = expectation_value(updated_MPS, operator, updated_MPS, operator_site)
            if i_trajectory == 1:
                average_exp_value = exp_value
            else:
                average_exp_value = (i_trajectory-1)/i_trajectory*avg_exp_values[j] + 1/i_trajectory*exp_value

            avg_exp_values[j] = average_exp_value

            single_trajectory_exp_values.append(exp_value)
        if i_trajectory == 1:
            plt.plot(times, single_trajectory_exp_values, linestyle='--', alpha=0.4, color=color, label=len(times)-1)
        else:
            plt.plot(times, single_trajectory_exp_values, linestyle='--', alpha=0.7, color=color)

    return avg_exp_values, times, fig, ax


def TN_MCWF_trajectory_error_test(system, i_trajectory, T, dt, operator, operator_site, force_noise=False, input_noise_list={}, input_exp_value=None, input_density_matrix=None):
    times = np.arange(0, T+dt, dt)
    stochastic_MPS = copy.deepcopy(system.MPS)
    for j in range(len(times)):
        if j == 0:
            continue
        updated_MPS, stochastic_MPS, _, _, _ = core_algorithm(stochastic_MPS, system, dt, i_trajectory, j, len(times), force_noise, input_noise_list, sampling=T)

    updated_MPS = site_canonical_form(updated_MPS, orthogonality_center=operator_site)
    exp_value = expectation_value(updated_MPS, operator, updated_MPS, operator_site)
    density_matrix = MPS_to_density_matrix(updated_MPS)

    if i_trajectory == 1:
        average_exp_value = exp_value
        average_density_matrix = density_matrix
    else:
        average_exp_value = (i_trajectory-1)/i_trajectory*input_exp_value + 1/i_trajectory*exp_value
        average_density_matrix = (i_trajectory-1)/i_trajectory*input_density_matrix + 1/i_trajectory*density_matrix

    return average_exp_value, average_density_matrix

