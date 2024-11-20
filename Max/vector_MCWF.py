import copy
import math
import numpy as np
import scipy
from classes import NoiseModel


def unitary_evolution(psi, H, T):
    """ Unitary time evolution of a quantum state for some Hamiltonian

    Args:
        psi: numpy array
            Initial state vector
        H: numpy array
            Hamiltonian matrix
        T: int
            Elapsed time

    Returns:
        psi: numpy array
            Evolved state vector
    """

    # H_sparse = scipy.sparse.csc_matrix(H)
    # U = scipy.linalg.expm(-1j*T*H_sparse)

    U = scipy.linalg.expm(-1j*T*H)
    psi = U @ psi

    return psi


def apply_jump_vector(state, updated_state, system, dt):
    """ Applies the quantum jump method for a single timestep

    Args:
        state: numpy array
            Initial state vector
        d: int
            Physical dimension
        L: int
            Number of sites
        H_eff: numpy array
            Non-Hermitian Hamiltonian
        dt: float
            Timestep
        Lindblad_operators: list of numpy arrays
            Possible operators being applied
        coupling_factors: list of floats
            Gammas for strength of each operator
        processes: list of str
            Noise processes

    Returns:
        updated_state: numpy array
            Evolved state vector
    """
    # Steps:
    # 1. Time evolve state with non-Hermitian Hamiltonian
    # 2. Check normalization of state norm = 1 - dp
    # 3. Select random epsilon [0,1] and compare it to jump probability dp
    # 4. If a jump occurs
    #   a. Select a random site
    #   b. Select a jump operator from a probability distribution
    #   c. Apply jump and normalize state
    # 5. If no jump occurs
    #   a. Normalize state

    # 1.
    # updated_state = unitary_evolution(state, H_eff, dt)

    # 2.
    # Check probablity of all operators
    operators = sum(system.operators.values(), [])
    coupling_factors = sum(system.coupling_factors.values(), [])

    delta_p_list = []
    for i in range(len(operators)):
        jumped_state = operators[i] @ state
        delta_p_operator = dt*coupling_factors[i]*np.vdot(jumped_state, jumped_state)
        delta_p_list.append(delta_p_operator)
    delta_p = np.sum(delta_p_list)
    probability_distribution = (delta_p_list/delta_p).astype(float)

    # 3.
    if np.random.rand() < delta_p:
        choices = [*range(len(probability_distribution))]
        choice = np.random.choice(choices, p=probability_distribution)

        # 4.
        jump_operator = operators[choice]
        jumped_state = jump_operator @ state
        updated_state = jumped_state / np.linalg.norm(jumped_state)

        # Should return the site that was jumped
        site_jumped = choice % system.L
        # Returns the type of jump for creating noise model
        jump_type = system.processes[math.floor(choice / system.L)]
        jump_occured = True
    # 5.
    else:
        updated_state = updated_state / np.linalg.norm(updated_state)

        # Used to signify no jump occured
        jump_operator = None
        site_jumped = None
        jump_occured = False
        jump_type = None

    return updated_state, jump_occured, site_jumped, jump_type, jump_operator


def force_apply_jump_vector(state, updated_state, noise_model, time, system):
    if time in noise_model.times:
        index = noise_model.times.index(time)
        jump_type = noise_model.jump_types[index]
        site = noise_model.sites[index]
        jump_operator = system.operators[jump_type][site]

        jumped_state = jump_operator @ state
        updated_state = jumped_state / np.linalg.norm(jumped_state)
        jump_occured = True

    else:
        updated_state = updated_state / np.linalg.norm(updated_state)

        jump_occured = False

    return updated_state, jump_occured


def vector_MCWF(system, num_trajectories, T, dt, operator, force_noise=False, input_noise_list={}):
    times = np.arange(0, T+dt, dt)
    exp_values = []
    for i in range(len(times)):
        exp_values.append(None)

    # Squeeze due to sparse matrix multiplication
    exp_values[0] = (np.conj(system.state_vector).T @ operator @ system.state_vector).squeeze().real

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
        updated_state = copy.deepcopy(system.state_vector)
    
        for j in range(len(times)):
            if j == 0:
                continue
            previous_state = copy.deepcopy(updated_state)
            updated_state = unitary_evolution(previous_state, system.H_eff, dt)

            jump_occured = False
            if system.processes and not force_noise:
                updated_state, jump_occured, site_jumped, jump_type, noise_operator = apply_jump_vector(previous_state, updated_state, system, dt)
                if jump_occured:
                    noise_times.append(j)
                    noise_sites.append(site_jumped)
                    noise_types.append(jump_type)
                    noise_operators.append(noise_operator)
            elif system.processes and force_noise:
                updated_state, jump_occured = force_apply_jump_vector(previous_state, updated_state, input_noise_list[i_trajectory-1], j, system)
            
            # Squeeze due to sparse matrix multiplication
            exp_value = (np.conj(updated_state).T @ operator @ updated_state).squeeze().real
            if i_trajectory == 1:
                average_exp_value = exp_value
            else:
                average_exp_value = (i_trajectory-1)/i_trajectory*exp_values[j] + 1/i_trajectory*exp_value

            exp_values[j] = average_exp_value

        noise_model = NoiseModel(noise_times, noise_types, noise_sites, noise_operators)
        output_noise_list.append(noise_model)

    return exp_values, times, output_noise_list


def vector_MCWF_trajectories(system, i_trajectory, T, dt, operator, input_exp_value, input_density_matrix):
    times = np.arange(0, T+dt, dt)
    updated_state = copy.deepcopy(system.state_vector)
    for j in range(len(times)):
        if j == 0:
            continue
        previous_state = copy.deepcopy(updated_state)

        updated_state = unitary_evolution(previous_state, system.H_eff, dt)
        # print(np.vdot(updated_state, updated_state))

        updated_state, jump_occured, site_jumped, jump_type, noise_operator = apply_jump_vector(previous_state, updated_state, system, dt)

    density_matrix = np.outer(updated_state, np.conj(updated_state))

    exp_value = (np.conj(updated_state).T @ operator @ updated_state).squeeze().real
    if i_trajectory == 1:
        average_exp_value = exp_value
        average_density_matrix = density_matrix
    else:
        average_exp_value = (i_trajectory-1)/i_trajectory*input_exp_value + 1/i_trajectory*exp_value
        average_density_matrix = (i_trajectory-1)/i_trajectory*input_density_matrix + 1/i_trajectory*density_matrix

    return average_exp_value, average_density_matrix

