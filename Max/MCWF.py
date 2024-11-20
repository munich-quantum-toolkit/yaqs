import copy
import math
import numpy as np
import opt_einsum as oe
import scipy.linalg

from operations import flip_network

from canonical_forms import check_canonical_form, shift_orthogonality_center_right, site_canonical_form, normalize
from metrics import scalar_product
from single_site_operators import create_excitation_operator, create_deexcitation_operator, create_pauli_z


def apply_jumps(MPS_previous_time, MPS_new_time, dt, system):
    operators = []
    delta_p_list = []

    # Put MPS in canonical form at first site
    # MPS_previous_time = site_canonical_form(MPS_previous_time, orthogonality_center=0)
    # check_canonical_form(MPS_previous_time)
    for process in system.processes:
        if process == 'relaxation':
            jump_operator = create_deexcitation_operator(system.d)
        elif process == 'dephasing':
            jump_operator = create_pauli_z(system.d)
        elif process == 'thermal':
            jump_operator = create_excitation_operator(system.d)
        
        for site in range(system.L):
            if system.model == 'Transmon':
                # Need to redefine for changing physical dimensions
                if site in system.model_params['resonator_sites']:
                    physical_dim = system.D
                else:
                    physical_dim = system.d

                if process == 'relaxation':
                    jump_operator = create_deexcitation_operator(physical_dim)
                elif process == 'dephasing':
                    jump_operator = create_pauli_z(physical_dim)
                elif process == 'thermal':
                    jump_operator = create_excitation_operator(physical_dim)

            jumped_state = jump(MPS_previous_time, site, jump_operator)
            delta_p_operator = dt*system.coupling_factors[process][site]*scalar_product(jumped_state, jumped_state, site)
            delta_p_list.append(delta_p_operator)

            if site != system.L-1:
                # Keep the MPS in site canonical form for the scalar product to work
                MPS_previous_time = shift_orthogonality_center_right(MPS_previous_time, current_orthogonality_center=site)
            else:
                # Reversing the coupling factors is necessary if they are not the
                # same for each site
                MPS_previous_time = flip_network(MPS_previous_time)
            operators.append(jump_operator)

    delta_p = np.sum(delta_p_list)
    probability_distribution = (delta_p_list/delta_p).astype(float)
    if np.random.rand() < delta_p:
        ### WIP
        if False: # system.L >= 100:
            choices = [*range(len(probability_distribution))]
            for jump_occuring in range(system.L//5 + 1):
                choice = np.random.choice(choices, p=probability_distribution)
                # Renormalize distribution
                site_jumped = choice % system.L
                for i, _ in enumerate(probability_distribution):
                    if i == site_jumped:
                        probability_distribution[i] = 0

                    elif i != 0 and i % (site_jumped + system.L) == 0:
                        probability_distribution[i] = 0

                probability_distribution = probability_distribution/np.sum(probability_distribution)
                jump_operator = operators[choice]
                updated_MPS = jump(MPS_previous_time, site_jumped, jump_operator)
                jump_type = system.processes[math.floor(choice / system.L)]
            updated_MPS = normalize(updated_MPS, form='B')

        else:
            choices = [*range(len(probability_distribution))]
            choice = np.random.choice(choices, p=probability_distribution)
            jump_operator = operators[choice]
            site_jumped = choice % system.L

            updated_MPS = jump(MPS_new_time, site_jumped, jump_operator)
            updated_MPS = normalize(updated_MPS, form='B')
            jump_type = system.processes[math.floor(choice / system.L)]
  
    else:
        # Normalize
        updated_MPS = normalize(MPS_new_time, form='B')

        jump_operator = None
        site_jumped = None
        jump_type = None

    return updated_MPS, site_jumped, jump_type, jump_operator


def force_apply_jumps(MPS_previous_time, MPS_new_time, noise_model, time):
    d = MPS_previous_time[0].shape[2]

    if time in noise_model.times:
        index = noise_model.times.index(time)
        jump_type = noise_model.jump_types[index]
        if jump_type == 'relaxation':
            jump_operator = create_deexcitation_operator(d)
        elif jump_type == 'dephasing':
            jump_operator = create_pauli_z(d)
        elif jump_type == 'thermal':
            jump_operator = create_excitation_operator(d)

        site = noise_model.sites[index]
        updated_MPS = jump(MPS_previous_time, site, jump_operator)
        updated_MPS = normalize(updated_MPS, form='B')

    else:
        updated_MPS = normalize(MPS_new_time, form='B')

    return updated_MPS


def apply_dissipation(MPS, dt, system, starting_point='left'):
    if starting_point == 'right':
        MPS = flip_network(MPS)

    for site, tensor in enumerate(MPS):
        physical_dim = tensor.shape[-1]

        if system.processes:
            k = 0
            for process in system.processes:
                if process == 'relaxation':
                    jump_operator = create_deexcitation_operator(physical_dim)
                elif process == 'dephasing':
                    jump_operator = create_pauli_z(physical_dim)
                elif process == 'thermal':
                    jump_operator = create_excitation_operator(physical_dim)

                if starting_point == 'left':
                    k += -1j*system.coupling_factors[process][site]/2 * np.conj(jump_operator).T @ jump_operator
                else:
                    k += -1j*system.coupling_factors[process][len(MPS)-1-site]/2 * np.conj(jump_operator).T @ jump_operator
            trotter = scipy.linalg.expm(-1j*k*dt)
            MPS[site] = oe.contract('ij, abj->abi', trotter, tensor)

        # Avoid normalizing the state by shifting further
        if site != len(MPS)-1:
            MPS = shift_orthogonality_center_right(MPS, current_orthogonality_center=site)

    if starting_point == 'right':
        MPS = flip_network(MPS)

    return MPS


def jump(MPS, site, operator):
    jumped_MPS = copy.deepcopy(MPS)
    jumped_MPS[site] = oe.contract('ij, abj->abi', operator, jumped_MPS[site])
    return jumped_MPS