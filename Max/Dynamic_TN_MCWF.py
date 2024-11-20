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
from Modified_MENDL_TwoSiteTDVP import integrate_local_twosite_modified
import pytenet as ptn

import csv


def write_to_csv(filepath, row):
    with open(filepath, 'a') as f:
        # create the csv writer
        writer = csv.writer(f)

        # write a row to the csv file
        writer.writerow(row)



# First order Trotter version
def core_algorithm_DynamicTDVP(stochastic_MPS, H, system, dt, i_trajectory, j_time, timesteps, max_bond_dim, force_noise=None, input_noise_list=None, sampling=None):
    MPS_previous_time = copy.deepcopy(stochastic_MPS)
    
    # for i in stochastic_MPS.A:
    #     print('shape before apply dissipation in core alg, Aaron ind:',i.shape)


    stochastic_MPS.A = apply_dissipation(stochastic_MPS.A, dt, system, starting_point='right')

    # for i in stochastic_MPS.A:
    #     print('shape after apply dissipation in core alg, Aaron ind:',i.shape)

    # change stochastic mps to Mendl ind
    for i in range(len(stochastic_MPS.A)):
         stochastic_MPS.A[i]= np.transpose(stochastic_MPS.A[i], (2,0,1))

    # for i in stochastic_MPS.A:
    #      print('shape after reshape in core alg, Mendl ind:',i.shape)


    # Check all bond dimensions, including the boundary bonds
    bond_dimensions = [stochastic_MPS.A[0].shape[1]]  # Left bond of first tensor
    bond_dimensions.extend(A.shape[2] for A in stochastic_MPS.A)  # Right bonds of all but last tensor
    

    current_max_bond_dim = max(bond_dimensions)

    if current_max_bond_dim < max_bond_dim:
        # Use two-site TDVP
        integrate_local_twosite_modified(H, stochastic_MPS, dt*1j, numsteps=1, numiter_lanczos=25, tol_split=1e-6, max_bond_dim=max_bond_dim)
        # for i in stochastic_MPS.A:
        #     print('shape after TDVP in core alg, Mendl ind:',i.shape)
    else:
        #print('Switched to 1TDVP')
        # Switch to one-site TDVP
        ptn.integrate_local_singlesite(H, stochastic_MPS, dt*1j, numsteps=1, numiter_lanczos=25)


    # back stochastic mps to aaron ind
    for i in range(len(stochastic_MPS.A)):
         stochastic_MPS.A[i]= np.transpose(stochastic_MPS.A[i], (1,2,0))

    # for i in stochastic_MPS.A:
    #     print('shape after TDVP in core alg, Aaron ind:',i.shape)
        

    # Checks if there are noise processes
    if system.processes:
        if not force_noise:
            stochastic_MPS.A, site_jumped, jump_type, noise_operator = apply_jumps(MPS_previous_time.A, stochastic_MPS.A, dt, system)
            # for i in stochastic_MPS.A:
            #     print('shape after apply jumps, aaron ind:',i.shape)

        elif force_noise:
            stochastic_MPS.A = force_apply_jumps(MPS_previous_time.A, stochastic_MPS.A, input_noise_list[i_trajectory-1], j_time)
            # for i in stochastic_MPS.A:
            #     print('shape after force_apply_jumps, aaron ind:',i.shape)
            site_jumped = None
            jump_type = None
            noise_operator = None
    else:
        site_jumped = None
        jump_type = None
        noise_operator = None

    updated_MPS = stochastic_MPS

    return updated_MPS, stochastic_MPS, site_jumped, jump_type, noise_operator




def TN_MCWF_DynamicTDVP(psi, H, system, num_trajectories, T, dt, max_bond_dim, operator, operator_site, force_noise=False, input_noise_list={}):
    MPS = copy.deepcopy(psi)

    times = np.arange(0, T+dt, dt)

    exp_values = []
    for i in range(len(times)):
        exp_values.append(None)

    # Site canonical form used to exploit structure for expectation value
    # Local
    # MPS = site_canonical_form(MPS, orthogonality_center=operator_site)
    # exp_values[0] = expectation_value(MPS, operator, MPS, None)
    # MPS = site_canonical_form(MPS, orthogonality_center=0)

    # change MPS to aaron ind
    for i in range(len(MPS.A)):
         MPS.A[i] = np.transpose(MPS.A[i], (1, 2, 0))
    # Global
   # MPS.A = site_canonical_form(MPS.A, orthogonality_center=0)
    MPS.A = site_canonical_form(MPS.A, orthogonality_center=operator_site)
    exp_value = expectation_value(MPS.A, operator, MPS.A, site=operator_site)
    print('timestep 0 exp val:',exp_value)
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
        print("Dynamic TDVP, Trajectory", i_trajectory, "of", num_trajectories)

        trajectory_exp_values = [exp_values[0]]
        stochastic_MPS = copy.deepcopy(psi)

        # change stochastic mps to aaron ind
        for i in range(len(stochastic_MPS.A)):
            stochastic_MPS.A[i] = np.transpose(stochastic_MPS.A[i], (1, 2, 0))
        for j in range(len(times)):
            #print("Dynamic TDVP, Trajectory", i_trajectory, " Timestep",  j+1, "of", len(times))
            if j == 0:
                continue
            updated_MPS, stochastic_MPS, site_jumped, jump_type, noise_operator = core_algorithm_DynamicTDVP(stochastic_MPS, H, system, dt, i_trajectory, j, len(times)-1, max_bond_dim, force_noise, input_noise_list)

            if site_jumped is not None:
                noise_times.append(j)
                noise_sites.append(site_jumped)
                noise_types.append(jump_type)
                noise_operators.append(noise_operator)
                # print('noise time:',j)
                # print('noise site:',site_jumped)
                # print('jump_type:', jump_type)
                # print('noise_operator:', noise_operator)



      
            updated_MPS.A = site_canonical_form(updated_MPS.A, orthogonality_center=operator_site)
            exp_value = expectation_value(updated_MPS.A, operator, updated_MPS.A, operator_site)
            updated_MPS.A = site_canonical_form(updated_MPS.A, orthogonality_center=0)



            if i_trajectory == 1:
                average_exp_value = exp_value
            else:
                average_exp_value = (i_trajectory-1)/i_trajectory*exp_values[j] + 1/i_trajectory*exp_value

            exp_values[j] = average_exp_value

            # print('at end of timestep', j+1)
            # for i in stochastic_MPS.A:
            #     print('shape at end of timestep stochastic MPS, Mendl ind:',i.shape)
            # for i in updated_MPS.A:
            #     print('shape at end of timestep updated MPS, Mendl ind:',i.shape)

            #print('end of timestep', j+1)

        noise_model = NoiseModel(noise_times, noise_types, noise_sites, noise_operators)
        output_noise_list.append(noise_model)
        #print('end of trajectory',i_trajectory)

    return exp_values, times, output_noise_list



'''Until here we have to adjust for MENDL TDVP'''
