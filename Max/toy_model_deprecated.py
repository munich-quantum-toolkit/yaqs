import numpy as np

# from stochastics import deexcitation_operator, excitation_operator, sigma_z
from Lindblad_functions import *
from tests import test_N

from Noise_Simulation.parameters import *

# TODO: Remove necessary functions for building new toy model
def find_N_needed(num_particles, T, dt, fidelity_limit, psi_local_initial, local_relaxation_op, local_thermal_op, local_dephasing_op, global_relaxation_gamma, global_thermal_gamma, global_dephasing_gamma):
    N_list = []
    L_list = range(1, 6)
    for L in L_list:
        psi_list = L*[psi_local_initial]

        relaxation_ops = create_list_of_local_operators(local_relaxation_op, L)
        relaxation_gammas = L*[global_relaxation_gamma]

        thermal_ops = create_list_of_local_operators(local_thermal_op, L)
        thermal_gammas = L*[global_thermal_gamma]

        dephasing_ops = create_list_of_local_operators(local_dephasing_op, L)
        dephasing_gammas = L*[global_dephasing_gamma]
        density_matrix = multi_site_tensor_product(density_matrix_subsystem, L)
        H = create_local_operator_chain(H_s, L)
        Lindblad_operators = [*relaxation_ops, *thermal_ops]
        coupling_factors = [*relaxation_gammas, *thermal_gammas]

        SO_system = System(rho=density_matrix, psi_list=[], H=H, L=len(psi_list), operators=Lindblad_operators, coupling_factors=coupling_factors)
        SU_system = System(rho=[], psi_list=psi_list, H=H_eff, L=len(psi_list), operators=[local_relaxation_op, local_thermal_op], coupling_factors=[global_relaxation_gamma, global_thermal_gamma])

        N = approximate(SU_system, SO_system, processes=['relaxation', 'thermal'], dt=dt, T=T, fidelity_limit=fidelity_limit)
        N_list.append(N)

    m = np.polyfit(L_list, N_list, deg=2)

    N_needed = m[0]*num_particles**2 + m[1]*num_particles + m[2]

    L_squared = np.array([L**2 for L in L_list])
    plt.title("Particles vs. Trajectories needed, F=%f, dt=%f, T=%f" % (fidelity_limit, dt, T))
    plt.xlabel("L")
    plt.ylabel("N")
    plt.plot(L_list, N_list, 'b')
    plt.plot(L_list, m[0]*L_squared + m[1]*L_list + m[2], 'r')
    plt.show()

    print(np.ceil(N_needed), " trajectories needed for ", num_particles, " particles", " @ dt=", dt, " T=", T)
    return int(np.ceil(N_needed))


def approximate(SU_system, SO_system, processes, dt, T, fidelity_limit, plot_on=False):
    num_trajectories = 1
    rho_stochastic_result, previous_input = stochastic_unraveling(SU_system, num_trajectories, T, dt, processes, previous_input='')
    rho_SO_full = time_evolution_superoperator(SO_system.rho, SO_system.H, SO_system.operators, SO_system.coupling_factors, T, plot_spectrum=False)
    fidelities = [fidelity(rho_SO_full, rho_stochastic_result)]
    previous_input = ''
    while fidelities[-1] < fidelity_limit:
        num_trajectories = num_trajectories+1
        rho_stochastic_result, previous_input = stochastic_unraveling(SU_system, num_trajectories, T, dt, processes, previous_input=previous_input)
        fidelities.append(fidelity(rho_SO_full, rho_stochastic_result))
    if plot_on:
        plt.title("Fidelity vs. N trajectories, L=%d, T=%f, dt=%f" % (SU_system.L, T, dt))
        plt.xlabel('N trajectories')
        plt.ylabel("Fidelity")
        plt.ylim(min(fidelities)-.005, 1.005)
        plt.title("Fidelity vs. N trajectories, L=%d, T=%f, dt=%f, N=%d (F=%f)" % (SU_system.L, T, dt, num_trajectories, fidelity_limit))
        plt.plot(range(num_trajectories), fidelities, marker='.', linestyle='')
        plt.show()
    return num_trajectories


# # Define system parameters
# L = 5  # Number of particles
# d = 2  # d-level system
# T = 1e-1
# dt = 1e-2
# fidelity_limit = 0.90

# # Initialize qubits
# psi_local_initial = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
# psi_list = L*[psi_local_initial]

# # Define qubit frequencies
# freq = 5e9
# freq_list = L*[freq]

# # Define operators and their gammas
# global_T1 = 27e-6
# global_T2star = 39e-6
# global_temp = 50e-3
# # global_emission_rate =

# global_relaxation_gamma = 1/np.sqrt(2*global_T1)
# global_thermal_gamma = 1/global_temp # TODO: Figure out the relation
# global_dephasing_gamma = 1/np.sqrt(2*global_T2star)
# # global_emission_gamma =

# local_relaxation_op = deexcitation_operator(d)
# local_thermal_op = excitation_operator(d)
# local_dephasing_op = sigma_z(d)
# # local_emission_gamma = deexcitation_operator(d)

# ### Used for superoperator check ###
# relaxation_ops = create_list_of_local_operators(local_relaxation_op, L)
# relaxation_gammas = L*[global_relaxation_gamma]

# thermal_ops = create_list_of_local_operators(local_thermal_op, L)
# thermal_gammas = L*[global_thermal_gamma]

# dephasing_ops = create_list_of_local_operators(local_dephasing_op, L)
# dephasing_gammas = L*[global_dephasing_gamma]


# ####################################

# # Create effective Hamiltonian
# H_s = H_system(freq, d)

# H_eff = H_s - 1j/2*(global_relaxation_gamma*np.conj(local_relaxation_op.T) @ local_relaxation_op
#                     + global_thermal_gamma*np.conj(local_thermal_op.T) @ local_thermal_op
#                     + global_dephasing_gamma*np.conj(local_dephasing_op.T) @ local_dephasing_op)

# # Compare with superoperator for low particle number
# density_matrix_subsystem = np.outer(psi_local_initial, np.conj(psi_local_initial))
# density_matrix = multi_site_tensor_product(density_matrix_subsystem, L)
# Lindblad_operators = [*relaxation_ops, *thermal_ops, *dephasing_ops]
# coupling_factors = [*relaxation_gammas, *thermal_gammas, *dephasing_gammas]
# H = create_local_operator_chain(H_s, L)

# ### Approximation with dephasing
# SO_system = System(rho=density_matrix, psi_list=[], H=H, L=len(psi_list), operators=Lindblad_operators, coupling_factors=coupling_factors)
# SU_system = System(rho=[], psi_list=psi_list, H=H_eff, L=len(psi_list), operators=[local_relaxation_op, local_thermal_op, local_dephasing_op], coupling_factors=[global_relaxation_gamma, global_thermal_gamma, global_dephasing_gamma])
test_N(SU_system, SO_system, processes=['relaxation', 'thermal', 'dephasing'], num_trajectories_list=range(1, 1000), dt=1e-5, T=1e-2)
###

### Solving for higher particle number
H_eff = H_s - 1j/2*(global_relaxation_gamma*np.conj(local_relaxation_op).T @ local_relaxation_op
                    + global_thermal_gamma*np.conj(local_thermal_op).T @ local_thermal_op)

Lindblad_operators = [*relaxation_ops, *thermal_ops]
coupling_factors = [*relaxation_gammas, *thermal_gammas]
# N_needed = find_N_needed(L, T, dt, fidelity_limit, psi_local_initial, local_relaxation_op, local_thermal_op, local_dephasing_op, global_relaxation_gamma, global_thermal_gamma, global_dephasing_gamma)
SO_system = System(rho=density_matrix, psi_list=[], H=H, L=len(psi_list), operators=Lindblad_operators, coupling_factors=coupling_factors)
SU_system = System(rho=[], psi_list=psi_list, H=H_eff, L=len(psi_list), operators=[local_relaxation_op, local_thermal_op], coupling_factors=[global_relaxation_gamma, global_thermal_gamma])
# test_N_expanded(SU_system, SO_system, processes=['relaxation', 'thermal'], num_trajectories_list=range(1, 100), dt=1e-6, T=1e-2)

# approximate(SU_system, SO_system, processes=['relaxation', 'thermal'], dt=1e-2, T=1e-1, fidelity_limit=0.90, plot_on=True)
# rho_stochastic_result, previous_input = stochastic_unraveling_expanded(SU_system, N_needed, T, dt, processes=['relaxation', 'thermal'])
# print(fidelity(rho_stochastic_result, density_matrix))
