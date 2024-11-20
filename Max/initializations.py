from tkinter import E
import numpy as np
import scipy.sparse
from classes import System
from single_site_operators import create_excitation_operator, create_deexcitation_operator, create_pauli_x, create_pauli_z, create_pauli_y
from tensor_products import create_local_operators_list, sum_local_operators, multi_site_tensor_product
from models import initialize_atomic_MPO, initialize_ising_MPO, initialize_transmon_MPO, initialize_Leggett_MPO
from operations import MPO_to_matrix


from single_site_operators import create_pauli_x, create_pauli_z, create_excitation_operator, create_deexcitation_operator

def initialize_heisenberg_xxz_mpo(d, num_sites, J, J_z, g):
    """ Initializes the Quantum Ising Model as a Matrix Product Operator
    Args:
        num_sites: Number of tensors in MPO
        g: Interaction parameter for S^z
        J: Interaction type for S^+S^- and S^-S^+ interaction
        J_z: Interaction type for Z interaction

    Hamiltonian in matrix form: 
        H =             H = sum J/2 X X + J/2 Y Y + D Z Z + h Z
            = sum J/2 (S_p S_m + S_m S_p) + J_z Z Z 

    Returns:
        MPO: List of tensors of length num_sites

                Left Bound MPO[1] has shape (right_bond
                                            x lower_phys_dim
                                            x upper_phys_dim)
                Inner MPO[i] has shape (left_bond
                                    x right_bond
                                    x lower_phys_dim
                                    x upper_phys_dim)
                Right Bound MPO[N] has shape (right_bond
                                            x lower_phys_dim
                                            x upper_phys_dim)

                Initialization done by hand
    """
    zero = np.zeros((d, d))
    identity = np.identity(d)
    pauli_x = create_pauli_x(d)
    pauli_y = create_pauli_y(d)
    pauli_z = create_pauli_z(d)
    S_p = pauli_x + 1j* pauli_y
    S_m = pauli_x - 1j* pauli_y

    # left_bound = np.array([identity, -J/2*S_p, -J_z*pauli_z, J/2*S_p, -g*pauli_z])
    # left_bound = np.expand_dims(left_bound, 0)
    # inner = np.array([np.array([identity, -J/2*S_p, -J_z*pauli_z, -J/2*S_m -g*pauli_z]),
    #                 np.array([zero, zero, zero, zero, S_m]),
    #                 np.array([zero, zero, zero, zero, pauli_z]),
    #                 np.array([zero, zero, zero, zero, S_p]),
    #                 np.array([zero, zero, zero, zero, identity])])


    left_bound = np.array([identity, -J*S_p, -J_z*pauli_z, J*S_p, -g*pauli_z])
    left_bound = np.expand_dims(left_bound, 0)
    inner = np.array([np.array([identity, -J*S_p, -J_z*pauli_z, -J*S_m -g*pauli_z]),
                    np.array([zero, zero, zero, zero, S_m]),
                    np.array([zero, zero, zero, zero, pauli_z]),
                    np.array([zero, zero, zero, zero, S_p]),
                    np.array([zero, zero, zero, zero, identity])])

    right_bound = np.array([[-g*pauli_z],
                            [S_m],
                            [pauli_z],
                            [S_p],
                            [identity]])
    MPO = [left_bound] + [inner]*(num_sites-2) + [right_bound]
    return MPO


def single_site_atomic_H(freq, d):
    """ System Hamiltonian used in literature H_0 = w_0 sigma_plus * sigma_minus

    Args:
        freq: int
            Frequency of system
        d: int
            Physical dimension

    Returns:
        H_0: numpy array
            System Hamiltonian
    """
    sigma_plus = create_excitation_operator(d)
    sigma_minus = create_deexcitation_operator(d)

    H_0 = freq * sigma_plus @ sigma_minus
    return H_0


def initialize_psi(psi_length):
    """ Generates a random normalized state of a given length

    Args:
        psi_length: desired vector length

    Returns:
        psi: normalized vector
    """
    # Generate random complex vector
    psi = np.random.rand(psi_length) + np.random.rand(psi_length)*1j
    # Normalize
    psi = psi/np.linalg.norm(psi)
    return psi


def initialize_Ising(d, L, J, g):
    """ Creates the Ising model for L sites in matrix form

    Args:
        L: int
            Number of sites
        J: float
            Interaction parameter
        g: float
            External field

    Returns:
        H: numpy array
            L-site Ising Hamiltonian in matrix form
    """
    H = np.zeros((d**L, d**L))

    pauli_z = create_pauli_z(d)
    pauli_x = create_pauli_x(d)
    Z = create_local_operators_list(pauli_z, L)
    X = create_local_operators_list(pauli_x, L)

    for i in range(L):
        if i != L-1:
            H += -J*Z[i] @ Z[i+1]
        H += -g*X[i]

    H = scipy.sparse.csr_matrix(H)
    return H




def create_system(model, d, L, max_bond_dimension, freq, T1, T2star, temperature, processes, model_params, initial_state='+X', D=None, calculate_exact=True, calculate_state_vector=True):
    """ Creates a system for input parameters

    Args:
        model: str
            Name of model
        L: int
            Number of sites
        d: int
            Physical dimension
        single_state: numpy array
            State of a single site for initialization
        freq: float
            Frequency of qubit
        T1: float
            Decoherence time of qubit
        T2star: float
            Dephasing time of qubit
        temperature: float
            External temperature
        processes: list of str
            Noise processes to be considered
        model_params: dict of float
            Parameters for model i.e. J and g for Ising

    Returns:
        system: System class
    """
    single_site_state = []
    if initial_state == '+X':
        for i in range(d):
            if i == 0 or i == 1:
                single_site_state.append(1/np.sqrt(2))
            else:
                single_site_state.append(0)
    elif initial_state == '+Z':
        for i in range(d):
            if i == 0:
                single_site_state.append(1)
            else:
                single_site_state.append(0)
    elif initial_state == '-Z':
        for i in range(d):
            if i == 1:
                single_site_state.append(1)
            else:
                single_site_state.append(0)
    elif initial_state == 'Random':
        single_site_state = initialize_psi(d)
    single_site_state = np.array(single_site_state)


    if model == 'Transmon':
        single_site_state = []
        for i in range(d):
            if i == 0:
                single_site_state.append(1)
            else:
                single_site_state.append(0)
            # single_site_state.append(1/np.sqrt(d))
        single_site_state = np.array(single_site_state)

        resonator_state = []
        for i in range(D):
            photons_in_resonator = 0
            if i == photons_in_resonator:
                resonator_state.append(1)
            else:
                resonator_state.append(0)
        resonator_state = np.array(resonator_state)
    else:
        resonator_state = None

    if model == 'Leggett':
        single_site_state = []
        for i in range(d):
            if i == 0:
                single_site_state.append(1)
            else:
                single_site_state.append(0)
        single_site_state = np.array(single_site_state)

    global_relaxation_gamma = 1/T1
    global_dephasing_gamma = 1/T2star
    global_thermal_gamma = temperature

    local_ops = {'relaxation': [], 'dephasing': [], 'thermal': [], 'x_dephasing': [], 'y_dephasing': []}
    local_gammas = {'relaxation': [], 'dephasing': [], 'thermal': [], 'x_dephasing': [], 'y_dephasing': []}

    if 'relaxation' in processes:
        # if model == 'Transmon':
        #     single_site_relaxation_op = create_deexcitation_operator(d)
        #     local_relaxation_gammas = L*[global_relaxation_gamma]
        #     if calculate_state_vector or calculate_exact:
        #         relaxation_ops = create_local_operators_list(single_site_relaxation_op, L, phys_dims_list=[d, D, d])
        #         local_ops['relaxation'] = relaxation_ops

        #     local_gammas['relaxation'] = local_relaxation_gammas

        #     single_site_relaxation_op = create_deexcitation_operator(D)
        #     relaxation_ops = create_local_operators_list(single_site_relaxation_op, L, phys_dims_list=[d, D, d])
        #     for site in range(L):
        #         if site in model_params['resonator_sites']:
        #             local_ops['relaxation'][site] = relaxation_ops[site]

        # else:
        single_site_relaxation_op = create_deexcitation_operator(d)
        local_relaxation_gammas = L*[global_relaxation_gamma]
        if calculate_state_vector or calculate_exact:
            relaxation_ops = create_local_operators_list(single_site_relaxation_op, L)
            local_ops['relaxation'] = relaxation_ops

        local_gammas['relaxation'] = local_relaxation_gammas


    if 'thermal' in processes:
        single_site_thermal_op = create_excitation_operator(d)
        local_thermal_gammas = L*[global_thermal_gamma]
        if calculate_state_vector or calculate_exact:
            thermal_ops = create_local_operators_list(single_site_thermal_op, L)
            local_ops['thermal'] = thermal_ops

        local_gammas['thermal'] = local_thermal_gammas

        # if model == 'Transmon':
        #     single_site_thermal_op = create_deexcitation_operator(D)
        #     thermal_ops = create_local_operators_list(single_site_thermal_op, L)
        #     for site in range(L):
        #         if site in model_params['resonator_sites']:
        #             local_ops['thermal'][site] = thermal_ops[site]
    
    if 'dephasing' in processes:
        single_site_dephasing_op = create_pauli_z(d)
        local_dephasing_gammas = L*[global_dephasing_gamma]
        if calculate_state_vector or calculate_exact:
            dephasing_ops = create_local_operators_list(single_site_dephasing_op, L)
            local_ops['dephasing'] = dephasing_ops

        local_gammas['dephasing'] = local_dephasing_gammas
        # if model == 'Transmon':
        #     single_site_dephasing_op = create_deexcitation_operator(D)
        #     dephasing_ops = create_local_operators_list(single_site_dephasing_op, L)
        #     for site in range(L):
        #         if site in model_params['resonator_sites']:
        #             local_ops['dephasing'][site] = dephasing_ops[site]
    if 'x_dephasing' in processes:
        single_site_dephasing_op = create_pauli_x(d)
        local_dephasing_gammas = L*[global_dephasing_gamma]
        if calculate_state_vector or calculate_exact:
            dephasing_ops = create_local_operators_list(single_site_dephasing_op, L)
            local_ops['x_dephasing'] = dephasing_ops

        local_gammas['x_dephasing'] = local_dephasing_gammas

    if 'y_dephasing' in processes:
        single_site_dephasing_op = create_pauli_y(d)
        local_dephasing_gammas = L*[global_dephasing_gamma]
        if calculate_state_vector or calculate_exact:
            dephasing_ops = create_local_operators_list(single_site_dephasing_op, L)
            local_ops['y_dephasing'] = dephasing_ops

        local_gammas['y_dephasing'] = local_dephasing_gammas

    # Initialization of effective Hamiltonian
    if model == 'Atomic':
        if calculate_state_vector or calculate_exact:
            H_s = single_site_atomic_H(freq, d)
            H_0 = sum_local_operators(H_s, L)
        else: 
            H_0 = None
        H_0_MPO = initialize_atomic_MPO(L, freq, d)
    elif model == 'Ising':
        J = model_params['J']
        g = model_params['g']
        if calculate_state_vector or calculate_exact:
            H_0 = initialize_Ising(d, L, model_params['J'], model_params['g'])
        else:
            H_0 = None
        H_0_MPO = initialize_ising_MPO(d, L, J, g)
        # if calculate_state_vector or calculate_exact:
        #     H_0 = MPO_to_matrix(H_0_MPO)
        # else:
        #     H_0 = None
    elif model == 'Heisenberg':
        J = model_params['J']
        J_z = model_params['J_z']
        g = model_params['g']
        H_0_MPO = initialize_heisenberg_xxz_mpo(d, L, J, J_z, g)
        H_0 = None

    elif model == 'Transmon':
        H_0_MPO = initialize_transmon_MPO(d, D, model_params['freq0'], model_params['anharmonicity0'], model_params['freq1'], model_params['anharmonicity1'], model_params['freq_resonator'], model_params['g0'], model_params['g1'])
        if calculate_state_vector or calculate_exact:
            H_0 = MPO_to_matrix(H_0_MPO)
        else:
            H_0 = None
    elif model == 'Leggett':
        H_0_MPO = initialize_Leggett_MPO(d, model_params['epsilon'], model_params['delta'], L)
        if calculate_state_vector or calculate_exact:
            H_0 = H_0_MPO
        else:
            H_0 = None
        print(H_0)
    # Required for stochastic unraveling
    if calculate_exact or calculate_state_vector:
        if L == 1:
            state_vector = single_site_state
        else:
            state_vector = multi_site_tensor_product(single_site_state, L).toarray().T
    else:
        state_vector = None

    # Required for superoperator
    if calculate_exact:
        density_matrix = np.outer(state_vector, np.conj(state_vector))
    else:
        density_matrix = None

    if D == None:
        D = d

    system = System(model, model_params, d, D, density_matrix, state_vector, resonator_state, max_bond_dimension, H_0, H_0_MPO, L, local_ops, local_gammas, single_site_state, T1, T2star, temperature, processes, calculate_exact, calculate_state_vector)
    return system
