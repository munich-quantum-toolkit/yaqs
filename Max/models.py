import numpy as np
from scipy.linalg import block_diag

from single_site_operators import create_pauli_x, create_pauli_z, create_excitation_operator, create_deexcitation_operator


def initialize_atomic_MPO(num_sites, freq, d):
    identity = np.identity(d)
    zero = np.zeros((d, d))

    inner = np.array([np.array([identity, freq*create_excitation_operator(d) @ create_deexcitation_operator(d)]),
                      np.array([zero, identity])])   

    left_bound = inner[0]
    left_bound = np.expand_dims(left_bound, 0)
    right_bound = inner[:, 1]
    right_bound = np.expand_dims(right_bound, 1)

    MPO = [left_bound] + [inner]*(num_sites-2) + [right_bound]




### Taken from Bachelor Thesis ###
def initialize_ising_MPO(d, num_sites, J, g):
    """ Initializes the Quantum Ising Model as a Matrix Product Operator
    Args:
        num_sites: Number of tensors in MPO
        g: Interaction parameter
        J: Interaction type, attached to first pauli_z in MPO

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
    pauli_z = create_pauli_z(d)

    left_bound = np.array([identity, -J*pauli_z, -g*pauli_x])
    left_bound = np.expand_dims(left_bound, 0)
    inner = np.array([np.array([identity, -J*pauli_z, -g*pauli_x]),
                      np.array([zero, zero, pauli_z]),
                      np.array([zero, zero, identity])])

    right_bound = np.array([[-g*pauli_x],
                            [pauli_z],
                            [identity]])
    MPO = [left_bound] + [inner]*(num_sites-2) + [right_bound]
    return MPO

def initialize_heisenberg_xxz_mpo(d, num_sites, J, J_z, g):
        """ Initializes the Quantum Ising Model as a Matrix Product Operator
        Args:
            num_sites: Number of tensors in MPO
            g: Interaction parameter for S^z
            J: Interaction type for S^+S^- and S^-S^+ interaction
            J_z: Interaction type for Z interaction

        Hamiltonian in matrix form: 
            H =             H = sum J X X + J Y Y + D Z Z + h Z
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



def initialize_transmon_MPO(d, D, freq0, anharmonicity0, freq1, anharmonicity1, freq_resonator, g0, g1):
    # a = create_deexcitation_operator(D)
    # a_dagger = create_excitation_operator(D)
    # b = create_deexcitation_operator(d)
    # b_dagger = create_excitation_operator(d)

    # qubit_identity = np.eye(d)
    # qubit_zeros = np.zeros((d, d))
    # resonator_identity = np.eye(D)
    # resonator_zeros = np.zeros((D, D))

    # qubit_0 = np.array([qubit_identity, freq0*(b_dagger @ b) + anharmonicity0/2*(b_dagger @ b) @ (b_dagger @ b - qubit_identity)])
    # qubit_0 = np.expand_dims(qubit_0, 0)
    # resonator = np.array([[resonator_identity, resonator_zeros], [resonator_zeros, resonator_identity]])
    # qubit_1 = np.array([[freq1*(b_dagger @ b) + anharmonicity1/2*(b_dagger @ b) @ (b_dagger @ b - qubit_identity)], [qubit_identity]])
    # qubit_MPO = [qubit_0, resonator, qubit_1]

    # qubit_0 = np.array([qubit_identity])
    # qubit_0 = np.expand_dims(qubit_0, axis=0)
    # resonator = np.array([freq_resonator*(a_dagger @ a)])
    # resonator = np.expand_dims(resonator, axis=0)
    # qubit_1 = np.array([qubit_identity])
    # qubit_1 = np.expand_dims(qubit_1, axis=0)
    # resonator_MPO = [qubit_0, resonator, qubit_1]

    # qubit_0 = np.array([qubit_identity, g0*(b_dagger + b)])
    # qubit_0 = np.expand_dims(qubit_0, 0)
    # resonator = np.array([[a_dagger + a, resonator_zeros], [resonator_zeros, a_dagger + a]])
    # qubit_1 = np.array([[g1*(b_dagger + b)], [qubit_identity]])
    # interaction_MPO = [qubit_0, resonator, qubit_1]

    # site0 = np.concatenate([qubit_MPO[0], resonator_MPO[0]], axis=1)
    # site0 = np.concatenate([site0, interaction_MPO[0]], axis=1)

    # zeros = np.zeros((qubit_MPO[1].shape[0], resonator_MPO[1].shape[1], resonator_MPO[1].shape[2], resonator_MPO[1].shape[3]))
    # site1_top = np.concatenate([qubit_MPO[1], zeros], axis=1)
    # zeros = np.zeros((resonator_MPO[1].shape[0], qubit_MPO[1].shape[1], resonator_MPO[1].shape[2], resonator_MPO[1].shape[3]))
    # site1_bottom = np.concatenate([zeros, resonator_MPO[1]], axis=1)
    # site1 = np.concatenate([site1_top, site1_bottom], axis=0)

    # zeros = np.zeros((site1.shape[0], interaction_MPO[1].shape[1], interaction_MPO[1].shape[2], interaction_MPO[1].shape[3]))
    # site1_top = np.concatenate([site1, zeros], axis=1)
    # zeros = np.zeros((interaction_MPO[1].shape[0], site1.shape[1], interaction_MPO[1].shape[2], interaction_MPO[1].shape[3]))
    # site1_bottom = np.concatenate([zeros, interaction_MPO[1]], axis=1)
    # site1 = np.concatenate([site1_top, site1_bottom], axis=0)

    # site2 = np.concatenate([qubit_MPO[2], resonator_MPO[2]], axis=0)
    # site2 = np.concatenate([site2, interaction_MPO[2]], axis=0)

    # full_MPO = [site0, site1, site2]

    a = create_deexcitation_operator(D)
    a_dagger = create_excitation_operator(D)
    b = create_deexcitation_operator(d)
    b_dagger = create_excitation_operator(d)

    qubit_identity = np.eye(d)
    qubit_zeros = np.zeros((d, d))
    resonator_identity = np.eye(D)
    resonator_zeros = np.zeros((D, D))

    B_0 = freq0*(b_dagger @ b) + anharmonicity0/2*(b_dagger @ b) @ (b_dagger @ b - qubit_identity)
    B_1 = freq1*(b_dagger @ b) + anharmonicity1/2*(b_dagger @ b) @ (b_dagger @ b - qubit_identity)
    M1 = np.array([B_0, qubit_identity, g0*(b_dagger + b), qubit_identity])
    M2 = np.array([np.array([resonator_identity, resonator_zeros, resonator_zeros, resonator_zeros]),
                   np.array([resonator_zeros, resonator_zeros, freq_resonator*(a_dagger @ a), resonator_zeros]),
                    np.array([a_dagger + a, resonator_zeros, resonator_zeros, resonator_zeros]),
                    np.array([resonator_zeros, a_dagger + a, resonator_zeros, resonator_identity])])

    M3 = np.array([qubit_identity, g1*(b_dagger + b), qubit_identity, B_1])
    M1 = np.expand_dims(M1, axis=0)
    M3 = np.expand_dims(M3, axis=1)

    full_MPO = [M1, M2, M3]

    return full_MPO

def initialize_Leggett_MPO(d, epsilon, delta, L):
    sigma_x = create_pauli_x(d)
    sigma_z = create_pauli_z(d)
    if L == 1:
        MPO = -1/2*delta*sigma_x + 1/2*epsilon*sigma_z
    return MPO


