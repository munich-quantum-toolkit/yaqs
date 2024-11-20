import math
import numpy as np
import opt_einsum as oe


def psi_to_MPS(psi, d):
    """ Decomposes a state into an MPS with physical dimension d at each site.
        Length of psi and the bond dimension must have the relation
        psi_length = d^num_sites

    Args:
        psi: state vector
        d: physical dimension at each site

    Returns:
        MPS: list of rank-3 tensors corresponding to each site
    """
    MPS = []

    # Check if decompose-able (psi_length = d^num_sites)
    # Checks if float from logarithm is not close to an integer (floor or ceiling)
    # The machine error from logarithms is too much to do this an easier way
    if math.isclose(np.log(len(psi))/np.log(d), math.floor(np.log(len(psi))/np.log(d))):
        num_sites = math.floor(np.log(len(psi))/np.log(d))
    elif math.isclose(np.log(len(psi))/np.log(d), math.ceil(np.log(len(psi))/np.log(d))):
        num_sites = math.ceil(np.log(len(psi))/np.log(d))
    else:
        raise NameError('State not decompose-able into qudits for given d')

    tensor = psi
    # Iteration starting at 1 for the exponential
    for i in range(1, num_sites+1):
        if i == 1:
            tensor = np.reshape(tensor, (d, len(psi)//d))
            U, S_vector, V = np.linalg.svd(tensor, full_matrices=0)
            site_tensor = np.expand_dims(U, 0)

            left_bond = 1
            # Left bond, right bond, phys dim
            site_tensor = np.transpose(site_tensor, (0, 2, 1))
            MPS.append(site_tensor)

            left_bond = len(S_vector)

        else:
            tensor = np.reshape(tensor, (d*left_bond, len(psi)//(d**i)))
            U, S_vector, V = np.linalg.svd(tensor, full_matrices=0)

            if i == num_sites+1:
                right_bond = 1
            else:
                right_bond = U.shape[1]
            site_tensor = np.reshape(U, (left_bond, d, right_bond))
            site_tensor = np.transpose(site_tensor, (0, 2, 1))
            left_bond = len(S_vector)
            MPS.append(site_tensor)

        tensor = np.diag(S_vector) @ V

    return MPS

# TODO: Docstring



def MPS_to_psi(MPS):
    for i, tensor in enumerate(MPS):
        if i == 0:
            psi = tensor
        if i == len(MPS)-1:
            break
        # Contract bond
        psi = np.einsum('ijk, jbc->ibkc', psi, MPS[i+1])
        # Combine physical dimensions
        psi = np.reshape(psi, (psi.shape[0], psi.shape[1], psi.shape[2]*psi.shape[3]))

    psi = np.reshape(psi, psi.size) # , order='C')

    return psi


def MPO_to_matrix(MPO):
    if MPO[0].ndim == 2:
        return MPO[0]

    for i, tensor in enumerate(MPO):
        if i == 0:
            mat = tensor
        if i == len(MPO)-1:
            break
        # Contract bond
        mat = np.einsum('ijkl, jbcd->ibkcld', mat, MPO[i+1])
        # Combine physical dimensions
        mat = np.reshape(mat, (mat.shape[0], mat.shape[1], mat.shape[2]*mat.shape[3], mat.shape[4]*mat.shape[5]))

    # Final left and right bonds should be 1
    mat = np.squeeze(mat, axis=(0, 1))
    # mat = np.reshape(mat, mat.size)
    return mat


def convert_MPS_MPO(network, d):
    """ Converts an MPS to an MPO by opening the physical dimension legs

    Args:
        MPS: list of rank-3 tensors with a physical dimension d^2
        d: desired bond dimension of MPS
    Returns:
        MPO: list of rank-4 tensors at each site
    """

    assert network[0].ndim == 3 or network[0].ndim == 4

    if network[0].ndim == 3:
        MPO = []
        for tensor in network:
            # Upper and lower physical dimensions may be incorrect
            tensor = np.reshape(tensor, (tensor.shape[0], tensor.shape[1], tensor.shape[2]//d, tensor.shape[2]//d))
            MPO.append(tensor)

        return MPO

    elif network[0].ndim == 4:
        MPS = []
        for tensor in network:
            tensor = np.reshape(tensor, (tensor.shape[0], tensor.shape[1], tensor.shape[2]*tensor.shape[3]))
            MPS.append(tensor)

        return MPS


def flip_network(network):
    """ Flips the bond dimensions in the network so that we can do operations
        from right to left

    Args:
        MPS: list of rank-3 tensors
    Returns:
        new_MPS: list of rank-3 tensors with bond dimensions reversed
                and sites reversed compared to input MPS
    """
    if network[0].ndim == 3:
        new_MPS = []
        for tensor in network:
            new_tensor = np.transpose(tensor, (1, 0, 2))
            new_MPS.append(new_tensor)

        new_MPS.reverse()
        return new_MPS

    elif network[0].ndim == 4:
        new_MPO = []
        for tensor in network:
            new_tensor = np.transpose(tensor, (1, 0, 2, 3))
            new_MPO.append(new_tensor)

        new_MPO.reverse()
        return new_MPO


def convert_local_operator_to_MPO(local_operator, site, d, L):
    identity = np.eye(d)

    tensor_identity = np.expand_dims(identity, (0, 1))
    # TODO: may need to transpose operator for up,down convention
    tensor_operator = np.expand_dims(local_operator, (0, 1))

    operator_MPO = []
    for i in range(L):
        if i == site:
            operator_MPO.append(tensor_operator)
        else:
            operator_MPO.append(tensor_identity)

    return operator_MPO


# TODO: Legitimately create the density matrix MPO
def MPS_to_density_matrix(MPS):
    psi = MPS_to_psi(MPS)
    density_matrix = np.outer(psi, np.conj(psi).T)
    return density_matrix

# def truncate(MPS, max_bond_dimension):
#     MPS = site_canonical_form(MPS, orthogonality_center=0)
#     for site in range(len(MPS)-1):
#         tensor1 = MPS[site]
#         tensor2 = MPS[site+1]

#         theta = oe.contract('ijk, jbc->ikcb', tensor1, tensor2)
#         theta = np.reshape(theta, (theta.shape[0]*theta.shape[1], theta.shape[2]*theta.shape[3]))
#         U, S_vector, V = np.linalg.svd(theta, full_matrices=0)

#         S_vector = [x for x in S_vector if x > np.finfo(float).eps]
#         S_vector = np.sort(S_vector)[::-1]
#         S_vector = S_vector[0:max_bond_dimension-1]

#         U = np.reshape(U, (tensor1.shape[0], tensor1.shape[2], len(S_vector)))
#         U = np.transpose(U, (0, 2, 1))
#         MPS[site] = U
#         M = S_vector @ V
#         M = np.reshape(M, (len(S_vector), tensor2.shape[1], tensor2.shape[2]))
#         MPS[site+1] = M
#     return MPS

