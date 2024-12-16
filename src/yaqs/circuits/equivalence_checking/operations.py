import numpy as np
import opt_einsum as oe
import time

from src.initialization import initialize_identity_MPO


def check_if_identity(MPO, fidelity):
    identity_MPO = initialize_identity_MPO(len(MPO))
    identity_MPS = convert_MPS_MPO(identity_MPO)

    # start_time = time.time()
    MPS = convert_MPS_MPO(MPO)
    # print("MPO->MPS conversion:", time.time()-start_time)

    # start_time = time.time()
    trace = scalar_product(MPS, identity_MPS)
    # print("Trace:", time.time()-start_time)

    # Calculate trace
    # print("Fidelity = ", np.round(np.abs(trace), 1) / 2**len(MPO))
    # Checks if trace is not a singular values for partial trace
    if trace.size != 1:
        return False
    if np.round(np.abs(trace), 1) / 2**len(MPO) > fidelity:
        return True
    else:
        # print("Fidelity = ", np.round(np.abs(trace), 1) / 2**len(MPO))
        return False


def convert_MPS_MPO(network):
    new_network = []

    if network[0].ndim == 4:
        for tensor in network:
            # Left phys, left bond, right phys, right bond (CW)
            new_tensor = np.transpose(tensor, (1, 0, 2, 3))
            # Left bond, left phys, right phys, right bond
            new_tensor = np.reshape(new_tensor, (new_tensor.shape[0], new_tensor.shape[1]*new_tensor.shape[2], new_tensor.shape[3]))
            # new_tensor = np.transpose(tensor, (1, 3, 0, 2))
            # new_tensor = np.reshape(new_tensor, (new_tensor.shape[0], new_tensor.shape[1], new_tensor.shape[2]*new_tensor.shape[3]))

            new_network.append(new_tensor)
    elif network[0].ndim == 3:
        for tensor in network:
            # Split physical dimension
            # Left bond, left phys, right phys, right bond
            # TODO: Not generalized for d-level
            new_tensor = np.reshape(tensor, (tensor.shape[0], 2, 2, tensor.shape[2]))

            # Left phys, left bond, right phys, right bond
            new_tensor = np.transpose(new_tensor, (1, 0, 2, 3))

            new_network.append(new_tensor)

    return new_network


def normalize(MPO, threshold):
    MPS = convert_MPS_MPO(MPO)

    for site, tensor in enumerate(MPS):
        dims = tensor.shape

        # Combine left bond and phys dim
        tensor_matrix = np.reshape(tensor, (dims[0]*dims[1], dims[2]))
        U, S_list, V = np.linalg.svd(tensor_matrix, full_matrices=False)
        S_list = S_list[S_list > threshold]

        # NOTE: Remove degenerate eigenvalues?
        # S_list = [np.round(s, 12) for s in S_list]
        # S_list = list(set(S_list))

        U = U[:, 0:len(S_list)]
        V = V[0:len(S_list), :]

        # Create site tensors
        U = np.reshape(U, (dims[0], dims[1], len(S_list)))

        M = np.diag(S_list) @ V
        # M = np.reshape(M, (len(S_list), dims[0], dims[2], dims[3]))
        # # chi, s, t, k -> k, chi, s, t
        # M = np.transpose(M, (3, 0, 1, 2))
        MPS[site] = U
        if site != len(MPO)-1:
            MPS[site+1] = oe.contract('ij, jbc->ibc', M, MPS[site+1])
        else:
            norm = M

    MPO = convert_MPS_MPO(MPS)
    return MPO


def flip_site(MPO, site, conjugate=False):
    MPO[site] = np.transpose(MPO[site], (2, 1, 0, 3))
    if conjugate:
        MPO[site] = np.conj(MPO[site])


def flip_network(MPO, conjugate=False):
    for site in range(len(MPO)):
       flip_site(MPO, site, conjugate)

def scalar_product(A, B, site=None):
    """ Calculates the scalar product of two Matrix Product States
        by contracting all positions vertically then horizontally.

    Args:
        A: first MPS
        B: second MPS

    Returns:
        result: Frobenius norm of A and B <A|B>
    """
    for i, tensor in enumerate(A):
        A[i] = np.conj(tensor)

    # Contract all sites
    if site == None:
        for site in range(len(A)):
            tensor = oe.contract('ikj, akb->iajb', A[site], B[site])
            if site == 0:
                result = tensor
            else:
                result = oe.contract('ijkl, klcd->ijcd', result, tensor)
        result = np.squeeze(result)

    # Used for ignoring other tensors if MPS is in canonical form
    else:
        # Single site operators
        result = oe.contract('ijk, ijk', A[site], B[site])

    return result
