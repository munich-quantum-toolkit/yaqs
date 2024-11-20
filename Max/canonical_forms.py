import copy
import numpy as np
import opt_einsum as oe

from operations import flip_network


# TODO: Most of these functions act on the argument MPS and do not even need to return anything
def shift_orthogonality_center_right(MPS, current_orthogonality_center):
    """ Left and right normalizes an MPS around a selected site

    Args:
        MPS: list of rank-3 tensors with a physical dimension d^2
        selected_site: site of matrix M around which we normalize
    Returns:
        new_MPS: list of rank-3 tensors at each site
    """

    tensor = MPS[current_orthogonality_center]
    old_dims = tensor.shape
    matricized_tensor = np.transpose(tensor, (0, 2, 1))
    matricized_tensor = np.reshape(matricized_tensor, (matricized_tensor.shape[0]*matricized_tensor.shape[1], matricized_tensor.shape[2]))
    Q, R = np.linalg.qr(matricized_tensor)
    Q = np.reshape(Q, (old_dims[0], old_dims[2], old_dims[1]))
    Q = np.transpose(Q, (0, 2, 1))
    MPS[current_orthogonality_center] = Q

    # If normalizing, we just throw away the R
    if current_orthogonality_center+1 < len(MPS):
        MPS[current_orthogonality_center+1] = oe.contract('ij, jbc->ibc', R, MPS[current_orthogonality_center+1])

    return MPS


def site_canonical_form(MPS, orthogonality_center):
    """ Left and right normalizes an MPS around a selected site

    Args:
        MPS: list of rank-3 tensors with a physical dimension d^2
        selected_site: site of matrix M around which we normalize
    Returns:
        new_MPS: list of rank-3 tensors at each site
    """
    def sweep_decomposition(MPS, orthogonality_center):
        for site, _ in enumerate(MPS):
            if site == orthogonality_center:
                break
            MPS = shift_orthogonality_center_right(MPS, site)
        return MPS

    MPS = sweep_decomposition(MPS, orthogonality_center)
    MPS = flip_network(MPS)
    flipped_orthogonality_center = len(MPS)-1-orthogonality_center
    MPS = sweep_decomposition(MPS, flipped_orthogonality_center)
    MPS = flip_network(MPS)

    return MPS


def normalize(MPS, form):
    if form == 'B':
        MPS = flip_network(MPS)
    
    MPS = site_canonical_form(MPS, orthogonality_center=len(MPS)-1)
    MPS = shift_orthogonality_center_right(MPS, len(MPS)-1)

    if form == 'B':
        MPS = flip_network(MPS)

    return MPS

def check_canonical_form(MPS):
    """ Checks what canonical form an MPS is in if any

    Args:
        MPS: list of rank-3 tensors with a physical dimension d^2
    """
    # Adapted from scalar product
    A = np.conj(MPS)
    B = MPS

    A_truth = []
    B_truth = []
    epsilon = 1e-12
    for i in range(len(A)):
        M = oe.contract('ijk, ibk->jb', A[i], B[i])
        M[M < epsilon] = 0
        test_identity = np.eye(M.shape[0], dtype=complex)
        A_truth.append(np.allclose(M, test_identity))

    for i in range(len(A)):
        M = oe.contract('ijk, ajk->ia', B[i], A[i])
        M[M < epsilon] = 0
        test_identity = np.eye(M.shape[0], dtype=complex)
        B_truth.append(np.allclose(M, test_identity))

    if all(A_truth):
        print("MPS is left (A) canonical.")
        print("MPS is site canonical at site % d" % (len(MPS)-1))

    elif all(B_truth):
        print("MPS is right (B) canonical.")
        print("MPS is site canonical at site 0")

    else:
        sites = []
        for truth_value in A_truth:
            if truth_value:
                sites.append(truth_value)
            else:
                break
        for truth_value in B_truth[len(sites):]:
            sites.append(truth_value)

        try:
            print("MPS is site canonical at site % d." % sites.index(False))
        except:
            form = []
            for i in range(len(A_truth)):
                if A_truth[i]:
                    form.append('A')
                elif B_truth[i]:
                    form.append('B')
            print("The MPS has the form: ", form)

    return
