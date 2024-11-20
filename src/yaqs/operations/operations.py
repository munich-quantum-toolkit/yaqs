import copy
import numpy as np
import opt_einsum as oe

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from yaqs.data_structures.MPO import MPO
    from yaqs.data_structures.MPS import MPS


def scalar_product(A: 'MPS', B: 'MPS', site: int=-1):
    """ Calculates the scalar product of two Matrix Product States
        by contracting all positions vertically then horizontally.

    Args:
        A: first MPS
        B: second MPS

    Returns:
        result: Frobenius norm of A and B <A|B>
    """
    A_copy = copy.deepcopy(A)
    B_copy = copy.deepcopy(B)
    for i, tensor in enumerate(A_copy.tensors):
        A_copy.tensors[i] = np.conj(tensor)

    # Contract all sites
    if site == -1:
        for site in range(A.length):
            tensor = oe.contract('abc, ade->bdce', A_copy.tensors[site], B_copy.tensors[site])
            # tensor = oe.contract('ijk, abk->iajb', A[site], B[site])
            if site == 0:
                result = tensor
            else:
                result = oe.contract('abcd, cdef->abef', result, tensor)
                # result = oe.contract('ijkl, klcd->ijcd', result, tensor)
        result = np.squeeze(result)

    # Used for ignoring other tensors if MPS is in canonical form
    else:
        # Single site operators
        print(site, A._check_canonical_form())
        # assert A._check_canonical_form() == site or A._check_canonical_form == 0 or A._check_canonical_form == A.length-1
        result = oe.contract('ijk, ijk', A_copy.tensors[site], B_copy.tensors[site])

    return result


def local_expval(state: 'MPS', operator: np.ndarray, site: int):
    """ Expectation value for a given MPS-MPO-MPS network

    Args:
        A: MPS
        B: MPO
        C: MPS

    Returns:
        E.real: real portion of calculated expectation value
    """
    # TODO: Could be more memory-efficient by not copying state
    state.set_canonical_form(site)
    temp_state = copy.deepcopy(state)
    temp_state.tensors[site] = oe.contract('ab, bcd->acd', operator, temp_state.tensors[site])
    E = scalar_product(temp_state, state, site)

    return E.real


# def variance(expval, MPS):
#     norm = scalar_product(MPS,MPS)
#     return norm - expval**2



# def fidelity(A, B):
#     """ Calculates the fidelity between matrix A and B

#     Args:
#         A: numpy array
#         B: numpy array

#     Returns:
#         fidelity: float
#     """
#     epsilon = 1e-12
#     eig_A, right_A = np.linalg.eigh(A)
#     eig_A[eig_A < epsilon] = 0
#     eig_A = np.real(np.diag(eig_A))

#     eig_B, right_B = np.linalg.eigh(B)
#     eig_B[eig_B < epsilon] = 0
#     eig_B = np.real(np.diag(eig_B))

#     interior = right_A @ np.sqrt(eig_A) @ np.conj(right_A.T) @ right_B @ np.sqrt(eig_B) @ np.conj(right_B.T)
#     fidelity = np.trace(np.abs(interior))**2

#     ### Original Method
#     # start = time.time()
#     # sqrtA = scipy.linalg.sqrtm(A)
#     # interior = sqrtA @ B @ sqrtA
#     # interior = scipy.linalg.sqrtm(interior)
#     # fidelity_2 = np.trace(interior)**2
#     # end = time.time()
#     # print("Time (No eigs): ", end-start)

#     return fidelity


# def sanity_checks(psi_length, d, psi, MPS):
#     # Check if decompose-able (psi_length = d^num_sites)
#     if not float(np.log(psi_length)/np.log(d)).is_integer():
#         print('Psi is NOT decompose-able into qudits for given d')
#     else:
#         print('Psi is decomposable into an MPS with the given d and psi_length')

#     # Check if normalized
#     epsilon = 1e-12
#     if np.linalg.norm(psi) < 1-epsilon or np.linalg.norm(psi) > 1+epsilon:
#         print('Psi is NOT normalized')
#     else:
#         print('Psi is normalized')

#     E = scalar_product(MPS, MPS)
#     if E[0] < 1-epsilon or E[0] > 1+epsilon:
#         print('The MPS is not normalized')
#     else:
#         print('The MPS is normalized')