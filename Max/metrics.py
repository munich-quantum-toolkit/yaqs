import numpy as np
import opt_einsum as oe


def scalar_product(A, B, site=None):
    """ Calculates the scalar product of two Matrix Product States
        by contracting all positions vertically then horizontally.

    Args:
        A: first MPS
        B: second MPS

    Returns:
        result: Frobenius norm of A and B <A|B>
    """
    A = np.conj(A)
    # Contract all sites
    if site == None:
        for site in range(len(A)):
            tensor = oe.contract('ijk, abk->iajb', A[site], B[site])
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


def expectation_value(A, B, C, site=None):
    """ Expectation value for a given MPS-MPO-MPS network

    Args:
        A: MPS
        B: MPO
        C: MPS

    Returns:
        E.real: real portion of calculated expectation value
    """

    C = np.conj(C)
    # Contract all sites
    if site == None:
        for site in range(len(A)):
            tensor = oe.contract('ijk, abkd, qrd->iaqjbr', A[site], B[site], C[site])
            if site == 0:
                E = tensor
            else:
                E = oe.contract('ijklmn, lmndef->ijkdef', E, tensor)
        E = E.squeeze()

    # Used for ignoring other tensors if MPS is in canonical form
    else:

        # Single site operators
        E = oe.contract('ijk, kd, ijd', A[site], B, C[site])
        # Two site operators
        # elif len(sites) == 2:
        #     for site in sites:
        #         if site == 0:
        #             E = tensor
        #         else:
        #             E = oe.contract('ijklmn, lmndef->ijkdef', E, tensor)

    # if E.ndim == 4:
    #     E = E[0][0][0][0].real
    # elif E.ndim == 6:
    #     E = E.squeeze()

    return E


def variance(expval, MPS):
    norm = scalar_product(MPS,MPS)
    return norm - expval**2



def fidelity(A, B):
    """ Calculates the fidelity between matrix A and B

    Args:
        A: numpy array
        B: numpy array

    Returns:
        fidelity: float
    """
    epsilon = 1e-12
    eig_A, right_A = np.linalg.eigh(A)
    eig_A[eig_A < epsilon] = 0
    eig_A = np.real(np.diag(eig_A))

    eig_B, right_B = np.linalg.eigh(B)
    eig_B[eig_B < epsilon] = 0
    eig_B = np.real(np.diag(eig_B))

    interior = right_A @ np.sqrt(eig_A) @ np.conj(right_A.T) @ right_B @ np.sqrt(eig_B) @ np.conj(right_B.T)
    fidelity = np.trace(np.abs(interior))**2

    ### Original Method
    # start = time.time()
    # sqrtA = scipy.linalg.sqrtm(A)
    # interior = sqrtA @ B @ sqrtA
    # interior = scipy.linalg.sqrtm(interior)
    # fidelity_2 = np.trace(interior)**2
    # end = time.time()
    # print("Time (No eigs): ", end-start)

    return fidelity


def sanity_checks(psi_length, d, psi, MPS):
    # Check if decompose-able (psi_length = d^num_sites)
    if not float(np.log(psi_length)/np.log(d)).is_integer():
        print('Psi is NOT decompose-able into qudits for given d')
    else:
        print('Psi is decomposable into an MPS with the given d and psi_length')

    # Check if normalized
    epsilon = 1e-12
    if np.linalg.norm(psi) < 1-epsilon or np.linalg.norm(psi) > 1+epsilon:
        print('Psi is NOT normalized')
    else:
        print('Psi is normalized')

    E = scalar_product(MPS, MPS)
    if E[0] < 1-epsilon or E[0] > 1+epsilon:
        print('The MPS is not normalized')
    else:
        print('The MPS is normalized')