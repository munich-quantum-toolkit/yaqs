import copy
import numpy as np
import opt_einsum as oe
import scipy.linalg

from .canonical_forms import check_canonical_form, site_canonical_form
from .operations import flip_network
from .metrics import scalar_product


# Acknowledgements to Kevin Hemery for this function, taken from Frank Pollmann's group
def evolve_lanczos(H, psiI, dt, krylovDim):
    if H.ndim == 6:
        H = np.reshape(H, (H.shape[0]*H.shape[1]*H.shape[2], H.shape[3]*H.shape[4]*H.shape[5]))
    elif H.ndim == 4:
        H = np.reshape(H, (H.shape[0]*H.shape[1], H.shape[2]*H.shape[3]))

    Dim = psiI.shape[0]
    if np.count_nonzero(psiI) == 0:
        return psiI

    if Dim >4:
        Vmatrix = np.zeros((Dim,krylovDim),dtype=np.complex128)

        psiI = psiI/np.linalg.norm(psiI)
        Vmatrix[:,0] = psiI

        alpha = np.zeros(krylovDim,dtype=np.complex128)
        beta = np.zeros(krylovDim,dtype=np.complex128)

        w = np.dot(H, psiI)

        alpha[0] = np.inner(np.conjugate(w),psiI)
        w = w -  alpha[0] * Vmatrix[:,0]
        beta[1] = np.linalg.norm(w)
        Vmatrix[:,1] = w/beta[1]

        for jj in range(1,krylovDim-1):
            w = np.dot(H, Vmatrix[:,jj]) - beta[jj]* Vmatrix[:,jj-1]
            alpha[jj] = np.real(np.inner(np.conjugate(w),Vmatrix[:,jj]))
            w = w -  alpha[jj] * Vmatrix[:,jj]
            beta[jj+1] = np.linalg.norm(w)
            Vmatrix[:,jj+1] = w/beta[jj+1]

        w = np.dot(H, Vmatrix[:,krylovDim-1]) - beta[krylovDim-1]*Vmatrix[:,krylovDim-2]
        alpha[krylovDim-1] = np.real(np.inner(np.conjugate(w),Vmatrix[:,krylovDim-1]))

        Tmatrix = np.diag(alpha,0) + np.diag(beta[1:krylovDim],1) + np.diag(beta[1:krylovDim],-1) 
    
        unitVector=np.zeros(krylovDim,dtype=complex)
        unitVector[0]=1.
            
        nans_in_alpha = np.isnan(alpha)
        if nans_in_alpha.any():
            final_Krylov_dim = krylovDim - np.count_nonzero(nans_in_alpha)
            Tmatrix = Tmatrix[0:final_Krylov_dim-1, 0:final_Krylov_dim-1]
            Vmatrix = Vmatrix[:, 0:final_Krylov_dim-1]
            unitVector = unitVector[0:final_Krylov_dim-1]
        subspaceFinal = np.dot(scipy.linalg.expm(-1j*dt*Tmatrix),unitVector)

        psiF = np.dot(Vmatrix,subspaceFinal)
    else:
        M = np.zeros([Dim,Dim],dtype=complex)
        for i in range(Dim):
            for j in range(Dim):
                vj = np.zeros(Dim);vj[j]=1.
                M[i,j] = np.dot(H, vj)[i]
        psiF = np.dot(scipy.linalg.expm(-1j*dt*M),psiI)

    return psiF


def contract_Hamiltonian(MPS, MPO, L, R, selected_site, env_type):
    """ Updates the Hamiltonian according to the environment

    Args:
        MPS: list of rank-3 tensors corresponding to each site
        MPO: list of rank-4 tensors corresponding to each site
        selected_site: site of matrix M around which we normalize
                       if env_type='bond', it will be around the bond following
                       this site
    Returns:
        H-prime: updated rank-6 Hamiltonian
                 dims = (Upper left, mid left, lower left,
                        Upper right, mid right, lower right)
                 where the upper bonds contract with our tensor
                 and the lower bonds are the leftovers
    """
    if env_type == 'site':
        # TODO: Y <-> Z ?
        # aiyckz
        H_prime = oe.contract('abc, bjyz, ijk->ckzaiy', L, MPO[selected_site], R)

    elif env_type == 'bond':
        H_prime = oe.contract('abc, ibk->ckai', L, R)

    return H_prime


def forward_evolution(MPS, MPO, L, R, dt, site):
    """ Forward evolution of the time-dependent Schrödinger equation
        used to optimize a site

    Args:
        MPS: list of rank-3 tensors
        MPO: list of rank-4 tensors
        dt: time step
        site: site we wish to optimize
    Returns:
        MPS: updated MPS with new site tensor
    """

    H_prime = contract_Hamiltonian(MPS, MPO, L, R, selected_site=site, env_type='site')
    M = MPS[site]

    old_dims = M.shape
    M_vec = np.reshape(M, (M.shape[0]*M.shape[1]*M.shape[2]))
    M_prime = evolve_lanczos(H_prime, M_vec, dt, krylovDim=5)

    M_prime = np.reshape(M_prime, old_dims)

    MPS[site] = M_prime

    return MPS


def create_bond_tensor(MPS, site, decomposition):
    # WIP: Copies added
    # M_prime =  copy.copy(MPS[site])
    # MPS = copy.copy(MPS)
    M_prime = MPS[site]

    old_dims = M_prime.shape
    M_prime = np.transpose(M_prime, (0, 2, 1))
    M_prime = np.reshape(M_prime, (M_prime.shape[0]*M_prime.shape[1], M_prime.shape[2]))

    if decomposition == 'SVD':
        U, S, V = np.linalg.svd(M_prime, full_matrices=False)

        # Allows for a changing bond dimension if bond is too high for site
        if U.shape[1] < old_dims[1]:
            U = np.reshape(U, (old_dims[0], old_dims[2], U.shape[1]))
        else:
            U = np.reshape(U, (old_dims[0], old_dims[2], old_dims[1]))
        U = np.transpose(U, (0, 2, 1))
        MPS[site] = U
        C = np.diag(S) @ V

    elif decomposition == 'QR':
        Q, R = np.linalg.qr(M_prime)
        Q = np.reshape(Q, (old_dims[0], old_dims[2], old_dims[1]))
        Q = np.transpose(Q, (0, 2, 1))
        MPS[site] = Q
        C = R

    return C


def backward_evolution(MPS, MPO, C, L, R, dt, site):
    """ Backward evolution of the time-dependent Schrödinger equation
        used to optimize a bond

    Args:
        MPS: list of rank-3 tensors
        MPO: list of rank-4 tensors
        dt: time step
        site: site BEFORE the bond we wish to optimize
    Returns:
        MPS: updated MPS with updated bond tensor multiplied into next site
    """

    H_prime = contract_Hamiltonian(MPS, MPO, L, R, selected_site=site, env_type='bond')

    old_dims = C.shape
    C_vec = np.reshape(C, (C.shape[0]*C.shape[1]))

    C_prime = evolve_lanczos(H_prime, C_vec, -dt, krylovDim=5)
    C_prime = np.reshape(C_prime, old_dims)

    MPS[site+1] = oe.contract('ij, jbc->ibc', C_prime, MPS[site+1])
    return MPS


def contract_left_environment(tensors, selected_site):
    """ Zipper contracts tensors to the left of a selected site.
        Input tensors are already from MPS-MPO-MPS site-wise contraction

    Args:
        tensors: list of rank-6 tensors with 3 left bonds and 3 right bonds
                 from contracting MPS-MPO-MPS site-wise
        selected_site: site of matrix M around which we normalize
    Returns:
        L: single rank-6 tensor
    """
    L = 1
    L = np.expand_dims(L, (0, 1, 2, 3, 4, 5))
    for site, tensor in enumerate(tensors):
        if site == selected_site:
            break

        if site == 0:
            if selected_site == 1:
                L = tensor
                break
            # Initializing our L tensor
            L = oe.contract('abcdef, deflmn->abclmn', tensor, tensors[1])
        elif site != selected_site-1:
            L = oe.contract('abcdef, deflmn->abclmn', L, tensors[site+1])

    return L


def initialize_environments(MPS, MPO):
    L = 1
    L = np.expand_dims(L, (0, 1, 2))
    
    R_list = []
    MPS = flip_network(MPS)
    MPO = flip_network(MPO)
    R = L
    R_list.append(R)
    for site in range(len(MPS)-1):
        R = update_environment(MPS, MPO, R, site)

        R_list.append(R)
    R_list = R_list[::-1]
    return L, R_list


def update_environment(MPS, MPO, L, site):
    L_updated = oe.contract('ijk, ibc, jxcz, kmz->bxm', L, MPS[site], MPO[site], np.conj(MPS[site]))
    return L_updated


def left_to_right_sweep(MPS, MPO, dt, R_list=[], decomposition='SVD'):
    if R_list == []:
        L, R_list = initialize_environments(MPS, MPO)
    else:
        L, _ = initialize_environments(MPS, MPO)

    L_list = [L]
    for site in range(len(MPS)-1):
        MPS = forward_evolution(MPS, MPO, L, R_list[site], 0.5*dt, site)

        # WIP: Norm should not change here
        # print("Before", scalar_product(MPS, MPS))
        C = create_bond_tensor(MPS, site, decomposition)
        # print("After", scalar_product(MPS, MPS))

        L = update_environment(MPS, MPO, L, site)
        L_list.append(L)
        MPS = backward_evolution(MPS, MPO, C, L, R_list[site], 0.5*dt, site)

    MPS = forward_evolution(MPS, MPO, L, R_list[len(MPS)-1], 0.5*dt, site=len(MPS)-1)
    return MPS, L_list


def single_sweep_TDVP(MPS, MPO, dt, starting_point='left'):
    """ Time-Dependent Variation Principle evolution of an input MPS according
        to an MPO

    Args:
        MPS: list of rank-3 tensors
        MPO: list of rank-4 tensors
    Returns:
    """

    # Check if physical dimensions are the same
    assert MPS[0].shape[2] == MPO[0].shape[3]

    if starting_point == 'right':
        MPS = flip_network(MPS)
        MPO = flip_network(MPO)

    # As long as we are in canonical form, we can ignore the lower
    # rows of the TDVP algorithm

    MPS, L_list = left_to_right_sweep(MPS, MPO, dt, decomposition='QR')

    MPS = flip_network(MPS)
    MPO = flip_network(MPO)

    MPS, _ = left_to_right_sweep(MPS, MPO, dt, R_list=L_list[::-1], decomposition='QR')

    MPS = flip_network(MPS)
    MPO = flip_network(MPO)

    if starting_point == 'right':
        MPS = flip_network(MPS)
        MPO = flip_network(MPO)

    return MPS

