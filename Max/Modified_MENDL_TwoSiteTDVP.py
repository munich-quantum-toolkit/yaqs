import numpy as np
import pytenet as ptn

def split_mps_tensor(A, qd0, qd1, qD, svd_distr, tol, max_bond_dim):
    """
    Split a MPS tensor with dimension `d0*d1 x D0 x D2` into two MPS tensors
    with dimensions `d0 x D0 x D1` and `d1 x D1 x D2`, respectively.
    """
    assert A.ndim == 3
    d0 = len(qd0)
    d1 = len(qd1)
    assert d0 * d1 == A.shape[0], 'physical dimension of MPS tensor must be equal to d0 * d1'
    # reshape as matrix and split by SVD
    A = A.reshape((d0, d1, A.shape[1], A.shape[2])).transpose((0, 2, 1, 3))
    s = A.shape
    q0 = ptn.qnumber_flatten([ qd0, qD[0]])
    q1 = ptn.qnumber_flatten([-qd1, qD[1]])
    A0, sigma, A1, qbond = split_matrix_svd(A.reshape((s[0]*s[1], s[2]*s[3])), q0, q1, tol, max_bond_dim)

    # Ensure qbond is truncated along with A0 and A1
    if max_bond_dim is not None and len(sigma) > max_bond_dim:
        A0 = A0[:, :max_bond_dim]
        A1 = A1[:max_bond_dim, :]
        sigma = sigma[:max_bond_dim]
        qbond = qbond[:max_bond_dim]



    A0.shape = (s[0], s[1], len(sigma))
    A1.shape = (len(sigma), s[2], s[3])
    # use broadcasting to distribute singular values
    if svd_distr == 'left':
        A0 = A0 * sigma
    elif svd_distr == 'right':
        A1 = A1 * sigma[:, None, None]
    elif svd_distr == 'sqrt':
        s = np.sqrt(sigma)
        A0 = A0 * s
        A1 = A1 * s[:, None, None]
    else:
        raise ValueError('svd_distr parameter must be "left", "right" or "sqrt".')
    # move physical dimension to the front
    A1 = A1.transpose((1, 0, 2))
    return (A0, A1, qbond)




def split_matrix_svd(A, q0, q1, tol, max_bond_dim):
    """
    Split a matrix by singular value decomposition,
    taking block sparsity structure dictated by quantum numbers into account,
    and truncate small singular values based on tolerance.
    """
    assert A.ndim == 2
    assert len(q0) == A.shape[0]
    assert len(q1) == A.shape[1]
    assert ptn.is_qsparse(A, [q0, -q1])

    # find common quantum numbers
    qis = np.intersect1d(q0, q1)

    if len(qis) == 0:
        assert np.linalg.norm(A) == 0
        # special case: no common quantum numbers;
        # use dummy intermediate dimension 1
        u = np.zeros((A.shape[0], 1), dtype=A.dtype)
        v = np.zeros((1, A.shape[1]), dtype=A.dtype)
        s = np.zeros(1)
        # single column of 'u' should have norm 1
        if A.shape[0] > 0:
            u[0, 0] = 1
        # ensure non-zero entry in 'u' formally matches quantum numbers
        q = q0[:1]
        # 'v' must remain zero matrix to satisfy quantum number constraints
        return (u, s, v, q)

    # require NumPy arrays for indexing
    q0 = np.array(q0)
    q1 = np.array(q1)

    # sort quantum numbers and arrange entries in A accordingly;
    # using mergesort to avoid permutations of identical quantum numbers
    idx0 = np.argsort(q0, kind='mergesort')
    idx1 = np.argsort(q1, kind='mergesort')
    if np.any(idx0 - np.arange(len(idx0))):
        # if not sorted yet...
        q0 = q0[idx0]
        A = A[idx0, :]
    if np.any(idx1 - np.arange(len(idx1))):
        # if not sorted yet...
        q1 = q1[idx1]
        A = A[:, idx1]

    # maximum intermediate dimension
    max_interm_dim = min(A.shape)

    # keep track of intermediate dimension
    D = 0

    # allocate memory for U and V matrices, singular values and
    # corresponding intermediate quantum numbers
    u = np.zeros((A.shape[0], max_interm_dim), dtype=A.dtype)
    v = np.zeros((max_interm_dim, A.shape[1]), dtype=A.dtype)
    s = np.zeros(max_interm_dim)
    q = np.zeros(max_interm_dim, dtype=q0.dtype)

    # for each shared quantum number...
    for qn in qis:
        # indices of current quantum number
        iqn = np.where(q0 == qn)[0]; i0 = iqn[0]; i1 = iqn[-1] + 1
        iqn = np.where(q1 == qn)[0]; j0 = iqn[0]; j1 = iqn[-1] + 1

        # perform SVD decomposition of current block
        usub, ssub, vsub = np.linalg.svd(A[i0:i1, j0:j1], full_matrices=False)

        # update intermediate dimension
        Dprev = D
        D += len(ssub)

        u[i0:i1, Dprev:D] = usub
        v[Dprev:D, j0:j1] = vsub
        s[Dprev:D] = ssub
        q[Dprev:D] = qn

    assert D <= max_interm_dim

    # use actual intermediate dimensions
    u = u[:, :D]
    v = v[:D, :]
    s = s[:D]
    q = q[:D]

    # truncate small singular values
    idx = ptn.retained_bond_indices(s, tol)
    u = u[:, idx]
    v = v[idx, :]
    s = s[idx]
    q = q[idx]

    # Additional truncation based on max_bond_dim
    if max_bond_dim is not None and len(s) > max_bond_dim:
        u = u[:, :max_bond_dim]
        v = v[:max_bond_dim, :]
        s = s[:max_bond_dim]
        q = q[:max_bond_dim]

    # undo sorting of quantum numbers
    if np.any(idx0 - np.arange(len(idx0))):
        u = u[np.argsort(idx0), :]
    if np.any(idx1 - np.arange(len(idx1))):
        v = v[:, np.argsort(idx1)]

    return (u, s, v, q)






def _local_hamiltonian_step(L, R, W, A, dt, numiter: int):
    """
    Local time step effected by Hamiltonian, based on a Lanczos iteration.
    """
    return ptn.expm_krylov(
        lambda x: ptn.apply_local_hamiltonian(L, R, W, x.reshape(A.shape)).reshape(-1),
            A.reshape(-1), -dt, numiter, hermitian=True).reshape(A.shape)


def _local_bond_step(L, R, C, dt, numiter: int):
    """
    Local "zero-site" bond step, based on a Lanczos iteration.
    """
    return ptn.expm_krylov(
        lambda x: ptn.apply_local_bond_contraction(L, R, x.reshape(C.shape)).reshape(-1),
            C.reshape(-1), -dt, numiter, hermitian=True).reshape(C.shape)


def contraction_operator_step_left(A, B, W, L):
    r"""
    Contraction step from left to right, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

     ______           _____
           \         /     \
          2|---   ---|1 B*2|---
           |         \__0__/
           |            |
           |
           |          __|__
           |         /  0  \
      L   1|---   ---|2 W 3|---
           |         \__1__/
           |            |
           |
           |          __|__
           |         /  0  \
          0|---   ---|1 A 2|---
     ______/         \_____/
    """
    assert A.ndim == 3
    assert B.ndim == 3
    assert W.ndim == 4
    assert L.ndim == 3
    # multiply with conjugated B tensor
    T = np.tensordot(L, B.conj(), axes=(2, 1))
    # multiply with W tensor
    T = np.tensordot(W, T, axes=((0, 2), (2, 1)))
    # multiply with A tensor
    Lnext = np.tensordot(A, T, axes=((0, 1), (0, 2)))
    return Lnext


def contraction_operator_step_right(A, B, W, R):
    r"""
    Contraction step from right to left, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

          _____           ______
         /     \         /
      ---|1 B*2|---   ---|2
         \__0__/         |
            |            |
                         |
          __|__          |
         /  0  \         |
      ---|2 W 3|---   ---|1   R
         \__1__/         |
            |            |
                         |
          __|__          |
         /  0  \         |
      ---|1 A 2|---   ---|0
         \_____/         \______
    """
    assert A.ndim == 3
    assert B.ndim == 3
    assert W.ndim == 4
    assert R.ndim == 3
    # multiply with A tensor
    T = np.tensordot(A, R, 1)
    # multiply with W tensor
    T = np.tensordot(W, T, axes=((1, 3), (0, 2)))
    # interchange levels 0 <-> 2 in T
    T = T.transpose((2, 1, 0, 3))
    # multiply with conjugated B tensor
    Rnext = np.tensordot(T, B.conj(), axes=((2, 3), (0, 2)))
    return Rnext




def integrate_local_twosite_modified(H, psi, dt, numsteps, numiter_lanczos= 25, tol_split = 1e-12, max_bond_dim=32):
    """
    Symmetric two-site TDVP integration.
    `psi` is overwritten in-place with the time-evolved state.

    Args:
        H: Hamiltonian as MPO
        psi: initial state as MPS
        dt: time step; for real-time evolution, use purely imaginary dt
        numsteps: number of time steps
        numiter_lanczos: number of Lanczos iterations for each site-local step
        tol_split: tolerance for SVD-splitting of neighboring MPS tensors

    Returns:
        float: norm of initial psi

    Reference:
        J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete
        Unifying time evolution and optimization with matrix product states
        Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
    """

    # number of lattice sites
    L = H.nsites
    assert L == psi.nsites
    assert L >= 2
    # for i in psi.A:
    #     print('shape before error, Mendl ind:',i.shape)
    # print(psi.qd)
    # for i in reversed(range(1, len(psi.A))):
    #     print(i, psi.qD[i:i+2])

    # right-normalize input matrix product state
    nrm = orthonormalize(psi,mode='right')

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = ptn.compute_right_operator_blocks(psi, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1]]], dtype=BR[0].dtype)

    # consistency check
    for i in range(len(BR)):
        assert ptn.is_qsparse(BR[i], [psi.qD[i+1], H.qD[i+1], -psi.qD[i+1]]), \
            'sparsity pattern of operator blocks must match quantum numbers'

    for n in range(numsteps):

        # sweep from left to right
        for i in range(L - 2):
            # merge neighboring tensors
            Am = ptn.merge_mps_tensor_pair(psi.A[i], psi.A[i+1])
            Hm = ptn.merge_mpo_tensor_pair(H.A[i], H.A[i+1])
            # evolve Am forward in time by half a time step
            Am = _local_hamiltonian_step(BL[i], BR[i+1], Hm, Am, 0.5*dt, numiter_lanczos)
            # split Am
            psi.A[i], psi.A[i+1], psi.qD[i+1] = split_mps_tensor(Am, psi.qd, psi.qd, [psi.qD[i], psi.qD[i+2]], 'right', tol=tol_split, max_bond_dim=max_bond_dim)
            # update the left blocks
            BL[i+1] = contraction_operator_step_left(psi.A[i], psi.A[i], H.A[i], BL[i])
            # evolve psi.A[i+1] backward in time by half a time step
            psi.A[i+1] = _local_hamiltonian_step(BL[i+1], BR[i+1], H.A[i+1], psi.A[i+1], -0.5*dt, numiter_lanczos)

        # rightmost tensor pair
        i = L - 2
        # merge neighboring tensors
        Am = ptn.merge_mps_tensor_pair(psi.A[i], psi.A[i+1])
        Hm = ptn.merge_mpo_tensor_pair(H.A[i], H.A[i+1])
        # evolve Am forward in time by a full time step
        Am = _local_hamiltonian_step(BL[i], BR[i+1], Hm, Am, dt, numiter_lanczos)
        # split Am
        psi.A[i], psi.A[i+1], psi.qD[i+1] = split_mps_tensor(Am, psi.qd, psi.qd, [psi.qD[i], psi.qD[i+2]], 'left', tol=tol_split,max_bond_dim=max_bond_dim)
        # update the right blocks
        BR[i] = contraction_operator_step_right(psi.A[i+1], psi.A[i+1], H.A[i+1], BR[i+1])

        # sweep from right to left
        for i in reversed(range(L - 2)):
            # evolve psi.A[i+1] backward in time by half a time step
            psi.A[i+1] = _local_hamiltonian_step(BL[i+1], BR[i+1], H.A[i+1], psi.A[i+1], -0.5*dt, numiter_lanczos)
            # merge neighboring tensors
            Am = ptn.merge_mps_tensor_pair(psi.A[i], psi.A[i+1])
            Hm = ptn.merge_mpo_tensor_pair(H.A[i], H.A[i+1])
            # evolve Am forward in time by half a time step
            Am = _local_hamiltonian_step(BL[i], BR[i+1], Hm, Am, 0.5*dt, numiter_lanczos)
            # split Am
            psi.A[i], psi.A[i+1], psi.qD[i+1] = split_mps_tensor(Am, psi.qd, psi.qd, [psi.qD[i], psi.qD[i+2]], 'left', tol=tol_split,max_bond_dim=max_bond_dim)
            # update the right blocks
            BR[i] = contraction_operator_step_right(psi.A[i+1], psi.A[i+1], H.A[i+1], BR[i+1])

    # return norm of initial psi
    return nrm





def qr(A, q0, q1):
    """
    Compute the block-wise QR decompositions of a matrix, taking block sparsity
    structure dictated by quantum numbers into account (that is, `A[i, j]` can
    only be non-zero if `q0[i] == q1[j]`).

    The resulting R matrix is not necessarily upper triangular due to
    reordering of entries.
    """
    assert A.ndim == 2
    assert len(q0) == A.shape[0]
    # print('len(q1)',len(q1))
    # print('A.shape[1]',A.shape[1])
    assert len(q1) == A.shape[1]
    assert ptn.is_qsparse(A, [q0, -q1])

    # find common quantum numbers
    qis = np.intersect1d(q0, q1)

    if len(qis) == 0:
        assert np.linalg.norm(A) == 0
        # special case: no common quantum numbers;
        # use dummy intermediate dimension 1 with all entries in 'R' set to zero
        Q = np.zeros((A.shape[0], 1), dtype=A.dtype)
        R = np.zeros((1, A.shape[1]), dtype=A.dtype)
        # single column of 'Q' should have norm 1
        Q[0, 0] = 1
        # ensure non-zero entry in 'Q' formally matches quantum numbers
        qinterm = q0[:1]
        return (Q, R, qinterm)

    # require NumPy arrays for indexing
    q0 = np.array(q0)
    q1 = np.array(q1)

    # sort quantum numbers and arrange entries in A accordingly;
    # using mergesort to avoid permutations of identical quantum numbers
    idx0 = np.argsort(q0, kind='mergesort')
    idx1 = np.argsort(q1, kind='mergesort')
    if np.any(idx0 - np.arange(len(idx0))):
        # if not sorted yet...
        q0 = q0[idx0]
        A = A[idx0, :]
    if np.any(idx1 - np.arange(len(idx1))):
        # if not sorted yet...
        q1 = q1[idx1]
        A = A[:, idx1]

    # maximum intermediate dimension
    max_interm_dim = min(A.shape)

    # keep track of intermediate dimension
    D = 0

    Q = np.zeros((A.shape[0], max_interm_dim), dtype=A.dtype)
    R = np.zeros((max_interm_dim, A.shape[1]), dtype=A.dtype)

    # corresponding intermediate quantum numbers
    qinterm = np.zeros(max_interm_dim, dtype=q0.dtype)

    # for each shared quantum number...
    for qn in qis:
        # indices of current quantum number
        iqn = np.where(q0 == qn)[0]; i0 = iqn[0]; i1 = iqn[-1] + 1
        iqn = np.where(q1 == qn)[0]; j0 = iqn[0]; j1 = iqn[-1] + 1

        # perform QR decomposition of current block
        Qsub, Rsub = np.linalg.qr(A[i0:i1, j0:j1], mode='reduced')

        # update intermediate dimension
        Dprev = D
        D += Qsub.shape[1]

        Q[i0:i1, Dprev:D] = Qsub
        R[Dprev:D, j0:j1] = Rsub
        qinterm[Dprev:D] = qn

    assert D <= max_interm_dim

    # use actual intermediate dimensions
    Q = Q[:, :D]
    R = R[:D, :]
    qinterm = qinterm[:D]

    # undo sorting of quantum numbers
    if np.any(idx0 - np.arange(len(idx0))):
        Q = Q[np.argsort(idx0), :]
    if np.any(idx1 - np.arange(len(idx1))):
        R = R[:, np.argsort(idx1)]

    return (Q, R, qinterm)


def local_orthonormalize_right_qr(A, Aprev, qd, qD):
    """
    Right-orthonormalize the local site tensor `A` by a QR decomposition,
    and update the tensor at the previous site.
    """
    # flip left and right virtual bond dimensions
    A = A.transpose((0, 2, 1))
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 3
    q0 = ptn.qnumber_flatten([qd, -qD[1]])
    Q, R, qbond = qr(A.reshape((s[0]*s[1], s[2])), q0, -qD[0])
    A = Q.reshape((s[0], s[1], Q.shape[1])).transpose((0, 2, 1))
    # update Aprev tensor: multiply with R from right
    Aprev = np.tensordot(Aprev, R, (2, 1))
    return (A, Aprev, -qbond)


def orthonormalize(self, mode='left'):
    """
    Left- or right-orthonormalize the MPS using QR decompositions.
    """
    if len(self.A) == 0:
        return 1

    if mode == 'left':
        for i in range(len(self.A) - 1):
            self.A[i], self.A[i+1], self.qD[i+1] = ptn.local_orthonormalize_left_qr(self.A[i], self.A[i+1], self.qd, self.qD[i:i+2])
        # last tensor
        self.A[-1], T, self.qD[-1] = ptn.local_orthonormalize_left_qr(self.A[-1], np.array([[[1]]]), self.qd, self.qD[-2:])
        # normalization factor (real-valued since diagonal of R matrix is real)
        assert T.shape == (1, 1, 1)
        nrm = T[0, 0, 0].real
        if nrm < 0:
            # flip sign such that normalization factor is always non-negative
            self.A[-1] = -self.A[-1]
            nrm = -nrm
        return nrm
    if mode == 'right':
        for i in reversed(range(1, len(self.A))):
            self.A[i], self.A[i-1], self.qD[i] = local_orthonormalize_right_qr(self.A[i], self.A[i-1], self.qd, self.qD[i:i+2])
        # first tensor
        self.A[0], T, self.qD[0] = local_orthonormalize_right_qr(self.A[0], np.array([[[1]]]), self.qd, self.qD[:2])
        # normalization factor (real-valued since diagonal of R matrix is real)
        assert T.shape == (1, 1, 1)
        nrm = T[0, 0, 0].real
        if nrm < 0:
            # flip sign such that normalization factor is always non-negative
            self.A[0] = -self.A[0]
            nrm = -nrm
        return nrm
    raise ValueError(f'mode = {mode} invalid; must be "left" or "right".')