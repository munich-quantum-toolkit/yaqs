from __future__ import annotations
import numpy as np
import opt_einsum as oe

from .matrix_exponential import expm_krylov

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..data_structures.networks import MPO, MPS


def split_mps_tensor(tensor: np.ndarray, svd_distribution: str, threshold: float=0) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a Matrix Product State (MPS) tensor into two tensors using singular value decomposition (SVD).

    The input tensor is assumed to have a composite physical index of dimension d0*d1 and virtual dimensions D0 and D2,
    i.e. its shape is (d0*d1, D0, D2). The function reshapes and splits it into two tensors:
      - A left tensor of shape (d0, D0, D1)
      - A right tensor of shape (d1, D1, D2)

    The parameter `svd_distribution` determines how the singular values are distributed between the left and right tensors.
    It can be:
        - 'left'  : Multiply the left tensor by the singular values.
        - 'right' : Multiply the right tensor by the singular values.
        - 'sqrt'  : Multiply both tensors by the square root of the singular values.

    Args:
        tensor: Input MPS tensor of shape (d0*d1, D0, D2).
        svd_distribution: How to distribute singular values ('left', 'right', or 'sqrt').
        threshold: Values in the singular value spectrum below this threshold are discarded.

    Returns:
        A tuple (A0, A1) of MPS tensors after the splitting.
    """
    # Check that the physical dimension can be equally split
    if tensor.shape[0] % 2 != 0:
        raise ValueError("The first dimension of the tensor must be divisible by 2.")

    # Reshape the tensor from (d0*d1, D0, D2) to (d0, d1, D0, D2) and then transpose to bring
    # the left virtual dimension next to the first physical index:
    # (d0, D0, d1, D2)
    d_physical = tensor.shape[0] // 2
    tensor_reshaped = tensor.reshape(d_physical, d_physical, tensor.shape[1], tensor.shape[2])
    tensor_transposed = tensor_reshaped.transpose((0, 2, 1, 3))
    shape_transposed = tensor_transposed.shape  # (d0, D0, d1, D2)

    # Merge the first two and last two indices so that we can perform an SVD:
    # The reshaped matrix has dimensions (d0*D0) x (d1*D2)
    matrix_for_svd = tensor_transposed.reshape((shape_transposed[0] * shape_transposed[1],
                                                 shape_transposed[2] * shape_transposed[3]))
    U, sigma, Vh = np.linalg.svd(matrix_for_svd, full_matrices=False)

    # Truncate the singular values below the threshold
    sigma = sigma[sigma > threshold]
    num_sv = len(sigma)

    # Truncate U and Vh (Vh is the conjugate-transpose of V)
    U = U[:, :num_sv]
    Vh = Vh[:num_sv, :]

    # Reshape U and Vh back to tensor form:
    # U goes to shape (d0, D0, num_sv)
    A0 = U.reshape((shape_transposed[0], shape_transposed[1], num_sv))
    # Vh is reshaped to (num_sv, d1, D2)
    A1 = Vh.reshape((num_sv, shape_transposed[2], shape_transposed[3]))

    # Distribute the singular values according to the chosen option
    if svd_distribution == 'left':
        # Multiply the left tensor by sigma (broadcasting over the singular value dimension)
        A0 = A0 * sigma
    elif svd_distribution == 'right':
        # Multiply the right tensor by sigma. We add extra dimensions for proper broadcasting.
        A1 = A1 * sigma[:, None, None]
    elif svd_distribution == 'sqrt':
        # Multiply both tensors by the square root of the singular values.
        sqrt_sigma = np.sqrt(sigma)
        A0 = A0 * sqrt_sigma
        A1 = A1 * sqrt_sigma[:, None, None]
    else:
        raise ValueError('svd_distribution parameter must be "left", "right", or "sqrt".')

    # Adjust the ordering of indices in A1 so that the physical dimension comes first:
    # Change from (num_sv, d1, D2) to (d1, num_sv, D2)
    A1 = A1.transpose((1, 0, 2))

    return A0, A1


def merge_mps_tensors(A0: np.ndarray, A1: np.ndarray) -> np.ndarray:
    """
    Merge two neighboring MPS tensors into one.

    The tensors A0 and A1 are contracted using opt_einsum. The contraction pattern is chosen so that
    the resulting tensor has its two physical dimensions still separated and later can be reshaped back if needed.

    Args:
        A0: Left MPS tensor.
        A1: Right MPS tensor.

    Returns:
        A merged MPS tensor.
    """
    # Contract over the common bond (index 3 in A0 and index 1 in A1)
    # The contraction indices:
    #   A0 indices: (physical_left, virtual_left, virtual_mid)
    #   A1 indices: (virtual_mid, physical_right, virtual_right)
    # The result has indices: (physical_left, physical_right, virtual_right)
    merged_tensor = oe.contract('abc,dce->adbe', A0, A1)
    # Combine the two physical dimensions into one by reshaping:
    merged_shape = merged_tensor.shape
    merged_tensor = merged_tensor.reshape((merged_shape[0]*merged_shape[1], merged_shape[2], merged_shape[3]))
    return merged_tensor


def merge_mpo_tensors(A0: np.ndarray, A1: np.ndarray) -> np.ndarray:
    """
    Merge two neighboring MPO tensors into one.

    The contraction is performed over the appropriate bond indices, and then the resulting tensor
    is reshaped to combine the physical indices from the two tensors.

    Args:
        A0: Left MPO tensor.
        A1: Right MPO tensor.

    Returns:
        A merged MPO tensor.
    """
    # Contract over the shared virtual bond (index 6 in A0 with index 6 in A1) as well as other matching indices.
    merged_tensor = oe.contract('acei,bdif->abcdef', A0, A1, optimize=True)
    # Reshape to combine the original physical indices:
    s = merged_tensor.shape
    merged_tensor = merged_tensor.reshape((s[0]*s[1], s[2]*s[3], s[4], s[5]))
    return merged_tensor


def update_right_environment(A: np.ndarray, B: np.ndarray, W: np.ndarray, R: np.ndarray) -> np.ndarray:
    r"""
    Perform a contraction step from right to left with an operator inserted.

    The network structure (indices shown for contracted bonds) is illustrated below:

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

    The steps are:
      1. Contract A with R over one bond.
      2. Contract the result with the MPO tensor W.
      3. Rearrange the resulting tensor dimensions.
      4. Contract with the conjugate of tensor B to produce the new operator block.

    Args:
        A: Tensor A with three indices.
        B: Tensor B with three indices (to be conjugated).
        W: MPO tensor with four indices.
        R: Right operator block with three indices.

    Returns:
        The updated operator block as a 3-index tensor.
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


def update_left_environment(A: np.ndarray, B: np.ndarray, W: np.ndarray, L: np.ndarray) -> np.ndarray:
    r"""
    Perform a contraction step from left to right with an operator inserted.

    The network structure is illustrated below:

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

    The steps are:
      1. Contract the left operator L with the conjugate of tensor B.
      2. Contract the result with the MPO tensor W.
      3. Contract with tensor A to yield the updated left operator block.

    Args:
        A: Tensor A with three indices.
        B: Tensor B with three indices (to be conjugated).
        W: MPO tensor with four indices.
        L: Left operator block with three indices.

    Returns:
        The updated left operator block.
    """
    # Step 1: Contract L with the conjugate of B.
    T = np.tensordot(L, B.conj(), axes=(2, 1))
    # Step 2: Contract with the MPO tensor W.
    T = np.tensordot(W, T, axes=((0, 2), (2, 1)))
    # Step 3: Contract the resulting tensor with tensor A.
    Lnext = np.tensordot(A, T, axes=((0, 1), (0, 2)))
    return Lnext


def initialize_right_environments(psi: MPS, op: MPO) -> np.ndarray:
    """
    Compute the right operator blocks (partial contractions) for the given MPS and MPO.

    Starting from the rightmost site, an identity operator is constructed and then
    the network is contracted site-by-site moving to the left.

    Args:
        psi: The matrix product state (MPS) representing the state.
        op: The matrix product operator (MPO) representing the Hamiltonian.

    Returns:
        A list of right operator blocks for each site.
    """
    num_sites = psi.length
    if num_sites != op.length:
        raise ValueError("The lengths of the state and the operator must match.")

    # Initialize the list to store right operator blocks.
    right_blocks = [None for _ in range(num_sites)]

    # Set up the rightmost operator block as an identity-like tensor.
    right_virtual_dim = psi.tensors[num_sites - 1].shape[2]
    mpo_right_dim = op.tensors[num_sites - 1].shape[3]
    right_identity = np.zeros((right_virtual_dim, mpo_right_dim, right_virtual_dim), dtype=complex)
    for i in range(right_virtual_dim):
        for a in range(mpo_right_dim):
            right_identity[i, a, i] = 1
    right_blocks[num_sites - 1] = right_identity

    # Propagate the contraction from right to left.
    for site in reversed(range(num_sites - 1)):
        # Use the next site's tensors and operator block for the contraction step.
        right_blocks[site] = update_right_environment(psi.tensors[site + 1], psi.tensors[site + 1], op.tensors[site + 1], right_blocks[site + 1])
    return right_blocks


def project_site(L: np.ndarray, R: np.ndarray, W: np.ndarray, A: np.ndarray) -> np.ndarray:
    r"""
    Apply the local Hamiltonian operator on a tensor A.

    The operation contracts the tensor network composed of the left operator block L,
    the MPO tensor W, the tensor A, and the right operator block R.

    The contraction sequence is:
      1. Contract A with R.
      2. Contract the result with W.
      3. Contract the outcome with L.
      4. Permute the resulting indices to match the output ordering.

    Args:
        L: Left block operator (3-index tensor).
        R: Right block operator (3-index tensor).
        W: MPO tensor (4-index tensor).
        A: Local MPS tensor (3-index tensor).

    Returns:
        The resulting tensor after applying the local Hamiltonian.
    """
    # Contract A with R over the appropriate index.
    T = np.tensordot(A, R, axes=1)
    # Contract with the MPO tensor W.
    T = np.tensordot(W, T, axes=((1, 3), (0, 2)))
    # Contract with the left operator block L.
    T = np.tensordot(T, L, axes=((2, 1), (0, 1)))
    # Permute indices to obtain the correct output order.
    T = T.transpose((0, 2, 1))
    return T


def project_bond(L: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    r"""
    Apply the "zero-site" bond contraction between two operator blocks L and R using a bond tensor C.

    The contraction sequence is:
      1. Contract C with the right operator block R.
      2. Contract the result with the left operator block L.

    Args:
        L: Left operator block (3-index tensor).
        R: Right operator block (3-index tensor).
        C: Bond tensor (2-index tensor).

    Returns:
        The resulting tensor from the bond contraction.
    """
    # Contract bond tensor C with R.
    T = np.tensordot(C, R, axes=1)
    # Contract with the left operator block L.
    T = np.tensordot(L, T, axes=((0, 1), (0, 1)))
    return T


def update_site(L: np.ndarray, R: np.ndarray, W: np.ndarray, A: np.ndarray, dt: float, numiter: int) -> np.ndarray:
    """
    Evolve the local MPS tensor A forward in time using the local Hamiltonian.

    This function applies a Lanczos-based exponential of the local Hamiltonian on tensor A.
    The effective operator is defined via a lambda function wrapping project_site.

    Args:
        L: Left operator block.
        R: Right operator block.
        W: Local MPO tensor.
        A: Local MPS tensor.
        dt: Time step for evolution.
        numiter: Number of Lanczos iterations.

    Returns:
        The updated MPS tensor after evolution.
    """
    # Flatten A into a vector, apply the exponential via expm_krylov, and reshape back.
    A_flat = A.reshape(-1)
    evolved_A_flat = expm_krylov(
        lambda x: project_site(L, R, W, x.reshape(A.shape)).reshape(-1),
        A_flat, dt, numiter
    )
    return evolved_A_flat.reshape(A.shape)


def update_bond(L: np.ndarray, R: np.ndarray, C: np.ndarray, dt: float, numiter: int) -> np.ndarray:
    """
    Evolve the bond tensor C using a Lanczos iteration for the "zero-site" bond contraction.

    Args:
        L: Left operator block.
        R: Right operator block.
        C: Bond tensor.
        dt: Time step for the bond evolution.
        numiter: Number of Lanczos iterations.

    Returns:
        The updated bond tensor after evolution.
    """
    C_flat = C.reshape(-1)
    evolved_C_flat = expm_krylov(
        lambda x: project_bond(L, R, x.reshape(C.shape)).reshape(-1),
        C_flat, dt, numiter
    )
    return evolved_C_flat.reshape(C.shape)


def single_site_TDVP(state: MPS, H: MPO, sim_params, numiter_lanczos: int=25):
    """
    Perform symmetric single-site Time-Dependent Variational Principle (TDVP) integration.

    The state (MPS) is evolved in time (in-place) by sequentially updating each site tensor.
    The evolution is split into left-to-right and right-to-left sweeps. In each sweep,
    local Hamiltonian evolution and bond updates are applied using Lanczos iterations.

    Args:
        H: Hamiltonian represented as an MPO.
        state: The initial state represented as an MPS.
        sim_params: Simulation parameters containing time step 'dt' (and possibly a threshold).
        numiter_lanczos: Number of Lanczos iterations to perform for each local update.

    Returns:
        None. The state is updated in place.

    Reference:
        J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete,
        "Unifying time evolution and optimization with matrix product states",
        Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
    """
    from ..data_structures.simulation_parameters import WeakSimParams, StrongSimParams
    num_sites = H.length
    if num_sites != state.length:
        raise ValueError("The state and Hamiltonian must have the same number of sites.")

    # Compute the right operator blocks for the entire chain.
    right_blocks = initialize_right_environments(state, H)

    # Initialize left operator blocks with an identity-like tensor for the first site.
    left_blocks = [None for _ in range(num_sites)]
    left_virtual_dim = state.tensors[0].shape[1]
    mpo_left_dim = H.tensors[0].shape[2]
    left_identity = np.zeros((left_virtual_dim, mpo_left_dim, left_virtual_dim), dtype=right_blocks[0].dtype)
    for i in range(left_virtual_dim):
        for a in range(mpo_left_dim):
            left_identity[i, a, i] = 1
    left_blocks[0] = left_identity

    # Adjust simulation time step if simulation parameters require a unit time step.
    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        sim_params.dt = 2

    # Left-to-right sweep: Update all sites except the last.
    for i in range(num_sites - 1):
        # Evolve the tensor at site i forward by half a time step.
        state.tensors[i] = update_site(left_blocks[i], right_blocks[i], H.tensors[i], state.tensors[i], 0.5*sim_params.dt, numiter_lanczos)

        # Left-orthonormalize the updated tensor via QR decomposition.
        tensor_shape = state.tensors[i].shape
        reshaped_tensor = state.tensors[i].reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        Q, C = np.linalg.qr(reshaped_tensor)
        state.tensors[i] = Q.reshape((tensor_shape[0], tensor_shape[1], Q.shape[1]))

        # Update the left operator block for the next site.
        left_blocks[i + 1] = update_left_environment(state.tensors[i], state.tensors[i], H.tensors[i], left_blocks[i])

        # Evolve the bond tensor C backward by half a time step.
        C = update_bond(left_blocks[i + 1], right_blocks[i], C, -0.5*sim_params.dt, numiter_lanczos)

        # Update the next site tensor by contracting it with the evolved bond tensor C.
        state.tensors[i + 1] = oe.contract(state.tensors[i + 1], (0, 3, 2), C, (1, 3), (0, 1, 2))

    # Guarantees unit time at final site for circuits
    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        sim_params.dt = 1

    # Evolve the last site tensor by a full time step.
    last = num_sites - 1
    state.tensors[last] = update_site(left_blocks[last], right_blocks[last], H.tensors[last], state.tensors[last], sim_params.dt, numiter_lanczos)

    # Only a single sweep is needed for circuits
    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        return

    # Right-to-left sweep: Update all sites except the first.
    for i in reversed(range(1, num_sites)):
        # Right-orthonormalize the tensor at site i.
        # First, transpose to swap left and right virtual bonds.
        state.tensors[i] = state.tensors[i].transpose((0, 2, 1))
        tensor_shape = state.tensors[i].shape
        reshaped_tensor = state.tensors[i].reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        Q, C = np.linalg.qr(reshaped_tensor)
        # Reshape Q and undo the transposition.
        state.tensors[i] = Q.reshape((tensor_shape[0], tensor_shape[1], Q.shape[1])).transpose((0, 2, 1))

        # Update the right operator block for the previous site.
        right_blocks[i - 1] = update_right_environment(state.tensors[i], state.tensors[i], H.tensors[i], right_blocks[i])

        # Evolve the bond tensor C backward by half a time step.
        C = C.transpose()  # Prepare bond tensor for contraction
        C = update_bond(left_blocks[i], right_blocks[i - 1], C, -0.5 * sim_params.dt, numiter_lanczos)

        # Update the previous site tensor by contracting with the evolved bond tensor.
        state.tensors[i - 1] = oe.contract(state.tensors[i - 1], (0, 1, 3), C, (3, 2), (0, 1, 2))

        # Evolve the previous site tensor forward by half a time step.
        state.tensors[i - 1] = update_site(left_blocks[i - 1], right_blocks[i - 1], H.tensors[i - 1], state.tensors[i - 1], 0.5*sim_params.dt, numiter_lanczos)


def two_site_TDVP(state: MPS, H: MPO, sim_params, numiter_lanczos: int=25):
    """
    Perform symmetric two-site TDVP integration.

    This function evolves the MPS by updating two neighboring sites simultaneously.
    The evolution includes merging the two site tensors, applying the local Hamiltonian,
    splitting the merged tensor back using an SVD (with a tolerance specified in sim_params),
    and updating the operator blocks in left-to-right and right-to-left sweeps.

    Args:
        H: Hamiltonian represented as an MPO.
        state: The initial state represented as an MPS.
        sim_params: Simulation parameters containing time step 'dt' and SVD threshold.
        numiter_lanczos: Number of Lanczos iterations for each local update.

    Returns:
        None. The state is updated in place.

    Reference:
        J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, F. Verstraete,
        "Unifying time evolution and optimization with matrix product states",
        Phys. Rev. B 94, 165116 (2016) (arXiv:1408.5056)
    """
    from ..data_structures.simulation_parameters import WeakSimParams, StrongSimParams
    num_sites = H.length
    if num_sites != state.length:
        raise ValueError("State and Hamiltonian must have the same number of sites")
    if num_sites < 2:
        raise ValueError("Hamiltonian is too short for a two-site update (2TDVP).")

    # Compute the right operator blocks.
    right_blocks = initialize_right_environments(state, H)

    # Initialize left operator blocks with an identity-like tensor.
    left_blocks = [None for _ in range(num_sites)]
    left_virtual_dim = state.tensors[0].shape[1]
    mpo_left_dim = H.tensors[0].shape[2]
    left_identity = np.zeros((left_virtual_dim, mpo_left_dim, left_virtual_dim), dtype=right_blocks[0].dtype)
    for i in range(left_virtual_dim):
        for a in range(mpo_left_dim):
            left_identity[i, a, i] = 1
    left_blocks[0] = left_identity

    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        sim_params.dt = 1

    # Left-to-right sweep for sites 0 to L-2.
    for i in range(num_sites - 2):
        # Merge tensors from site i and i+1 into a single tensor.
        merged_tensor = merge_mps_tensors(state.tensors[i], state.tensors[i + 1])
        # Similarly, merge the corresponding MPO tensors.
        merged_mpo = merge_mpo_tensors(H.tensors[i], H.tensors[i + 1])
        # Evolve the merged tensor forward by half a time step.
        merged_tensor = update_site(left_blocks[i], right_blocks[i + 1], merged_mpo, merged_tensor, 0.5*sim_params.dt, numiter_lanczos)
        # Split the merged tensor back into two tensors.
        state.tensors[i], state.tensors[i + 1] = split_mps_tensor( merged_tensor, 'right', threshold=sim_params.threshold)
        # Update the left operator block for site i+1.
        left_blocks[i + 1] = update_left_environment(state.tensors[i], state.tensors[i], H.tensors[i], left_blocks[i])
        # Evolve the tensor at site i+1 backward by half a time step.
        state.tensors[i + 1] = update_site(left_blocks[i + 1], right_blocks[i + 1], H.tensors[i + 1], state.tensors[i + 1], -0.5*sim_params.dt, numiter_lanczos)

    # Process the rightmost pair (sites L-2 and L-1)
    i = num_sites - 2
    merged_tensor = merge_mps_tensors(state.tensors[i], state.tensors[i + 1])
    merged_mpo = merge_mpo_tensors(H.tensors[i], H.tensors[i + 1])
    # Evolve the merged tensor forward by a full time step.
    merged_tensor = update_site(left_blocks[i], right_blocks[i + 1], merged_mpo, merged_tensor, sim_params.dt, numiter_lanczos)
    # Split the merged tensor using a 'left' distribution of singular values.
    state.tensors[i], state.tensors[i + 1] = split_mps_tensor(merged_tensor, 'left', threshold=sim_params.threshold)
    # Update the right operator block for site i.
    right_blocks[i] = update_right_environment(state.tensors[i + 1], state.tensors[i + 1], H.tensors[i + 1], right_blocks[i + 1])

    # Right-to-left sweep.
    for i in reversed(range(num_sites - 2)):
        # Evolve the tensor at site i+1 backward by half a time step.
        state.tensors[i + 1] = update_site(left_blocks[i + 1], right_blocks[i + 1], H.tensors[i + 1], state.tensors[i + 1], -0.5*sim_params.dt, numiter_lanczos)
        # Merge the tensors at sites i and i+1.
        merged_tensor = merge_mps_tensors(state.tensors[i], state.tensors[i + 1])
        merged_mpo = merge_mpo_tensors(H.tensors[i], H.tensors[i + 1])
        # Evolve the merged tensor forward by half a time step.
        merged_tensor = update_site(left_blocks[i], right_blocks[i + 1], merged_mpo, merged_tensor, 0.5 * sim_params.dt, numiter_lanczos)
        # Split the merged tensor using a 'left' distribution.
        state.tensors[i], state.tensors[i + 1] = split_mps_tensor(merged_tensor, 'left', threshold=sim_params.threshold)
        # Update the right operator block.
        right_blocks[i] = update_right_environment(state.tensors[i + 1], state.tensors[i + 1], H.tensors[i + 1], right_blocks[i + 1])
