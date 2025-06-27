# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""1TDVP + 2TDVP implementation with helper functions.

This module implements functions for performing time evolution on Matrix Product States (MPS)
using the Time-Dependent Variational Principle (TDVP). It provides utilities for:
  - Splitting and merging MPS tensors via singular value decomposition (SVD).
  - Updating local MPS tensors and bond tensors using Lanczos-based approximations of the matrix exponential.
  - Constructing effective local operators through contractions with MPO tensors and environment blocks.
  - Performing single-site and two-site TDVP integration schemes to evolve the MPS in time.

These methods are designed for simulating the dynamics of quantum many-body systems and are based on
techniques described in Haegeman et al., Phys. Rev. B 94, 165116 (2016).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import opt_einsum as oe

from ..data_structures.simulation_parameters import PhysicsSimParams, StrongSimParams, WeakSimParams
from .matrix_exponential import expm_krylov

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..data_structures.networks import MPO, MPS


def split_mps_tensor(
    tensor: NDArray[np.complex128],
    svd_distribution: str,
    sim_params: PhysicsSimParams | StrongSimParams | WeakSimParams,
    *,
    dynamic: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Split a Matrix Product State (MPS) tensor into two tensors using singular value decomposition (SVD).

    The input tensor is assumed to have a composite physical index of dimension d0*d1 and virtual dimensions D0 and D2,
    i.e. its shape is (d0*d1, D0, D2). The function reshapes and splits it into two tensors:
      - A left tensor of shape (d0, D0, num_sv)
      - A right tensor of shape (d1, num_sv, D2)
    where num_sv is the number of singular values retained after truncation.

    The parameter `svd_distribution` determines how the singular values are distributed between
    the left and right tensors.
    It can be:
        - "left"  : Multiply the left tensor by the singular values.
        - "right" : Multiply the right tensor by the singular values.
        - "sqrt"  : Multiply both tensors by the square root of the singular values.

    Args:
        tensor (NDArray[np.complex128]): Input MPS tensor of shape (d0*d1, D0, D2).
        svd_distribution (str): How to distribute singular values ("left", "right", or "sqrt").
        sim_params: Simulation parameters containing threshold and max bond dimension
        dynamic: Determines if bond dimension is handled by dynamic TDVP (True) or truncation (False).

    Returns:
        tuple[NDArray[np.complex128], NDArray[np.complex128]]:
            A tuple (A0, A1) of MPS tensors after splitting.

    Raises:
        ValueError: If physical dimension can not be split in half evenly.
    """
    # Check that the physical dimension can be equally split
    if tensor.shape[0] % 2 != 0:
        msg = "The first dimension of the tensor must be divisible by 2."
        raise ValueError(msg)

    # Reshape the tensor from (d0*d1, D0, D2) to (d0, d1, D0, D2) and then transpose to bring
    # the left virtual dimension next to the first physical index: (d0, D0, d1, D2)
    d_physical = tensor.shape[0] // 2
    tensor_reshaped = tensor.reshape(d_physical, d_physical, tensor.shape[1], tensor.shape[2])
    tensor_transposed = tensor_reshaped.transpose((0, 2, 1, 3))
    shape_transposed = tensor_transposed.shape  # (d0, D0, d1, D2)

    # Merge the first two and last two indices for SVD: matrix of shape (d0*D0) x (d1*D2)
    theta_mat = tensor_transposed.reshape(
        shape_transposed[0] * shape_transposed[1],
        shape_transposed[2] * shape_transposed[3],
    )
    u_mat, s_vec, v_mat = np.linalg.svd(theta_mat, full_matrices=False)

    # Handled by dynamic TDVP
    keep = min(len(s_vec), sim_params.max_bond_dim)
    if not dynamic:
        discard = 0.0
        min_keep = min(len(s_vec), sim_params.min_bond_dim)  # Prevents pathological dimension-1 truncation
        for idx, s in enumerate(reversed(s_vec)):
            discard += s**2
            if discard >= sim_params.threshold:
                keep = max(len(s_vec) - idx, min_keep)
                break
        if sim_params.max_bond_dim is not None:
            keep = min(keep, sim_params.max_bond_dim)

    left_tensor = u_mat[:, :keep]
    s_vec = s_vec[:keep]
    right_tensor = v_mat[:keep, :]

    # Reshape U and Vh back to tensor form:
    # U to shape (d0, D0, num_sv)
    left_tensor = left_tensor.reshape((shape_transposed[0], shape_transposed[1], keep))
    # Vh reshaped to (num_sv, d1, D2)
    right_tensor = right_tensor.reshape((keep, shape_transposed[2], shape_transposed[3]))

    # Distribute the singular values according to the chosen option
    if svd_distribution == "left":
        left_tensor *= s_vec
    elif svd_distribution == "right":
        right_tensor *= s_vec[:, None, None]
    elif svd_distribution == "sqrt":
        sqrt_sigma = np.sqrt(s_vec)
        left_tensor *= sqrt_sigma
        right_tensor *= sqrt_sigma[:, None, None]
    else:
        msg = "svd_distribution parameter must be left, right, or sqrt."
        raise ValueError(msg)

    # Adjust the ordering of indices in A1 so that the physical dimension comes first:
    right_tensor = right_tensor.transpose((1, 0, 2))
    return left_tensor, right_tensor


def merge_mps_tensors(
    left_tensor: NDArray[np.complex128], right_tensor: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Merge two neighboring MPS tensors into one.

    The tensors left_tensor and right_tensor are contracted using opt_einsum. The contraction is performed
    over the common bond, and the resulting tensor is reshaped to combine the two physical dimensions into one.

    Args:
        left_tensor (NDArray[np.complex128]): Left MPS tensor.
        right_tensor (NDArray[np.complex128]): Right MPS tensor.

    Returns:
        NDArray[np.complex128]: The merged MPS tensor.
    """
    # Contract over the common bond (index 2 in A0 and index 0 in A1) using specified contraction pattern.
    merged_tensor = oe.contract("abc,dce->adbe", left_tensor, right_tensor)
    merged_shape = merged_tensor.shape
    # Reshape to combine the two physical dimensions.
    return merged_tensor.reshape((merged_shape[0] * merged_shape[1], merged_shape[2], merged_shape[3]))


def merge_mpo_tensors(
    left_tensor: NDArray[np.complex128], right_tensor: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Merge two neighboring MPO tensors into one.

    The function contracts left_tensor and right_tensor over their shared virtual bond and
    then reshapes the result to combine the physical indices appropriately.

    Args:
        left_tensor (NDArray[np.complex128]): Left MPO tensor.
        right_tensor (NDArray[np.complex128]): Right MPO tensor.

    Returns:
        NDArray[np.complex128]: The merged MPO tensor.
    """
    merged_tensor = oe.contract("acei,bdif->abcdef", left_tensor, right_tensor, optimize=True)
    dims = merged_tensor.shape
    return merged_tensor.reshape((dims[0] * dims[1], dims[2] * dims[3], dims[4], dims[5]))


def update_right_environment(
    ket: NDArray[np.complex128],
    bra: NDArray[np.complex128],
    op: NDArray[np.complex128],
    right_env: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    r"""Perform a contraction step from right to left with an operator inserted.

    The procedure involves:
      1. Contracting tensor A with the right operator block R.
      2. Contracting the result with the MPO tensor W.
      3. Permuting the indices.
      4. Contracting with the conjugate of tensor B to obtain the updated right environment.

    Args:
        ket (NDArray[np.complex128]): Tensor A (3-index tensor).
        bra (NDArray[np.complex128]): Tensor B (3-index tensor), to be conjugated.
        op (NDArray[np.complex128]): MPO tensor (4-index tensor).
        right_env (NDArray[np.complex128]): Right operator block (3-index tensor).

    Returns:
        NDArray[np.complex128]: The updated right operator block.
    """
    assert ket.ndim == 3
    assert bra.ndim == 3
    assert op.ndim == 4
    assert right_env.ndim == 3
    tensor = np.tensordot(ket, right_env, axes=1)
    tensor = np.tensordot(op, tensor, axes=((1, 3), (0, 2)))
    tensor = tensor.transpose((2, 1, 0, 3))
    return np.tensordot(tensor, bra.conj(), axes=((2, 3), (0, 2)))


def update_left_environment(
    ket: NDArray[np.complex128],
    bra: NDArray[np.complex128],
    op: NDArray[np.complex128],
    left_env: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    r"""Perform a contraction step from left to right with an operator inserted.

    The process contracts:
      1. The left operator L with the conjugate of tensor B.
      2. The result with the MPO tensor W.
      3. Finally, contracts with tensor A to produce the updated left environment.

    Args:
        ket (NDArray[np.complex128]): Tensor A (3-index tensor).
        bra (NDArray[np.complex128]): Tensor B (3-index tensor), to be conjugated.
        op (NDArray[np.complex128]): MPO tensor (4-index tensor).
        left_env (NDArray[np.complex128]): Left operator block (3-index tensor).

    Returns:
        NDArray[np.complex128]: The updated left operator block.
    """
    tensor = np.tensordot(left_env, bra.conj(), axes=(2, 1))
    tensor = np.tensordot(op, tensor, axes=((0, 2), (2, 1)))
    return np.tensordot(ket, tensor, axes=((0, 1), (0, 2)))


def initialize_right_environments(psi: MPS, op: MPO) -> NDArray[np.complex128]:
    """Compute the right operator blocks (partial contractions) for the given MPS and MPO.

    Starting from the rightmost site, an identity-like tensor is constructed and then
    the network is contracted site-by-site moving to the left to produce a list of right operator blocks.

    Args:
        psi (MPS): The Matrix Product State representing the quantum state.
        op (MPO): The Matrix Product Operator representing the Hamiltonian.

    Returns:
        NDArray[np.complex128]: A list (of length equal to the number of sites) containing the right operator blocks.

    Raises:
        ValueError: If state and operator length does not match.
    """
    num_sites = psi.length
    if num_sites != op.length:
        msg = "The lengths of the state and the operator must match."
        raise ValueError(msg)

    right_blocks = [None for _ in range(num_sites)]
    right_virtual_dim = psi.tensors[num_sites - 1].shape[2]
    mpo_right_dim = op.tensors[num_sites - 1].shape[3]
    right_identity = np.zeros((right_virtual_dim, mpo_right_dim, right_virtual_dim), dtype=complex)
    for i in range(right_virtual_dim):
        for a in range(mpo_right_dim):
            right_identity[i, a, i] = 1
    right_blocks[num_sites - 1] = right_identity

    for site in reversed(range(num_sites - 1)):
        right_blocks[site] = update_right_environment(
            psi.tensors[site + 1], psi.tensors[site + 1], op.tensors[site + 1], right_blocks[site + 1]
        )
    return right_blocks


def project_site(
    left_env: NDArray[np.complex128],
    right_env: NDArray[np.complex128],
    op: NDArray[np.complex128],
    ket: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    r"""Apply the local Hamiltonian operator on a tensor A.

    The function contracts the local MPS tensor A with the right environment R, then with the MPO tensor W,
    and finally with the left environment L, to yield the effective local Hamiltonian action.

    Args:
        left_env (NDArray[np.complex128]): Left operator block (3-index tensor).
        right_env (NDArray[np.complex128]): Right operator block (3-index tensor).
        op (NDArray[np.complex128]): MPO tensor (4-index tensor).
        ket (NDArray[np.complex128]): Local MPS tensor (3-index tensor).

    Returns:
        NDArray[np.complex128]: The resulting tensor after applying the local Hamiltonian.
    """
    tensor = np.tensordot(ket, right_env, axes=1)
    tensor = np.tensordot(op, tensor, axes=((1, 3), (0, 2)))
    tensor = np.tensordot(tensor, left_env, axes=((2, 1), (0, 1)))
    return tensor.transpose((0, 2, 1))


def project_bond(
    left_env: NDArray[np.complex128], right_env: NDArray[np.complex128], bond_tensor: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    r"""Apply the "zero-site" bond contraction between two operator blocks L and R using a bond tensor C.

    The contraction is performed in two steps:
      1. Contract the bond tensor C with the right operator block R.
      2. Contract the resulting tensor with the left operator block L.

    Args:
        left_env (NDArray[np.complex128]): Left operator block (3-index tensor).
        right_env (NDArray[np.complex128]): Right operator block (3-index tensor).
        bond_tensor (NDArray[np.complex128]): Bond tensor (2-index tensor).

    Returns:
        NDArray[np.complex128]: The resulting tensor from the bond contraction.
    """
    tensor = np.tensordot(bond_tensor, right_env, axes=1)
    return np.tensordot(left_env, tensor, axes=((0, 1), (0, 1)))


def update_site(
    left_env: NDArray[np.complex128],
    right_env: NDArray[np.complex128],
    op: NDArray[np.complex128],
    ket: NDArray[np.complex128],
    dt: float,
    lanczos_iterations: int,
) -> NDArray[np.complex128]:
    """Evolve the local MPS tensor A forward in time using the local Hamiltonian.

    The function flattens tensor A, applies a Lanczos-based approximation of the matrix exponential
    to evolve it by time dt, and then reshapes the result back to the original tensor shape.

    Args:
        left_env (NDArray[np.complex128]): Left operator block.
        right_env (NDArray[np.complex128]): Right operator block.
        op (NDArray[np.complex128]): Local MPO tensor.
        ket (NDArray[np.complex128]): Local MPS tensor.
        dt (float): Time step for evolution.
        lanczos_iterations (int): Number of Lanczos iterations.

    Returns:
        NDArray[np.complex128]: The updated MPS tensor after evolution.
    """
    ket_flat = ket.reshape(-1)
    evolved_ket_flat = expm_krylov(
        lambda x: project_site(left_env, right_env, op, x.reshape(ket.shape)).reshape(-1),
        ket_flat,
        dt,
        lanczos_iterations,
    )
    return evolved_ket_flat.reshape(ket.shape)


def update_bond(
    left_env: NDArray[np.complex128],
    right_env: NDArray[np.complex128],
    bond_tensor: NDArray[np.complex128],
    dt: float,
    lanczos_iterations: int,
) -> NDArray[np.complex128]:
    """Evolve the bond tensor C using a Lanczos iteration for the "zero-site" bond contraction.

    The bond tensor C is flattened, evolved via the Krylov subspace approximation of the matrix exponential,
    and then reshaped back to its original form.

    Args:
        left_env (NDArray[np.complex128]): Left operator block.
        right_env (NDArray[np.complex128]): Right operator block.
        bond_tensor (NDArray[np.complex128]): Bond tensor.
        dt (float): Time step for the bond evolution.
        lanczos_iterations (int): Number of Lanczos iterations.

    Returns:
        NDArray[np.complex128]: The updated bond tensor after evolution.
    """
    bond_tensor_flat = bond_tensor.reshape(-1)
    evolved_bond_tensor_flat = expm_krylov(
        lambda x: project_bond(left_env, right_env, x.reshape(bond_tensor.shape)).reshape(-1),
        bond_tensor_flat,
        dt,
        lanczos_iterations,
    )
    return evolved_bond_tensor_flat.reshape(bond_tensor.shape)


def single_site_tdvp(
    state: MPS,
    hamiltonian: MPO,
    sim_params: PhysicsSimParams | StrongSimParams | WeakSimParams,
    numiter_lanczos: int = 25,
) -> None:
    """Perform symmetric single-site Time-Dependent Variational Principle (TDVP) integration.

    The function evolves the MPS state in time by sequentially updating each site tensor using
    local Hamiltonian evolution and bond updates. The process includes a left-to-right sweep followed by
    an optional right-to-left sweep for full integration.

    Args:
        state (MPS): The initial state represented as an MPS.
        hamiltonian (MPO): Hamiltonian represented as an MPO.
        sim_params (PhysicsSimParams | StrongSimParams | WeakSimParams):
            Simulation parameters containing the time step 'dt' (and possibly a threshold for SVD truncation).
        numiter_lanczos (int, optional): Number of Lanczos iterations for each local update. Defaults to 25.

    Raises:
        ValueError: If Hamiltonian is invalid length.
    """
    num_sites = hamiltonian.length
    if num_sites != state.length:
        msg = "The state and Hamiltonian must have the same number of sites."
        raise ValueError(msg)

    right_blocks = initialize_right_environments(state, hamiltonian)

    left_blocks = [None for _ in range(num_sites)]
    left_virtual_dim = state.tensors[0].shape[1]
    mpo_left_dim = hamiltonian.tensors[0].shape[2]
    left_identity = np.zeros((left_virtual_dim, mpo_left_dim, left_virtual_dim), dtype=right_blocks[0].dtype)
    for i in range(left_virtual_dim):
        for a in range(mpo_left_dim):
            left_identity[i, a, i] = 1
    left_blocks[0] = left_identity

    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        sim_params.dt = 2

    # Left-to-right sweep: Update sites 0 to L-2.
    for i in range(num_sites - 1):
        state.tensors[i] = update_site(
            left_blocks[i],
            right_blocks[i],
            hamiltonian.tensors[i],
            state.tensors[i],
            0.5 * sim_params.dt,
            numiter_lanczos,
        )
        tensor_shape = state.tensors[i].shape
        reshaped_tensor = state.tensors[i].reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        site_tensor, bond_tensor = np.linalg.qr(reshaped_tensor)
        state.tensors[i] = site_tensor.reshape((tensor_shape[0], tensor_shape[1], site_tensor.shape[1]))
        left_blocks[i + 1] = update_left_environment(
            state.tensors[i], state.tensors[i], hamiltonian.tensors[i], left_blocks[i]
        )
        bond_tensor = update_bond(
            left_blocks[i + 1], right_blocks[i], bond_tensor, -0.5 * sim_params.dt, numiter_lanczos
        )
        state.tensors[i + 1] = oe.contract(state.tensors[i + 1], (0, 3, 2), bond_tensor, (1, 3), (0, 1, 2))

    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        sim_params.dt = 1

    last = num_sites - 1
    state.tensors[last] = update_site(
        left_blocks[last],
        right_blocks[last],
        hamiltonian.tensors[last],
        state.tensors[last],
        sim_params.dt,
        numiter_lanczos,
    )

    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        return

    # Right-to-left sweep: Update sites 1 to L-1.
    for i in reversed(range(1, num_sites)):
        state.tensors[i] = state.tensors[i].transpose((0, 2, 1))
        tensor_shape = state.tensors[i].shape
        reshaped_tensor = state.tensors[i].reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        site_tensor, bond_tensor = np.linalg.qr(reshaped_tensor)
        state.tensors[i] = site_tensor.reshape((tensor_shape[0], tensor_shape[1], site_tensor.shape[1])).transpose((
            0,
            2,
            1,
        ))
        right_blocks[i - 1] = update_right_environment(
            state.tensors[i], state.tensors[i], hamiltonian.tensors[i], right_blocks[i]
        )
        bond_tensor = bond_tensor.transpose()
        bond_tensor = update_bond(
            left_blocks[i], right_blocks[i - 1], bond_tensor, -0.5 * sim_params.dt, numiter_lanczos
        )
        state.tensors[i - 1] = oe.contract(state.tensors[i - 1], (0, 1, 3), bond_tensor, (3, 2), (0, 1, 2))
        state.tensors[i - 1] = update_site(
            left_blocks[i - 1],
            right_blocks[i - 1],
            hamiltonian.tensors[i - 1],
            state.tensors[i - 1],
            0.5 * sim_params.dt,
            numiter_lanczos,
        )


def two_site_tdvp(
    state: MPS,
    hamiltonian: MPO,
    sim_params: PhysicsSimParams | StrongSimParams | WeakSimParams,
    numiter_lanczos: int = 25,
    *,
    dynamic: bool = False,
) -> None:
    """Perform symmetric two-site TDVP integration.

    This function evolves the MPS by updating two neighboring sites simultaneously. The evolution involves:
      - Merging the two site tensors.
      - Applying the local Hamiltonian evolution on the merged tensor.
      - Splitting the merged tensor back into two tensors via SVD, using a specified singular value distribution.
      - Updating the operator blocks via left-to-right and right-to-left sweeps.

    Args:
        state (MPS): The initial state represented as an MPS.
        hamiltonian (MPO): Hamiltonian represented as an MPO.
        sim_params (PhysicsSimParams | StrongSimParams | WeakSimParams):
            Simulation parameters containing the time step 'dt' and SVD threshold.
        numiter_lanczos (int, optional): Number of Lanczos iterations for each local update. Defaults to 25.
        dynamic: Determines if bond dimension is handled by dynamic TDVP (True) or truncation (False).

    Raises:
        ValueError: If Hamiltonian is invalid length.
    """
    num_sites = hamiltonian.length
    if num_sites != state.length:
        msg = "State and Hamiltonian must have the same number of sites"
        raise ValueError(msg)
    if num_sites < 2:
        msg = "Hamiltonian is too short for a two-site update (2TDVP)."
        raise ValueError(msg)

    right_blocks = initialize_right_environments(state, hamiltonian)

    left_blocks = [None for _ in range(num_sites)]
    left_virtual_dim = state.tensors[0].shape[1]
    mpo_left_dim = hamiltonian.tensors[0].shape[2]
    left_identity = np.zeros((left_virtual_dim, mpo_left_dim, left_virtual_dim), dtype=right_blocks[0].dtype)
    for i in range(left_virtual_dim):
        for a in range(mpo_left_dim):
            left_identity[i, a, i] = 1
    left_blocks[0] = left_identity

    # Adjust simulation time step if simulation parameters require a unit time step.
    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        sim_params.dt = 2

    # Left-to-right sweep for sites 0 to L-2.
    for i in range(num_sites - 2):
        merged_tensor = merge_mps_tensors(state.tensors[i], state.tensors[i + 1])
        merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i], hamiltonian.tensors[i + 1])
        merged_tensor = update_site(
            left_blocks[i], right_blocks[i + 1], merged_mpo, merged_tensor, 0.5 * sim_params.dt, numiter_lanczos
        )
        state.tensors[i], state.tensors[i + 1] = split_mps_tensor(merged_tensor, "right", sim_params, dynamic=dynamic)
        left_blocks[i + 1] = update_left_environment(
            state.tensors[i], state.tensors[i], hamiltonian.tensors[i], left_blocks[i]
        )
        state.tensors[i + 1] = update_site(
            left_blocks[i + 1],
            right_blocks[i + 1],
            hamiltonian.tensors[i + 1],
            state.tensors[i + 1],
            -0.5 * sim_params.dt,
            numiter_lanczos,
        )

    # Guarantees unit time at final site for circuits
    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        sim_params.dt = 1

    i = num_sites - 2
    merged_tensor = merge_mps_tensors(state.tensors[i], state.tensors[i + 1])
    merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i], hamiltonian.tensors[i + 1])
    merged_tensor = update_site(
        left_blocks[i], right_blocks[i + 1], merged_mpo, merged_tensor, sim_params.dt, numiter_lanczos
    )
    # Only a single sweep is needed for circuits
    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        state.tensors[i], state.tensors[i + 1] = split_mps_tensor(merged_tensor, "right", sim_params, dynamic=dynamic)
        return

    state.tensors[i], state.tensors[i + 1] = split_mps_tensor(merged_tensor, "left", sim_params, dynamic=dynamic)
    right_blocks[i] = update_right_environment(
        state.tensors[i + 1], state.tensors[i + 1], hamiltonian.tensors[i + 1], right_blocks[i + 1]
    )

    # Right-to-left sweep.
    for i in reversed(range(num_sites - 2)):
        state.tensors[i + 1] = update_site(
            left_blocks[i + 1],
            right_blocks[i + 1],
            hamiltonian.tensors[i + 1],
            state.tensors[i + 1],
            -0.5 * sim_params.dt,
            numiter_lanczos,
        )
        merged_tensor = merge_mps_tensors(state.tensors[i], state.tensors[i + 1])
        merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i], hamiltonian.tensors[i + 1])
        merged_tensor = update_site(
            left_blocks[i], right_blocks[i + 1], merged_mpo, merged_tensor, 0.5 * sim_params.dt, numiter_lanczos
        )
        state.tensors[i], state.tensors[i + 1] = split_mps_tensor(merged_tensor, "left", sim_params, dynamic=dynamic)
        right_blocks[i] = update_right_environment(
            state.tensors[i + 1], state.tensors[i + 1], hamiltonian.tensors[i + 1], right_blocks[i + 1]
        )


def local_dynamic_tdvp(
    state: MPS,
    hamiltonian: MPO,
    sim_params: PhysicsSimParams | StrongSimParams | WeakSimParams,
    numiter_lanczos: int = 25,
) -> None:
    """Perform a dynamic TDVP sweep: at each bond.

    Local dynamic TDVP sweep. If the current bond dimension
    exceeds max_bond_dim, apply a local single-site TDVP step; otherwise,
    apply a two-site TDVP step.

    Args:
        state (MPS): MPS state to evolve.
        hamiltonian (MPO): MPO Hamiltonian.
        sim_params: Simulation parameters including dt and threshold.
        max_bond_dim (int): Maximum allowed bond dimension for two-site updates.
        numiter_lanczos (int): Lanczos iterations per local update.

    Raises:
        ValueError: If Hamiltonian is invalid length.
    """
    num_sites = hamiltonian.length
    if num_sites != state.length:
        msg = "State and Hamiltonian must have the same length"
        raise ValueError(msg)

    # Prepare environments
    right_blocks = initialize_right_environments(state, hamiltonian)
    left_blocks = [None] * num_sites
    # build identity for left_blocks[0]
    chi0 = state.tensors[0].shape[1]
    mpo_dim = hamiltonian.tensors[0].shape[2]
    eye = np.zeros((chi0, mpo_dim, chi0), dtype=right_blocks[0].dtype)
    for i in range(chi0):
        eye[i, :, i] = 1
    left_blocks[0] = eye

    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        sim_params.dt = 2

    # ----- LEFT-TO-RIGHT DYNAMIC SWEEP -----
    lock_final_site = False
    for i in range(num_sites):
        # current bond dimension between i and i+1
        bond_dim = state.tensors[i].shape[2]
        if bond_dim >= sim_params.max_bond_dim or lock_final_site:
            state.tensors[i] = update_site(
                left_blocks[i],
                right_blocks[i],
                hamiltonian.tensors[i],
                state.tensors[i],
                0.5 * sim_params.dt,
                numiter_lanczos,
            )
            if i != num_sites - 1:
                tensor_shape = state.tensors[i].shape
                reshaped_tensor = state.tensors[i].reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
                site_tensor, bond_tensor = np.linalg.qr(reshaped_tensor)
                state.tensors[i] = site_tensor.reshape((tensor_shape[0], tensor_shape[1], site_tensor.shape[1]))
                left_blocks[i + 1] = update_left_environment(
                    state.tensors[i], state.tensors[i], hamiltonian.tensors[i], left_blocks[i]
                )
                bond_tensor = update_bond(
                    left_blocks[i + 1], right_blocks[i], bond_tensor, -0.5 * sim_params.dt, numiter_lanczos
                )
                state.tensors[i + 1] = oe.contract(state.tensors[i + 1], (0, 3, 2), bond_tensor, (1, 3), (0, 1, 2))
            if i == num_sites - 2:
                # Guarantees final site is 1TDVP
                lock_final_site = True
        # Will be encountered at final site in loop due to dummy dimension
        elif i == num_sites - 1:
            continue
        elif i == num_sites - 2:
            merged_tensor = merge_mps_tensors(state.tensors[i], state.tensors[i + 1])
            merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i], hamiltonian.tensors[i + 1])
            merged_tensor = update_site(
                left_blocks[i], right_blocks[i + 1], merged_mpo, merged_tensor, 0.5 * sim_params.dt, numiter_lanczos
            )

            state.tensors[i], state.tensors[i + 1] = split_mps_tensor(merged_tensor, "right", sim_params, dynamic=True)
            right_blocks[i] = update_right_environment(
                state.tensors[i + 1], state.tensors[i + 1], hamiltonian.tensors[i + 1], right_blocks[i + 1]
            )
            left_blocks[i + 1] = update_left_environment(
                state.tensors[i], state.tensors[i], hamiltonian.tensors[i], left_blocks[i]
            )

        else:
            merged_tensor = merge_mps_tensors(state.tensors[i], state.tensors[i + 1])
            merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i], hamiltonian.tensors[i + 1])
            merged_tensor = update_site(
                left_blocks[i], right_blocks[i + 1], merged_mpo, merged_tensor, 0.5 * sim_params.dt, numiter_lanczos
            )
            state.tensors[i], state.tensors[i + 1] = split_mps_tensor(merged_tensor, "right", sim_params, dynamic=True)
            left_blocks[i + 1] = update_left_environment(
                state.tensors[i], state.tensors[i], hamiltonian.tensors[i], left_blocks[i]
            )
            state.tensors[i + 1] = update_site(
                left_blocks[i + 1],
                right_blocks[i + 1],
                hamiltonian.tensors[i + 1],
                state.tensors[i + 1],
                -0.5 * sim_params.dt,
                numiter_lanczos,
            )

    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        return

    # ----- RIGHT-TO-LEFT DYNAMIC SWEEP -----
    lock_final_site = False
    for i in reversed(range(num_sites)):
        bond_dim = state.tensors[i].shape[1]
        if bond_dim >= sim_params.max_bond_dim or lock_final_site:
            state.tensors[i] = update_site(
                left_blocks[i],
                right_blocks[i],
                hamiltonian.tensors[i],
                state.tensors[i],
                0.5 * sim_params.dt,
                numiter_lanczos,
            )
            if i != 0:
                state.tensors[i] = state.tensors[i].transpose((0, 2, 1))
                tensor_shape = state.tensors[i].shape
                reshaped_tensor = state.tensors[i].reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
                site_tensor, bond_tensor = np.linalg.qr(reshaped_tensor)
                state.tensors[i] = site_tensor.reshape((
                    tensor_shape[0],
                    tensor_shape[1],
                    site_tensor.shape[1],
                )).transpose((
                    0,
                    2,
                    1,
                ))
                right_blocks[i - 1] = update_right_environment(
                    state.tensors[i], state.tensors[i], hamiltonian.tensors[i], right_blocks[i]
                )
                bond_tensor = bond_tensor.transpose()
                bond_tensor = update_bond(
                    left_blocks[i], right_blocks[i - 1], bond_tensor, -0.5 * sim_params.dt, numiter_lanczos
                )
                state.tensors[i - 1] = oe.contract(state.tensors[i - 1], (0, 1, 3), bond_tensor, (3, 2), (0, 1, 2))

                if i == 1:
                    lock_final_site = True
        elif i == 0:
            # Will be encountered at final site in loop due to dummy dimension
            continue
        else:
            merged_tensor = merge_mps_tensors(state.tensors[i - 1], state.tensors[i])
            merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i - 1], hamiltonian.tensors[i])
            merged_tensor = update_site(
                left_blocks[i - 1], right_blocks[i], merged_mpo, merged_tensor, 0.5 * sim_params.dt, numiter_lanczos
            )
            state.tensors[i - 1], state.tensors[i] = split_mps_tensor(merged_tensor, "left", sim_params, dynamic=True)
            right_blocks[i - 1] = update_right_environment(
                state.tensors[i], state.tensors[i], hamiltonian.tensors[i], right_blocks[i]
            )
            # No backwards evolution at final site
            if i != 1:
                state.tensors[i - 1] = update_site(
                    left_blocks[i - 1],
                    right_blocks[i - 1],
                    hamiltonian.tensors[i - 1],
                    state.tensors[i - 1],
                    -0.5 * sim_params.dt,
                    numiter_lanczos,
                )


def global_dynamic_tdvp(
    state: MPS, hamiltonian: MPO, sim_params: PhysicsSimParams | StrongSimParams | WeakSimParams
) -> None:
    """Perform a dynamic Time-Dependent Variational Principle (TDVP) evolution of the system state.

    This function evolves the state by choosing between a two-site TDVP (2TDVP) and a single-site TDVP (1TDVP)
    based on the current maximum bond dimension of the MPS. The decision is made by comparing the state's bond
    dimension (obtained via `state.write_max_bond_dim()`) to the maximum allowed bond dimension specified in
    `sim_params`.

    Args:
        state (MPS): The Matrix Product State representing the current state of the system.
        hamiltonian (MPO): The Matrix Product Operator representing the Hamiltonian of the system.
        sim_params (PhysicsSimParams | StrongSimParams | WeakSimParams): Simulation parameters containing settings
            such as the maximum allowable bond dimension for the MPS.
    """
    current_max_bond_dim = state.write_max_bond_dim()
    if current_max_bond_dim < sim_params.max_bond_dim:
        # Perform 2TDVP when the current bond dimension is within the allowed limit
        two_site_tdvp(state, hamiltonian, sim_params, dynamic=True)
    else:
        # Perform 1TDVP when the bond dimension exceeds the allowed limit
        single_site_tdvp(state, hamiltonian, sim_params)
