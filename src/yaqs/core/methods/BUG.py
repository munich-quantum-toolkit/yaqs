

import numpy as np
from numpy.linalg import qr

from yaqs.core.methods.TDVP import (initialize_right_environments,
                                    update_site,
                                    update_left_environment,
                                    update_right_environment)
from yaqs.core.data_structures.simulation_parameters import (WeakSimParams,
                                                             StrongSimParams)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple
    from yaqs.core.data_structures.networks import MPO, MPS

def _right_qr(ps_tensor: np.ndarray
              ) -> Tuple[np.ndarray,np.ndarray]:
    """
    Performs the QR decompositoin of an MPS tensor.

    Args:
        ps_tensor: The tensor to be decomposed.

    Returns:
        np.ndarray: The Q tensor with the left virtual leg and the physical
            leg (phys,left,new).
        np.ndarray: The R matrix with the right virtual leg (new,right).

    """
    old_shape = ps_tensor.shape
    qr_shape = (old_shape[0]*old_shape[1],old_shape[2])
    ps_tensor = ps_tensor.reshape(qr_shape)
    q_matrix, r_matrix = qr(ps_tensor)
    new_shape = (old_shape[0],old_shape[1],-1)
    q_matrix = q_matrix.reshape(new_shape)
    return q_matrix, r_matrix

def left_right_local_update(site: int,
                            state: MPS,
                            H: MPO,
                            left_envs: list[np.ndarray],
                            right_envs: list[np.ndarray],
                            sim_params,
                            numiter: int
                            ):
    """
    Run the local update for a site in the BUG algorithm.

    This expands the bond to the right of the site and 
    """
    old_tensor = state[site]
    updated_tensor = update_site(left_envs[site],
                                    right_envs[site],
                                    H[site],
                                    old_tensor,
                                    sim_params.dt,
                                    numiter)
    stacked_tensor = np.stack((old_tensor,updated_tensor),
                              axis=-1)
    new_q, _ = _right_qr(stacked_tensor)
    old_q, old_r = _right_qr(old_tensor)
    # Obtain the basis change matrix by contracting the Q tensors
    # We merely need to contract these, as everything to the left is
    # in caonical form
    basis_change_m = np.tensordot(new_q.conj(),
                                  old_q,
                                  axes=([0,1],[0,1]))
    # We need to have the next tensor in canonical form
    new_next_tensor = np.tensordot(old_r,
                                   state[site+1],
                                   axes=(1,1)) # legs (left,phys,right)
    new_next_tensor = new_next_tensor.transpose(1,0,2)
    # Absorb basis change tensor to left leg of next site
    new_next_tensor = np.tensordot(basis_change_m,
                                   new_next_tensor,
                                   axis=(1,1)) # legs (left,phys,right)
    new_next_tensor = new_next_tensor.transpose(1,0,2)
    state[site] = updated_tensor
    state[site+1] = new_next_tensor
    new_env = update_left_environment(updated_tensor,
                                      updated_tensor,
                                      H[site],
                                      left_envs[site])
    left_envs[site+1] = new_env

def _left_qr(ps_tensor: np.ndarray
              ) -> Tuple[np.ndarray,np.ndarray]:
    """
    Performs the QR decompositoin of an MPS tensor.

    Args:
        ps_tensor: The tensor to be decomposed.

    Returns:
        np.ndarray: The Q tensor with the physical leg and the right virtual
            leg (phys,new,right).
        np.ndarray: The R matrix with the left virtual leg (left,new).
    
    """
    old_shape = ps_tensor.shape
    ps_tensor = ps_tensor.transpose(0,2,1)
    qr_shape = (old_shape[0]*old_shape[2],old_shape[1])
    ps_tensor = ps_tensor.reshape(qr_shape)
    q_matrix, r_matrix = qr(ps_tensor)
    q_tensor = q_matrix.reshape((old_shape[0],old_shape[2],-1))
    q_tensor = q_tensor.transpose(0,2,1)
    r_matrix = r_matrix.T
    return q_tensor, r_matrix

def right_left_local_update(site: int,
                            state: MPS,
                            H: MPO,
                            left_envs: list[np.ndarray],
                            right_envs: list[np.ndarray],
                            sim_params,
                            numiter: int
                            ):
    """
    Run the local update for a site in the BUG algorithm.

    This expands the bond to the left of the site and 
    """
    old_tensor = state[site]
    updated_tensor = update_site(left_envs[site],
                                    right_envs[site],
                                    H[site],
                                    old_tensor,
                                    sim_params.dt,
                                    numiter)
    stacked_tensor = np.stack((old_tensor,updated_tensor),
                              axis=1)
    new_q, _ = _left_qr(stacked_tensor)
    old_q, old_r = _left_qr(old_tensor)
    # Obtain the basis change matrix by contracting the Q tensors
    # We merely need to contract these, as everything to the right is
    # in caonical form
    basis_change_m = np.tensordot(new_q.conj(),
                                  old_q,
                                  axes=([0,2],[0,2]))
    # We need to have the next tensor in canonical form
    new_next_tensor = np.tensordot(state[site-1],
                                   old_r,
                                   axes=(2,0)) # correct order
    # Absorb basis change tensor to right leg of next site
    new_next_tensor = np.tensordot(new_next_tensor,
                                   basis_change_m,
                                   axis=(2,0)) # correct order
    state[site] = updated_tensor
    state[site-1] = new_next_tensor
    new_env = update_right_environment(updated_tensor,
                                      updated_tensor,
                                      H[site],
                                      right_envs[site])
    right_envs[site-1] = new_env

def BUG(state: MPS,
        H: MPO,
        sim_params,
        numiter_lanczos: int=25):
    """
    Performs the Basis-Update and Galerkin Method for an MPS.

    Args:
        H: Hamiltonian represented as an MPO.
        state: The initial state represented as an MPS.
        sim_params: Simulation parameters containing time step 'dt' and SVD threshold.
        numiter_lanczos: Number of Lanczos iterations for each local update.

    Returns:
        None. The state is updated in place.

    """
    num_sites = H.length
    if num_sites != state.length:
        raise ValueError("State and Hamiltonian must have the same number of sites")
    if num_sites < 2:
        raise ValueError("Hamiltonian is too short for a two-site update (BUG).")

    # Compute the right operator blocks.
    right_blocks = initialize_right_environments(state, H)

    # Initialize left operator blocks with an identity-like tensor.
    left_blocks = [None for _ in range(num_sites)]
    left_virtual_dim = state.tensors[0].shape[1]
    mpo_left_dim = H.tensors[0].shape[2]
    left_identity = np.zeros((left_virtual_dim, mpo_left_dim, left_virtual_dim),
                             dtype=right_blocks[0].dtype)
    for i in range(left_virtual_dim):
        for a in range(mpo_left_dim):
            left_identity[i, a, i] = 1
    left_blocks[0] = left_identity

    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        sim_params.dt = 1

    