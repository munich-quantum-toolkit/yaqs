
from __future__ import annotations
from dataclasses import dataclass
from copy import copy

import numpy as np
from numpy.linalg import qr, svd

from .tdvp import (update_site,
                    update_left_environment,
                    update_right_environment)
from ..data_structures.simulation_parameters import (WeakSimParams,
                                                     StrongSimParams)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple, List
    from ..data_structures.networks import MPO, MPS

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

def _prepare_canonical_site_tensors(state: MPS,
                                    H: MPO
                                    ) -> Tuple[List[np.ndarray],List[np.ndarray]]:
    """
    We need to get the original tensor when every site is the caonical form.

    Assumes the MPS is in the left-canonical form.

    Args:
        state: The MPS.
        H: The MPO.

    Returns:
        List[np.ndarray]: The list of the canonical site tensors.
        List[np.ndarray]: The list of the left environments.

    """
    # This will merely do a shallow copy of the MPS.
    canon_tensors = copy(state.tensors)
    left_end_dimension = state.tensors[0].shape[1]
    left_envs = [np.eye(left_end_dimension).reshape(left_end_dimension,1,left_end_dimension)]
    for i, local_tensor in enumerate(canon_tensors[1:],
                                     start=1):
        left_tensor = canon_tensors[i-1]
        left_q, left_r = _right_qr(left_tensor)
        # Legs of right_r: (new, old_right)
        local_tensor = np.tensordot(left_r,
                                    local_tensor,
                                    axes=(1,1))
        # Leg order of local_tensor: (left, phys, right)
        local_tensor = local_tensor.transpose(1,0,2)
        # Correct leg order: (phys, left, right) and orth center
        canon_tensors[i] = local_tensor
        new_env = update_left_environment(left_q,
                                          left_q,
                                          H.tensors[i-1],
                                          left_envs[i-1])
        left_envs.append(new_env)
    return canon_tensors, left_envs

def _choose_stack_tensor(site: int,
                         canon_center_tensors: List[np.ndarray],
                         state: MPS
                         ) -> np.ndarray:
    """
    Return the non-update tensor that should be used for the stacking step.

    If the site is the last one and thus the leaf site, we need to choose the
    MPS tensor, when the MPS was in left-canonical form. Otherwise, we choose
    the MPS tensor, when the local site was the orthognality center.

    Args:
        site: The site to be updated.
        canon_center_tensors: The canonical site tensors.
        state: The MPS.
    
    Returns:
        np.ndarray: The tensor to be stacked.

    """
    if site == state.num_sites() - 1:
        # This is the only leaf case.
        old_stack_tensor = state.tensors[site]
    else:
        old_stack_tensor = canon_center_tensors[site]
    return old_stack_tensor

def _find_new_q(old_stack_tensor: np.ndarray,
                updated_tensor: np.ndarray
                ) -> np.ndarray:
    """
    Finds the new Q tensor after the update with enlarged left virtual leg.

    Args:
        old_stack_tensor: The tensor to be stacked with the updated tensor.
        updated_tensor: The tensor after the update.

    Returns:
        np.ndarray: The new Q tensor with MPS leg order (phys, left, right).
    
    """
    stacked_tensor = np.concatenate((old_stack_tensor, updated_tensor),
                                    axis=1)
    new_q, _ = _left_qr(stacked_tensor)
    return new_q

def _build_basis_change_tensor(old_q: np.ndarray,
                               new_q: np.ndarray,
                               old_m: np.ndarray
                               ) -> np.ndarray:
    """
    Build a new basis change tensor M.

    Args:
        old_q: The old tensor of the site, when the MPS was in left-canonical
            form. The leg order is (phys, left, right).
        new_q: The extended local base tensor after the update. Same leg order
            as an MPS tensor. The leg order is (phys, left, right).
        old_m: The basis change matrix of the site to the right. The leg order
            is (old,new).
    
    Returns:
        np.ndarray: The basis change tensor M. The leg order is (old,new).
   
    """
    new_m = np.tensordot(old_q,
                         old_m,
                         axes=(2,0))
    new_m = np.tensordot(new_m,
                         new_q.conj(),
                         axes=([0,2],[0,2]))
    return new_m

def _local_update(state: MPS,
                    H: MPO,
                    left_blocks: List[np.ndarray],
                    right_block: np.ndarray,
                    canon_center_tensors: List[np.ndarray],
                    site: int,
                    right_m_block: np.ndarray,
                    sim_params,
                    numiter_lanczos: int
                    ) -> Tuple[np.ndarray,np.ndarray]:
    """
    Updates a single site of the MPS.

    Args:
        state: The MPS.
        H: The MPO.
        left_blocks: The left environments.
        right_block: The right environment.
        canon_center_tensors: The canonical site tensors.
        site: The site to be updated.
        right_m_block: The basis update matrix of the site to the right.
        sim_params: Simulation parameters.
        numiter_lanczos: Number of Lanczos iterations.
    
    Returns:
        np.ndarray: The basis update matrix of this site.
        np.ndarray: The right environment of this site.
    """
    old_tensor = canon_center_tensors[site]
    updated_tensor = update_site(left_blocks[site],
                                 right_block,
                                 H.tensors[site],
                                 old_tensor,
                                 sim_params.dt,
                                 numiter_lanczos)
    old_stack_tensor = _choose_stack_tensor(site,
                                            canon_center_tensors,
                                            state)
    new_q = _find_new_q(old_stack_tensor,
                        updated_tensor)
    old_q = state.tensors[site]
    basis_change_m = _build_basis_change_tensor(old_q,
                                                new_q,
                                                right_m_block)
    state.tensors[site] = new_q
    canon_center_tensors[site-1] = np.tensordot(canon_center_tensors[site-1],
                                                basis_change_m,
                                                axes=(2,0))
    new_right_block = update_right_environment(new_q,
                                               new_q,
                                               H.tensors[site],
                                               right_block
                                               )
    return basis_change_m, new_right_block

def _right_svd(ps_tensor: np.ndarray
              ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Performs the singular value decomposition of an MPS tensor.

    Args:
        ps_tensor: The tensor to be decomposed.

    Returns:
        np.ndarray: The U tensor with the left virtual leg and the physical
            leg (phys,left,new).
        np.ndarray: The S vector with the singular values.
        np.ndarray: The V matrix with the right virtual leg (new,right).

    """
    old_shape = ps_tensor.shape
    svd_shape = (old_shape[0]*old_shape[1],old_shape[2])
    ps_tensor = ps_tensor.reshape(svd_shape)
    u_matrix, s_vector, v_matrix = svd(ps_tensor,
                                       full_matrices=False)
    new_shape = (old_shape[0],old_shape[1],-1)
    u_tensor = u_matrix.reshape(new_shape)
    return u_tensor, s_vector, v_matrix

def _truncated_right_svd(ps_tensor: np.ndarray,
                         threshold: float,
                         max_dim: int
                         ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    Performs the truncated singular value decomposition of an MPS tensor.

    Args:
        ps_tensor: The tensor to be decomposed.
        threshold: The truncation threshold.
        max_dim: The maximum number of singular values to keep.
    
    Returns:
        np.ndarray: The U tensor with the left virtual leg and the physical
            leg (phys,left,new).
        np.ndarray: The S vector with the singular values.
        np.ndarray: The V matrix with the right virtual leg (new,right).
    
    """
    u_matrix, s_vector, v_matrix = _right_svd(ps_tensor)
    cut_sum = 0
    thresh_sq = threshold**2
    cut_index = 1
    for i, s_val in enumerate(np.flip(s_vector)):
        cut_sum += s_val**2
        if cut_sum >= thresh_sq:
            cut_index = len(s_vector) - i
            break
    cut_index = min(cut_index, max_dim)
    u_tensor = u_matrix[:,:,:cut_index]
    s_vector = s_vector[:cut_index]
    v_matrix = v_matrix[:cut_index,:]
    return u_tensor, s_vector, v_matrix

@dataclass
class TruncationParams:
    threshold: float = 1e-16
    max_dim: int = 1000

def truncate(state: MPS,
             svd_params):
    """
    Truncates the MPS in place.

    Args:
        state: The MPS.
        svd_params: The truncation parameters.
    
    """
    if state.length == 1:
        return
    for i, tensor in enumerate(state.tensors[:-1]):
        _, _, v_matrix = _truncated_right_svd(tensor,
                                              svd_params.threshold,
                                              svd_params.max_dim)
        # Pull v into left leg of next tensor.
        new_next = np.tensordot(v_matrix,
                                state.tensors[i+1],
                                axes=(1,1))
        new_next = new_next.transpose(1,0,2)
        state.tensors[i+1] = new_next
        # Pull v^dag into current tensor.
        state.tensors[i] = np.tensordot(state.tensors[i],
                                        v_matrix.conj(), # No transpose, put correct axes instead
                                        axes=(2,1))

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

    if isinstance(sim_params, (WeakSimParams, StrongSimParams)):
        sim_params.dt = 1

    canon_center_tensors, left_envs = _prepare_canonical_site_tensors(state,
                                                                      H)
    right_end_dimension = state.tensors[-1].shape[2]
    right_block = np.eye(right_end_dimension).reshape(right_end_dimension,1,right_end_dimension)
    right_m_block = np.eye(right_end_dimension)
    # Sweep from right to left.
    for site in range(num_sites-1,0,-1):
        right_m_block, right_block = _local_update(state,
                                                   H,
                                                   left_envs,
                                                   right_block,
                                                   canon_center_tensors,
                                                   site,
                                                   right_m_block,
                                                   sim_params,
                                                   numiter_lanczos)
    # Update the first site.
    updated_tensor = update_site(left_envs[0],
                                    right_block,
                                    H.tensors[0],
                                    canon_center_tensors[0],
                                    sim_params.dt,
                                    numiter_lanczos)
    state.tensors[0] = updated_tensor
    # Truncation
    truncate(state, sim_params)
