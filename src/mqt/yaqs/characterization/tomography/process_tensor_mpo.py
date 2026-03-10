"""Process-Tensor MPO utilities for efficient multi-time estimates."""

import numpy as np
from numpy.typing import NDArray
from mqt.yaqs.core.data_structures.networks import MPO


def rank1_upsilon_mpo_term(
    rho_final: NDArray[np.complex128],
    dual_ops: list[NDArray[np.complex128]],
    weight: float = 1.0,
) -> MPO:
    """Build a rank-1 MPO term representing a single sample's contribution to Upsilon.

    The comb Choi operator lives on sites [F, P1, P2, ..., Pk].
    Site 0 (F) is the final state density matrix (2x2).
    Sites 1..k are the dual operators (4x4) corresponding to interventions.

    Args:
        rho_final: (2x2) density matrix of the final state.
        dual_ops: list of k (4x4) dual matrices for the time steps.
        weight: Scalar importance weight for this sample.

    Returns:
        MPO with bond dimension 1 representing `weight * (rho_final ⊗ D1 ⊗ D2 ⊗ ... ⊗ Dk)`.
    """
    k = len(dual_ops)
    length = k + 1
    phys_dims = [2] + [4] * k

    # Build local tensors with dummy bond dims of 1
    # YAQS MPO tensor order: (phys_out, phys_in, chi_left, chi_right)
    tensors = []
    t0 = (weight * rho_final).reshape(2, 2, 1, 1)
    tensors.append(t0)

    for D in dual_ops:
        tD = D.reshape(4, 4, 1, 1)
        tensors.append(tD)

    m = MPO()
    # `custom` sets tensors, length, physical_dimension, checks bounds
    m.custom(tensors, transpose=False)
    m.physical_dimension = phys_dims

    return m


def upsilon_mpo_to_dense(mpo: MPO) -> NDArray[np.complex128]:
    """Convert a process-tensor MPO back to its dense matrix representation.

    Args:
        mpo: The MPO representing the comb Choi operator.

    Returns:
        Dense matrix of shape (2*4^k, 2*4^k)
    """
    return mpo.to_matrix()
