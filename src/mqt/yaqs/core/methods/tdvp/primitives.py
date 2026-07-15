# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Low-level TDVP primitives.

MPO environment blocks, dense effective operators, and local tensor updates.
Integrator loops live in :mod:`mqt.yaqs.core.methods.tdvp.integrators`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
import numpy as np
import opt_einsum as oe

from ..matrix_exponential import expm_krylov
from .numba import build_dense_heff_site_numba

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ...data_structures.mpo import MPO
    from ...data_structures.mps import MPS


DENSE_THRESHOLD = 128
NUMBA_DENSE_HEFF_MIN_DIM = 16

__all__ = [
    "build_dense_heff_bond",
    "build_dense_heff_site",
    "initialize_right_environments",
    "merge_mpo_tensors",
    "project_bond",
    "project_site",
    "update_bond",
    "update_left_environment",
    "update_right_environment",
    "update_site",
]


# --- MPO environment blocks ---


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
    merged_tensor = np.asarray(
        oe.contract("acei,bdif->abcdef", left_tensor, right_tensor, optimize=True), dtype=np.complex128
    )
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
    return np.asarray(np.tensordot(tensor, bra.conj(), axes=((2, 3), (0, 2))), dtype=np.complex128)


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
    return np.asarray(np.tensordot(ket, tensor, axes=((0, 1), (0, 2))), dtype=np.complex128)


def initialize_right_environments(psi: MPS, op: MPO) -> list[NDArray[np.complex128]]:
    """Compute the right operator blocks (partial contractions) for the given MPS and MPO.

    Starting from the rightmost site, an identity-like tensor is constructed and then
    the network is contracted site-by-site moving to the left to produce a list of right operator blocks.

    Args:
        psi (MPS): The Matrix Product MPS representing the quantum state.
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

    right_blocks: list[NDArray[np.complex128]] = [np.empty((0, 0, 0), dtype=np.complex128) for _ in range(num_sites)]
    right_virtual_dim = psi.tensors[num_sites - 1].shape[2]
    mpo_right_dim = op.tensors[num_sites - 1].shape[3]
    right_identity = np.zeros((right_virtual_dim, mpo_right_dim, right_virtual_dim), dtype=np.complex128)
    for i in range(right_virtual_dim):
        for a in range(mpo_right_dim):
            right_identity[i, a, i] = 1
    right_blocks[num_sites - 1] = right_identity

    for site in reversed(range(num_sites - 1)):
        right_blocks[site] = update_right_environment(
            psi.tensors[site + 1], psi.tensors[site + 1], op.tensors[site + 1], right_blocks[site + 1]
        )
    return right_blocks


# --- Matrix-free projectors ---


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
    return np.asarray(tensor.transpose((0, 2, 1)), dtype=np.complex128)


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
    return np.asarray(np.tensordot(left_env, tensor, axes=((0, 1), (0, 1))), dtype=np.complex128)


# --- Dense effective operator ---


def build_dense_heff_site(
    left_env: NDArray[np.complex128],  # shape (a, l, A)
    right_env: NDArray[np.complex128],  # shape (b, r, B)
    op: NDArray[np.complex128],  # shape (o, p, l, r)
) -> NDArray[np.complex128]:
    r"""Construct the dense effective operator for a single-site Hamiltonian update.

    This function builds the dense matrix representation ``H_eff`` of the linear
    map implemented by :func:`project_site`.  The operator is defined implicitly by

        Y = project_site(left_env, right_env, op, X),

    where ``X`` is a local MPS tensor of shape ``(p, a, b)`` and ``Y`` is the
    resulting tensor of shape ``(o, A, B)``.

    The returned matrix ``H_eff`` satisfies

        vec(Y) = H_eff @ vec(X),

    where ``vec`` denotes NumPy row-major flattening (``reshape(-1)``).

    The dense operator is constructed directly via a single multi-index tensor
    contraction over the MPO tensor and the left and right operator blocks.
    This avoids the explicit application of ``project_site`` to each basis
    vector of the local tensor space and is algebraically equivalent to the
    generic basis-expansion construction.

    Args:
        left_env (NDArray[np.complex128]):
            Left operator block, a 3-index tensor of shape ``(a, l, A)``.
        right_env (NDArray[np.complex128]):
            Right operator block, a 3-index tensor of shape ``(b, r, B)``.
        op (NDArray[np.complex128]):
            Local MPO tensor, a 4-index tensor of shape ``(o, p, l, r)``.

    Returns:
        NDArray[np.complex128]:
            A dense matrix ``H_eff`` of shape ``(o * A * B, p * a * b)`` such that
            applying ``H_eff @ vec(X)`` reproduces the action of
            ``project_site(left_env, right_env, op, X)`` exactly.

    Notes:
        - The index ordering of ``H_eff`` is consistent with NumPy row-major
          flattening of both input and output tensors.
        - This function is intended for small local Hilbert spaces, where
          explicitly materializing the dense effective operator is efficient.
          For larger local dimensions, the matrix-free projector path should be
          preferred.

    """
    left_env = np.asarray(left_env, dtype=np.complex128)
    right_env = np.asarray(right_env, dtype=np.complex128)
    op = np.asarray(op, dtype=np.complex128)
    d = max(left_env.shape[0], left_env.shape[2], right_env.shape[0], right_env.shape[2])
    if d >= NUMBA_DENSE_HEFF_MIN_DIM and numba.get_num_threads() > 1:
        return build_dense_heff_site_numba(left_env, right_env, op)
    # h[o,A,B,p,a,b] = sum_{l,r} op[o,p,l,r] * left_env[a,l,A] * right_env[b,r,B]
    h6 = np.einsum("oplr,alA,brB->oABpab", op, left_env, right_env, optimize=True)
    o_dim, a_dim_out, b_dim_out, p_dim, a_dim_in, b_dim_in = h6.shape
    return np.asarray(
        h6.reshape(o_dim * a_dim_out * b_dim_out, p_dim * a_dim_in * b_dim_in),
        dtype=np.complex128,
    )


def build_dense_heff_bond(
    left_env: NDArray[np.complex128],  # shape (u, a, p)
    right_env: NDArray[np.complex128],  # shape (v, a, w)
) -> NDArray[np.complex128]:
    r"""Construct the dense effective operator for the bond contraction.

    This function builds the dense matrix representation ``H_eff`` of the linear
    map implemented by :func:`project_bond`.  The operator is defined implicitly by

        Y = project_bond(left_env, right_env, C),

    where ``C`` is a bond tensor of shape ``(u, v)`` and ``Y`` is the resulting
    tensor of shape ``(p, w)``.

    The returned matrix ``H_eff`` satisfies

        vec(Y) = H_eff @ vec(C),

    where ``vec`` denotes NumPy row-major flattening (``reshape(-1)``).

    This implementation constructs ``H_eff`` directly using a single tensor
    contraction, avoiding the explicit application of ``project_bond`` to each
    basis vector of the local space.  The result is algebraically equivalent to
    the generic basis-expansion construction but significantly more efficient.

    Args:
        left_env (NDArray[np.complex128]):
            Left operator block, a 3-index tensor of shape ``(u, a, p)``.
        right_env (NDArray[np.complex128]):
            Right operator block, a 3-index tensor of shape ``(v, a, w)``.

    Returns:
        NDArray[np.complex128]:
            A dense matrix ``H_eff`` of shape ``(p * w, u * v)`` such that
            applying ``H_eff @ vec(C)`` reproduces the action of
            ``project_bond(left_env, right_env, C)`` exactly.

    Notes:
        This function is intended for small local bond dimensions, where
        explicitly materializing the dense effective operator is efficient and
        typically faster than repeated tensor contractions inside a Krylov
        iteration.  For large bond dimensions, the matrix-free projector path
        should be preferred.

    """
    left_env = np.asarray(left_env, dtype=np.complex128)
    right_env = np.asarray(right_env, dtype=np.complex128)
    # h[p,w,u,v] = sum_a left_env[u,a,p] * right_env[v,a,w]
    h4 = np.einsum("uap,vaw->pwuv", left_env, right_env, optimize=True)
    p_dim, w_dim, u_dim, v_dim = h4.shape
    return np.asarray(h4.reshape(p_dim * w_dim, u_dim * v_dim), dtype=np.complex128)


def _build_dense_effective_operator(
    projector: Callable[..., NDArray[np.complex128]],
    proj_args: tuple[NDArray[np.complex128], ...],
    tensor_shape: tuple[int, ...],
) -> NDArray[np.complex128]:
    r"""Construct a dense matrix representation of the local effective operator.

    The operator is defined implicitly by the projector:

        Y = projector(*proj_args, X)

    where X and Y are local tensors of shape ``tensor_shape``. This function
    builds the unique dense matrix ``H_eff`` such that:

        vec(Y) = H_eff @ vec(X)

    for every possible local tensor X. This is accomplished by applying the
    projector to each basis vector ``e_j`` of the flattened local space and
    storing the resulting column ``vec(Y_j)``. Since the operator is
    reconstructed directly from the projector, the result is guaranteed to
    match the behavior of ``project_site`` or ``project_bond`` exactly,
    independent of index order or contraction details.

    Args:
        projector:
            A function implementing the local operator action, e.g.
            ``project_site(left_env, right_env, op, ket)`` or
            ``project_bond(left_env, right_env, bond_tensor)``.
        proj_args:
            Extra positional arguments passed to ``projector`` before the tensor X.
            These typically include the left environment, right environment, and
            (for ``project_site``) the local MPO tensor.
        tensor_shape:
            The shape of the local tensor X. The flattened operator ``H_eff`` will
            have dimension ``n_loc = prod(tensor_shape)``.

    Returns:
        NDArray[np.complex128]: A dense matrix of shape ``(n_loc, n_loc)`` such that
        applying ``H_eff @ vec(X)`` reproduces the action of
        ``projector(*proj_args, X)`` to machine precision.

    Notes:
        This method is intended for small local dimensions, where explicitly
        materializing ``H_eff`` is efficient and typically faster than repeated
        tensor contractions inside a Krylov iteration. For large local
        dimensions, the matrix-free path (projector applied directly inside
        Lanczos) is preferred.

    """
    # Fast paths
    if projector is project_site:
        left_env, right_env, op = proj_args
        return build_dense_heff_site(left_env, right_env, op)

    if projector is project_bond:
        left_env, right_env = proj_args
        return build_dense_heff_bond(left_env, right_env)

    # Generic fallback (slow but general)
    n_loc = int(np.prod(tensor_shape))
    h_eff = np.empty((n_loc, n_loc), dtype=np.complex128)
    e = np.zeros(n_loc, dtype=np.complex128)
    for j in range(n_loc):
        e[:] = 0.0
        e[j] = 1.0
        x_tensor = e.reshape(tensor_shape)
        y_tensor = projector(*proj_args, x_tensor)
        h_eff[:, j] = y_tensor.reshape(-1)

    return h_eff


def _evolve_local_tensor_krylov(
    projector: Callable[..., NDArray[np.complex128]],
    tensor: NDArray[np.complex128],
    dt: float,
    proj_args: tuple[NDArray[np.complex128], ...],
    dense_threshold: int = DENSE_THRESHOLD,
    *,
    krylov_tol: float,
) -> NDArray[np.complex128]:
    """Evolve a local tensor with a matrix-free Krylov exponential.

    Args:
        projector: Function implementing the local operator action on a tensor,
            e.g. :func:`project_site` or :func:`project_bond`.
        tensor: Tensor to evolve (arbitrary shape).
        dt: Time step for evolution.
        proj_args: Extra arguments passed to ``projector`` before the tensor.
        dense_threshold: Maximum flattened size for building a dense effective
            operator instead of a matrix-free projector.
        krylov_tol: Tolerance for the adaptive Krylov/Lanczos matrix exponential.

    Returns:
        Evolved tensor with the same shape as ``tensor``.

    """
    tensor_shape = tensor.shape
    tensor_flat = tensor.reshape(-1)
    n_loc = tensor_flat.size

    if n_loc <= dense_threshold:
        # Build dense H_eff once from environments + MPO
        h_eff = _build_dense_effective_operator(projector, proj_args, tensor_shape)

        def apply_effective_operator(x_flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
            """Apply the pre-built dense effective operator.

            Returns:
                Flattened image of ``x_flat`` under ``h_eff``.

            """
            return h_eff @ x_flat

    else:
        # Matrix-free projector path
        def apply_effective_operator(x_flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
            """Apply the matrix-free local projector.

            Returns:
                Flattened image of ``x_flat`` under the local projector.

            """
            x_tensor = x_flat.reshape(tensor_shape)
            y_tensor = projector(*proj_args, x_tensor)
            return y_tensor.reshape(-1)

    evolved_flat = expm_krylov(apply_effective_operator, tensor_flat, dt, tol=krylov_tol)
    return evolved_flat.reshape(tensor_shape)


# --- Local TDVP updates ---


def update_site(
    left_env: NDArray[np.complex128],
    right_env: NDArray[np.complex128],
    op: NDArray[np.complex128],
    ket: NDArray[np.complex128],
    dt: float,
    *,
    krylov_tol: float,
) -> NDArray[np.complex128]:
    """Evolve the local MPS tensor A forward in time using the local Hamiltonian.

    The function flattens tensor A, applies a Lanczos-based approximation of the matrix exponential
    to evolve it by time dt, and then reshapes the result back to the original tensor shape.

    Args:
        left_env: Left operator block.
        right_env: Right operator block.
        op: Local MPO tensor.
        ket: Local MPS tensor.
        dt: Time step for evolution.
        krylov_tol: Tolerance for the adaptive Krylov/Lanczos matrix exponential.

    Returns:
        The updated MPS tensor after evolution.

    """
    proj_args = (left_env, right_env, op)
    return _evolve_local_tensor_krylov(
        projector=project_site,
        tensor=ket,
        dt=dt,
        proj_args=proj_args,
        krylov_tol=krylov_tol,
    )


def update_bond(
    left_env: NDArray[np.complex128],
    right_env: NDArray[np.complex128],
    bond_tensor: NDArray[np.complex128],
    dt: float,
    *,
    krylov_tol: float,
) -> NDArray[np.complex128]:
    """Evolve the bond tensor C using a Lanczos iteration for the "zero-site" bond contraction.

    The bond tensor C is flattened, evolved via the Krylov subspace approximation of the matrix exponential,
    and then reshaped back to its original form.

    Args:
        left_env: Left operator block.
        right_env: Right operator block.
        bond_tensor: Bond tensor.
        dt: Time step for the bond evolution.
        krylov_tol: Tolerance for the adaptive Krylov/Lanczos matrix exponential.

    Returns:
        The updated bond tensor after evolution.

    """
    proj_args = (left_env, right_env)
    return _evolve_local_tensor_krylov(
        projector=project_bond,
        tensor=bond_tensor,
        dt=dt,
        proj_args=proj_args,
        krylov_tol=krylov_tol,
    )
