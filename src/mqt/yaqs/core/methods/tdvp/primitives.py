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
    """Contract the shared bond of two neighboring MPO tensors.

    Args:
        left_tensor: Left tensor with axes ``(out, in, left, right)``.
        right_tensor: Right tensor with the same axis convention.

    Returns:
        Tensor with paired output and input axes and the two exterior bonds.
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
    """Extend a right operator environment one site to the left.

    Args:
        ket: MPS tensor with shape ``(p, a, b)``.
        bra: MPS tensor with shape ``(o, A, B)``; conjugated during contraction.
        op: MPO tensor with shape ``(o, p, l, r)``.
        right_env: Operator environment with shape ``(b, r, B)``.

    Returns:
        Left operator environment with shape ``(a, l, A)``.
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
    """Extend a left operator environment one site to the right.

    Args:
        ket: MPS tensor with shape ``(p, a, b)``.
        bra: MPS tensor with shape ``(o, A, B)``; conjugated during contraction.
        op: MPO tensor with shape ``(o, p, l, r)``.
        left_env: Operator environment with shape ``(a, l, A)``.

    Returns:
        Right operator environment with shape ``(b, r, B)``.
    """
    tensor = np.tensordot(left_env, bra.conj(), axes=(2, 1))
    tensor = np.tensordot(op, tensor, axes=((0, 2), (2, 1)))
    return np.asarray(np.tensordot(ket, tensor, axes=((0, 1), (0, 2))), dtype=np.complex128)


def initialize_right_environments(psi: MPS, op: MPO) -> list[NDArray[np.complex128]]:
    """Build operator environments to the right of each MPS site.

    Args:
        psi: MPS whose bra and ket are contracted.
        op: MPO with the same length.

    Returns:
        One right environment per site; the final entry is the boundary identity.

    Raises:
        ValueError: If state and operator lengths differ.
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
    """Apply the effective single-site Hamiltonian to a local MPS tensor.

    Args:
        left_env: Left operator environment with shape ``(a, l, A)``.
        right_env: Right operator environment with shape ``(b, r, B)``.
        op: MPO tensor with shape ``(o, p, l, r)``.
        ket: Local tensor with shape ``(p, a, b)``.

    Returns:
        Updated tensor with shape ``(o, A, B)``. See :func:`build_dense_heff_site`
        for the equivalent dense operator.
    """
    tensor = np.tensordot(ket, right_env, axes=1)
    tensor = np.tensordot(op, tensor, axes=((1, 3), (0, 2)))
    tensor = np.tensordot(tensor, left_env, axes=((2, 1), (0, 1)))
    return np.asarray(tensor.transpose((0, 2, 1)), dtype=np.complex128)


def project_bond(
    left_env: NDArray[np.complex128], right_env: NDArray[np.complex128], bond_tensor: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Apply the effective zero-site Hamiltonian to a bond tensor.

    Args:
        left_env: Left operator environment with shape ``(u, a, p)``.
        right_env: Right operator environment with shape ``(v, a, w)``.
        bond_tensor: Bond tensor with shape ``(u, v)``.

    Returns:
        Updated bond tensor with shape ``(p, w)``. See :func:`build_dense_heff_bond`
        for the equivalent dense operator.
    """
    tensor = np.tensordot(bond_tensor, right_env, axes=1)
    return np.asarray(np.tensordot(left_env, tensor, axes=((0, 1), (0, 1))), dtype=np.complex128)


# --- Dense effective operator ---


def build_dense_heff_site(
    left_env: NDArray[np.complex128],  # shape (a, l, A)
    right_env: NDArray[np.complex128],  # shape (b, r, B)
    op: NDArray[np.complex128],  # shape (o, p, l, r)
) -> NDArray[np.complex128]:
    """Build the dense operator for :func:`project_site`.

    The contraction is::

        H_eff[o,A,B,p,a,b] = sum_{l,r} op[o,p,l,r] * left_env[a,l,A] * right_env[b,r,B]

    ``vec(Y) = H_eff @ vec(X)`` uses C-order flattening for input ``(p, a, b)``
    and output ``(o, A, B)``. The dense matrix alone occupies
    ``16 * (o*A*B) * (p*a*b)`` bytes; use :func:`project_site` for matrix-free updates.

    Args:
        left_env: Left operator environment with shape ``(a, l, A)``.
        right_env: Right operator environment with shape ``(b, r, B)``.
        op: MPO tensor with shape ``(o, p, l, r)``.

    Returns:
        Complex128 matrix with shape ``(o*A*B, p*a*b)``.
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
    """Build the dense operator for :func:`project_bond`.

    The contraction is::

        H_eff[p,w,u,v] = sum_a left_env[u,a,p] * right_env[v,a,w]

    ``vec(Y) = H_eff @ vec(C)`` uses C-order flattening for input ``(u, v)``
    and output ``(p, w)``. Dense storage requires ``16 * p*w*u*v`` bytes;
    use :func:`project_bond` for matrix-free updates.

    Args:
        left_env: Left operator environment with shape ``(u, a, p)``.
        right_env: Right operator environment with shape ``(v, a, w)``.

    Returns:
        Complex128 matrix with shape ``(p*w, u*v)``.
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
    """Build a dense operator from a local projector.

    Uses :func:`build_dense_heff_site` or :func:`build_dense_heff_bond` when
    applicable; otherwise applies the projector to each local basis tensor.
    Input and output tensors use C-order flattening.

    Args:
        projector: Linear map called as ``projector(*proj_args, X)``.
        proj_args: Arguments preceding the input tensor.
        tensor_shape: Common shape of input and output tensors.

    Returns:
        Complex128 matrix ``H_eff`` with shape ``(n, n)``, where
        ``n = prod(tensor_shape)``, satisfying ``vec(Y) = H_eff @ vec(X)``.
        Dense storage requires ``16 * n**2`` bytes, excluding workspace.
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
    """Evolve a local MPS tensor with :func:`project_site` and a Krylov exponential.

    Args:
        left_env: Left operator block.
        right_env: Right operator block.
        op: Local MPO tensor.
        ket: Local MPS tensor.
        dt: Time step for evolution.
        krylov_tol: Tolerance for the adaptive Krylov/Lanczos matrix exponential.

    Returns:
        Evolved tensor with the same shape as ``ket``.
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
    """Evolve a bond tensor with :func:`project_bond` and a Krylov exponential.

    Args:
        left_env: Left operator block.
        right_env: Right operator block.
        bond_tensor: Bond tensor.
        dt: Time step for the bond evolution.
        krylov_tol: Tolerance for the adaptive Krylov/Lanczos matrix exponential.

    Returns:
        Evolved tensor with the same shape as ``bond_tensor``.
    """
    proj_args = (left_env, right_env)
    return _evolve_local_tensor_krylov(
        projector=project_bond,
        tensor=bond_tensor,
        dt=dt,
        proj_args=proj_args,
        krylov_tol=krylov_tol,
    )
