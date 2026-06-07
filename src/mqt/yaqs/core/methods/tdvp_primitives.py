# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Low-level TDVP primitives.

MPO environment blocks, dense effective operators, local tensor updates, and
shared sweep helpers (split adapter, fixed-χ bond handling, canonicalization).
Integrator loops live in :mod:`mqt.yaqs.core.methods.tdvp`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import opt_einsum as oe

from ..data_structures.simulation_parameters import AnalogSimParams, StrongSimParams, WeakSimParams
from .decompositions import split_two_site
from .matrix_exponential import expm_krylov

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ..data_structures.mpo import MPO
    from ..data_structures.mps import MPS
    from .decompositions import SvdDistribution, TruncMode


DENSE_THRESHOLD = 128


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

    right_blocks = [np.empty((0, 0, 0), dtype=np.complex128) for _ in range(num_sites)]
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


# --- Dense effective operator ---


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
    return np.tensordot(left_env, tensor, axes=((0, 1), (0, 1)))


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
    """Generic helper to evolve a local tensor with a matrix-free Krylov exponential.

    Args:
        projector: Function implementing the local operator action on a tensor,
            e.g. project_site(left_env, right_env, op, ket) or
                 project_bond(left_env, right_env, bond_tensor).
        tensor: Tensor to evolve (arbitrary shape).
        dt: Time step for evolution.
        proj_args: Extra arguments passed to `projector` before the tensor.
        dense_threshold: int, optional. Maximum size of flattened tensor to use dense operator.
        krylov_tol: float. Tolerance for the adaptive Krylov/Lanczos matrix exponential.

    Returns:
        The evolved tensor with the same shape as `tensor`.
    """
    tensor_shape = tensor.shape
    tensor_flat = tensor.reshape(-1)
    n_loc = tensor_flat.size

    if n_loc <= dense_threshold:
        # Build dense H_eff once from environments + MPO
        h_eff = _build_dense_effective_operator(projector, proj_args, tensor_shape)

        def apply_effective_operator(x_flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return h_eff @ x_flat

    else:
        # Matrix-free projector path
        def apply_effective_operator(x_flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
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


# --- Sweep helpers ---


def _bond_dim_at_or_above_cap(bond_dim: int, max_bond_dim: int | None) -> bool:
    """Return True when a finite ``max_bond_dim`` cap is reached."""
    return max_bond_dim is not None and bond_dim >= max_bond_dim


def _prepare_substep_evolution_dt(
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    step_scale: float,
) -> float:
    """Return the TDVP evolution timestep for the current symmetric substep."""
    if not isinstance(sim_params, (StrongSimParams, WeakSimParams)):
        return float(sim_params.dt) * step_scale
    return step_scale


def _split_two_site_tdvp(
    merged: NDArray[np.complex128],
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    physical_dimensions: list[int],
    svd_distribution: str,
    *,
    dynamic: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Split a merged two-site tensor using TDVP simulation truncation policy.

    Thin adapter around :func:`mqt.yaqs.core.methods.decompositions.split_two_site`.
    When ``dynamic`` is True, no ``max_bond_dim`` cap is applied during truncation
    (bond growth is handled by the dynamic TDVP sweep). Otherwise the cap is
    ``sim_params.max_bond_dim``.

    Args:
        merged: Two-site tensor ``(d_left * d_right, D0, D2)``.
        sim_params: Simulation parameters with ``svd_threshold``, ``trunc_mode``,
            and ``max_bond_dim``.
        physical_dimensions: ``[d_left, d_right]`` physical dimensions.
        svd_distribution: How to absorb singular values (``"left"``, ``"right"``, ``"sqrt"``).
        dynamic: If True, pass ``max_bond_dim=None`` to truncation (dynamic TDVP path).

    Returns:
        Left and right MPS site tensors after split and truncation.
    """
    return split_two_site(
        merged,
        physical_dimensions,
        svd_distribution=cast("SvdDistribution", svd_distribution),
        trunc_mode=cast("TruncMode", sim_params.trunc_mode),
        threshold=sim_params.svd_threshold,
        max_bond_dim=None if dynamic and sim_params.max_bond_dim is None else sim_params.max_bond_dim,
    )


def _sync_bond_dim(state: MPS, bond_index: int, target_dim: int) -> None:
    """Set both tensors on an internal bond to share dimension ``target_dim``."""
    left = state.tensors[bond_index]
    right = state.tensors[bond_index + 1]
    chi_out = int(left.shape[2])
    chi_in = int(right.shape[1])
    if chi_out == target_dim and chi_in == target_dim:
        return
    if chi_out > target_dim:
        state.tensors[bond_index] = left[:, :, :target_dim]
    elif chi_out < target_dim:
        state._ensure_internal_bond_dims((bond_index,), target_dim, max_dim=target_dim)
    right = state.tensors[bond_index + 1]
    chi_in = int(right.shape[1])
    if chi_in > target_dim:
        state.tensors[bond_index + 1] = right[:, :target_dim, :]
    elif chi_in < target_dim:
        state._ensure_internal_bond_dims((bond_index,), target_dim, max_dim=target_dim)


def _contract_bond_target_dim(
    state: MPS,
    bond_index: int,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> int:
    """Return the shared bond dimension to use before a bond transfer contraction."""
    chi_left = int(state.tensors[bond_index].shape[2])
    chi_right = int(state.tensors[bond_index + 1].shape[1])
    chi_target = max(chi_left, chi_right)
    cap = sim_params.max_bond_dim
    if cap is not None:
        chi_target = min(chi_target, cap)
    return max(chi_target, 1)


def _bond_dims_mismatched(state: MPS, bond_index: int) -> bool:
    """Return whether neighboring tensors disagree on a bond dimension."""
    return int(state.tensors[bond_index].shape[2]) != int(state.tensors[bond_index + 1].shape[1])


def _enforce_global_bond_cap(
    state: MPS,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> None:
    """Truncate all internal bonds to ``max_bond_dim`` before a fixed-χ sweep."""
    cap = sim_params.max_bond_dim
    if cap is None:
        return
    for bond in range(state.length - 1):
        chi_out = int(state.tensors[bond].shape[2])
        chi_in = int(state.tensors[bond + 1].shape[1])
        if chi_out > cap or chi_in > cap:
            _sync_bond_dim(state, bond, cap)
    state.normalize()


def _resize_bond(
    bond_tensor: NDArray[np.complex128],
    *,
    lead: int | None = None,
    trail: int | None = None,
) -> NDArray[np.complex128]:
    """Resize leading and/or trailing axes of a bond transfer matrix.

    Returns:
        Resized bond tensor.
    """
    out = bond_tensor
    if lead is not None:
        current = int(out.shape[0])
        if current != lead:
            if current > lead:
                out = out[:lead, :]
            else:
                padded = np.zeros((lead, out.shape[1]), dtype=out.dtype)
                padded[:current, :] = out
                out = padded
    if trail is not None:
        current = int(out.shape[1])
        if current != trail:
            if current > trail:
                out = out[:, :trail]
            else:
                padded = np.zeros((out.shape[0], trail), dtype=out.dtype)
                padded[:, :current] = out
                out = padded
    return out


def _canonicalize_site_ltr(
    tensor: NDArray[np.complex128],
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Right-canonicalize a site tensor and return the bond transfer matrix.

    Returns:
        Right-canonical site tensor and outgoing bond transfer matrix.
    """
    tensor_shape = tensor.shape
    reshaped = tensor.reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
    site_tensor, bond_tensor = np.linalg.qr(reshaped)
    chi_out = int(site_tensor.shape[1])
    site_tensor = site_tensor.reshape((tensor_shape[0], tensor_shape[1], chi_out))
    cap = sim_params.max_bond_dim
    if cap is not None and chi_out > cap:
        site_tensor = site_tensor[:, :, :cap]
        bond_tensor = bond_tensor[:cap, :]
    return site_tensor, bond_tensor


def _canonicalize_site_rtl(
    tensor: NDArray[np.complex128],
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Left-canonicalize a site tensor (RTL sweep) and return the bond transfer matrix.

    Returns:
        Left-canonical site tensor and incoming bond transfer matrix.
    """
    tensor_t = tensor.transpose((0, 2, 1))
    tensor_shape = tensor_t.shape
    reshaped = tensor_t.reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
    site_tensor, bond_tensor = np.linalg.qr(reshaped)
    site_tensor = site_tensor.reshape((tensor_shape[0], tensor_shape[1], site_tensor.shape[1])).transpose((
        0,
        2,
        1,
    ))
    cap = sim_params.max_bond_dim
    if cap is not None and int(site_tensor.shape[1]) > cap:
        site_tensor = site_tensor[:, :cap, :]
        bond_tensor = bond_tensor[:cap, :]
    return site_tensor, bond_tensor.transpose()
