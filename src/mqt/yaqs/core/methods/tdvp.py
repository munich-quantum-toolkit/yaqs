# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""1TDVP + 2TDVP implementation with helper functions.

This module implements functions for performing time evolution on Matrix Product
States (MPS) using the Time-Dependent Variational Principle (TDVP). It provides
utilities for:

- Updating local MPS tensors and bond tensors using Lanczos-based approximations
  of the matrix exponential.
- Constructing effective local operators through contractions with MPO tensors
  and environment blocks.
- Performing single-site and two-site TDVP integration schemes to evolve the
  MPS in time.

Two-site MPS merge/split with SVD truncation lives in :mod:`mqt.yaqs.core.methods.decompositions`.

These methods are designed for simulating the dynamics of quantum many-body systems and are based on
techniques described in Haegeman et al., Phys. Rev. B 94, 165116 (2016).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import opt_einsum as oe

from ..data_structures.simulation_parameters import StrongSimParams, WeakSimParams
from .decompositions import merge_two_site, split_two_site
from .matrix_exponential import expm_krylov

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ..data_structures.mpo import MPO
    from ..data_structures.mps import MPS
    from ..data_structures.simulation_parameters import AnalogSimParams
    from .decompositions import SvdDistribution, TruncMode


def _bond_dim_at_or_above_cap(bond_dim: int, max_bond_dim: int | None) -> bool:
    """Return True when a finite ``max_bond_dim`` cap is reached."""
    return max_bond_dim is not None and bond_dim >= max_bond_dim


DENSE_THRESHOLD = 128


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
            ``max_bond_dim``, and ``min_bond_dim``.
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
        max_bond_dim=None if dynamic else sim_params.max_bond_dim,
        min_bond_dim=sim_params.min_bond_dim,
    )


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


def _build_dense_effective_hamiltonian(
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
        h_eff = _build_dense_effective_hamiltonian(projector, proj_args, tensor_shape)

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


def _build_tdvp_sweep_plan(
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> list[float]:
    """Build the sweep plan for one evolution step (gate or analog ``dt``).

    Each substep is symmetric (LTR then RTL) at ``step_time / tdvp_sweeps``.

    Returns:
        Ordered step-scale factors for one batched kernel call.
    """
    step_scale = 1.0 / sim_params.tdvp_sweeps
    return [step_scale for _ in range(sim_params.tdvp_sweeps)]


def _run_sweeps(
    evolve_once: Callable[..., None],
    state: MPS,
    hamiltonian: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    /,
    *args: object,
    **kwargs: object,
) -> None:
    """Run ``sim_params.tdvp_sweeps`` TDVP substeps per evolution step.

    Public TDVP entry points validate inputs and delegate here. Private ``_*_sweep``
    kernels perform one symmetric substep and accept ``step_scale``; they must not
    re-enter the public wrappers.

    Substep geometry (``_build_tdvp_sweep_plan``): ``tdvp_sweeps`` symmetric substeps
    (LTR then RTL each) at ``step_time / tdvp_sweeps`` for analog ``dt`` and digital gates.
    """
    sweep_plan = _build_tdvp_sweep_plan(sim_params)
    evolve_once(
        state,
        hamiltonian,
        sim_params,
        *args,
        sweep_plan=sweep_plan,
        **kwargs,
    )


def _single_site_tdvp_sweep(
    state: MPS,
    hamiltonian: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    *,
    step_scale: float = 1.0,
    sweep_plan: list[float] | None = None,
) -> None:
    if sweep_plan is not None:
        for plan_step_scale in sweep_plan:
            _single_site_tdvp_sweep(
                state,
                hamiltonian,
                sim_params,
                step_scale=plan_step_scale,
            )
        return

    num_sites = hamiltonian.length

    right_blocks = initialize_right_environments(state, hamiltonian)

    left_blocks = [np.empty((0, 0, 0), dtype=np.complex128) for _ in range(num_sites)]
    left_virtual_dim = state.tensors[0].shape[1]
    mpo_left_dim = hamiltonian.tensors[0].shape[2]
    left_identity = np.zeros((left_virtual_dim, mpo_left_dim, left_virtual_dim), dtype=right_blocks[0].dtype)
    for i in range(left_virtual_dim):
        for a in range(mpo_left_dim):
            left_identity[i, a, i] = 1
    left_blocks[0] = left_identity

    substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, step_scale)

    # Left-to-right sweep: Update sites 0 to L-2.
    for i in range(num_sites - 1):
        state.tensors[i] = update_site(
            left_blocks[i],
            right_blocks[i],
            hamiltonian.tensors[i],
            state.tensors[i],
            0.5 * substep_evolution_dt,
            krylov_tol=sim_params.krylov_tol,
        )
        tensor_shape = state.tensors[i].shape
        reshaped_tensor = state.tensors[i].reshape((tensor_shape[0] * tensor_shape[1], tensor_shape[2]))
        site_tensor, bond_tensor = np.linalg.qr(reshaped_tensor)
        state.tensors[i] = site_tensor.reshape((tensor_shape[0], tensor_shape[1], site_tensor.shape[1]))
        left_blocks[i + 1] = update_left_environment(
            state.tensors[i], state.tensors[i], hamiltonian.tensors[i], left_blocks[i]
        )
        bond_tensor = update_bond(
            left_blocks[i + 1],
            right_blocks[i],
            bond_tensor,
            -0.5 * substep_evolution_dt,
            krylov_tol=sim_params.krylov_tol,
        )
        state.tensors[i + 1] = oe.contract(state.tensors[i + 1], (0, 3, 2), bond_tensor, (1, 3), (0, 1, 2))

    last = num_sites - 1
    state.tensors[last] = update_site(
        left_blocks[last],
        right_blocks[last],
        hamiltonian.tensors[last],
        state.tensors[last],
        substep_evolution_dt,
        krylov_tol=sim_params.krylov_tol,
    )

    substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, step_scale)

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
            left_blocks[i],
            right_blocks[i - 1],
            bond_tensor,
            -0.5 * substep_evolution_dt,
            krylov_tol=sim_params.krylov_tol,
        )
        state.tensors[i - 1] = oe.contract(state.tensors[i - 1], (0, 1, 3), bond_tensor, (3, 2), (0, 1, 2))
        state.tensors[i - 1] = update_site(
            left_blocks[i - 1],
            right_blocks[i - 1],
            hamiltonian.tensors[i - 1],
            state.tensors[i - 1],
            0.5 * substep_evolution_dt,
            krylov_tol=sim_params.krylov_tol,
        )


def single_site_tdvp(
    state: MPS,
    hamiltonian: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> None:
    """Perform symmetric single-site Time-Dependent Variational Principle (TDVP) integration.

    The function evolves the MPS state in time by sequentially updating each site tensor using
    local Hamiltonian evolution and bond updates. The process includes a left-to-right sweep followed by
    an optional right-to-left sweep for full integration.

    Args:
        state: The initial state represented as an MPS.
        hamiltonian: Hamiltonian represented as an MPO.
        sim_params: Simulation parameters with ``dt``, ``svd_threshold``, ``krylov_tol``,
            ``trunc_mode``, and bond limits.

    Raises:
        ValueError: If Hamiltonian is invalid length.
    """
    num_sites = hamiltonian.length
    if num_sites != state.length:
        msg = "The state and Hamiltonian must have the same number of sites."
        raise ValueError(msg)

    _run_sweeps(_single_site_tdvp_sweep, state, hamiltonian, sim_params)


def _two_site_tdvp_sweep(
    state: MPS,
    hamiltonian: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    *,
    dynamic: bool = False,
    step_scale: float = 1.0,
    sweep_plan: list[float] | None = None,
) -> None:
    num_sites = hamiltonian.length
    plan = sweep_plan if sweep_plan is not None else [step_scale]

    right_blocks = initialize_right_environments(state, hamiltonian)
    left_blocks = [np.empty((0, 0, 0), dtype=np.complex128) for _ in range(num_sites)]
    left_virtual_dim = state.tensors[0].shape[1]
    mpo_left_dim = hamiltonian.tensors[0].shape[2]
    left_identity = np.zeros((left_virtual_dim, mpo_left_dim, left_virtual_dim), dtype=right_blocks[0].dtype)
    for i in range(left_virtual_dim):
        for a in range(mpo_left_dim):
            left_identity[i, a, i] = 1
    left_blocks[0] = left_identity

    for plan_step_scale in plan:
        substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, plan_step_scale)

        for i in range(num_sites - 2):
            merged_tensor = merge_two_site(state.tensors[i], state.tensors[i + 1])
            merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i], hamiltonian.tensors[i + 1])
            merged_tensor = update_site(
                left_blocks[i],
                right_blocks[i + 1],
                merged_mpo,
                merged_tensor,
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            state.tensors[i], state.tensors[i + 1] = _split_two_site_tdvp(
                merged_tensor,
                sim_params,
                [state.physical_dimensions[i], state.physical_dimensions[i + 1]],
                "right",
                dynamic=dynamic,
            )
            left_blocks[i + 1] = update_left_environment(
                state.tensors[i], state.tensors[i], hamiltonian.tensors[i], left_blocks[i]
            )
            state.tensors[i + 1] = update_site(
                left_blocks[i + 1],
                right_blocks[i + 1],
                hamiltonian.tensors[i + 1],
                state.tensors[i + 1],
                -0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )

        i = num_sites - 2
        merged_tensor = merge_two_site(state.tensors[i], state.tensors[i + 1])
        merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i], hamiltonian.tensors[i + 1])
        merged_tensor = update_site(
            left_blocks[i],
            right_blocks[i + 1],
            merged_mpo,
            merged_tensor,
            substep_evolution_dt,
            krylov_tol=sim_params.krylov_tol,
        )
        state.tensors[i], state.tensors[i + 1] = _split_two_site_tdvp(
            merged_tensor,
            sim_params,
            [state.physical_dimensions[i], state.physical_dimensions[i + 1]],
            "left",
            dynamic=dynamic,
        )

        right_blocks[i] = update_right_environment(
            state.tensors[i + 1], state.tensors[i + 1], hamiltonian.tensors[i + 1], right_blocks[i + 1]
        )

        substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, plan_step_scale)

        if num_sites - 2 == 0:
            i = num_sites - 2
            merged_tensor = merge_two_site(state.tensors[i], state.tensors[i + 1])
            merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i], hamiltonian.tensors[i + 1])
            merged_tensor = update_site(
                left_blocks[i],
                right_blocks[i + 1],
                merged_mpo,
                merged_tensor,
                substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            state.tensors[i], state.tensors[i + 1] = _split_two_site_tdvp(
                merged_tensor,
                sim_params,
                [state.physical_dimensions[i], state.physical_dimensions[i + 1]],
                "right",
                dynamic=dynamic,
            )
            continue

        for i in reversed(range(num_sites - 2)):
            state.tensors[i + 1] = update_site(
                left_blocks[i + 1],
                right_blocks[i + 1],
                hamiltonian.tensors[i + 1],
                state.tensors[i + 1],
                -0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            merged_tensor = merge_two_site(state.tensors[i], state.tensors[i + 1])
            merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i], hamiltonian.tensors[i + 1])
            merged_tensor = update_site(
                left_blocks[i],
                right_blocks[i + 1],
                merged_mpo,
                merged_tensor,
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            state.tensors[i], state.tensors[i + 1] = _split_two_site_tdvp(
                merged_tensor,
                sim_params,
                [state.physical_dimensions[i], state.physical_dimensions[i + 1]],
                "left",
                dynamic=dynamic,
            )
            right_blocks[i] = update_right_environment(
                state.tensors[i + 1], state.tensors[i + 1], hamiltonian.tensors[i + 1], right_blocks[i + 1]
            )


def two_site_tdvp(
    state: MPS,
    hamiltonian: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
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
        state: The initial state represented as an MPS.
        hamiltonian: Hamiltonian represented as an MPO.
        sim_params: Simulation parameters with ``dt``, ``svd_threshold``, ``krylov_tol``,
            ``trunc_mode``, ``max_bond_dim``, and related truncation settings.
        dynamic: If True, bond growth is handled by dynamic TDVP without an intermediate cap.

    Raises:
        ValueError: If Hamiltonian is invalid length.
    """
    num_sites = hamiltonian.length
    if num_sites != state.length:
        msg = "MPS and Hamiltonian must have the same number of sites"
        raise ValueError(msg)
    if num_sites < 2:
        msg = "Hamiltonian is too short for a two-site update (2TDVP)."
        raise ValueError(msg)

    _run_sweeps(_two_site_tdvp_sweep, state, hamiltonian, sim_params, dynamic=dynamic)


def _local_dynamic_tdvp_sweep(
    state: MPS,
    hamiltonian: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    *,
    step_scale: float = 1.0,
    sweep_plan: list[float] | None = None,
) -> None:
    if sweep_plan is not None:
        for plan_step_scale in sweep_plan:
            _local_dynamic_tdvp_sweep(
                state,
                hamiltonian,
                sim_params,
                step_scale=plan_step_scale,
            )
        return

    num_sites = hamiltonian.length

    right_blocks = initialize_right_environments(state, hamiltonian)
    left_blocks = [np.empty((0, 0, 0), dtype=np.complex128) for _ in range(num_sites)]
    chi0 = state.tensors[0].shape[1]
    mpo_dim = hamiltonian.tensors[0].shape[2]
    eye = np.zeros((chi0, mpo_dim, chi0), dtype=np.complex128)
    for i in range(chi0):
        eye[i, :, i] = 1
    left_blocks[0] = eye

    substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, step_scale)

    # ----- LEFT-TO-RIGHT DYNAMIC SWEEP -----
    lock_final_site = False
    for i in range(num_sites):
        bond_dim = state.tensors[i].shape[2]
        if _bond_dim_at_or_above_cap(bond_dim, sim_params.max_bond_dim) or lock_final_site:
            state.tensors[i] = update_site(
                left_blocks[i],
                right_blocks[i],
                hamiltonian.tensors[i],
                state.tensors[i],
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
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
                    left_blocks[i + 1],
                    right_blocks[i],
                    bond_tensor,
                    -0.5 * substep_evolution_dt,
                    krylov_tol=sim_params.krylov_tol,
                )
                state.tensors[i + 1] = oe.contract(state.tensors[i + 1], (0, 3, 2), bond_tensor, (1, 3), (0, 1, 2))
            if i == num_sites - 2:
                lock_final_site = True
        elif i == num_sites - 1:
            continue
        elif i == num_sites - 2:
            merged_tensor = merge_two_site(state.tensors[i], state.tensors[i + 1])
            merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i], hamiltonian.tensors[i + 1])
            merged_tensor = update_site(
                left_blocks[i],
                right_blocks[i + 1],
                merged_mpo,
                merged_tensor,
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )

            state.tensors[i], state.tensors[i + 1] = _split_two_site_tdvp(
                merged_tensor,
                sim_params,
                [state.physical_dimensions[i], state.physical_dimensions[i + 1]],
                "right",
                dynamic=True,
            )
            right_blocks[i] = update_right_environment(
                state.tensors[i + 1], state.tensors[i + 1], hamiltonian.tensors[i + 1], right_blocks[i + 1]
            )
            left_blocks[i + 1] = update_left_environment(
                state.tensors[i], state.tensors[i], hamiltonian.tensors[i], left_blocks[i]
            )

        else:
            merged_tensor = merge_two_site(state.tensors[i], state.tensors[i + 1])
            merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i], hamiltonian.tensors[i + 1])
            merged_tensor = update_site(
                left_blocks[i],
                right_blocks[i + 1],
                merged_mpo,
                merged_tensor,
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            state.tensors[i], state.tensors[i + 1] = _split_two_site_tdvp(
                merged_tensor,
                sim_params,
                [state.physical_dimensions[i], state.physical_dimensions[i + 1]],
                "right",
                dynamic=True,
            )
            left_blocks[i + 1] = update_left_environment(
                state.tensors[i], state.tensors[i], hamiltonian.tensors[i], left_blocks[i]
            )
            state.tensors[i + 1] = update_site(
                left_blocks[i + 1],
                right_blocks[i + 1],
                hamiltonian.tensors[i + 1],
                state.tensors[i + 1],
                -0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
    substep_evolution_dt = _prepare_substep_evolution_dt(sim_params, step_scale)

    # ----- RIGHT-TO-LEFT DYNAMIC SWEEP -----
    lock_final_site = False
    for i in reversed(range(num_sites)):
        bond_dim = state.tensors[i].shape[1]
        if _bond_dim_at_or_above_cap(bond_dim, sim_params.max_bond_dim) or lock_final_site:
            state.tensors[i] = update_site(
                left_blocks[i],
                right_blocks[i],
                hamiltonian.tensors[i],
                state.tensors[i],
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
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
                    left_blocks[i],
                    right_blocks[i - 1],
                    bond_tensor,
                    -0.5 * substep_evolution_dt,
                    krylov_tol=sim_params.krylov_tol,
                )
                state.tensors[i - 1] = oe.contract(state.tensors[i - 1], (0, 1, 3), bond_tensor, (3, 2), (0, 1, 2))

                if i == 1:
                    lock_final_site = True
        elif i == 0:
            continue
        else:
            merged_tensor = merge_two_site(state.tensors[i - 1], state.tensors[i])
            merged_mpo = merge_mpo_tensors(hamiltonian.tensors[i - 1], hamiltonian.tensors[i])
            merged_tensor = update_site(
                left_blocks[i - 1],
                right_blocks[i],
                merged_mpo,
                merged_tensor,
                0.5 * substep_evolution_dt,
                krylov_tol=sim_params.krylov_tol,
            )
            state.tensors[i - 1], state.tensors[i] = _split_two_site_tdvp(
                merged_tensor,
                sim_params,
                [state.physical_dimensions[i - 1], state.physical_dimensions[i]],
                "left",
                dynamic=True,
            )
            right_blocks[i - 1] = update_right_environment(
                state.tensors[i], state.tensors[i], hamiltonian.tensors[i], right_blocks[i]
            )
            if i != 1:
                state.tensors[i - 1] = update_site(
                    left_blocks[i - 1],
                    right_blocks[i - 1],
                    hamiltonian.tensors[i - 1],
                    state.tensors[i - 1],
                    -0.5 * substep_evolution_dt,
                    krylov_tol=sim_params.krylov_tol,
                )


def local_dynamic_tdvp(
    state: MPS,
    hamiltonian: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
) -> None:
    """Perform a dynamic TDVP sweep: at each bond.

    Local dynamic TDVP sweep. If the current bond dimension
    exceeds max_bond_dim, apply a local single-site TDVP step; otherwise,
    apply a two-site TDVP step.

    Args:
        state: MPS state to evolve.
        hamiltonian: MPO Hamiltonian.
        sim_params: Simulation parameters including ``dt``, ``svd_threshold``, ``krylov_tol``,
            and ``max_bond_dim``.

    Raises:
        ValueError: If Hamiltonian is invalid length.
    """
    num_sites = hamiltonian.length
    if num_sites != state.length:
        msg = "MPS and Hamiltonian must have the same length"
        raise ValueError(msg)

    if num_sites == 1:
        single_site_tdvp(state, hamiltonian, sim_params)
        return

    _run_sweeps(_local_dynamic_tdvp_sweep, state, hamiltonian, sim_params)


def _global_dynamic_tdvp_sweep(
    state: MPS,
    hamiltonian: MPO,
    sim_params: AnalogSimParams | StrongSimParams | WeakSimParams,
    *,
    step_scale: float = 1.0,
    sweep_plan: list[float] | None = None,
) -> None:
    """Run one global dynamic TDVP substep, routing to the appropriate sweep kernel."""
    current_max_bond_dim = state.get_max_bond()
    if sim_params.max_bond_dim is None or current_max_bond_dim < sim_params.max_bond_dim:
        _two_site_tdvp_sweep(
            state,
            hamiltonian,
            sim_params,
            dynamic=True,
            step_scale=step_scale,
            sweep_plan=sweep_plan,
        )
    else:
        _single_site_tdvp_sweep(
            state,
            hamiltonian,
            sim_params,
            step_scale=step_scale,
            sweep_plan=sweep_plan,
        )


def global_dynamic_tdvp(
    state: MPS, hamiltonian: MPO, sim_params: AnalogSimParams | StrongSimParams | WeakSimParams
) -> None:
    """Perform a dynamic Time-Dependent Variational Principle (TDVP) evolution of the system state.

    This function evolves the state by choosing between a two-site TDVP (2TDVP) and a single-site TDVP (1TDVP)
    based on the current maximum bond dimension of the MPS. The decision is made by comparing the state's bond
    dimension (obtained via `state.get_max_bond()`) to the maximum allowed bond dimension specified in
    `sim_params`.

    Args:
        state: The MPS representing the current state of the system.
        hamiltonian: The MPO representing the Hamiltonian of the system.
        sim_params: Simulation parameters including ``max_bond_dim``, ``svd_threshold``,
            and ``krylov_tol``.
    """
    if state.length == 1:
        single_site_tdvp(state, hamiltonian, sim_params)
        return

    _run_sweeps(_global_dynamic_tdvp_sweep, state, hamiltonian, sim_params)
