# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact Lindblad master-equation evolution for ``representation='density_matrix'``.

This module integrates the ensemble-averaged master equation (deterministic in rho):

    drho/dt = -i[H, rho] + sum_k ( L_k rho L_k^dag - 0.5 {L_k^dag L_k, rho} )

MPS/MPO specify the initial pure state and Hamiltonian; the state is carried as a
dense ``dim x dim`` matrix with ``dim = 2^N``. With ``noise_model=None`` (or zero
strengths), the dissipator vanishes and evolution is unitary on rho.

Because ``H`` and the jump operators are time-independent, the generator is fixed.
For small systems we precompute ``exp(L dt)`` where ``L`` is the Liouvillian
superoperator (``d vec(rho)/dt = L @ vec(rho)``). Larger systems fall back to
adaptive RK45 on the same RHS. Suitable for ``N <= 10``; use ``representation='mps'``
or ``'vector'`` for larger lattices.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.sparse
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.mpo import MPO
    from ..core.data_structures.mps import MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import AnalogSimParams

from ..core import linalg
from .utils import _embed_observable_sparse, _embed_operator_sparse

# Maximum length of vec(rho) = dim**2 for storing dense exp(L * dt).
# vec_dim=4096 corresponds to N=6 qubits (propagator ~256 MB); N=7+ uses ODE fallback.
MAX_LIOUVILLIAN_VECTOR_DIM = 4096


@dataclass
class LindbladContext:
    """Pre-computed operators for one density-matrix evolution run."""

    rho_initial: NDArray[np.complex128]
    dim: int
    h_mat: scipy.sparse.spmatrix
    jump_ops: list[scipy.sparse.spmatrix]
    # sum_k L_k^dag L_k, used in the anti-commutator term -0.5 {sum_k L_k^dag L_k, rho}.
    l_dag_l_sum: scipy.sparse.csr_matrix
    embedded_observables: list[scipy.sparse.spmatrix | NDArray[np.complex128] | None]
    sim_params: AnalogSimParams
    is_unitary: bool = False
    # exp(L * dt) with L the Liouvillian superoperator, if vec(rho) is small enough.
    step_propagator: NDArray[np.complex128] | None = None


def preprocess_lindblad(
    initial_state: MPS | None,
    hamiltonian: MPO | None,
    noise_model: NoiseModel | None,
    sim_params: AnalogSimParams,
    *,
    rho_initial: NDArray[np.complex128] | None = None,
    num_sites: int | None = None,
    h_sparse: scipy.sparse.spmatrix | None = None,
) -> LindbladContext:
    """Pre-compute operators and optional fixed-step propagator for Lindblad evolution.

    Args:
        initial_state: The initial MPS state (converted to ``rho = |psi><psi|`` when no
            ``rho_initial`` is passed), or ``None`` when ``rho_initial`` is supplied.
        hamiltonian: The Hamiltonian MPO (ignored if ``h_sparse`` is set).
        noise_model: The noise model.
        sim_params: Simulation parameters.
        rho_initial: Optional density matrix (square) or flattened vector for vec(rho).
        num_sites: Number of lattice sites when ``initial_state`` is ``None``.
        h_sparse: Pre-materialized sparse Hamiltonian (skips ``hamiltonian.to_sparse_matrix()``).

    Returns:
        LindbladContext ready for time evolution.

    Raises:
        ValueError: If neither ``initial_state`` nor ``rho_initial`` is provided, or if
            ``num_sites`` is missing when only ``rho_initial`` is given.
    """
    if initial_state is not None:
        num_sites = initial_state.length
    elif rho_initial is not None:
        if num_sites is None:
            msg = "num_sites is required when preprocess_lindblad is called with rho_initial only."
            raise ValueError(msg)
    else:
        msg = "preprocess_lindblad requires initial_state or rho_initial."
        raise ValueError(msg)

    dim = 2**num_sites

    if num_sites > 10:
        msg = (
            f"System size {num_sites} exceeds the recommended limit (10) for representation='density_matrix'. "
            "Density-matrix evolution uses dense-like scaling (2^2N elements). "
            "Simulation may be very slow or run out of memory. "
            "Consider using representation='mps' for larger systems."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # 1. Initial state as flattened vec(rho) (column-major order of the matrix).
    if rho_initial is not None:
        rho_arr = np.asarray(rho_initial, dtype=np.complex128)
        if rho_arr.ndim == 2:
            if rho_arr.shape != (dim, dim):
                msg = f"rho_initial shape {rho_arr.shape} does not match ({dim}, {dim})."
                raise ValueError(msg)
            rho_mat = rho_arr
        else:
            if rho_arr.size != dim * dim:
                msg = f"rho_initial size {rho_arr.size} does not match Hilbert dimension {dim * dim}."
                raise ValueError(msg)
            rho_mat = rho_arr.reshape(dim, dim, order="F")
        trace = np.trace(rho_mat)
        if np.isclose(trace, 0.0):
            msg = "rho_initial must have non-zero trace."
            raise ValueError(msg)
        if not np.isclose(trace, 1.0):
            rho_mat /= trace
        rho_vec = np.asarray(rho_mat.flatten(order="F"), dtype=np.complex128)
    else:
        assert initial_state is not None
        psi = initial_state.to_vec()
        rho_vec = np.asarray(np.outer(psi, psi.conj()).flatten(order="F"), dtype=np.complex128)

    # 2. Hamiltonian as sparse matrix on the full Hilbert space.
    if h_sparse is not None:
        h_mat = scipy.sparse.csr_matrix(h_sparse)
    elif hamiltonian is not None:
        h_mat = hamiltonian.to_sparse_matrix()
    else:
        msg = "preprocess_lindblad requires hamiltonian or h_sparse."
        raise ValueError(msg)

    # 3. Jump operators L_k = sqrt(gamma) * op on the full space.
    jump_ops: list[scipy.sparse.spmatrix] = []
    if noise_model is not None:
        for process in noise_model.processes:
            strength = process["strength"]
            if strength <= 0:
                continue
            op_full = _embed_operator_sparse(process, num_sites)
            jump_ops.append(np.sqrt(strength) * op_full)

    is_unitary = len(jump_ops) == 0

    # 4. Precompute sum_k L_k^dag L_k for the dissipator anti-commutator (skipped if noiseless).
    l_dag_l_sum = scipy.sparse.csr_matrix((dim, dim), dtype=np.complex128)
    if jump_ops:
        for op in jump_ops:
            op_csr = cast("Any", op)
            l_dag_l_sum += op_csr.conj().T @ op_csr

    # 5. Embed observables; MPS-only diagnostics are not traced on rho.
    embedded_observables: list[scipy.sparse.spmatrix | NDArray[np.complex128] | None] = []
    for obs in sim_params.sorted_observables:
        if obs.gate.name in {"entropy", "schmidt_spectrum"}:
            embedded_observables.append(None)
        else:
            embedded_observables.append(_embed_observable_sparse(obs, num_sites))

    # 6. Fixed-step propagator exp(L dt) when vec(rho) fits in memory (time-independent generator).
    step_propagator: NDArray[np.complex128] | None = None
    vec_dim = dim * dim
    if vec_dim <= MAX_LIOUVILLIAN_VECTOR_DIM:
        liouvillian = _build_liouvillian_superoperator(dim, h_mat, jump_ops, l_dag_l_sum)
        step_propagator = linalg.expm(liouvillian * sim_params.dt)

    return LindbladContext(
        rho_initial=rho_vec,
        dim=dim,
        h_mat=h_mat,
        jump_ops=jump_ops,
        l_dag_l_sum=l_dag_l_sum,
        embedded_observables=embedded_observables,
        sim_params=sim_params,
        is_unitary=is_unitary,
        step_propagator=step_propagator,
    )


def _lindblad_rhs_flat(
    rho_flat: NDArray[np.complex128],
    dim: int,
    h_mat: scipy.sparse.spmatrix,
    jump_ops: list[scipy.sparse.spmatrix],
    l_dag_l_sum: scipy.sparse.csr_matrix,
) -> NDArray[np.complex128]:
    """Evaluate drho/dt flattened, matching the Lindblad master equation.

    Used both by the ODE integrator and when building the Liouvillian column-wise.

    Returns:
        Flattened time derivative of the density matrix.
    """
    rho = rho_flat.reshape((dim, dim), order="F")
    h_any = cast("Any", h_mat)

    # Unitary part: -i [H, rho] = -i (H rho - rho H).
    drho = -1j * (h_any @ rho - rho @ h_any)

    # Dissipative part: sum_k L_k rho L_k^dag - 0.5 {sum_k L_k^dag L_k, rho}.
    for l_op in jump_ops:
        l_op_any = cast("Any", l_op)
        l_dag = l_op_any.conj().T
        drho += l_op_any @ rho @ l_dag
    l_sum_any = cast("Any", l_dag_l_sum)
    ac_term = 0.5 * (l_sum_any @ rho + rho @ l_sum_any)
    drho -= ac_term

    return drho.flatten(order="F")


def _build_liouvillian_superoperator(
    dim: int,
    h_mat: scipy.sparse.spmatrix,
    jump_ops: list[scipy.sparse.spmatrix],
    l_dag_l_sum: scipy.sparse.csr_matrix,
) -> NDArray[np.complex128]:
    """Build dense Liouvillian L such that d vec(rho)/dt = L @ vec(rho).

    Columns are obtained by applying ``_lindblad_rhs_flat`` to basis vectors, so the
    superoperator matches the ODE RHS exactly without hand-derived Kronecker formulas.

    Returns:
        Dense Liouvillian superoperator of shape ``(dim**2, dim**2)``.
    """
    vec_dim = dim * dim

    def rhs(rho_flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return _lindblad_rhs_flat(rho_flat, dim, h_mat, jump_ops, l_dag_l_sum)

    liouvillian = np.zeros((vec_dim, vec_dim), dtype=np.complex128)
    for k in range(vec_dim):
        basis = np.zeros(vec_dim, dtype=np.complex128)
        basis[k] = 1.0
        liouvillian[:, k] = rhs(basis)
    return liouvillian


def _measure_rho(
    rho_flat: NDArray[np.complex128],
    dim: int,
    ctx: LindbladContext,
    obs_results: NDArray[np.float64],
    t_idx: int,
) -> None:
    """Record <O> = Tr(O rho) for each observable at time index ``t_idx``."""
    rho_t = rho_flat.reshape((dim, dim), order="F")
    for i, op_mat in enumerate(ctx.embedded_observables):
        if op_mat is not None:
            op_any = cast("Any", op_mat)
            val = np.trace(op_any @ rho_t)
            obs_results[i, t_idx] = val.real
        else:
            obs_results[i, t_idx] = 0.0


def _rho_vec_at_elapsed_time(ctx: LindbladContext) -> NDArray[np.complex128]:
    """Evolve ``vec(rho)`` to ``sim_params.elapsed_time`` (not ``times[-1]``).

    Args:
        ctx: Preprocessed Lindblad context.

    Returns:
        Flattened density matrix at ``sim_params.elapsed_time``.

    Raises:
        RuntimeError: If the ODE integration fails on the large-system fallback path.
    """
    sim_params = ctx.sim_params
    dim = ctx.dim
    target_t = sim_params.elapsed_time
    if target_t <= 0.0:
        return ctx.rho_initial.copy()

    dt = sim_params.dt
    n_full = int(target_t // dt)
    remainder = target_t - n_full * dt

    if ctx.step_propagator is not None:
        rho_vec = ctx.rho_initial.copy()
        for _ in range(n_full):
            rho_vec = ctx.step_propagator @ rho_vec
        if remainder > 1e-12:
            liouvillian = _build_liouvillian_superoperator(dim, ctx.h_mat, ctx.jump_ops, ctx.l_dag_l_sum)
            rho_vec = linalg.expm(liouvillian * remainder) @ rho_vec
        return rho_vec

    def lindblad_rhs(_t: float, rho_flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return _lindblad_rhs_flat(rho_flat, dim, ctx.h_mat, ctx.jump_ops, ctx.l_dag_l_sum)

    result = solve_ivp(
        lindblad_rhs,
        (0.0, target_t),
        ctx.rho_initial,
        t_eval=[target_t],
        method="RK45",
        rtol=sim_params.svd_threshold,
        atol=sim_params.svd_threshold * 1e-2,
    )
    if not result.success:
        msg = (
            f"Lindblad integration to elapsed_time={target_t} failed: {result.message} "
            f"(rtol={sim_params.svd_threshold}, atol={sim_params.svd_threshold * 1e-2})"
        )
        raise RuntimeError(msg)
    return result.y.T[0]


def _evolve_with_propagator(ctx: LindbladContext) -> NDArray[np.float64]:
    """Evolve rho on the fixed grid ``sim_params.times`` via vec(rho) <- exp(L dt) vec(rho).

    Matches the user ``dt`` step used elsewhere in YAQS (not adaptive substeps).
    Noiseless runs use the same map but without separate dissipator terms in L.

    Args:
        ctx: Preprocessed Lindblad context.

    Returns:
        Observable expectation values at each sampled time on ``sim_params.times``.
    """
    sim_params = ctx.sim_params
    dim = ctx.dim
    assert ctx.step_propagator is not None

    num_obs = len(sim_params.sorted_observables)
    num_steps = len(sim_params.times)
    num_cols = num_steps if sim_params.sample_timesteps else 1
    obs_results = np.zeros((num_obs, num_cols), dtype=np.float64)

    rho_vec = ctx.rho_initial.copy()
    if sim_params.sample_timesteps:
        _measure_rho(rho_vec, dim, ctx, obs_results, 0)

    for t_idx in range(1, num_steps):
        rho_vec = ctx.step_propagator @ rho_vec
        if sim_params.sample_timesteps:
            _measure_rho(rho_vec, dim, ctx, obs_results, t_idx)

    if not sim_params.sample_timesteps:
        _measure_rho(rho_vec, dim, ctx, obs_results, 0)

    return obs_results


def _evolve_with_ode(ctx: LindbladContext) -> NDArray[np.float64]:
    """Adaptive RK45 integration when ``vec(rho)`` is too large to store ``exp(L dt)``.

    Uses the same ``_lindblad_rhs_flat`` as the propagator path; tolerances come from
    ``sim_params.svd_threshold``.

    Args:
        ctx: Preprocessed Lindblad context.

    Returns:
        Observable expectation values at each sampled time on ``sim_params.times``.

    Raises:
        RuntimeError: If the ODE integration fails.
    """
    sim_params = ctx.sim_params
    dim = ctx.dim

    # Ensure t_span covers t_eval (sim_params.times) even if floats drift slightly.
    t_end = max(sim_params.elapsed_time, sim_params.times[-1] + 1e-9)
    t_span = (0.0, t_end)
    t_eval = sim_params.times

    def lindblad_rhs(_t: float, rho_flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        # Time argument unused: generator is autonomous (time-independent H and jumps).
        return _lindblad_rhs_flat(rho_flat, dim, ctx.h_mat, ctx.jump_ops, ctx.l_dag_l_sum)

    result = solve_ivp(
        lindblad_rhs,
        t_span,
        ctx.rho_initial,
        t_eval=t_eval,
        method="RK45",
        rtol=sim_params.svd_threshold,
        atol=sim_params.svd_threshold * 1e-2,
    )

    if not result.success:
        msg = (
            f"Lindblad integration failed: {result.message} "
            f"(rtol={sim_params.svd_threshold}, atol={sim_params.svd_threshold * 1e-2}, t_span={t_span})"
        )
        raise RuntimeError(msg)

    # result.y has shape (dim^2, num_time_points).
    num_obs = len(sim_params.sorted_observables)
    if sim_params.sample_timesteps:
        obs_results = np.zeros((num_obs, len(result.t)), dtype=np.float64)
        for t_idx, rho_flat_t in enumerate(result.y.T):
            _measure_rho(rho_flat_t, dim, ctx, obs_results, t_idx)
    else:
        obs_results = np.zeros((num_obs, 1), dtype=np.float64)
        _measure_rho(result.y.T[-1], dim, ctx, obs_results, 0)

    return obs_results


def lindblad_evolve(ctx: LindbladContext) -> tuple[NDArray[np.float64], None, NDArray[np.complex128] | None]:
    """Evolve a preprocessed Lindblad context and return observable trajectories.

    Args:
        ctx: Preprocessed Lindblad context.

    Returns:
        tuple[NDArray[np.float64], None, NDArray[np.complex128] | None]: Observable data,
        no diagnostics, and optional final density matrix at ``sim_params.elapsed_time``
        when ``get_state`` is set.
    """
    obs = _evolve_with_propagator(ctx) if ctx.step_propagator is not None else _evolve_with_ode(ctx)
    if ctx.sim_params.get_state:
        rho_vec = _rho_vec_at_elapsed_time(ctx)
        rho_mat = rho_vec.reshape((ctx.dim, ctx.dim), order="F")
        return obs, None, rho_mat
    return obs, None, None


def lindblad(
    args: tuple[int, MPS, NoiseModel | None, AnalogSimParams, MPO],
) -> tuple[NDArray[np.float64], None, NDArray[np.complex128] | None]:
    """Run an exact Lindblad master-equation simulation.

    Args:
        args: A tuple containing:
            - int: Trajectory identifier (unused; evolution is deterministic in rho).
            - MPS: The initial state.
            - NoiseModel | None: The noise model.
            - AnalogSimParams: Simulation parameters.
            - MPO: The Hamiltonian.

    Returns:
        tuple[NDArray[np.float64], None, NDArray[np.complex128] | None]: Observable data,
        no diagnostics, and optional final density matrix when ``get_state`` is set.
    """
    _i, initial_state, noise_model, sim_params, hamiltonian = args
    ctx = preprocess_lindblad(initial_state, hamiltonian, noise_model, sim_params)
    return lindblad_evolve(ctx)
