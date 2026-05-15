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
import scipy.linalg
import scipy.sparse
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.networks import MPO, MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import AnalogSimParams

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
    initial_state: MPS,
    hamiltonian: MPO,
    noise_model: NoiseModel | None,
    sim_params: AnalogSimParams,
) -> LindbladContext:
    """Pre-compute operators and optional fixed-step propagator for Lindblad evolution.

    Args:
        initial_state: The initial MPS state (converted to rho = |psi><psi|).
        hamiltonian: The Hamiltonian MPO.
        noise_model: The noise model.
        sim_params: Simulation parameters.

    Returns:
        LindbladContext ready for time evolution.
    """
    num_sites = initial_state.length
    dim = 2**num_sites

    if num_sites > 10:
        msg = (
            f"System size {num_sites} exceeds the recommended limit (10) for representation='density_matrix'. "
            "Density-matrix evolution uses dense-like scaling (2^2N elements). "
            "Simulation may be very slow or run out of memory. "
            "Consider using representation='mps' for larger systems."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # 1. Initial state: pure |psi><psi| from MPS (flattened column-major vec(rho)).
    psi = initial_state.to_vec()
    rho_initial = np.outer(psi, psi.conj()).flatten()

    # 2. Hamiltonian as sparse matrix on the full Hilbert space.
    h_mat = hamiltonian.to_sparse_matrix()

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
        if obs.gate.name in {"runtime_cost", "max_bond", "total_bond", "entropy", "schmidt_spectrum"}:
            embedded_observables.append(None)
        else:
            embedded_observables.append(_embed_observable_sparse(obs, num_sites))

    # 6. Fixed-step propagator exp(L dt) when vec(rho) fits in memory (time-independent generator).
    step_propagator: NDArray[np.complex128] | None = None
    vec_dim = dim * dim
    if vec_dim <= MAX_LIOUVILLIAN_VECTOR_DIM:
        liouvillian = _build_liouvillian_superoperator(dim, h_mat, jump_ops, l_dag_l_sum)
        step_propagator = scipy.linalg.expm(liouvillian * sim_params.dt)

    return LindbladContext(
        rho_initial=rho_initial,
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
    rho = rho_flat.reshape((dim, dim))
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

    return drho.flatten()


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
    rho_t = rho_flat.reshape((dim, dim))
    for i, op_mat in enumerate(ctx.embedded_observables):
        if op_mat is not None:
            op_any = cast("Any", op_mat)
            val = np.trace(op_any @ rho_t)
            obs_results[i, t_idx] = val.real
        else:
            obs_results[i, t_idx] = 0.0


def _evolve_with_propagator(ctx: LindbladContext) -> NDArray[np.float64]:
    """Evolve rho on the fixed grid ``sim_params.times`` via vec(rho) <- exp(L dt) vec(rho).

    Matches the user ``dt`` step used elsewhere in YAQS (not adaptive substeps).
    Noiseless runs use the same map but without separate dissipator terms in L.

    Returns:
        Observable expectation values at each sampled time.
    """
    sim_params = ctx.sim_params
    dim = ctx.dim
    assert ctx.step_propagator is not None

    num_obs = len(sim_params.sorted_observables)
    num_steps = len(sim_params.times)
    obs_results = np.zeros((num_obs, num_steps), dtype=np.float64)

    rho_vec = ctx.rho_initial.copy()
    if sim_params.sample_timesteps:
        _measure_rho(rho_vec, dim, ctx, obs_results, 0)

    for t_idx in range(1, num_steps):
        rho_vec = ctx.step_propagator @ rho_vec
        if sim_params.sample_timesteps or t_idx == num_steps - 1:
            _measure_rho(rho_vec, dim, ctx, obs_results, t_idx)

    return obs_results


def _evolve_with_ode(ctx: LindbladContext) -> NDArray[np.float64]:
    """Adaptive RK45 integration when ``vec(rho)`` is too large to store ``exp(L dt)``.

    Uses the same ``_lindblad_rhs_flat`` as the propagator path; tolerances come from
    ``sim_params.threshold``.

    Returns:
        Observable expectation values at each sampled time.

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
        rtol=sim_params.threshold,
        atol=sim_params.threshold * 1e-2,
    )

    if not result.success:
        msg = (
            f"Lindblad integration failed: {result.message} "
            f"(rtol={sim_params.threshold}, atol={sim_params.threshold * 1e-2}, t_span={t_span})"
        )
        raise RuntimeError(msg)

    # result.y has shape (dim^2, num_time_points).
    num_obs = len(sim_params.sorted_observables)
    obs_results = np.zeros((num_obs, len(result.t)), dtype=np.float64)

    for t_idx, rho_flat_t in enumerate(result.y.T):
        _measure_rho(rho_flat_t, dim, ctx, obs_results, t_idx)

    return obs_results


def lindblad(
    args: tuple[int, MPS, NoiseModel | None, AnalogSimParams, MPO],
) -> NDArray[np.float64]:
    """Run an exact Lindblad master-equation simulation.

    Args:
        args: A tuple containing:
            - int: Trajectory identifier (unused; evolution is deterministic in rho).
            - MPS: The initial state.
            - NoiseModel | None: The noise model.
            - AnalogSimParams: Simulation parameters.
            - MPO: The Hamiltonian.

    Returns:
        An array of expectation values for each observable over time.
    """
    _i, initial_state, noise_model, sim_params, hamiltonian = args
    ctx = preprocess_lindblad(initial_state, hamiltonian, noise_model, sim_params)

    if ctx.step_propagator is not None:
        return _evolve_with_propagator(ctx)
    return _evolve_with_ode(ctx)
