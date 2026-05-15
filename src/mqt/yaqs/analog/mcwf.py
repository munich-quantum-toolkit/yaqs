# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Monte Carlo Wavefunction (MCWF) evolution for the vector representation.

Converts MPS/MPO inputs into dense operators and evolves the state vector under
an effective non-Hermitian Hamiltonian (with optional quantum jumps). For noiseless
runs, evolution is unitary; for larger systems prefer ``representation='mps'``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.linalg
import scipy.sparse

from ..core.methods.matrix_exponential import expm_arnoldi, expm_krylov

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.networks import MPO, MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import AnalogSimParams

from .utils import _embed_observable_sparse, _embed_operator_sparse

# Maximum Hilbert-space dimension for storing a dense time-step propagator U_step.
MAX_PRECOMPUTE_DIM = 4096


@dataclass
class MCWFContext:
    """Context for MCWF simulation containing pre-computed sparse operators."""

    psi_initial: NDArray[np.complex128]
    heff: scipy.sparse.spmatrix
    jump_ops: list[scipy.sparse.spmatrix]
    embedded_observables: list[scipy.sparse.spmatrix | NDArray[np.complex128] | None]
    sim_params: AnalogSimParams
    is_unitary: bool = False
    step_propagator: scipy.sparse.csr_matrix | None = None
    output_state: NDArray[np.complex128] | None = None


def preprocess_mcwf(
    initial_state: MPS,
    hamiltonian: MPO,
    noise_model: NoiseModel | None,
    sim_params: AnalogSimParams,
) -> MCWFContext:
    """Pre-compute dense operators and initial state for MCWF simulation.

    Args:
        initial_state: The initial MPS state.
        hamiltonian: The Hamiltonian MPO.
        noise_model: The noise model.
        sim_params: Simulation parameters.

    Returns:
        MCWFContext containing dense arrays ready for trajectory simulation.
    """
    # Check dimensions
    num_sites = initial_state.length
    dim = 2**num_sites

    # Limit system size to avoid OOM
    if num_sites > 14:
        msg = (
            f"System size {num_sites} is too large for representation='vector' even with sparse matrices. "
            "Simulation may be very slow or run out of memory. "
            "Consider using representation='mps' for larger systems."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # 1. Initial State to Vector
    psi = initial_state.to_vec()
    psi /= np.linalg.norm(psi)

    # 2. Convert Hamiltonian MPO to sparse matrix
    h_mat = hamiltonian.to_sparse_matrix()

    # 3. Prepare Jump Operators
    jump_ops: list[scipy.sparse.spmatrix] = []
    if noise_model is not None:
        for process in noise_model.processes:
            strength = process["strength"]
            if strength <= 0:
                continue
            op_full = _embed_operator_sparse(process, num_sites)
            jump_ops.append(np.sqrt(strength) * op_full)

    is_unitary = len(jump_ops) == 0

    # 4. Construct Effective Hamiltonian
    heff = h_mat.copy()
    if jump_ops:
        sum_ldag_l = scipy.sparse.csr_matrix((dim, dim), dtype=complex)
        for op in jump_ops:
            op_csr = cast("Any", op)
            sum_ldag_l += op_csr.conj().T @ op_csr
        heff -= 0.5j * sum_ldag_l

    # 5. Precompute time-step propagator U_step = exp(-i * H_eff * dt) when affordable
    step_propagator: scipy.sparse.csr_matrix | None = None
    if dim <= MAX_PRECOMPUTE_DIM:
        u_dense = scipy.linalg.expm(-1j * sim_params.dt * heff.toarray())
        step_propagator = scipy.sparse.csr_matrix(u_dense)

    # 6. Prepare Observables
    embedded_observables: list[scipy.sparse.spmatrix | NDArray[np.complex128] | None] = []
    for obs in sim_params.sorted_observables:
        if obs.gate.name in {"runtime_cost", "max_bond", "total_bond", "entropy", "schmidt_spectrum"}:
            embedded_observables.append(None)
        else:
            op = _embed_observable_sparse(obs, num_sites)
            embedded_observables.append(op)

    return MCWFContext(
        psi_initial=psi,
        heff=heff,
        jump_ops=jump_ops,
        embedded_observables=embedded_observables,
        sim_params=sim_params,
        is_unitary=is_unitary,
        step_propagator=step_propagator,
    )


def _apply_noisy_step(
    psi: NDArray[np.complex128],
    psi_next: NDArray[np.complex128],
    ctx: MCWFContext,
    rng: np.random.Generator,
) -> NDArray[np.complex128]:
    """Apply norm check, optional quantum jump, and renormalization for one MCWF step.

    Returns:
        Normalized state vector after the non-unitary step (with or without jump).
    """
    norm_sq = np.vdot(psi_next, psi_next).real
    p_jump = 1.0 - norm_sq

    if rng.random() >= p_jump:
        return psi_next / np.sqrt(norm_sq)

    param_psi = psi
    normalization_sum = 0.0
    weights: list[float] = []
    for op in ctx.jump_ops:
        op_sparse = cast("Any", op)
        l_psi = op_sparse.dot(param_psi)
        w = np.vdot(l_psi, l_psi).real
        weights.append(w)
        normalization_sum += w

    if normalization_sum < 1e-15:
        return psi_next / np.sqrt(norm_sq)

    weights_arr = np.array(weights, dtype=np.float64)
    weights_arr /= normalization_sum
    k_idx = rng.choice(len(ctx.jump_ops), p=weights_arr)
    jump_op = cast("Any", ctx.jump_ops[k_idx])
    jumped = jump_op.dot(param_psi)
    return jumped / np.linalg.norm(jumped)


def mcwf(args: tuple[int, MCWFContext]) -> NDArray[np.float64]:
    """Run a single Monte Carlo Wavefunction trajectory using pre-computed context.

    Args:
        args: A tuple containing:
            - int: Trajectory identifier.
            - MCWFContext: Pre-computed simulation context.

    Returns:
        An array of expectation values for each observable over time.
    """
    _traj_idx, ctx = args
    sim_params = ctx.sim_params
    dt = sim_params.dt

    psi = ctx.psi_initial.copy()
    rng = np.random.default_rng()

    num_obs = len(sim_params.sorted_observables)
    num_steps = len(sim_params.times)
    results = np.zeros((num_obs, num_steps), dtype=np.float64)

    def measure(current_psi: NDArray[np.complex128], t_idx: int) -> None:
        for i, op_mat in enumerate(ctx.embedded_observables):
            if op_mat is not None:
                if scipy.sparse.issparse(op_mat):
                    op_mat_sparse = cast("Any", op_mat)
                    val = np.vdot(current_psi, op_mat_sparse.dot(current_psi))
                else:
                    op_mat_dense = cast("NDArray[np.complex128]", op_mat)
                    val = np.vdot(current_psi, op_mat_dense @ current_psi)
                results[i, t_idx] = val.real
            else:
                results[i, t_idx] = 0.0

    if sim_params.sample_timesteps:
        measure(psi, 0)

    for t_idx in range(1, num_steps):
        if ctx.step_propagator is not None:
            if ctx.is_unitary:
                psi = ctx.step_propagator @ psi
            else:
                psi_before = psi
                psi_next = ctx.step_propagator @ psi_before
                psi = _apply_noisy_step(psi_before, psi_next, ctx, rng)
        elif ctx.is_unitary:
            psi = expm_krylov(lambda v: ctx.heff @ v, psi, dt)
        else:
            psi_next = expm_arnoldi(lambda v: ctx.heff @ v, psi, dt)
            psi = _apply_noisy_step(psi, psi_next, ctx, rng)

        if sim_params.sample_timesteps or t_idx == num_steps - 1:
            measure(psi, t_idx)

    if sim_params.get_state:
        ctx.output_state = psi

    return results
