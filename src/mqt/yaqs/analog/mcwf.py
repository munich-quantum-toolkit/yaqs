# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Monte Carlo Wavefunction (MCWF) Solver.

This module provides a Monte Carlo Wavefunction (or Quantum Jump) solver for small quantum systems.
It converts the Matrix Product State (MPS) and Matrix Product Operator (MPO) representations
into dense vectors and matrices, and evolves the wavefunction under an effective non-Hermitian Hamiltonian:
    H_eff = H - 0.5 * i * sum_k (L_k^dag L_k)
stochastically applying quantum jumps.

This solver scales exponentially with system size and is intended primarily for benchmarking
and validating the Tensor Jump Method (TJM) on small systems (N <= 8-10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.networks import MPO, MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import AnalogSimParams


from dataclasses import dataclass

from .utils import _embed_observable, _embed_operator


@dataclass
class MCWFContext:
    """Context for MCWF simulation containing pre-computed dense operators."""

    psi_initial: NDArray[np.complex128]
    u_eff: NDArray[np.complex128]
    jump_ops: list[NDArray[np.complex128]]
    embedded_observables: list[NDArray[np.complex128] | None]
    sim_params: AnalogSimParams


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

    Raises:
        ValueError: If the system size is too large.
    """
    # Check dimensions
    num_sites = initial_state.length
    dim = 2**num_sites
    if num_sites > 10:
        msg = f"System size too large for MCWF solver (N={num_sites}, dim={dim})."
        raise ValueError(msg)

    # 1. Initial State to Vector
    psi = initial_state.to_vec()
    psi /= np.linalg.norm(psi)

    # 2. Convert Hamiltonian MPO to dense matrix
    h_mat = hamiltonian.to_matrix()

    # 3. Prepare Jump Operators
    jump_ops = []
    if noise_model is not None:
        for process in noise_model.processes:
            strength = process["strength"]
            if strength <= 0:
                continue
            op_full = _embed_operator(process, num_sites)
            jump_ops.append(np.sqrt(strength) * op_full)

    # 4. Construct Effective Hamiltonian
    heff = h_mat.copy()
    if jump_ops:
        sum_ldag_l = np.zeros((dim, dim), dtype=complex)
        for op in jump_ops:
            sum_ldag_l += op.conj().T @ op
        heff -= 0.5j * sum_ldag_l

    # Pre-compute time evolution operator
    u_eff = expm(-1j * heff * sim_params.dt)

    # 5. Prepare Observables
    embedded_observables: list[NDArray[np.complex128] | None] = []
    for obs in sim_params.sorted_observables:
        if obs.gate.name in {"runtime_cost", "max_bond", "total_bond", "entropy", "schmidt_spectrum"}:
            embedded_observables.append(None)
        else:
            op = _embed_observable(obs, num_sites)
            embedded_observables.append(op)

    return MCWFContext(
        psi_initial=psi,
        u_eff=u_eff,
        jump_ops=jump_ops,
        embedded_observables=embedded_observables,
        sim_params=sim_params,
    )


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

    # Copy initial state for this trajectory
    psi = ctx.psi_initial.copy()

    rng = np.random.default_rng()

    # Storage for results
    num_obs = len(sim_params.sorted_observables)
    num_steps = len(sim_params.times)
    results = np.zeros((num_obs, num_steps), dtype=np.float64)

    # Helper to measure
    def measure(current_psi: NDArray[np.complex128], t_idx: int) -> None:
        for i, op_mat in enumerate(ctx.embedded_observables):
            if op_mat is not None:
                val = np.vdot(current_psi, op_mat @ current_psi)
                results[i, t_idx] = val.real
            else:
                results[i, t_idx] = 0.0

    # Initial measurement
    if sim_params.sample_timesteps:
        measure(psi, 0)

    # Time evolution loop
    for t_idx in range(1, num_steps):
        # 1. Evolve with H_eff
        psi_next = ctx.u_eff @ psi

        # 2. Norm check
        norm_sq = np.vdot(psi_next, psi_next).real
        p_jump = 1.0 - norm_sq

        # 3. Random number for jump
        r = rng.random()

        if r < p_jump:
            # Jump occurs!
            weights = []
            param_psi = psi  # Use state at start of step

            normalization_sum = 0.0
            for op in ctx.jump_ops:
                l_psi = op @ param_psi
                w = np.vdot(l_psi, l_psi).real
                weights.append(w)
                normalization_sum += w

            if normalization_sum < 1e-15:
                psi = psi_next / np.sqrt(norm_sq)
            else:
                weights = np.array(weights)
                weights /= normalization_sum

                k_idx = rng.choice(len(ctx.jump_ops), p=weights)

                psi = ctx.jump_ops[k_idx] @ param_psi
                psi /= np.linalg.norm(psi)
        else:
            # No jump
            psi = psi_next / np.sqrt(norm_sq)

        # Measurement
        if sim_params.sample_timesteps or t_idx == num_steps - 1:
            measure(psi, t_idx)

    return results
