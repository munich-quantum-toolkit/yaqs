# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Exact Lindblad Solver.

This module provides an exact Lindblad master equation solver for small quantum systems.
It converts the Matrix Product State (MPS) and Matrix Product Operator (MPO) representations
into dense matrices and solves the Lindblad master equation:
    drho/dt = -i[H, rho] + sum_k (L_k rho L_k^dag - 0.5 * {L_k^dag L_k, rho})

It is suitable for small systems (N <= 12).
For larger systems or when stochastic trajectories are preferred, consider
using the MCWF solver or the Tensor Jump Method (TJM).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.networks import MPO, MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import AnalogSimParams


from .utils import _embed_observable_sparse, _embed_operator_sparse


def lindblad(
    args: tuple[int, MPS, NoiseModel | None, AnalogSimParams, MPO],
) -> NDArray[np.float64]:
    """Run an exact Lindblad master equation simulation.

    Args:
        args: A tuple containing:
            - int: Trajectory identifier (unused).
            - MPS: The initial state.
            - NoiseModel | None: The noise model.
            - AnalogSimParams: Simulation parameters.
            - MPO: The Hamiltonian.

    Returns:
        An array of expectation values for each observable over time.

    Raises:
        RuntimeError: If the integration fails.
    """
    _i, initial_state, noise_model, sim_params, hamiltonian = args

    # Check dimensions
    num_sites = initial_state.length
    dim = 2**num_sites

    # Limit system size to avoid OOM
    if num_sites > 9:
        msg = (
            f"System size {num_sites} exceeds the recommended limit (14) for the exact Lindblad solver. "
            "Lindblad uses dense-like scaling for the density matrix (2^2N elements). "
            "Simulation may be very slow or run out of memory. "
            "Consider using the TJM solver for larger systems."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # 1. Initial State to Rho
    psi = initial_state.to_vec()
    rho_initial = np.outer(psi, psi.conj())

    # 2. Convert Hamiltonian MPO to sparse matrix
    h_mat = hamiltonian.to_sparse_matrix()

    # 3. Prepare Jump Operators
    jump_ops = []
    if noise_model is not None:
        for process in noise_model.processes:
            strength = process["strength"]
            if strength <= 0:
                continue

            # Convert local operator to full-space sparse operator
            op_full = _embed_operator_sparse(process, num_sites)

            # Scale by sqrt(gamma)
            jump_ops.append(np.sqrt(strength) * op_full)

    # 4. Define Lindblad Superoperator for ODE solver
    # drho/dt = -i[H, rho] + sum( L rho L+ - 0.5 {L+ L, rho} )

    # Preconstruct L_dag + L sum for efficiency (dissipator anti-commutator part)
    l_dag_l_sum = scipy.sparse.csr_matrix((dim, dim), dtype=complex)
    for l_op in jump_ops:
        l_dag_l_sum += l_op.conj().T @ l_op

    # 5. Integrate
    # Ensure t_span covers t_eval (sim_params.times) even if floats drift slightly
    t_end = max(sim_params.elapsed_time, sim_params.times[-1] + 1e-9)
    t_span = (0, t_end)
    t_eval = sim_params.times

    def lindblad_rhs(_t: float, rho_flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        rho = rho_flat.reshape((dim, dim))

        # Unitary part: -i [H, rho]
        # Commutator [H, rho] = H rho - rho H
        drho = -1j * (h_mat @ rho - rho @ h_mat)

        # Dissipative part
        # 1. Accumulate sum( L rho L+ )
        for l_op in jump_ops:
            l_dag = l_op.conj().T
            drho += l_op @ rho @ l_dag

        # 2. Subtract anti-commutator term once: - 0.5 { sum(L+ L), rho }
        # Note: l_dag_l_sum = sum( L+ L ) was precomputed outside
        ac_term = 0.5 * (l_dag_l_sum @ rho + rho @ l_dag_l_sum)
        drho -= ac_term

        return drho.flatten()

    result = solve_ivp(
        lindblad_rhs,
        t_span,
        rho_initial.flatten(),
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

    # 6. Compute Observables
    # result.y has shape (dim^2, num_time_points)
    num_obs = len(sim_params.sorted_observables)
    obs_results = np.zeros((num_obs, len(result.t)), dtype=np.float64)

    # Pre-embed observable operators to full space
    embedded_observables: list[scipy.sparse.spmatrix | NDArray[np.complex128] | None] = []
    for obs in sim_params.sorted_observables:
        if obs.gate.name in {"runtime_cost", "max_bond", "total_bond", "entropy", "schmidt_spectrum"}:
            # These are structural/diagnostic metrics not straightforwardly defined on rho
            embedded_observables.append(None)
        else:
            op = _embed_observable_sparse(obs, num_sites)
            embedded_observables.append(op)

    for t_idx, rho_flat_t in enumerate(result.y.T):
        rho_t = rho_flat_t.reshape((dim, dim))

        for i, _obs in enumerate(sim_params.sorted_observables):
            op_mat = embedded_observables[i]
            if op_mat is not None:
                # Expectation <O> = Tr(O rho)
                val = np.trace(op_mat @ rho_t)
                obs_results[i, t_idx] = val.real
            else:
                # Handle special observables if needed, or leave as 0/NaN
                obs_results[i, t_idx] = 0.0

    return obs_results
