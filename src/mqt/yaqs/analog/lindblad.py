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

This solver scales exponentially with system size and is intended primarily for benchmarking
and validating the Tensor Jump Method (TJM) on small systems (N <= 8-10).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.networks import MPO, MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import AnalogSimParams, Observable


def _kron_all(ops: list[NDArray[np.complex128]]) -> NDArray[np.complex128]:
    """Compute the Kronecker product of a list of matrices.

    Args:
        ops: A list of matrices (2D numpy arrays) to compute the product of.

    Returns:
        The resulting matrix from the Kronecker product of all input matrices.
    """
    res = ops[0]
    for op in ops[1:]:
        res = np.kron(res, op)
    return res.astype(np.complex128)


def _embed_operator(process: dict, num_sites: int) -> NDArray[np.complex128]:
    """Embeds a local noise process operator into the full Hilbert space (size 2^N).

    Args:
        process: A dictionary defining the noise process, including its matrix and target sites.
        num_sites: The total number of sites in the system.

    Returns:
        The operator embedded into the full Hilbert space.

    Raises:
        NotImplementedError: If the process definition involves unsupported structures.
    """
    sites = process["sites"]

    # If it's a "matrix" based process (1-site or adjacent 2-site)
    if "matrix" in process:
        op_local = process["matrix"]

        # If 1-site
        if len(sites) == 1:
            site = sites[0]
            # Construct I x ... x op x ... x I
            ops = [np.eye(2, dtype=complex) for _ in range(num_sites)]
            ops[site] = op_local
            return _kron_all(ops)

        # If 2-site (adjacent)
        if len(sites) == 2:
            s1, s2 = sorted(sites)
            assert s2 == s1 + 1, "Matrix-based 2-site op must be adjacent for simple kron construction"

            # Construct I x ... x op_2site x ... x I
            # Note: The 4x4 matrix `op_local` corresponds to sites (s1, s2)

            # Divide system into: [0...s1-1], [s1, s2], [s2+1...N-1]
            left_id = np.eye(2**s1, dtype=complex) if s1 > 0 else np.eye(1, dtype=complex)
            right_id = (
                np.eye(2 ** (num_sites - 1 - s2), dtype=complex) if s2 < num_sites - 1 else np.eye(1, dtype=complex)
            )

            # Full op = L x Op x R
            res = np.kron(left_id, op_local)
            return np.kron(res, right_id)

    # If it's a factors-based process (non-adjacent or crosstalk)
    if "factors" in process:
        # factors is tuple (op1, op2) acting on (site1, site2)
        op1, op2 = process["factors"]
        s1, s2 = sites  # Keep original order if crucial, but factors usually map 1-1 to sites

        ops = [np.eye(2, dtype=complex) for _ in range(num_sites)]
        ops[s1] = op1
        ops[s2] = op2
        return _kron_all(ops)

    msg = f"Cannot embed operator for process: {process}"
    raise NotImplementedError(msg)


def _embed_observable(obs: Observable, num_sites: int) -> NDArray[np.complex128]:
    """Embeds an observable into the full Hilbert space.

    Args:
        obs: The observable to embed.
        num_sites: The total number of sites in the system.

    Returns:
        The observable operator embedded into the full Hilbert space.

    Raises:
        NotImplementedError: If the observable targets non-adjacent sites or >2 sites.
    """
    sites = obs.sites
    if isinstance(sites, int):
        sites = [sites]

    # If 1-site
    if len(sites) == 1:
        site = sites[0]
        ops = [np.eye(2, dtype=complex) for _ in range(num_sites)]
        ops[site] = obs.gate.matrix
        return _kron_all(ops)

    # If 2-site
    if len(sites) == 2:
        s1, s2 = sorted(sites)
        if s2 == s1 + 1:
            # Adjacent 2-site observable (local gate matrix is 4x4)
            op_local = obs.gate.matrix
            left_id = np.eye(2**s1, dtype=complex) if s1 > 0 else np.eye(1, dtype=complex)
            right_id = (
                np.eye(2 ** (num_sites - 1 - s2), dtype=complex) if s2 < num_sites - 1 else np.eye(1, dtype=complex)
            )

            res = np.kron(left_id, op_local)
            return np.kron(res, right_id)
        msg = "Non-adjacent 2-site observables not yet supported in exact solver."
        raise NotImplementedError(msg)

    msg = f"Unsupported observable site count: {len(sites)}"
    raise NotImplementedError(msg)


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
        ValueError: If the system size is too large for the exact Lindblad solver.
    """
    _i, initial_state, noise_model, sim_params, hamiltonian = args

    # 1. Initial State to Rho
    psi = initial_state.to_vec()
    rho_initial = np.outer(psi, psi.conj())
    dim = rho_initial.shape[0]
    num_sites = initial_state.length

    # 2. Convert Hamiltonian MPO to dense matrix
    h_mat = hamiltonian.to_matrix()

    # 3. Prepare Jump Operators
    jump_ops = []
    if noise_model is not None:
        for process in noise_model.processes:
            strength = process["strength"]
            if strength <= 0:
                continue

            # Convert local operator to full-space operator
            op_full = _embed_operator(process, num_sites)

            # Scale by sqrt(gamma)
            jump_ops.append(np.sqrt(strength) * op_full)

    # 4. Define Lindblad Superoperator for ODE solver
    # drho/dt = -i[H, rho] + sum( L rho L+ - 0.5 {L+ L, rho} )

    # Preconstruct L_dag + L sum for efficiency (dissipator anti-commutator part)
    l_dag_l_sum = np.zeros((dim, dim), dtype=complex)
    for l_op in jump_ops:
        l_dag_l_sum += l_op.conj().T @ l_op

    # 5. Integrate
    # Ensure t_span covers t_eval (sim_params.times) even if floats drift slightly
    t_end = max(sim_params.elapsed_time, sim_params.times[-1] + 1e-9)
    t_span = (0, t_end)
    t_eval = sim_params.times

    # Check dimensions
    if num_sites > 10:
        msg = f"System size too large for exact Lindblad solver (N={num_sites}, dim={dim})."
        raise ValueError(msg)

    def lindblad_rhs(rho_flat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        rho = rho_flat.reshape((dim, dim))

        # Unitary part: -i [H, rho]
        # Commutator [H, rho] = H rho - rho H
        drho = -1j * (h_mat @ rho - rho @ h_mat)

        # Dissipative part
        for l_op in jump_ops:
            l_dag = l_op.conj().T
            # L rho L+
            term1 = l_op @ rho @ l_dag
            # - 0.5 {L+ L, rho}
            term2 = 0.5 * (l_dag_l_sum @ rho + rho @ l_dag_l_sum)
            drho += term1 - term2

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

    # 6. Compute Observables
    # result.y has shape (dim^2, num_time_points)
    num_obs = len(sim_params.sorted_observables)
    obs_results = np.zeros((num_obs, len(result.t)), dtype=np.float64)

    # Pre-embed observable operators to full space
    embedded_observables: list[NDArray[np.complex128] | None] = []
    for obs in sim_params.sorted_observables:
        if obs.gate.name in {"runtime_cost", "max_bond", "total_bond", "entropy", "schmidt_spectrum"}:
            # These are structural/diagnostic metrics not straightforwardly defined on rho
            embedded_observables.append(None)
        else:
            op = _embed_observable(obs, num_sites)
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
