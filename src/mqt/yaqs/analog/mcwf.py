# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Monte Carlo wavefunction (MCWF) evolution for ``representation='vector'``.

This module implements a stochastic unraveling of the Lindblad master equation for
small systems. MPS/MPO inputs are contracted to a dense state vector and sparse
operators; the trajectory is evolved in Hilbert space (not in a tensor network).

Effective non-Hermitian dynamics between jumps:

    H_eff = H - (i/2) sum_k L_k^dag L_k

with a state vector updated by ``exp(-i H_eff dt)`` and occasional quantum jumps
``L_k``. When there is no noise, ``H_eff = H`` and evolution is unitary.

Hamiltonians are time-independent in YAQS today, so for fixed ``dt`` the map
``U_step = exp(-i H_eff dt)`` can be precomputed once (see ``MAX_PRECOMPUTE_DIM``).
Per-step Arnoldi is only used when the Hilbert-space dimension is too large to
store ``U_step``. For larger lattices use ``representation='mps'`` instead.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.sparse

from ..core import linalg
from ..core.data_structures.state_utils import resolve_physical_dimensions
from ..core.methods.matrix_exponential import expm_arnoldi, expm_krylov
from ..core.random_utils import make_trajectory_rng

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..core.data_structures.mpo import MPO
    from ..core.data_structures.mps import MPS
    from ..core.data_structures.noise_model import NoiseModel
    from ..core.data_structures.simulation_parameters import AnalogSimParams

from .utils import _embed_observable_sparse, _embed_operator_sparse

# Maximum Hilbert-space dimension dim = 2^N for storing dense U_step (dim x dim).
# N=12 -> dim=4096 (~256 MB per propagator); N=14 falls back to per-step Krylov.
MAX_PRECOMPUTE_DIM = 4096


@dataclass
class MCWFContext:
    """Pre-computed data for one MCWF trajectory (shared across parallel workers via preprocess)."""

    psi_initial: NDArray[np.complex128]
    heff: scipy.sparse.spmatrix
    jump_ops: list[scipy.sparse.spmatrix]
    embedded_observables: list[scipy.sparse.spmatrix | NDArray[np.complex128] | None]
    sim_params: AnalogSimParams
    # True when there is no dissipative part (no jump operators); skips jump/RNG logic.
    is_unitary: bool = False
    # exp(-i H_eff dt) as dense matrix, if dim <= MAX_PRECOMPUTE_DIM; else None.
    step_propagator: NDArray[np.complex128] | None = None
    output_state: NDArray[np.complex128] | None = None


def preprocess_mcwf(
    initial_state: MPS | None,
    hamiltonian: MPO | None,
    noise_model: NoiseModel | None,
    sim_params: AnalogSimParams,
    *,
    psi_initial: NDArray[np.complex128] | None = None,
    num_sites: int | None = None,
    physical_dimensions: int | list[int] | None = None,
    h_sparse: scipy.sparse.spmatrix | None = None,
) -> MCWFContext:
    """Pre-compute dense operators and initial state for MCWF simulation.

    Called once per :meth:`Simulator.run` before trajectory workers start (see
    ``preprocess_mcwf`` in ``simulator.py``).

    Args:
        initial_state: The initial MPS state, or ``None`` when ``psi_initial`` is supplied.
        hamiltonian: The Hamiltonian MPO (ignored if ``h_sparse`` is set).
        noise_model: The noise model.
        sim_params: Simulation parameters.
        psi_initial: Optional pre-encoded dense state vector (unit norm applied here).
        num_sites: Number of lattice sites when ``initial_state`` is ``None``.
        physical_dimensions: Per-site physical dimensions used to validate dense
            vector and sparse Hamiltonian sizes. Defaults to qubits.
        h_sparse: Pre-materialized sparse Hamiltonian (skips ``hamiltonian.to_sparse_matrix()``).

    Returns:
        MCWFContext containing dense arrays ready for trajectory simulation.

    Raises:
        ValueError: If neither ``initial_state`` nor ``psi_initial`` is provided, if
            ``num_sites`` is missing when only ``psi_initial`` is given, if
            ``psi_initial`` has the wrong Hilbert-space size, or if ``psi_initial`` has zero norm.
    """
    if initial_state is not None:
        num_sites = initial_state.length
        physical_dimensions = initial_state.physical_dimensions
    elif psi_initial is not None:
        if num_sites is None:
            msg = "num_sites is required when preprocess_mcwf is called with psi_initial only."
            raise ValueError(msg)
    else:
        msg = "preprocess_mcwf requires initial_state or psi_initial."
        raise ValueError(msg)

    dim = int(np.prod(resolve_physical_dimensions(num_sites, physical_dimensions), dtype=np.int64))

    if dim > 2**14:
        msg = (
            f"Hilbert-space dimension {dim} is large for representation='vector' even with sparse matrices. "
            "Simulation may be very slow or run out of memory. "
            "Consider using representation='mps' for larger systems."
        )
        warnings.warn(msg, RuntimeWarning, stacklevel=2)

    # 1. Initial state |psi> as dense vector.
    if psi_initial is not None:
        psi = np.asarray(psi_initial, dtype=np.complex128).reshape(-1)
        if psi.size != dim:
            msg = f"psi_initial size {psi.size} does not match Hilbert dimension {dim}."
            raise ValueError(msg)
        norm = np.linalg.norm(psi)
        if np.isclose(norm, 0.0):
            msg = "psi_initial must have non-zero norm."
            raise ValueError(msg)
        psi /= norm
    else:
        assert initial_state is not None
        psi = initial_state.to_vec()
        psi /= np.linalg.norm(psi)

    # 2. Hamiltonian as sparse matrix on the full Hilbert space.
    if h_sparse is not None:
        h_mat = scipy.sparse.csr_matrix(h_sparse)
        if h_mat.shape != (dim, dim):
            msg = f"h_sparse must have shape ({dim}, {dim}), got {h_mat.shape}."
            raise ValueError(msg)
    elif hamiltonian is not None:
        h_mat = hamiltonian.to_sparse_matrix()
    else:
        msg = "preprocess_mcwf requires hamiltonian or h_sparse."
        raise ValueError(msg)

    # 3. Jump operators L_k = sqrt(gamma) * op embedded on the full space.
    jump_ops: list[scipy.sparse.spmatrix] = []
    if noise_model is not None:
        for process in noise_model.processes:
            strength = process["strength"]
            if strength <= 0:
                continue
            op_full = _embed_operator_sparse(process, num_sites)
            jump_ops.append(np.sqrt(strength) * op_full)

    is_unitary = len(jump_ops) == 0

    # 4. Effective Hamiltonian for the no-jump evolution between stochastic events.
    heff = h_mat.copy()
    if jump_ops:
        sum_ldag_l = scipy.sparse.csr_matrix((dim, dim), dtype=complex)
        for op in jump_ops:
            op_csr = cast("Any", op)
            sum_ldag_l += op_csr.conj().T @ op_csr
        heff -= 0.5j * sum_ldag_l

    # 5. Fixed-step propagator U_step = exp(-i H_eff dt) (time-independent H in YAQS).
    step_propagator: NDArray[np.complex128] | None = None
    if dim <= MAX_PRECOMPUTE_DIM:
        h_dense = heff.toarray()
        if linalg.ishermitian(h_dense):
            step_propagator = linalg.expm_hermitian(h_dense, sim_params.dt)
        else:
            step_propagator = linalg.expm(-1j * sim_params.dt * h_dense)

    # 6. Observables embedded on the full space; diagnostics are not defined on |psi>.
    embedded_observables: list[scipy.sparse.spmatrix | NDArray[np.complex128] | None] = []
    for obs in sim_params.sorted_observables:
        if obs.gate.name in {"entropy", "schmidt_spectrum"}:
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
    """MCWF no-jump / jump decision and renormalization for one time step.

    ``psi_next`` is the result of non-unitary evolution ``exp(-i H_eff dt) |psi>``
    (not yet normalized). Norm loss ``1 - ||psi_next||^2`` is the jump probability.

    Returns:
        Normalized state vector after the step (with or without jump).
    """
    norm_sq = np.vdot(psi_next, psi_next).real
    p_jump = 1.0 - norm_sq

    if rng.random() >= p_jump:
        # No jump: renormalize the evolved state.
        return psi_next / np.sqrt(norm_sq)

    # Jump occurred: choose channel k with probability proportional to ||L_k |psi>||^2.
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
        # Degenerate case: fall back to no-jump renormalization.
        return psi_next / np.sqrt(norm_sq)

    weights_arr = np.array(weights, dtype=np.float64)
    weights_arr /= normalization_sum
    k_idx = rng.choice(len(ctx.jump_ops), p=weights_arr)
    jump_op = cast("Any", ctx.jump_ops[k_idx])
    jumped = jump_op.dot(param_psi)
    return jumped / np.linalg.norm(jumped)


def mcwf(args: tuple[int, MCWFContext]) -> tuple[NDArray[np.float64], None, NDArray[np.complex128] | None]:
    """Run a single Monte Carlo wavefunction trajectory.

    Args:
        args: A tuple containing:
            - int: Trajectory identifier (used for RNG seeding in parallel runs).
            - MCWFContext: Pre-computed simulation context from ``preprocess_mcwf``.

    Returns:
        An array of expectation values for each observable over time.
    """
    traj_idx, ctx = args
    sim_params = ctx.sim_params
    dt = sim_params.dt

    psi = ctx.psi_initial.copy()
    if sim_params.random_seed is not None:
        rng = make_trajectory_rng(traj_idx, base_seed=sim_params.random_seed)
    else:
        rng = np.random.default_rng()

    num_obs = len(sim_params.sorted_observables)
    num_steps = len(sim_params.times)
    num_cols = num_steps if sim_params.sample_timesteps else 1
    results = np.zeros((num_obs, num_cols), dtype=np.float64)

    def measure(current_psi: NDArray[np.complex128], col: int) -> None:
        for i, op_mat in enumerate(ctx.embedded_observables):
            if op_mat is not None:
                if scipy.sparse.issparse(op_mat):
                    op_mat_sparse = cast("Any", op_mat)
                    val = np.vdot(current_psi, op_mat_sparse.dot(current_psi))
                else:
                    op_mat_dense = cast("NDArray[np.complex128]", op_mat)
                    val = np.vdot(current_psi, op_mat_dense @ current_psi)
                results[i, col] = val.real
            else:
                results[i, col] = 0.0

    if sim_params.sample_timesteps:
        measure(psi, 0)

    for t_idx in range(1, num_steps):
        if ctx.step_propagator is not None:
            # Fast path: one application of precomputed U_step = exp(-i H_eff dt).
            if ctx.is_unitary:
                psi = ctx.step_propagator @ psi
            else:
                psi_before = psi
                psi_next = ctx.step_propagator @ psi_before
                psi = _apply_noisy_step(psi_before, psi_next, ctx, rng)
        elif ctx.is_unitary:
            # Noiseless but Hilbert space too large to store U_step: Hermitian Lanczos per step.
            psi = expm_krylov(lambda v: ctx.heff @ v, psi, dt)  # ty: ignore[unsupported-operator]
        else:
            # Noisy and no stored U_step: general non-Hermitian Arnoldi per step.
            psi_next = expm_arnoldi(lambda v: ctx.heff @ v, psi, dt)  # ty: ignore[unsupported-operator]
            psi = _apply_noisy_step(psi, psi_next, ctx, rng)

        if sim_params.sample_timesteps:
            measure(psi, t_idx)
        elif t_idx == num_steps - 1:
            measure(psi, 0)

    return results, None, psi if sim_params.get_state else None
