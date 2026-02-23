# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Automated Process Tomography Module.

This module provides functions to perform process tomography on a quantum system
modeled by an MPO (Matrix Product Operator) evolution. It reconstructs the
single-qubit process tensor (Choi matrix or Process map) by evolving a set of
basis states and measuring the output state in the Pauli basis.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.simulator import WORKER_CTX, available_cpus, run_backend_parallel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.networks import MPO
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

from mqt.yaqs.tomography.process_tensor import ProcessTensor


def _get_basis_states() -> list[tuple[str, NDArray[np.complex128]]]:
    """Returns the 6 single-qubit basis states for tomography.

    Returns:
        list of tuples (name, density_matrix_4x1_vector).
    """
    # Define the 6 basis states
    # Z basis
    psi_0 = np.array([1, 0], dtype=complex)
    psi_1 = np.array([0, 1], dtype=complex)
    # X basis
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    # Y basis
    psi_i_plus = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    psi_i_minus = np.array([1, -1j], dtype=complex) / np.sqrt(2)

    states = [
        ("zeros", psi_0),
        ("ones", psi_1),
        ("x+", psi_plus),
        ("x-", psi_minus),
        ("y+", psi_i_plus),
        ("y-", psi_i_minus),
    ]

    # Convert to density matrices
    basis_set = []
    for name, psi in states:
        rho = np.outer(psi, psi.conj())
        basis_set.append((name, psi, rho))
    return basis_set


def _calculate_dual_frame(basis_matrices: list[NDArray[np.complex128]]) -> list[NDArray[np.complex128]]:
    """Calculates the dual frame for the given basis states.

    The dual frame {D_k} allows reconstruction of any operator A via:
    A = sum_k Tr(D_k^dag A) F_k
    or
    A = sum_k Tr(F_k^dag A) D_k  <-- this is what we use if we treat basis_matrices as input basis F_k.

    If we have input states rho_in^k, and we measure output states rho_out^k,
    the map is E(rho) = sum_k Tr(D_k^dag rho) rho_out^k.

    Args:
        basis_matrices (list): List of density matrices (2x2) forming the frame.

    Returns:
        list: List of dual matrices D_k.
    """
    # Stack matrices as columns of a Frame Operator F
    # Shape (4, 6) for single qubit (dim=2^2=4)
    dim = basis_matrices[0].shape[0]
    dim * dim

    # F matrix: columns are vectorized density matrices
    F = np.column_stack([m.reshape(-1) for m in basis_matrices])

    # Calculate dual frame using Moore-Penrose pseudoinverse
    # Vectorized: |Rho>> = sum_k (<<D_k|Rho>>) |F_k>>
    #                    = sum_k |F_k>> <<D_k| |Rho>>

    F_pinv = np.linalg.pinv(F)
    D_dag = F_pinv
    D = D_dag.conj().T

    # Unpack columns of D into matrices
    return [D[:, k].reshape(dim, dim) for k in range(D.shape[1])]


def _reprepare_site_zero(mps: MPS, new_state: np.ndarray, rng: np.random.Generator, meas_basis: str = "Z") -> int:
    """Reprepare site 0 with selective (trajectory-resolved) intervention.

    This implements a clean trajectory-consistent operation:
    1. Sample measurement outcome m with Born probabilities
    2. Project onto |m⟩ in the chosen basis and keep conditional environment branch
    3. Reprepare to new_state while preserving the branch's bond vector

    Args:
        mps: The MPS to modify (modified in-place)
        new_state: The new state vector for site 0, shape (d,)
        rng: Random number generator for sampling
        meas_basis: The basis to measure in ("X", "Y", or "Z"). Defaults to "Z".

    Returns:
        int: The measurement outcome (0 or 1 for qubits)
    """
    # 1. Perform in-place projective measurement
    # This also handles shifting the center to site 0 correctly.
    outcome = mps.measure(site=0, basis=meas_basis, rng=rng)

    # 2. Extract environment part and reprepare with new_state
    # The state is now |outcome> \otimes |env_conditional_normalized>
    # The environment vector is sitting in mps.tensors[0][outcome, 0, :]
    env_vec_cond = mps.tensors[0][outcome, 0, :]
    d = new_state.shape[0]
    chi = env_vec_cond.shape[0]

    # Construct new tensor: |new_state> \otimes |env_conditional_normalized>
    new_tensor = np.zeros((d, 1, chi), dtype=np.complex128)
    for s in range(d):
        new_tensor[s, 0, :] = new_state[s] * env_vec_cond

    mps.tensors[0] = new_tensor

    # Final Renormalization
    final_norm = mps.norm()
    if abs(final_norm) > 1e-15:
        mps.tensors[0] /= final_norm

    return outcome


def _reprepare_site_zero_vector(
    psi: NDArray[np.complex128], new_state: NDArray[np.complex128], rng: np.random.Generator, meas_basis: str = "Z"
) -> tuple[NDArray[np.complex128], int]:
    """Reprepare site 0 for a dense vector state with selective intervention.

    Args:
        psi: The current state vector, shape (2^N,)
        new_state: The new state vector for site 0, shape (2,)
        rng: Random number generator for sampling
        meas_basis: The basis to measure in ("X", "Y", or "Z"). Defaults to "Z".

    Returns:
        tuple: (updated_psi, outcome)
    """
    dim_total = psi.shape[0]
    dim_env = dim_total // 2
    psi_reshaped = psi.reshape(2, dim_env)

    # 1. Basis Rotation
    # Extract the site 0 part (shape 2, dim_env)
    # To measure in X or Y, we apply the inverse rotation H or H S^\dagger to site 0,
    # which is equivalent to rotating the state forward before Z measurement.
    if meas_basis == "X":
        # H = 1/sqrt(2) [1  1]
        #               [1 -1]
        rot = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        psi_reshaped = rot @ psi_reshaped
    elif meas_basis == "Y":
        # H S^\dagger = 1/sqrt(2) [1 -i]
        #                         [1  i]
        rot = np.array([[1, -1j], [1, 1j]], dtype=np.complex128) / np.sqrt(2)
        psi_reshaped = rot @ psi_reshaped
    elif meas_basis != "Z":
        msg = f"Unsupported measurement basis: {meas_basis}"
        raise ValueError(msg)

    # 2. Born probabilities (in the rotated basis)
    p0 = np.linalg.norm(psi_reshaped[0, :]) ** 2
    p1 = np.linalg.norm(psi_reshaped[1, :]) ** 2
    norm = p0 + p1
    if norm > 1e-15:
        p0 /= norm
        p1 /= norm
    else:
        p0, p1 = 0.5, 0.5  # Neutral choice for zero vector

    # 3. Sample outcome
    outcome = rng.choice([0, 1], p=[p0, p1])

    # 4. Project and reprepare
    env_cond = psi_reshaped[outcome, :].copy()
    env_norm = np.linalg.norm(env_cond)
    if env_norm > 1e-15:
        env_cond /= env_norm

    new_psi = np.outer(new_state, env_cond).flatten()
    return new_psi, outcome


def _reconstruct_state(expectations: dict[str, float]) -> NDArray[np.complex128]:
    """Reconstructs single-qubit density matrix from Pauli expectations.

    rho = 0.5 * (I + <X>X + <Y>Y + <Z>Z)
    """
    I = np.eye(2, dtype=complex)
    x = X().matrix
    y = Y().matrix
    z = Z().matrix

    return 0.5 * (I + expectations["x"] * x + expectations["y"] * y + expectations["z"] * z)


def _tomography_trajectory_worker(job_idx: int) -> tuple[int, int, list[NDArray[np.complex128]]]:
    """Worker function for a single tomography trajectory.

    Args:
        job_idx: Flat index mapping to (seq_idx, traj_idx).

    Returns:
        tuple: (seq_idx, traj_idx, list of output states/results)
    """
    import numpy as np

    from mqt.yaqs.analog.analog_tjm import analog_tjm_1, analog_tjm_2
    from mqt.yaqs.analog.mcwf import mcwf
    from mqt.yaqs.core.data_structures.networks import MPS

    # 1. Decode job
    num_trajectories = WORKER_CTX["num_trajectories"]
    seq_idx = job_idx // num_trajectories
    traj_idx = job_idx % num_trajectories

    # 2. Get segment parameters
    operator = WORKER_CTX["operator"]
    sim_params = WORKER_CTX["sim_params"]
    timesteps = WORKER_CTX["timesteps"]
    basis_set = WORKER_CTX["basis_set"]
    worker_sequences = WORKER_CTX["worker_sequences"]
    noise_model = WORKER_CTX["noise_model"]

    prep_seq, meas_seq = worker_sequences[seq_idx]
    rng = np.random.default_rng(traj_idx)

    # 3. Initialize state
    initial_idx = prep_seq[0]
    _, psi_0, _ = basis_set[initial_idx]

    # Persistent state (either MPS or Vector)
    is_mcwf = sim_params.solver == "MCWF"
    current_state: MPS | NDArray[np.complex128]

    if is_mcwf:
        # MCWF uses dense vectors. Initialize product state |psi_0> \otimes |0...0>
        num_sites = operator.length
        current_state = np.array([1.0], dtype=np.complex128)
        # Site 0
        current_state = np.kron(current_state, psi_0)
        # Sites 1..N-1
        for _ in range(1, num_sites):
            current_state = np.kron(current_state, np.array([1.0, 0.0], dtype=np.complex128))
    else:
        # TJM uses MPS
        current_state = MPS(length=operator.length, state="zeros")
        current_state.tensors[0] = np.expand_dims(psi_0, axis=(1, 2)).astype(np.complex128)

    # Storage for output states (each step)
    trajectory_results = []

    # Initial measurement (t=0)
    def get_rho_site_zero(state):
        if isinstance(state, np.ndarray):
            # Dense vector: reshape to (2, 2^(N-1)) and compute partial trace
            rho = state.reshape(2, -1)
            return rho @ rho.conj().T
        rx = state.expect(Observable(X(), sites=[0]))
        ry = state.expect(Observable(Y(), sites=[0]))
        rz = state.expect(Observable(Z(), sites=[0]))
        return _reconstruct_state({"x": rx, "y": ry, "z": rz})

    trajectory_results.append(get_rho_site_zero(current_state))

    # 4. Multi-step loop
    for step_i, duration in enumerate(timesteps):
        # Segment Simulation
        step_params = copy.deepcopy(sim_params)
        step_params.elapsed_time = duration
        step_params.dt = sim_params.dt
        step_params.num_traj = 1
        step_params.show_progress = False
        step_params.get_state = True

        # Handle times array correctly (linspace to avoid drift)
        n_steps = int(np.round(duration / step_params.dt))
        step_params.times = np.linspace(0, n_steps * step_params.dt, n_steps + 1)

        if is_mcwf:
            static_ctx = WORKER_CTX["mcwf_static_ctx"]

            dynamic_ctx = copy.copy(static_ctx)
            dynamic_ctx.psi_initial = current_state
            dynamic_ctx.sim_params = step_params

            mcwf((0, dynamic_ctx))
            current_state = dynamic_ctx.output_state
        else:
            # TJM
            backend = analog_tjm_1 if step_params.order == 1 else analog_tjm_2
            backend((0, current_state, noise_model, step_params, operator))
            current_state = step_params.output_state

        # Measure AFTER evolution
        trajectory_results.append(get_rho_site_zero(current_state))

        # Intervention
        if step_i < len(timesteps) - 1:
            next_idx = prep_seq[step_i + 1]
            meas_basis = meas_seq[step_i]
            _, psi_next, _ = basis_set[next_idx]

            if is_mcwf:
                current_state, _ = _reprepare_site_zero_vector(current_state, psi_next, rng, meas_basis=meas_basis)
            else:
                _reprepare_site_zero(current_state, psi_next, rng, meas_basis=meas_basis)

    return (seq_idx, traj_idx, trajectory_results)


def run(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    num_trajectories: int = 100,
    noise_model: NoiseModel | None = None,
    measurement_bases: list[str] | str | None = None,
) -> ProcessTensor:
    """Run Process Tomography / Process Tensor Tomography using parallelized backend.

    Reconstructs a single-step or multi-step restricted process tensor (state-injection comb)
    under the chosen intervention set. Measures outputs at each time step and applies
    system-only interventions to preserve environment correlations.

    Args:
        operator: System evolution MPO.
        sim_params: Simulation parameters.
        timesteps: List of time durations for each evolution segment.
                   If None, defaults to [sim_params.elapsed_time] (standard 1-step tomography).
        num_trajectories: Number of trajectories to average per sequence.
        noise_model: Noise model to apply. If None, uses sim_params.noise_model.
        measurement_bases: Bases to use for measurement interventions.
                           Can be a single string (e.g., "Z") or a list (e.g., ["X", "Y", "Z"]).
                           If None, defaults to ["X", "Y", "Z"].

    Returns:
        ProcessTensor object representing the final-time map conditioned on preparation sequences.
    """
    import itertools

    from mqt.yaqs.analog.mcwf import preprocess_mcwf

    # 0. Handle default timesteps
    if timesteps is None:
        timesteps = [sim_params.elapsed_time]

    sim_params.get_state = True
    
    # Deferred output validation (safety check)
    assert sim_params.get_state or sim_params.observables, "No output specified: either observables or get_state must be set."

    # 1. Prepare Basis and Duals
    basis_set = _get_basis_states()
    basis_rhos = [b[2] for b in basis_set]
    _calculate_dual_frame(basis_rhos)

    num_steps = len(timesteps)
    prep_basis_indices = list(range(len(basis_set)))
    prep_sequences = list(itertools.product(prep_basis_indices, repeat=num_steps))
    
    # Handle measurement bases
    if measurement_bases is None:
        measurement_bases = ["Z"]
    if isinstance(measurement_bases, str):
        measurement_bases = [measurement_bases]

    if num_steps > 1:
        meas_sequences = list(itertools.product(measurement_bases, repeat=num_steps - 1))
    else:
        # Standard tomography (1-step) has no interventions
        meas_sequences = [()]

    worker_sequences = list(itertools.product(prep_sequences, meas_sequences))
    num_prep_sequences = len(prep_sequences)
    num_worker_sequences = len(worker_sequences)
    num_meas_sequences = len(meas_sequences)

    # 2. Prepare Simulation Context
    if noise_model is None:
        noise_model = sim_params.noise_model

    mcwf_static_ctx = None
    if sim_params.solver == "MCWF":
        # Pre-calculate jump ops and effective Hamiltonian once
        # Using a dummy initial state as it will be updated per worker/trajectory
        dummy_mps = MPS(length=operator.length, state="zeros")
        mcwf_static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, sim_params)

    payload = {
        "operator": operator,
        "sim_params": sim_params,
        "timesteps": timesteps,
        "basis_set": basis_set,
        "worker_sequences": worker_sequences,
        "noise_model": noise_model,
        "num_trajectories": num_trajectories,
        "mcwf_static_ctx": mcwf_static_ctx,
    }

    # 3. Parallel Execution
    total_jobs = num_worker_sequences * num_trajectories
    max_workers = max(1, available_cpus() - 1)

    # Storage: prep_sequence_idx -> list of [num_steps+1] averaged density matrices
    aggregated_outputs = [
        [np.zeros((2, 2), dtype=np.complex128) for _ in range(num_steps + 1)] for _ in range(num_prep_sequences)
    ]

    results_iterator = run_backend_parallel(
        worker_fn=_tomography_trajectory_worker,
        payload=payload,
        n_jobs=total_jobs,
        max_workers=max_workers,
        show_progress=sim_params.show_progress,
        desc="Simulating Tomography Trajectories",
    )

    # Average over BOTH trajectories AND measurement bases for the same prep sequence
    total_averages = num_trajectories * num_meas_sequences
    for _job_idx, (worker_seq_idx, _traj_idx, trajectory_rhos) in results_iterator:
        prep_seq_idx = worker_seq_idx // num_meas_sequences
        for t_idx, rho in enumerate(trajectory_rhos):
            aggregated_outputs[prep_seq_idx][t_idx] += rho / total_averages

    # 4. Construct Process Tensor
    num_frame_states = len(basis_set)
    tensor_shape = [4] + [num_frame_states] * num_steps
    process_tensor = np.zeros(tensor_shape, dtype=complex)

    for prep_seq_idx, avg_rhos in enumerate(aggregated_outputs):
        seq = prep_sequences[prep_seq_idx]
        rho_final = avg_rhos[-1]
        rho_vec = rho_final.reshape(-1)

        idx = (slice(None), *tuple(seq))
        process_tensor[idx] = rho_vec

    pt = ProcessTensor(process_tensor, timesteps)
    pt.data = process_tensor
    return pt
