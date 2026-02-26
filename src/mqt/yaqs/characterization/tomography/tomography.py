# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Automated Process Tomography Module.

This module provides functions to perform process tomography on a quantum system
modeled by an MPO (Matrix Product Operator) evolution. It reconstructs the
single-qubit restricted process tensor (Choi matrix or Process map) by evolving a set of
basis states and measuring the output state in the Pauli basis.
"""

from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, cast

import numpy as np

from mqt.yaqs.analog.analog_tjm import analog_tjm_1, analog_tjm_2
from mqt.yaqs.analog.mcwf import mcwf, preprocess_mcwf
from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.simulator import WORKER_CTX, available_cpus, run_backend_parallel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.networks import MPO
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams

from .process_tensor import ProcessTensor


def get_basis_states() -> list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]]:
    """Return the 4 minimal single-qubit basis states for tomography.

    Returns:
        List of tuples (name, state_vector, density_matrix).
    """
    # Define the 4 basis states
    # Z basis
    psi_0 = np.array([1, 0], dtype=complex)
    psi_1 = np.array([0, 1], dtype=complex)
    # X basis
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    # Y basis
    psi_i_plus = np.array([1, 1j], dtype=complex) / np.sqrt(2)

    states = [
        ("zeros", psi_0),
        ("ones", psi_1),
        ("x+", psi_plus),
        ("y+", psi_i_plus),
    ]

    # Convert to density matrices
    basis_set: list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]] = []
    for name, psi in states:
        rho = np.outer(psi, psi.conj())
        basis_set.append((name, psi, rho))
    return basis_set


def get_choi_basis() -> tuple[list[NDArray[np.complex128]], list[tuple[int, int]]]:
    r"""Generate the 16 basis CP maps (Choi matrices) from the 4 basis states.

    A basis CP map is A_{p,m}(rho) = Tr(E_m rho) rho_p.
    Its Choi matrix is B_{p,m} = rho_p \\otimes E_m^T,
    because J(A) = sum_{ij} A(|i><j|) otimes |i><j| = rho_p otimes sum_{ij} Tr(E_m |i><j|) |i><j|
    = rho_p otimes E_m^T.

    Returns:
        tuple containing:
        - List of 16 Choi matrices (4x4 each).
        - List of (prep_idx, meas_idx) tuples corresponding to each Choi matrix.
    """
    basis_set = get_basis_states()
    choi_matrices = []
    indices = []

    for p, (_, _, rho_p) in enumerate(basis_set):
        for m, (_, _, e_m) in enumerate(basis_set):
            b_pm = np.kron(rho_p, e_m.T)
            choi_matrices.append(b_pm)
            indices.append((p, m))

    return choi_matrices, indices


def calculate_dual_choi_basis(basis_matrices: list[NDArray[np.complex128]]) -> list[NDArray[np.complex128]]:
    """Calculate the dual frame for the given Choi basis matrices.

    The dual frame {D_k} allows reconstruction of any map or process tensor component.

    Args:
        basis_matrices (list): List of Choi matrices (4x4) forming the generalized basis.

    Returns:
        list[NDArray[np.complex128]]: List of dual Choi matrices D_k.
    """
    # Stack matrices as columns of a Frame Operator F
    # Shape (16, 16) for Choi basis maps
    dim = basis_matrices[0].shape[0]

    # frame_matrix: columns are vectorized density matrices
    frame_matrix = np.column_stack([m.reshape(-1) for m in basis_matrices])

    # Calculate dual frame using Moore-Penrose pseudoinverse
    # Vectorized: |Rho>> = sum_k (<<D_k|Rho>>) |F_k>>
    #                    = sum_k |F_k>> <<D_k| |Rho>>

    frame_pinv = np.linalg.pinv(frame_matrix)
    dual_frame_dag = frame_pinv
    dual_frame = dual_frame_dag.conj().T

    # Unpack columns of dual_frame into matrices
    return [dual_frame[:, k].reshape(dim, dim) for k in range(dual_frame.shape[1])]


def _reprepare_site_zero_forced(
    mps: MPS, proj_state: NDArray[np.complex128], new_state: NDArray[np.complex128]
) -> float:
    """Project site 0 onto proj_state and reprepare new_state deterministically.

    Args:
        mps: The MPS to modify (modified in-place)
        proj_state: The state vector to project site 0 onto (e.g., measurement outcome)
        new_state: The new state vector to prepare for site 0

    Returns:
        float: The probability of this projection occurring.
    """
    # Force right-canonical form at site 0 to extract proper environment state
    mps.set_canonical_form(orthogonality_center=0)
    T = mps.tensors[0]

    # Contract site 0 with <proj_state|
    # T shape: (d, 1, chi). proj_state shape: (d,)
    env_vec = np.einsum("s c, s -> c", T[:, 0, :], proj_state.conj())
    prob = float(np.linalg.norm(env_vec) ** 2)

    if prob > 1e-15:
        env_vec /= np.sqrt(prob)

    d = new_state.shape[0]
    chi = env_vec.shape[0]
    new_tensor = np.zeros((d, 1, chi), dtype=np.complex128)
    for s in range(d):
        new_tensor[s, 0, :] = new_state[s] * env_vec

    mps.tensors[0] = new_tensor

    # Final Renormalization
    final_norm = mps.norm()
    if abs(final_norm) > 1e-15:
        mps.tensors[0] /= final_norm

    return prob


def _reprepare_site_zero_vector_forced(
    state_vec: NDArray[np.complex128],
    proj_state: NDArray[np.complex128],
    new_state: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], float]:
    """Reprepare site 0 for a dense vector state deterministically.

    Args:
        state_vec: The dense state vector (length 2^N)
        proj_state: The state vector to project site 0 onto
        new_state: The new state vector to prepare for site 0

    Returns:
        tuple[NDArray[np.complex128], float]: (new_state_vector, probability)
    """
    dim_total = state_vec.shape[0]
    dim_env = dim_total // 2
    psi_reshaped = state_vec.reshape(2, dim_env)

    # Project site 0
    env_vec = proj_state.conj() @ psi_reshaped
    prob = float(np.linalg.norm(env_vec) ** 2)

    if prob > 1e-15:
        env_vec /= np.sqrt(prob)

    new_psi = np.outer(new_state, env_vec).flatten()
    return new_psi, prob


def _reconstruct_state(expectations: dict[str, float]) -> NDArray[np.complex128]:
    """Reconstruct single-qubit density matrix from Pauli expectations.

    Args:
        expectations: Dictionary of Pauli expectations.

    Returns:
        NDArray[np.complex128]: The reconstructed single-qubit density matrix.
    """
    eye = np.eye(2, dtype=complex)
    x_matrix = X().matrix
    y_matrix = Y().matrix
    z_matrix = Z().matrix

    return 0.5 * (eye + expectations["x"] * x_matrix + expectations["y"] * y_matrix + expectations["z"] * z_matrix)


def _tomography_sequence_worker(job_idx: int) -> tuple[int, int, list[NDArray[np.complex128]], float]:
    """Worker function for a single tomography sequence.

    Args:
        job_idx: Flat index mapping to (seq_idx, traj_idx).

    Returns:
        tuple[int, int, list[NDArray[np.complex128]], float]: (seq_idx, traj_idx, [final_rho], prob weight)
    """
    # 1. Decode job
    num_trajectories = WORKER_CTX["num_trajectories"]
    seq_idx = job_idx // num_trajectories
    traj_idx = job_idx % num_trajectories

    # 2. Get segment parameters
    operator = WORKER_CTX["operator"]
    sim_params = WORKER_CTX["sim_params"]
    timesteps = WORKER_CTX["timesteps"]
    basis_set = WORKER_CTX["basis_set"]
    choi_indices = WORKER_CTX["choi_indices"]
    worker_sequences = WORKER_CTX["worker_sequences"]
    noise_model = WORKER_CTX["noise_model"]

    alpha_seq = worker_sequences[seq_idx]

    # 3. Initialize state to |0...0>
    is_mcwf = sim_params.solver == "MCWF"
    current_state: MPS | NDArray[np.complex128]

    if is_mcwf:
        num_sites = operator.length
        current_state = np.array([1.0], dtype=np.complex128)
        for _ in range(num_sites):
            current_state = np.kron(current_state, np.array([1.0, 0.0], dtype=np.complex128))
    else:
        current_state = MPS(length=operator.length, state="zeros")

    sequence_weight = 1.0

    # We only need the final output state after all evolution steps
    def _get_rho_site_zero(state: MPS | NDArray[np.complex128]) -> NDArray[np.complex128]:
        if isinstance(state, np.ndarray):
            rho = np.reshape(state, (2, -1))
            return rho @ rho.conj().T
        assert isinstance(state, MPS)
        trace = float(state.norm() ** 2)
        rx = state.expect(Observable(X(), sites=[0]))
        ry = state.expect(Observable(Y(), sites=[0]))
        rz = state.expect(Observable(Z(), sites=[0]))
        # Scale the normalized Pauli reconstruction by the actual trace
        return trace * _reconstruct_state({
            "x": rx / trace if trace > 1e-15 else 0,
            "y": ry / trace if trace > 1e-15 else 0,
            "z": rz / trace if trace > 1e-15 else 0,
        })

    # 4. Multi-step loop
    for step_i, duration in enumerate(timesteps):
        # Apply intervention for this slot
        alpha_t = alpha_seq[step_i]
        p_t, m_t = choi_indices[alpha_t]
        _, psi_next, _ = basis_set[p_t]
        _, psi_proj, _ = basis_set[m_t]

        if is_mcwf:
            assert isinstance(current_state, np.ndarray)
            current_state, step_prob = _reprepare_site_zero_vector_forced(
                cast("NDArray[np.complex128]", current_state), psi_proj, psi_next
            )
        else:
            assert isinstance(current_state, MPS)
            step_prob = _reprepare_site_zero_forced(current_state, psi_proj, psi_next)

        sequence_weight *= step_prob

        # Skip evolution if branch is physically dead
        if sequence_weight < 1e-15:
            break

        # Segment Simulation
        step_params = copy.deepcopy(sim_params)
        step_params.elapsed_time = duration
        step_params.dt = sim_params.dt
        step_params.num_traj = 1
        step_params.show_progress = False
        step_params.get_state = True

        n_steps = int(np.round(duration / step_params.dt))
        step_params.times = np.linspace(0, n_steps * step_params.dt, n_steps + 1)

        if is_mcwf:
            static_ctx = WORKER_CTX["mcwf_static_ctx"]
            dynamic_ctx = copy.copy(static_ctx)
            dynamic_ctx.psi_initial = current_state
            dynamic_ctx.sim_params = step_params

            mcwf((traj_idx, dynamic_ctx))
            assert dynamic_ctx.output_state is not None
            current_state = cast("NDArray[np.complex128]", dynamic_ctx.output_state)
        else:
            backend = analog_tjm_1 if step_params.order == 1 else analog_tjm_2
            assert isinstance(current_state, MPS)
            backend((traj_idx, current_state, noise_model, step_params, operator))
            assert step_params.output_state is not None
            current_state = cast("MPS", step_params.output_state)

    sequence_results = [_get_rho_site_zero(current_state)]
    return (seq_idx, traj_idx, sequence_results, sequence_weight)


def run(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    num_trajectories: int = 100,
    noise_model: NoiseModel | None = None,
) -> ProcessTensor:
    """Run Process Tomography / Process Tensor Tomography using parallelized backend.

    Reconstructs a single-step or multi-step process tensor (state-injection comb)
    under the chosen intervention set. Measures outputs at each time step and applies
    system-only interventions to preserve environment correlations.
    Measurements and preparations are applied deterministically across all combinations
    of the basis set.

    Args:
        operator: System evolution MPO.
        sim_params: Simulation parameters.
        timesteps: List of time durations for each evolution segment.
                   If None, defaults to [sim_params.elapsed_time] (standard 1-step tomography).
        num_trajectories: Number of trajectories to average per sequence (for noise unravelling).
        noise_model: Noise model to apply. If None, uses sim_params.noise_model.

    Returns:
        ProcessTensor object representing the final-time map conditioned on preparation sequences.
    """
    if timesteps is None:
        timesteps = [sim_params.elapsed_time]

    sim_params.get_state = True
    assert sim_params.get_state or sim_params.observables, (
        "No output specified: either observables or get_state must be set."
    )

    # 1. Prepare Basis and Duals
    basis_set = get_basis_states()
    choi_basis, choi_indices = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    # Sanity check duality
    for i in range(16):
        for j in range(16):
            inner = np.trace(duals[i].conj().T @ choi_basis[j])
            expected = 1.0 if i == j else 0.0
            if not np.isclose(inner, expected, atol=1e-10):
                msg = f"Dual basis creation failed: <D_{i}|B_{j}> = {inner}"
                raise ValueError(msg)

    num_steps = len(timesteps)
    alpha_indices = list(range(16))
    worker_sequences = list(itertools.product(alpha_indices, repeat=num_steps))
    rng = np.random.default_rng()
    rng.shuffle(worker_sequences)
    num_worker_sequences = len(worker_sequences)

    # 2. Prepare Simulation Context
    if noise_model is None:
        noise_model = sim_params.noise_model

    if noise_model is None:
        num_trajectories = 1

    mcwf_static_ctx = None
    if sim_params.solver == "MCWF":
        dummy_mps = MPS(length=operator.length, state="zeros")
        mcwf_static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, sim_params)

    payload = {
        "operator": operator,
        "sim_params": sim_params,
        "timesteps": timesteps,
        "basis_set": basis_set,
        "choi_indices": choi_indices,
        "worker_sequences": worker_sequences,
        "noise_model": noise_model,
        "num_trajectories": num_trajectories,
        "mcwf_static_ctx": mcwf_static_ctx,
    }

    # 3. Parallel Execution
    total_jobs = num_worker_sequences * num_trajectories
    max_workers = max(1, available_cpus() - 1)

    aggregated_outputs = [np.zeros((2, 2), dtype=np.complex128) for _ in range(num_worker_sequences)]
    aggregated_weights = np.zeros(num_worker_sequences, dtype=np.float64)

    results_iterator = run_backend_parallel(
        worker_fn=_tomography_sequence_worker,
        payload=payload,
        n_jobs=total_jobs,
        max_workers=max_workers,
        show_progress=sim_params.show_progress,
        desc="Simulating Tomography Sequences",
    )

    for _job_idx, (worker_seq_idx, _traj_idx, sequence_rhos, sequence_weight) in results_iterator:
        rho_final = sequence_rhos[0]
        aggregated_outputs[worker_seq_idx] += rho_final * sequence_weight
        aggregated_weights[worker_seq_idx] += sequence_weight

    # Average normalized to the number of trajectories
    for i in range(num_worker_sequences):
        aggregated_outputs[i] /= num_trajectories
        aggregated_weights[i] /= num_trajectories

    # 4. Construct Tensor
    # Tensor shape is [4, 16] for 1 step, [4, 16, 16] for 2 steps, etc.
    # Indices are: final_output, alpha_0, alpha_1, ...
    tensor_shape = [4] + [16] * num_steps
    process_tensor_data = np.zeros(tensor_shape, dtype=np.complex128)
    weights_shape = [16] * num_steps
    process_tensor_weights = np.zeros(weights_shape, dtype=np.float64)

    for worker_seq_idx, avg_rho in enumerate(aggregated_outputs):
        seq_tuple = worker_sequences[worker_seq_idx]
        rho_vec = avg_rho.reshape(-1)

        idx = (slice(None), *seq_tuple)
        process_tensor_data[idx] = rho_vec
        process_tensor_weights[seq_tuple] = aggregated_weights[worker_seq_idx]

    return ProcessTensor(
        tensor=process_tensor_data,
        weights=process_tensor_weights,
        timesteps=timesteps,
        choi_duals=duals,
        choi_indices=choi_indices,
        choi_basis=choi_basis,
    )
