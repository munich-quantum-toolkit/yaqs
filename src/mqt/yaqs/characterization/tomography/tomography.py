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
from typing import TYPE_CHECKING, Literal, cast

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

from .process_tensor import ProcessTensor, canonicalize_upsilon


def _random_pure_state(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Generate a random pure state vector for continuous tomography."""
    z = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    z /= np.linalg.norm(z)
    return z


def _sample_intervention_candidate_local(
    current_state: MPS | NDArray[np.complex128],
    rng: np.random.Generator,
    candidates: list[tuple[NDArray[np.complex128], NDArray[np.complex128]]],
    is_mcwf: bool,
) -> tuple[MPS | NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], float]:
    """Sample an intervention from a candidate pool using local importance sampling."""
    num_candidates = len(candidates)
    p_j = np.zeros(num_candidates, dtype=np.float64)
    next_states = []

    for j, (psi_meas, psi_prep) in enumerate(candidates):
        if is_mcwf:
            assert isinstance(current_state, np.ndarray)
            n_st, p = _reprepare_site_zero_vector_forced(current_state, psi_meas, psi_prep)
            next_states.append(n_st)
            p_j[j] = p
        else:
            from mqt.yaqs.core.data_structures.networks import MPS
            assert isinstance(current_state, MPS)
            cand_state = current_state.copy()
            p = _reprepare_site_zero_forced(cand_state, psi_meas, psi_prep)
            next_states.append(cand_state)
            p_j[j] = p

    sum_p = p_j.sum()
    if sum_p < 1e-300:
        # Fallback to uniform
        q_j = np.ones(num_candidates) / num_candidates
        inc_weight = sum_p
    else:
        q_j = p_j / sum_p
        inc_weight = sum_p

    # Sample candidate
    J = rng.choice(num_candidates, p=q_j)
    psi_meas, psi_prep = candidates[J]
    
    return next_states[J], psi_meas, psi_prep, float(inc_weight)


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
    t_mps = mps.tensors[0]

    # Contract site 0 with <proj_state|
    # T shape: (d, 1, chi). proj_state shape: (d,)
    env_vec = np.einsum("s c, s -> c", t_mps[:, 0, :], proj_state.conj())
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


def _tomography_sequence_worker(
    job_idx: int,
) -> tuple[int, int, tuple[int, ...], NDArray[np.complex128], float, list[tuple[NDArray[np.complex128], NDArray[np.complex128]]] | None]:
    """Worker function for a single tomography sequence.

    Args:
        job_idx: Flat index mapping to (seq_idx/s_idx, traj_idx).

    Returns:
        tuple[int, int, tuple, NDArray, float, list | None]: (s_idx, traj_idx, alpha_seq, rho_final, sequence_weight, chosen_continuous_states)
    """
    # 1. Decode job
    num_trajectories = WORKER_CTX["num_trajectories"]
    s_idx = job_idx // num_trajectories
    traj_idx = job_idx % num_trajectories

    # 2. Get segment parameters
    operator = WORKER_CTX["operator"]
    sim_params = WORKER_CTX["sim_params"]
    timesteps = WORKER_CTX["timesteps"]
    basis_set = WORKER_CTX["basis_set"]
    choi_indices = WORKER_CTX["choi_indices"]
    worker_sequences = WORKER_CTX.get("worker_sequences")
    sampled_sequences = WORKER_CTX.get("sampled_sequences")
    continuous_states = WORKER_CTX.get("continuous_states")
    continuous_candidates = WORKER_CTX.get("continuous_candidates")
    ensemble = WORKER_CTX.get("ensemble", "discrete")
    sampling = WORKER_CTX.get("sampling", "uniform")
    noise_model = WORKER_CTX["noise_model"]
    
    worker_seeds = WORKER_CTX.get("worker_seeds")
    worker_seed = worker_seeds[job_idx] if worker_seeds is not None else 42 + job_idx
    rng = np.random.default_rng(worker_seed)

    alpha_seq = tuple()
    if ensemble == "discrete":
        if sampled_sequences is not None:
            alpha_seq = sampled_sequences[s_idx]
        else:
            assert worker_sequences is not None
            alpha_seq = worker_sequences[s_idx]

    # Determinism guard: when no noise, traj_idx must be 0 and results must be identical
    if noise_model is None:
        assert num_trajectories == 1, (
            f"Expected num_trajectories=1 when noise_model is None, got {num_trajectories}. "
            "Setting num_trajectories=1 in run/run_mc_upsilon ensures deterministic evolution."
        )

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

    # We need to collect the chosen states for the continuous dual builder
    chosen_continuous_states: list[tuple[NDArray[np.complex128], NDArray[np.complex128]]] | None = [] if ensemble == "continuous" else None

    # 4. Multi-step loop
    for step_i, duration in enumerate(timesteps):
        # Apply intervention for this slot
        if ensemble == "continuous":
            if sampling == "uniform":
                assert continuous_states is not None
                psi_meas, psi_prep = continuous_states[s_idx][step_i]
                psi_proj = psi_meas
                psi_next = psi_prep

                if is_mcwf:
                    assert isinstance(current_state, np.ndarray)
                    current_state, step_prob = _reprepare_site_zero_vector_forced(
                        cast("NDArray[np.complex128]", current_state), psi_proj, psi_next
                    )
                else:
                    assert isinstance(current_state, MPS)
                    step_prob = _reprepare_site_zero_forced(current_state, psi_proj, psi_next)

                sequence_weight *= step_prob
                chosen_continuous_states.append((psi_meas, psi_prep))
            elif sampling == "candidate_local":
                assert continuous_candidates is not None
                candidates = continuous_candidates[s_idx][step_i]
                current_state, psi_meas, psi_prep, inc_weight = _sample_intervention_candidate_local(
                    current_state, rng, candidates, is_mcwf
                )
                sequence_weight *= inc_weight
                chosen_continuous_states.append((psi_meas, psi_prep))
        else:
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

    rho_final = _get_rho_site_zero(current_state)
    return (s_idx, traj_idx, tuple(alpha_seq), rho_final, sequence_weight, chosen_continuous_states)


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

    Raises:
        ValueError: If dual basis creation fails.
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

    for _job_idx, (worker_seq_idx, _traj_idx, _alpha, rho_final, sequence_weight, _seq_states) in results_iterator:
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


def _apply_dual_transform(D: NDArray[np.complex128], transform: str) -> NDArray[np.complex128]:
    """Apply a transformation to a dual matrix."""
    if transform == "id":
        return D
    if transform == "T":
        return D.T
    if transform == "conj":
        return D.conj()
    if transform == "dag":
        return D.conj().T
    raise ValueError(f"Invalid dual_transform: {transform}. Must be one of {{'id', 'T', 'conj', 'dag'}}.")


def run_mc_upsilon(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    num_sequences: int = 256,
    num_trajectories: int = 100,
    noise_model: NoiseModel | None = None,
    seed: int | None = None,
    dual_transform: str = "id",
    replace: bool = True,
    ensemble: Literal["discrete", "continuous"] = "discrete",
    sampling: str = "uniform",
    num_candidates: int = 8,
) -> tuple[NDArray[np.complex128], dict]:
    """Run Monte Carlo Process Tomography (Phase A: Uniform Sampling).

    Accumulates the Upsilon matrix directly by sampling intervention sequences.

    Args:
        operator: System evolution MPO.
        sim_params: Simulation parameters.
        timesteps: List of time durations for each evolution segment.
        num_sequences: Number of Monte Carlo sequence samples.
        num_trajectories: Number of trajectories to average per sequence (unravelling).
        noise_model: Noise model to apply.
        seed: Random seed for sampling.
        dual_transform: Transformation to apply to dual matrices (one of {"id", "T", "conj", "dag"}).
        replace: Whether to sample sequences with replacement.
        ensemble: Basis sequence ensemble to sample from ("discrete" or "continuous").
        sampling: Sampling mode ("uniform" or "candidate_local").
        num_candidates: Number of candidates for local importance sampling.

    Returns:
        tuple containing:
        - Upsilon_hat: Reconstructed Upsilon matrix (2*4^k, 2*4^k).
        - meta: Dictionary containing simulation metadata.
    """
    if timesteps is None:
        timesteps = [sim_params.elapsed_time]

    k = len(timesteps)
    rng = np.random.default_rng(seed)

    # 1. Prepare Basis and Duals
    basis_set = get_basis_states()
    choi_basis, choi_indices = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)

    # 2. Sample Sequences
    continuous_states: list[list[tuple[NDArray[np.complex128], NDArray[np.complex128]]]] | None = None
    continuous_candidates: list[list[list[tuple[NDArray[np.complex128], NDArray[np.complex128]]]]] | None = None

    if ensemble == "continuous":
        if sampling not in {"uniform", "candidate_local"}:
            raise ValueError('sampling must be "uniform" or "candidate_local"')
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")

        if sampling == "uniform":
            continuous_states = []
            for _ in range(num_sequences):
                seq_states = []
                for _ in range(k):
                    seq_states.append((_random_pure_state(rng), _random_pure_state(rng)))
                continuous_states.append(seq_states)
        elif sampling == "candidate_local":
            continuous_candidates = []
            for _ in range(num_sequences):
                seq_cands = []
                for _ in range(k):
                    step_cands = []
                    for _ in range(num_candidates):
                        step_cands.append((_random_pure_state(rng), _random_pure_state(rng)))
                    seq_cands.append(step_cands)
                continuous_candidates.append(seq_cands)

        sampled_sequences = [tuple()] * num_sequences  # Dummy sequence list
        inv_q = 1.0  # Continuous Haar duals already integrate directly to the target

    elif replace:
        # Sample with replacement
        sampled_sequences = [tuple(rng.integers(0, 16, size=k).tolist()) for _ in range(num_sequences)]
        inv_q = 16**k
    else:
        # Unique sampling
        total_possible = 16**k
        if num_sequences > total_possible:
            num_sequences = total_possible

        sampled_indices = rng.choice(total_possible, size=num_sequences, replace=False)

        # Convert indices to tuples
        sampled_sequences = []
        for idx in sampled_indices:
            seq = [0] * k
            temp_idx = idx
            for i in range(k - 1, -1, -1):
                seq[i] = int(temp_idx % 16)
                temp_idx //= 16
            sampled_sequences.append(tuple(seq))
        inv_q = 16**k

    # 3. Prepare Simulation Context
    if noise_model is None:
        noise_model = sim_params.noise_model

    if noise_model is None:
        num_trajectories = 1

    mcwf_static_ctx = None
    if sim_params.solver == "MCWF":
        dummy_mps = MPS(length=operator.length, state="zeros")
        mcwf_static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, sim_params)

    total_jobs = num_sequences * num_trajectories
    worker_seeds = rng.integers(0, 2**31 - 1, size=total_jobs).tolist()

    payload = {
        "operator": operator,
        "sim_params": sim_params,
        "timesteps": timesteps,
        "basis_set": basis_set,
        "choi_indices": choi_indices,
        "sampled_sequences": sampled_sequences,
        "continuous_states": continuous_states,
        "continuous_candidates": continuous_candidates,
        "ensemble": ensemble,
        "sampling": sampling,
        "noise_model": noise_model,
        "num_trajectories": num_trajectories,
        "mcwf_static_ctx": mcwf_static_ctx,
        "worker_seeds": worker_seeds,
    }

    # 4. Parallel Execution
    max_workers = max(1, available_cpus() - 1)

    avg_rho_weighted = np.zeros((num_sequences, 2, 2), dtype=np.complex128)

    # 5. Build Upsilon
    dim_p = 4**k
    upsilon = np.zeros((2 * dim_p, 2 * dim_p), dtype=np.complex128)

    results_iterator = run_backend_parallel(
        worker_fn=_tomography_sequence_worker,
        payload=payload,
        n_jobs=total_jobs,
        max_workers=max_workers,
        show_progress=sim_params.show_progress,
        desc=f"Simulating {num_sequences} MC Sequences",
    )

    for _job_idx, (s_idx, _traj_idx, _alpha, rho_final, sequence_weight, seq_states) in results_iterator:
        if ensemble == "continuous":
            assert seq_states is not None
            def _get_dual(psi_m: NDArray[np.complex128], psi_p: NDArray[np.complex128]) -> NDArray[np.complex128]:
                P = np.outer(psi_m, psi_m.conj())
                Q = np.outer(psi_p, psi_p.conj())
                D_Q = 2.0 * (3.0 * Q - np.eye(2, dtype=np.complex128))
                D_PT = 2.0 * (3.0 * P.T - np.eye(2, dtype=np.complex128))
                return np.kron(D_Q, D_PT).T

            past = _apply_dual_transform(_get_dual(*seq_states[0]), dual_transform)
            for psi_meas, psi_prep in seq_states[1:]:
                past = np.kron(past, _apply_dual_transform(_get_dual(psi_meas, psi_prep), dual_transform))
            
            # Form expectation per-trajectory to capture candidate dependencies correctly
            upsilon += np.kron(rho_final * sequence_weight, past) * (inv_q / (num_sequences * num_trajectories))
        else:
            avg_rho_weighted[s_idx] += rho_final * sequence_weight

    if ensemble == "discrete":
        avg_rho_weighted /= num_trajectories
        for s_idx, alpha_seq in enumerate(sampled_sequences):
            past = _apply_dual_transform(duals[alpha_seq[0]], dual_transform)
            for a in alpha_seq[1:]:
                past = np.kron(past, _apply_dual_transform(duals[a], dual_transform))
            rho_w = avg_rho_weighted[s_idx]
            upsilon += np.kron(rho_w, past) * inv_q

        upsilon /= num_sequences

    # NOTE: This is the raw estimator Υ_hat(N). Targets the POPULATION TOTAL
    # T = Σ_{all α} p(α)·(ρ_out(α) ⊗ D(α)), NOT the mean μ = T/16^k.
    # inv_q = 16^k is the Horvitz–Thompson correction; both replace=True and
    # replace=False are unbiased for T (inclusion prob = n/M for replace=False).
    # It is NOT trace-normalized or PSD-projected. Callers should apply
    # canonicalize_upsilon(...) as needed for their use case.
    meta = {
        "k": k,
        "num_sequences": num_sequences,
        "num_trajectories": num_trajectories,
        "dual_transform": dual_transform,
        "replace": replace,
        "seed": seed,
        "ensemble": ensemble,
    }

    return upsilon, meta


def _sis_evolve_worker(job_idx: int) -> tuple[int, NDArray[np.complex128]]:
    """Parallel worker: run one MCWF evolution for one SIS particle.

    Reads from WORKER_CTX:
        ``mcwf_static_ctx``: static MCWF context shared across all particles.
        ``step_params``: simulation parameters for this segment.
        ``particle_states``: list of (particle_idx, psi_initial) pairs.

    Args:
        job_idx: Index into ``particle_states``.

    Returns:
        tuple[int, NDArray]: (particle_idx, evolved_psi)
    """
    particle_idx, psi_initial = WORKER_CTX["particle_states"][job_idx]
    static_ctx = WORKER_CTX["mcwf_static_ctx"]
    step_params = WORKER_CTX["step_params"]

    dynamic_ctx = copy.copy(static_ctx)
    dynamic_ctx.psi_initial = psi_initial
    dynamic_ctx.sim_params = step_params
    mcwf((0, dynamic_ctx))
    assert dynamic_ctx.output_state is not None
    return particle_idx, dynamic_ctx.output_state  # type: ignore[return-value]


def run_sis_upsilon(
    operator: MPO,
    sim_params: AnalogSimParams,
    timesteps: list[float] | None = None,
    num_particles: int = 256,
    noise_model: NoiseModel | None = None,
    dual_transform: str = "T",
    seed: int | None = None,
    ess_threshold: float = 0.5,
    resample: bool = True,
    proposal: str = "local",
    mixture_eps: float = 0.1,
    eps_schedule: list[float] | None = None,
    floor_eps: float = 0.0,
    stratify_step1: bool = True,
    parallel: bool = True,
    rejuvenate: bool = False,
    rejuvenate_exact: bool = False,
) -> tuple[NDArray[np.complex128], dict]:
    """Run Sequential Importance Sampling (SIS/SMC) Process Tomography.

    Propagates ``num_particles`` particles through k timesteps. At each step,
    alpha_t is sampled from proposal q_t and importance weights are updated.

    **Proposal options** (``proposal``):

    - ``"uniform"``: q_t = 1/16.  Weight = 16·p_t.  High variance.
    - ``"local"``: q_t ∝ p_t.  Weight = Z_t (marginal likelihood).  ESS ≈ N
      but particles can cluster (low path diversity).  Use ``stratify_step1``
      or ``floor_eps`` to maintain coverage.
    - ``"mixture"``: q_t = (1-ε)·q_opt + ε/16.  Balances efficiency and
      diversity. ``eps_schedule`` allows per-step decay, e.g. [0.3, 0.1, 0.05].

    **Diversity controls**:

    - ``stratify_step1=True`` (default): allocates ``N//16`` particles per
      alpha_1, guaranteeing all basis elements are covered at step 0.
    - ``floor_eps > 0``: enforces q_t(alpha) >= floor_eps/16 for all alphas
      before renormalizing.  Prevents complete clustering without the full
      overhead of the mixture proposal.
    - ``eps_schedule``: per-step epsilon list for "mixture" proposal.  If
      shorter than k, the last value is repeated.

    **Parallel evolution** (``parallel=True``, default): uses
    ``run_backend_parallel`` to evolve all N particles simultaneously per step.
    For N ≤ 32 on Windows, sequential is preferred to avoid spawn overhead;
    the auto-threshold is applied automatically.

    **Rejuvenation** (``rejuvenate=True``): after resampling, propose an
    alpha_τ flip at a random historical step and accept/reject via MH.

    .. warning::
        For k > 2, ``rejuvenate=True`` uses a **single-step MH ratio
        approximation**, which may bias the estimator. For exact MH, set
        ``rejuvenate_exact=True`` (only practical for k ≤ 3 as it requires
        full path re-evolution from the checkpoint at τ).

    Args:
        operator: System evolution MPO.
        sim_params: Simulation parameters (must use solver="MCWF").
        timesteps: List of segment durations.
        num_particles: Number of SMC particles.
        noise_model: If None, evolution is deterministic (no jumps) and MCWF
            reduces to exact Schrödinger evolution. With ``noise_model`` set,
            each particle carries **one stochastic trajectory**; k=1 exactness
            via ``stratify_step1`` only removes α₁ sampling variance, not
            trajectory variance. For noisy systems, increase ``num_particles``
            to average over both sources of variance.
        dual_transform: One of "id", "T", "conj", "dag".
        seed: Random seed.
        ess_threshold: Resample when ESS < ess_threshold * N.
        resample: Enable systematic resampling.
        proposal: "uniform", "local", or "mixture".
        mixture_eps: Uniform floor for mixture proposal (constant ε).
        eps_schedule: Per-step epsilon schedule for "mixture" (overrides mixture_eps).
        floor_eps: Enforce q_t(a) >= floor_eps/16 for all a (post-renorm).
        stratify_step1: Allocate N//16 particles per alpha_1 at step 0.
        parallel: Use run_backend_parallel for particle evolution.
        rejuvenate: Enable MH rejuvenation after resampling.
        rejuvenate_exact: Use exact path-weight MH (only practical for k≤3).

    Returns:
        tuple:
        - upsilon: Raw Υ_hat (NOT trace-normalized).
        - meta: Dict with ess_history, weight_cv, max_mean_ratio, unique_paths,
                alpha_entropy, resampling_steps, total_weight, trace_upsilon_raw.
    """
    if timesteps is None:
        timesteps = [sim_params.elapsed_time]
    k = len(timesteps)
    rng = np.random.default_rng(seed)

    if proposal not in ("uniform", "local", "mixture"):
        raise ValueError(f"proposal must be 'uniform', 'local', or 'mixture', got '{proposal}'.")

    # Resolve noise model; enforce determinism when no noise (item 6)
    if noise_model is None:
        noise_model = sim_params.noise_model
    if noise_model is None:
        pass  # num_trajectories=1 enforced implicitly (MCWF with no noise is deterministic)

    if sim_params.solver != "MCWF":
        raise ValueError(f"run_sis_upsilon requires solver='MCWF', got '{sim_params.solver}'.")

    # ── Basis ──────────────────────────────────────────────────────────────────
    basis_set = get_basis_states()
    choi_basis, choi_indices = get_choi_basis()
    duals = calculate_dual_choi_basis(choi_basis)
    dual_mats = [_apply_dual_transform(duals[a], dual_transform) for a in range(16)]

    # ── Static MCWF context ────────────────────────────────────────────────────
    dummy_mps = MPS(length=operator.length, state="zeros")
    static_ctx = preprocess_mcwf(dummy_mps, operator, noise_model, sim_params)

    n_hilbert = 2**operator.length
    dim_p = 4**k
    dual_dim = duals[0].shape[0]  # 4

    # ── Initialize particles ───────────────────────────────────────────────────
    psi_zero = np.zeros(n_hilbert, dtype=np.complex128)
    psi_zero[0] = 1.0

    states: NDArray[np.complex128] = np.tile(psi_zero, (num_particles, 1))
    weights = np.ones(num_particles, dtype=np.float64)
    # explicit list comprehension avoids list-multiplication aliasing
    past_ops: list[NDArray[np.complex128]] = [np.eye(dual_dim, dtype=np.complex128) for _ in range(num_particles)]
    alpha_seqs: list[list[int]] = [[] for _ in range(num_particles)]

    # Checkpoints for rejuvenation: state just BEFORE reprepare at each step
    # checkpoints[t][i] = psi of particle i just before step t's reprepare
    checkpoints: list[NDArray[np.complex128]] = []  # length = k

    # Per-step diagnostics
    ess_history: list[float] = []
    weight_cv: list[float] = []
    max_mean_ratio: list[float] = []
    unique_paths: list[int] = []
    alpha_entropy: list[float] = []   # Shannon entropy of alpha_t histogram (bits; max=4)
    resampling_steps: list[int] = []
    trace_before_resampling: list[float] = []

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _probs_for_state(psi: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Compute step_prob for all 16 alphas on a single state (scalar)."""
        dim_env = psi.shape[0] // 2
        psi_r = psi.reshape(2, dim_env)
        probs = np.empty(16, dtype=np.float64)
        for a in range(16):
            psi_proj = basis_set[choi_indices[a][1]][1]
            env_vec = psi_proj.conj() @ psi_r
            probs[a] = float(np.linalg.norm(env_vec)**2)
        return probs

    # Precompute site-0 projectors for all 16 interventions: shape (16, 2)
    _projs_batch = np.array(
        [basis_set[choi_indices[a][1]][1] for a in range(16)], dtype=np.complex128
    )  # (16, 2)

    def _probs_for_all_particles(psis: NDArray[np.complex128]) -> NDArray[np.float64]:
        """Vectorized proposal probabilities for all N particles simultaneously.

        Args:
            psis: (N, n_hilbert) state array.

        Returns:
            probs: (N, 16) float64 array of step_prob per particle per alpha.
        """
        dim_env = psis.shape[1] // 2
        psi_r = psis.reshape(-1, 2, dim_env)           # (N, 2, dim_env)
        # env[n, a, e] = Σ_s proj[a, s].conj * psi_r[n, s, e]
        env = np.einsum('as,nse->nae', _projs_batch.conj(), psi_r)  # (N, 16, dim_env)
        return (np.abs(env) ** 2).sum(axis=-1)          # (N, 16)

    def _apply_floor(q: NDArray[np.float64]) -> NDArray[np.float64]:
        """Enforce q_t(alpha) >= floor_eps/16 then renormalize. Returns new array."""
        if floor_eps > 0:
            q = np.maximum(q, floor_eps / 16.0)
            q = q / q.sum()  # new array, never in-place
        return q

    def _safe_normalize(q: NDArray[np.float64]) -> NDArray[np.float64] | None:
        """Normalize q to a valid probability vector, or return None on failure.

        Returns None when q is all-zero, overflows, or contains non-finite values.
        Callers should fall back to uniform sampling when None is returned.
        """
        s = float(q.sum())
        if not np.isfinite(s) or s < 1e-300:
            return None
        qn = q / s
        if np.any(~np.isfinite(qn)):
            return None
        return qn

    def _evolve(psi: NDArray[np.complex128], duration: float) -> NDArray[np.complex128]:
        """Run one MCWF evolution segment deterministically."""
        sp = copy.deepcopy(sim_params)
        sp.elapsed_time = duration
        sp.dt = sim_params.dt
        sp.num_traj = 1
        sp.show_progress = False
        sp.get_state = True
        n_steps = max(1, int(np.round(duration / sp.dt)))
        sp.times = np.linspace(0, n_steps * sp.dt, n_steps + 1)
        dctx = copy.copy(static_ctx)
        dctx.psi_initial = psi
        dctx.sim_params = sp
        mcwf((0, dctx))
        assert dctx.output_state is not None
        return dctx.output_state  # type: ignore[return-value]

    def _reprepare_and_evolve(psi: NDArray[np.complex128],
                              alpha_t: int, duration: float) -> tuple[NDArray[np.complex128], float]:
        """One full intervention + evolution step. Returns (new_state, step_prob)."""
        p_idx, m_idx = choi_indices[alpha_t]
        psi_new, sp = _reprepare_site_zero_vector_forced(psi, basis_set[m_idx][1], basis_set[p_idx][1])
        psi_evolved = _evolve(psi_new, duration)
        return psi_evolved, sp

    def _full_path_weight(particle_psi_start: NDArray[np.complex128],
                          alphas: list[int]) -> float:
        """Compute cumulative step_prob for a complete alpha sequence from psi_start."""
        psi = particle_psi_start
        total = 1.0
        for t_idx, a in enumerate(alphas):
            p_all_a = _probs_for_state(psi)
            total *= p_all_a[a]
            if total < 1e-50:
                return 0.0
            psi_new, sp = _reprepare_site_zero_vector_forced(psi, basis_set[choi_indices[a][1]][1], basis_set[choi_indices[a][0]][1])
            psi = _evolve(psi_new, timesteps[t_idx])
        return total

    # ── resolve per-step epsilon schedule for 'mixture' ──────────────────────
    def _eps_for_step(t: int) -> float:
        if eps_schedule:
            return float(eps_schedule[min(t, len(eps_schedule) - 1)])
        return mixture_eps

    # ── build step_params template (same for every evolution segment) ─────────
    # (populated per-step inside the loop so duration can vary)
    max_workers = max(1, available_cpus() - 1)
    use_parallel = parallel and num_particles > 32

    # ── SMC loop ───────────────────────────────────────────────────────────────
    for step_i, duration in enumerate(timesteps):
        # Save checkpoint: states BEFORE reprepare at this step
        checkpoints.append(states.copy())

        eps_t = _eps_for_step(step_i)

        # ── Stratified initialisation at step 0 (item B) ─────────────────────
        # Allocate N//16 particles per alpha_1 to guarantee full basis coverage.
        if step_i == 0 and stratify_step1:
            base_count = num_particles // 16
            remainder = num_particles - 16 * base_count
            # Build deterministic alpha_t assignments
            stratified_alphas: list[int] = []
            for a_val in range(16):
                count = base_count + (1 if a_val < remainder else 0)
                stratified_alphas.extend([a_val] * count)
            # Shuffle so particle ordering is random
            rng.shuffle(stratified_alphas)  # type: ignore[arg-type]
            forced_alpha0 = stratified_alphas
        else:
            forced_alpha0 = None

        new_states = np.empty_like(states)
        reprepared: list[NDArray[np.complex128]] = []  # psi_new before evolution

        # Batch-compute all step_probs in one einsum: (N, 16) — avoids 16N Python loop
        P_all = _probs_for_all_particles(states)  # (N, 16)

        for i in range(num_particles):
            psi = states[i]
            p_all = P_all[i]          # already (16,) float64, same values as _probs_for_state
            z_t = float(p_all.sum())

            if forced_alpha0 is not None:
                # Stratified: forced choice of alpha
                alpha_t = forced_alpha0[i]
                p_idx, m_idx = choi_indices[alpha_t]
                psi_new, step_prob = _reprepare_site_zero_vector_forced(
                    psi, basis_set[m_idx][1], basis_set[p_idx][1])
                # q = 1/16 (stratified allocation) → w *= p/(1/16)
                weights[i] *= step_prob / max(1.0 / 16.0, 1e-30)

            elif proposal == "uniform":
                alpha_t = int(rng.integers(0, 16))
                p_idx, m_idx = choi_indices[alpha_t]
                psi_new, step_prob = _reprepare_site_zero_vector_forced(
                    psi, basis_set[m_idx][1], basis_set[p_idx][1])
                weights[i] *= step_prob * 16.0

            elif proposal == "local":
                if floor_eps > 0:
                    qn = _safe_normalize(_apply_floor(
                        p_all / z_t if z_t > 1e-300 else np.full(16, 1.0 / 16.0)))
                else:
                    qn = _safe_normalize(p_all)  # unnorm OK; _safe_normalize normalizes

                if qn is None:
                    # Pathological state: uniform fallback, weight → 0
                    alpha_t = int(rng.integers(0, 16))
                    weights[i] *= p_all[alpha_t] / max(1.0 / 16.0, 1e-30)
                elif floor_eps > 0:
                    # floored proposal ≠ p/z → exact p/q
                    alpha_t = int(rng.choice(16, p=qn))
                    weights[i] *= p_all[alpha_t] / max(float(qn[alpha_t]), 1e-30)
                else:
                    # pure local: q = p/z → w *= z_t (variance-free)
                    alpha_t = int(rng.choice(16, p=qn))
                    weights[i] *= z_t

                p_idx, m_idx = choi_indices[alpha_t]
                psi_new, _ = _reprepare_site_zero_vector_forced(
                    psi, basis_set[m_idx][1], basis_set[p_idx][1])

            else:  # mixture
                if z_t > 1e-300:
                    q_opt = p_all / z_t
                    q_mix = (1.0 - eps_t) * q_opt + eps_t / 16.0
                    q_mix = _apply_floor(q_mix)
                    qn = _safe_normalize(q_mix)
                else:
                    qn = None  # dead particle

                if qn is None:
                    # Pathological state: uniform fallback
                    alpha_t = int(rng.integers(0, 16))
                    q_alpha = 1.0 / 16.0
                else:
                    alpha_t = int(rng.choice(16, p=qn))
                    q_alpha = float(qn[alpha_t])

                p_idx, m_idx = choi_indices[alpha_t]
                psi_new, step_prob = _reprepare_site_zero_vector_forced(
                    psi, basis_set[m_idx][1], basis_set[p_idx][1])
                weights[i] *= step_prob / max(q_alpha, 1e-30)

            if step_i == 0:
                past_ops[i] = dual_mats[alpha_t].copy()
            else:
                past_ops[i] = np.kron(past_ops[i], dual_mats[alpha_t])

            alpha_seqs[i].append(alpha_t)
            reprepared.append(psi_new)

        # ── Evolve particles (parallel or sequential) ─────────────────────────
        n_steps_seg = max(1, int(np.round(duration / sim_params.dt)))
        step_params_seg = copy.deepcopy(sim_params)
        step_params_seg.elapsed_time = duration
        step_params_seg.num_traj = 1
        step_params_seg.show_progress = False
        step_params_seg.get_state = True
        step_params_seg.times = np.linspace(0, n_steps_seg * sim_params.dt, n_steps_seg + 1)

        alive_jobs = [
            (i, reprepared[i])
            for i in range(num_particles)
            if np.linalg.norm(reprepared[i]) > 1e-30
        ]

        if use_parallel and alive_jobs:
            evo_payload = {
                "mcwf_static_ctx": static_ctx,
                "step_params": step_params_seg,
                "particle_states": alive_jobs,
            }
            results_iter = run_backend_parallel(
                worker_fn=_sis_evolve_worker,
                payload=evo_payload,
                n_jobs=len(alive_jobs),
                max_workers=max_workers,
                show_progress=False,
                desc="",
            )
            new_states = np.tile(np.zeros_like(reprepared[0]), (num_particles, 1)).reshape(num_particles, -1)
            for _, (particle_idx, evolved_psi) in results_iter:
                new_states[particle_idx] = evolved_psi
        else:
            new_states = np.empty_like(states)
            for i, psi_new in enumerate(reprepared):
                if np.linalg.norm(psi_new) > 1e-30:
                    new_states[i] = _evolve(psi_new, duration)
                else:
                    new_states[i] = psi_new

        states = new_states

        # ── Per-step diagnostics ──────────────────────────────────────────────
        w_sum = float(weights.sum())
        if w_sum < 1e-30:
            weights[:] = 1.0
            states = np.tile(psi_zero, (num_particles, 1))
            past_ops = [np.eye(dual_dim, dtype=np.complex128) for _ in range(num_particles)]  # no aliasing
            alpha_seqs = [[] for _ in range(num_particles)]
            checkpoints.clear()
            ess_history.append(float(num_particles))
            weight_cv.append(0.0)
            max_mean_ratio.append(1.0)
            unique_paths.append(num_particles)
            alpha_entropy.append(4.0)  # max entropy as sentinel
            trace_before_resampling.append(0.0)
            continue

        w_norm = weights / w_sum
        ess = float(1.0 / np.sum(w_norm**2))
        w_mean = float(weights.mean())
        cv = float(weights.std() / max(w_mean, 1e-30))
        mm = float(weights.max() / max(w_mean, 1e-30))
        n_unique = len({tuple(s) for s in alpha_seqs})

        # Alpha histogram entropy at this step (bits; max = log2(16) = 4)
        step_alphas = [seq[step_i] for seq in alpha_seqs]
        counts = np.bincount(step_alphas, minlength=16).astype(float)
        counts_norm = counts / counts.sum()
        ent = float(-np.sum(counts_norm * np.log2(np.where(counts_norm > 0, counts_norm, 1))))

        ess_history.append(ess)
        weight_cv.append(cv)
        max_mean_ratio.append(mm)
        unique_paths.append(n_unique)
        alpha_entropy.append(ent)
        trace_before_resampling.append(w_sum / num_particles)

        # ── Resampling ────────────────────────────────────────────────────────
        do_resample = resample and ess < ess_threshold * num_particles
        if do_resample:
            positions = (rng.random() + np.arange(num_particles)) / num_particles
            cumsum = np.cumsum(w_norm)
            idxs = np.searchsorted(cumsum, positions)
            states = states[idxs]
            past_ops = [past_ops[j] for j in idxs]
            alpha_seqs = [alpha_seqs[j][:] for j in idxs]
            checkpoints = [cp[idxs] for cp in checkpoints]
            weights[:] = w_sum / num_particles
            resampling_steps.append(step_i)

            # ── MH Rejuvenation ───────────────────────────────────────────────
            if rejuvenate and step_i > 0:
                n_accepted = 0
                for i in range(num_particles):
                    tau = int(rng.integers(0, step_i + 1))
                    alpha_old = alpha_seqs[i][tau]
                    alpha_new = int(rng.integers(0, 16))
                    if alpha_new == alpha_old:
                        continue

                    psi_at_tau = checkpoints[tau][i]
                    p_all_at_tau = _probs_for_state(psi_at_tau)
                    w_old = p_all_at_tau[alpha_old]
                    w_new = p_all_at_tau[alpha_new]

                    # Exact MH: full re-evolution from checkpoint τ, using new α_τ'
                    # then propagating forward with the same future alphas {α_{τ+1},...}
                    # to obtain the exact forward path weight ratio.
                    # Note: this requires O(step_i - tau) additional _evolve calls;
                    #       only feasible for k ≤ 3.
                    # did_exact: track whether we computed psi_curr_new (guards against
                    # NameError when w_old <= 1e-30 with rejuvenate_exact=True)
                    did_exact = rejuvenate_exact and (w_old > 1e-30)
                    if did_exact:
                        # Re-evolve forward from checkpoint[tau] with alpha_new,
                        # then propagate future steps using the SAME future alphas.
                        # Convention: psi_curr_new is always the POST-EVOLVE state
                        # entering step tau2, i.e. pre-reprepare for tau2. This
                        # matches the main loop's checkpoint convention.
                        # ── Forward path weight for NEW proposal ─────────────────
                        p_idx_n, m_idx_n = choi_indices[alpha_new]
                        psi_curr_new, _ = _reprepare_site_zero_vector_forced(
                            psi_at_tau, basis_set[m_idx_n][1], basis_set[p_idx_n][1])
                        if np.linalg.norm(psi_curr_new) > 1e-30:
                            psi_curr_new = _evolve(psi_curr_new, timesteps[tau])
                        w_new_full = w_new  # p_tau(alpha_new)
                        for tau2 in range(tau + 1, step_i + 1):
                            p2_new = _probs_for_state(psi_curr_new)
                            a2 = alpha_seqs[i][tau2]
                            w_new_full *= p2_new[a2]
                            if w_new_full < 1e-50:
                                break
                            # Continue propagating for subsequent steps
                            if tau2 < step_i:
                                p2_idx, m2_idx = choi_indices[a2]
                                psi_next_new, _ = _reprepare_site_zero_vector_forced(
                                    psi_curr_new, basis_set[m2_idx][1], basis_set[p2_idx][1])
                                psi_curr_new = _evolve(psi_next_new, timesteps[tau2]) if np.linalg.norm(psi_next_new) > 1e-30 else psi_next_new

                        # ── Forward path weight for OLD alpha (re-evolve too) ────
                        p_idx_o, m_idx_o = choi_indices[alpha_old]
                        psi_curr_old, _ = _reprepare_site_zero_vector_forced(
                            psi_at_tau, basis_set[m_idx_o][1], basis_set[p_idx_o][1])
                        if np.linalg.norm(psi_curr_old) > 1e-30:
                            psi_curr_old = _evolve(psi_curr_old, timesteps[tau])
                        w_old_full = w_old  # p_tau(alpha_old)
                        for tau2 in range(tau + 1, step_i + 1):
                            p2_old = _probs_for_state(psi_curr_old)
                            a2 = alpha_seqs[i][tau2]
                            w_old_full *= p2_old[a2]
                            if w_old_full < 1e-50:
                                break
                            if tau2 < step_i:
                                p2_idx, m2_idx = choi_indices[a2]
                                psi_next_old, _ = _reprepare_site_zero_vector_forced(
                                    psi_curr_old, basis_set[m2_idx][1], basis_set[p2_idx][1])
                                psi_curr_old = _evolve(psi_next_old, timesteps[tau2]) if np.linalg.norm(psi_next_old) > 1e-30 else psi_next_old

                        accept_prob = min(1.0, w_new_full / max(w_old_full, 1e-50))
                    else:
                        # ⚠ Approximate: single-step ratio only.  May bias estimator.
                        accept_prob = min(1.0, w_new / max(w_old, 1e-50))

                    if rng.random() < accept_prob:
                        alpha_seqs[i][tau] = alpha_new
                        d_t = dual_mats[alpha_seqs[i][0]].copy()
                        for t2 in range(1, step_i + 1):
                            d_t = np.kron(d_t, dual_mats[alpha_seqs[i][t2]])
                        past_ops[i] = d_t

                        # Also update states[i] to the re-evolved terminal state.
                        # After the weight loop, psi_curr_new is the state BEFORE
                        # reprep at step_i (it's the post-evolve state from step_i-1).
                        # Apply the step_i reprep+evolve with the same alpha_{step_i}.
                        if did_exact:
                            a_term = alpha_seqs[i][step_i]
                            pt_idx, mt_idx = choi_indices[a_term]
                            psi_term, _ = _reprepare_site_zero_vector_forced(
                                psi_curr_new, basis_set[mt_idx][1], basis_set[pt_idx][1])
                            states[i] = _evolve(psi_term, timesteps[step_i]) if np.linalg.norm(psi_term) > 1e-30 else psi_term

                        n_accepted += 1

    # ── Build Upsilon ──────────────────────────────────────────────────────────
    # Υ_hat = (1/N) Σ_i w_i * kron(ρ_i, past_i)
    # This targets the POPULATION TOTAL:
    #   T = Σ_{all α} p(α) · (ρ_out(α) ⊗ D(α))
    # NOT the mean μ = T / 16^k. Both MC and SIS use inv_q = 16^k (or Z_t for
    # local) so they are both unbiased estimators of T. Callers that compare
    # against a full-enumeration reference Υ_ref should ensure Υ_ref is also
    # the raw total (not divided by 16^k).
    upsilon = np.zeros((2 * dim_p, 2 * dim_p), dtype=np.complex128)
    for i in range(num_particles):
        rho_mat = np.reshape(states[i], (2, -1))
        rho_i = rho_mat @ rho_mat.conj().T
        upsilon += weights[i] * np.kron(rho_i, past_ops[i])
    upsilon /= float(num_particles)

    meta = {
        "k": k,
        "num_particles": num_particles,
        "proposal": proposal,
        "mixture_eps": mixture_eps,
        "dual_transform": dual_transform,
        "ess_threshold": ess_threshold,
        "resampling_steps": resampling_steps,
        "ess_history": ess_history,
        "weight_cv": weight_cv,
        "max_mean_ratio": max_mean_ratio,
        "unique_paths": unique_paths,
        "alpha_entropy": alpha_entropy,
        "trace_before_resampling": trace_before_resampling,
        "total_weight": float(weights.sum()) / num_particles,
        "trace_upsilon_raw": float(np.trace(upsilon).real),
        "seed": seed,
    }
    return upsilon, meta

