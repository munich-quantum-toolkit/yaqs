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
import opt_einsum as oe
from qiskit.quantum_info import DensityMatrix, entropy

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
import mqt.yaqs.simulator as simulator

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.networks import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel


def _vec_to_rho(vec4: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Convert a 4-element vector to a 2x2 density matrix."""
    rho = vec4.reshape(2, 2)
    rho = 0.5 * (rho + rho.conj().T)
    tr = np.trace(rho)
    if abs(tr) > 1e-15:
        rho = rho / tr
    return rho


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
    dim_sq = dim * dim
    
    # F matrix: columns are vectorized density matrices
    F = np.column_stack([m.reshape(-1) for m in basis_matrices])
    
    # Calculate dual frame using Moore-Penrose pseudoinverse
    # F_dual = (F F^dag)^-1 F  if F is invertible/overcomplete? 
    # Actually, simply D = (F^dag)+ ?
    # Let's verify: We want Rho = sum_k Tr(D_k^dag Rho) F_k
    # Vectorized: |Rho>> = sum_k (<<D_k|Rho>>) |F_k>>
    #                    = sum_k |F_k>> <<D_k| |Rho>>
    # So we need sum_k |F_k>> <<D_k| = Identity.
    # Matrix form: F * D^dag = I.
    # So D^dag = F_pinv.
    # D = (F_pinv)^dag.
    
    F_pinv = np.linalg.pinv(F)
    D_dag = F_pinv
    D = D_dag.conj().T
    
    # Unpack columns of D into matrices
    duals = [D[:, k].reshape(dim, dim) for k in range(D.shape[1])]
    return duals


def _reprepare_site_zero_pure_env_approx(mps: MPS, new_state: np.ndarray) -> None:
    """Reprepare site 0 using pure-state approximation of environment.
    
    **WARNING**: This is an approximation! It approximates the unconditional environment
    marginal by a single pure vector. This is exact only if the environment marginal
    is rank-1 (pure). For mixed environment states, this introduces error.
    
    For exact nonselective operation, use MPO (mixed-state) representation.
    For trajectory-consistent operation, use _reprepare_site_zero_selective.
    
    Args:
        mps: The MPS to modify (modified in-place)
        new_state: The new state vector for site 0, shape (d,)
    """
    # Shift orthogonality center to site 0 for clean bond structure
    # Cost: O(L) per intervention; can be optimized by tracking center
    for i in range(mps.length - 1, 0, -1):
        mps.shift_orthogonality_center_left(i, decomposition="QR")
    
    old_tensor = mps.tensors[0]  # shape (d, 1, chi)
    d, left_bond, chi = old_tensor.shape
    
    # Assertions for gauge and shape consistency
    assert left_bond == 1, f"Expected left bond=1 after shifting to site 0, got {left_bond}"
    assert d == new_state.shape[0], f"Physical dim mismatch: tensor={d}, state={new_state.shape[0]}"
    
    # Extract environment vectors for each physical state
    env_vecs = old_tensor[:, 0, :]  # shape (d, chi)
    
    # Compute weights from current physical state (probabilities)
    weights = np.linalg.norm(env_vecs, axis=1) ** 2
    weights = weights / (np.sum(weights) + 1e-15)  # Normalize
    
    # Average environment vector (pure-state approximation of mixed marginal)
    avg_env = np.zeros(chi, dtype=np.complex128)
    for s in range(d):
        if weights[s] > 1e-12:
            env_normalized = env_vecs[s] / (np.linalg.norm(env_vecs[s]) + 1e-15)
            avg_env += np.sqrt(weights[s]) * env_normalized
    
    # Normalize the average environment
    avg_env = avg_env / (np.linalg.norm(avg_env) + 1e-15)
    
    # Construct new tensor: new_state[s] * avg_env for each s
    new_tensor = np.zeros((d, 1, chi), dtype=np.complex128)
    for s in range(d):
        new_tensor[s, 0, :] = new_state[s] * avg_env
    
    mps.tensors[0] = new_tensor
    
    # Renormalize MPS to prevent norm drift
    norm = mps.norm()
    if abs(norm) > 1e-15:
        mps.tensors[0] = mps.tensors[0] / norm


def _reprepare_site_zero_selective(mps: MPS, new_state: np.ndarray, rng: np.random.Generator) -> int:
    """Reprepare site 0 with selective (trajectory-resolved) intervention.
    
    This implements a clean trajectory-consistent operation:
    1. Sample measurement outcome m with Born probabilities
    2. Project onto |m⟩ and keep conditional environment branch
    3. Reprepare to new_state while preserving the branch's bond vector
    
    Args:
        mps: The MPS to modify (modified in-place)
        new_state: The new state vector for site 0, shape (d,)
        rng: Random number generator for sampling
        
    Returns:
        int: The measurement outcome (0 or 1 for qubits)
    """
    # Shift orthogonality center to site 0
    for i in range(mps.length - 1, 0, -1):
        mps.shift_orthogonality_center_left(i, decomposition="QR")
    
    old_tensor = mps.tensors[0]  # shape (d, 1, chi)
    d, left_bond, chi = old_tensor.shape
    
    # Assertions
    assert left_bond == 1, f"Expected left bond=1, got {left_bond}"
    assert d == new_state.shape[0], f"Physical dim mismatch"
    
    # Extract environment vectors (rows of the site-0 tensor)
    env_vecs = old_tensor[:, 0, :]  # shape (d, chi)
    
    # Compute Born probabilities from tensor norms (valid if center is at 0)
    probs_norm = np.linalg.norm(env_vecs, axis=1) ** 2
    probs_norm = probs_norm / (np.sum(probs_norm) + 1e-15)
    
    # --- DEBUG: Verify matching with global expectation ---
    if np.abs(mps.norm() - 1.0) > 1e-6:
         print(f"WARNING: MPS norm deviates from 1.0 before intervention: {mps.norm():.6f}")

    # Calculate <Z> on site 0 => P0 - P1
    z_expect = mps.expect(Observable(Z(), sites=[0]))
    p0_expect = 0.5 * (1.0 + z_expect)
    p0_norm = probs_norm[0]
    
    if abs(p0_norm - p0_expect) > 1e-8:
        print(f"WARNING: Probability mismatch! Norm={p0_norm:.6f}, Expect={p0_expect:.6f}")
    # ----------------------------------------------------
    
    # Sample outcome
    outcome = rng.choice(d, p=probs_norm)
    
    # --- Projective Collapse ---
    # Zero out the non-outcome branch
    mask = np.zeros(d, dtype=complex)
    mask[outcome] = 1.0
    
    # Apply projection to tensor
    mps.tensors[0] = old_tensor * mask[:, np.newaxis, np.newaxis]
    
    # Renormalize the entire MPS to fix the norm of the collapsed branch
    # This divides by ||env_outcome|| effectively
    norm = mps.norm()
    if norm < 1e-15:
         raise RuntimeError("Collapsed state has zero norm.")
    
    mps.tensors[0] /= norm
    
    # --- Reprepare ---
    # The state is now |outcome> \otimes |env_conditional_normalized>
    # We want to replace it with |new_state> \otimes |env_conditional_normalized>
    # The environment vector is sitting in mps.tensors[0][outcome, 0, :]
    env_vec_cond = mps.tensors[0][outcome, 0, :]
    
    # Construct new tensor
    new_tensor = np.zeros((d, 1, chi), dtype=np.complex128)
    for s in range(d):
        new_tensor[s, 0, :] = new_state[s] * env_vec_cond
        
    mps.tensors[0] = new_tensor
    
    # Final Renormalization (Critical Fix)
    # Ensure global norm is 1 after inserting new state
    final_norm = mps.norm()
    if abs(final_norm) > 1e-15:
        mps.tensors[0] /= final_norm
    
    return outcome


def _reconstruct_state(expectations: dict[str, float]) -> NDArray[np.complex128]:
    """Reconstructs single-qubit density matrix from Pauli expectations.

    rho = 0.5 * (I + <X>X + <Y>Y + <Z>Z)
    """
    I = np.eye(2, dtype=complex)
    x = X().matrix
    y = Y().matrix
    z = Z().matrix
    
    rho = 0.5 * (I + expectations["x"] * x + expectations["y"] * y + expectations["z"] * z)
    return rho


def run(operator: MPO, sim_params: AnalogSimParams) -> NDArray[np.complex128]:
    """Run process tomography for the given operator and parameters.

    Args:
        operator: The MPO describing the system evolution.
        sim_params: Simulation parameters. WARNING: Observables in these params 
                    will be ignored/overridden for the tomography process.

    Returns:
        NDArray: The process tensor Lambda defining the map E(rho) = Tr_env(U rho_sys x 0_env U^dag).
                 The tensor is constructed such that E(rho) = sum_k Tr(D_k^dag rho) rho_out_k.
    """
    length = operator.length
    if length < 1:
        msg = "Operator must have at least length 1."
        raise ValueError(msg)
    
    print("Starting Process Tomography...")
    print(f"System size: {length}")
    print("Preparing 6 basis states (Z, X, Y)...")

    # 1. Prepare Basis States
    basis_set = _get_basis_states()
    basis_names = [b[0] for b in basis_set]
    basis_rhos = [b[2] for b in basis_set]
    
    # 2. Calculate Dual Frame
    duals = _calculate_dual_frame(basis_rhos)
    
    output_rhos = []

    # 3. Evolution Loop
    for i, (name, _, _) in enumerate(basis_set):
        print(f"Simulating basis state {i+1}/6: {name}")
        
        # Prepare MPS: Site 0 is basis state, others |0>
        # We use the state name for init if supported, or manually set tensor
        mps = MPS(length=length, state="zeros")
        
        target_state_mps = MPS(length=1, state=name)
        # Ensure shape (2, 1, 1) and type
        mps.tensors[0] = target_state_mps.tensors[0].astype(np.complex128).reshape(2, 1, 1)
        
        # Set up observables for X, Y, Z on site 0
        tomo_params = copy.deepcopy(sim_params)
        tomo_params.observables = [
            Observable(X(), sites=[0]),
            Observable(Y(), sites=[0]),
            Observable(Z(), sites=[0]),
        ]
        tomo_params.sorted_observables = tomo_params.observables

        # Run Simulation
        simulator.run(mps, operator, tomo_params)
        
        # Collect Results (expectations at final time)
        res_x = tomo_params.observables[0].results[-1]
        res_y = tomo_params.observables[1].results[-1]
        res_z = tomo_params.observables[2].results[-1]
        
        expectations = {"x": res_x, "y": res_y, "z": res_z}
        rho_out = _reconstruct_state(expectations)
        output_rhos.append(rho_out)

    # 4. Construct Process Tensor
    # S = sum_k |rho_out_k>> <<D_k|
    dim = 2
    superop = np.zeros((4, 4), dtype=complex)
    for rho_out, dual in zip(output_rhos, duals):
        rho_vec = rho_out.reshape(-1, 1) # column vector
        dual_vec = dual.reshape(-1, 1)
        superop += rho_vec @ dual_vec.conj().T
        
    print("Process Tomography Complete.")
    
    # 5. Simple Verification (Log fidelity with Identity if nearly identity)
    
    # Test with a random state (not in basis)
    test_psi = np.array([np.cos(0.4), np.exp(1j*0.2)*np.sin(0.4)])
    test_rho_in = np.outer(test_psi, test_psi.conj())
    
    # Predict using Process Tensor
    test_rho_vec = test_rho_in.reshape(-1)
    predicted_rho_vec = superop @ test_rho_vec
    predicted_rho = predicted_rho_vec.reshape(2, 2)
    
    # Run Actual Simulation for this state
    print("Verifying with a test state...")
    mps_test = MPS(length=length, state="zeros")

    # Manually set first qubit
    tensor = np.expand_dims(test_psi, axis=(1, 2)) # (2, 1, 1)
    mps_test.tensors[0] = tensor
    
    test_params = copy.deepcopy(sim_params)
    test_params.observables = [
        Observable(X(), sites=[0]),
        Observable(Y(), sites=[0]),
        Observable(Z(), sites=[0]),
    ]
    test_params.sorted_observables = test_params.observables
    simulator.run(mps_test, operator, test_params)

    rx = test_params.observables[0].results[-1]
    ry = test_params.observables[1].results[-1]
    rz = test_params.observables[2].results[-1]
    actual_rho = _reconstruct_state({"x": rx, "y": ry, "z": rz})
    
    # Compute Frobenius distance
    diff = np.linalg.norm(predicted_rho - actual_rho)
    print(f"Verification Frobenius Distance: {diff:.6e}")

    return superop.reshape(2, 2, 2, 2)


class ProcessTensor:
    """Class to represent a Process Tensor.
    
    A Process Tensor generalizes the concept of a quantum channel to multiple time steps,
    capturing non-Markovian memory effects.
    
    Attributes:
        tensor (NDArray): The raw tensor data.
        timesteps (list[float]): The time points where interventions/measurements occurred.
        basis_set (str): The basis used for reconstruction (default "Pauli").
    """
    
    def __init__(self, tensor: NDArray[np.complex128], timesteps: list[float]):
        self.tensor = tensor
        self.timesteps = timesteps
        self.rank = len(tensor.shape) // 2
        
    def as_matrix_final_output(self) -> NDArray[np.complex128]:
        """Convert to matrix view (final output vs all inputs).
        
        Returns matrix of shape (4, N^k) where:
        - 4: final output (vectorized 2x2 density matrix)
        - N: number of frame states per input slot
        - k: number of steps
        
        For Pauli frame (N=6), k=2 gives shape (4, 36).
        """
        k = len(self.timesteps)
        # Shape is [4, N, N, ...] -> reshape to [4, N^k]
        num_inputs = np.prod(self.tensor.shape[1:])  # Product of all input dimensions
        return self.tensor.reshape(4, num_inputs)
    
    def to_choi_matrix(self) -> NDArray[np.complex128]:
        """Deprecated: use as_matrix_final_output() instead."""
        return self.as_matrix_final_output()
    
    def predict_final_state(self, rho_sequence: list[NDArray[np.complex128]], duals: list[NDArray[np.complex128]]) -> NDArray[np.complex128]:
        """Predict final state for a sequence of input states using dual-frame contraction.
        
        This method enables prediction on held-out state sequences not used during tomography.
        It computes dual-frame coefficients for each input state and contracts them with the
        process tensor to predict the final output state.
        
        Args:
            rho_sequence: List of density matrices (2x2), one per time step
            duals: Dual frame matrices from tomography (same as used in reconstruction)
            
        Returns:
            Predicted final density matrix (2x2)
            
        Example:
            >>> # Build PT from tomography
            >>> pt = run_process_tensor_tomography(...)
            >>> # Predict on new sequence
            >>> rho0 = np.array([[0.5, 0.5], [0.5, 0.5]])  # |+⟩⟨+|
            >>> rho1 = np.array([[1, 0], [0, 0]])  # |0⟩⟨0|
            >>> duals = _calculate_dual_frame(_get_pauli_basis_set())
            >>> predicted = pt.predict_final_state([rho0, rho1], duals)
        """
        if len(rho_sequence) != len(self.timesteps):
            msg = f"Sequence length {len(rho_sequence)} must match number of timesteps {len(self.timesteps)}"
            raise ValueError(msg)
        
        # Compute coefficients: c_k = Tr(D_k† ρ) for each input state
        coeffs_per_step = []
        for rho in rho_sequence:
            coeffs = [np.trace(d.conj().T @ rho) for d in duals]
            coeffs_per_step.append(coeffs)
        
        # Contract with tensor: ρ_pred = Σ_{i0,...,ik} c_i0^(0) ... c_ik^(k) T[:,i0,...,ik]
        result = np.zeros(4, dtype=complex)
        for idx in np.ndindex(*self.tensor.shape[1:]):  # Iterate over all input frame indices
            # Compute product of coefficients for this index combination
            coeff = 1.0
            for step, frame_idx in enumerate(idx):
                coeff *= coeffs_per_step[step][frame_idx]
            # Add weighted contribution from this tensor element
            result += coeff * self.tensor[(slice(None),) + idx]
        
        return result.reshape(2, 2)

    def holevo_information(self, p: dict[tuple[int, ...], float] | None = None, base: int = 2) -> float:
        """Calculate the Holevo information of the process tensor.
        
        Args:
            p: Optional dictionary mapping sequence tuples to probabilities.
               If None, assumes uniform distribution over all input sequences.
            base: Logarithm base for entropy calculation (default 2 for bits).
            
        Returns:
            float: The Holevo information.
        """
        import itertools
        
        # Tensor shape: (4, N, N, ..., N)
        # N is number of basis states (6 for Pauli)
        # k is number of steps (input slots)
        out_dim = self.tensor.shape[0]
        assert out_dim == 4
        N = self.tensor.shape[1]
        k = self.tensor.ndim - 1
        
        # Generate all input sequences
        seqs = list(itertools.product(range(N), repeat=k))
        
        if p is None:
            p = {seq: 1.0 / len(seqs) for seq in seqs}
            
        # Build ensemble and average output
        rhos = {}
        rho_avg = np.zeros((2, 2), dtype=np.complex128)
        
        for seq in seqs:
            # Extract output vector for this sequence
            vec = self.tensor[(slice(None),) + seq]  # shape (4,)
            rho = _vec_to_rho(vec)
            rhos[seq] = rho
            rho_avg += p[seq] * rho
            
        # Compute Entropies
        # S(rho_avg)
        S_avg = entropy(DensityMatrix(rho_avg), base=base)
        
        # sum p_i S(rho_i)
        S_cond = 0.0
        for seq in seqs:
            S_cond += p[seq] * entropy(DensityMatrix(rhos[seq]), base=base)
            
        return S_avg - S_cond

    def holevo_information_conditional(self, fixed_step: int, fixed_idx: int, base: int = 2) -> float:
        """Calculate the Holevo information conditioned on a fixed input at a specific step.
        
        Args:
            fixed_step: The time step index to fix (0-indexed).
            fixed_idx: The basis state index to fix at that step.
            base: Logarithm base for entropy calculation (default 2 for bits).
            
        Returns:
            float: The conditional Holevo information.
        """
        import itertools
        
        out_dim = self.tensor.shape[0]
        assert out_dim == 4
        N = self.tensor.shape[1]
        k = self.tensor.ndim - 1
        
        if fixed_step < 0 or fixed_step >= k:
            raise ValueError(f"fixed_step {fixed_step} out of bounds for {k} steps.")
        if fixed_idx < 0 or fixed_idx >= N:
            raise ValueError(f"fixed_idx {fixed_idx} out of bounds for {N} basis states.")

        # Generate all sequences of length k where seq[fixed_step] == fixed_idx
        # We can optimize this by only generating the varying parts, but filtering is easier to write clearly
        seqs = [seq for seq in itertools.product(range(N), repeat=k) if seq[fixed_step] == fixed_idx]
        
        if not seqs:
            return 0.0

        # Uniform probability over this subset
        p_val = 1.0 / len(seqs)
        p = {seq: p_val for seq in seqs}
        
        # Build ensemble and average output
        rhos = {}
        rho_avg = np.zeros((2, 2), dtype=np.complex128)
        
        for seq in seqs:
            vec = self.tensor[(slice(None),) + seq]
            rho = _vec_to_rho(vec)
            rhos[seq] = rho
            rho_avg += p[seq] * rho
            
        # Compute Entropies
        S_avg = entropy(DensityMatrix(rho_avg), base=base)
        
        S_cond = 0.0
        for seq in seqs:
            S_cond += p[seq] * entropy(DensityMatrix(rhos[seq]), base=base)
            
        return S_avg - S_cond
    
    def __repr__(self) -> str:
        return f"<ProcessTensor steps={len(self.timesteps)}, shape={self.tensor.shape}>"


def run_process_tensor_tomography(
    operator: MPO, 
    sim_params: AnalogSimParams, 
    timesteps: list[float], 
    num_trajectories: int = 100,
    mode: str = "selective",
    noise_model: NoiseModel | None = None,
) -> ProcessTensor:
    """Run multi-step Process Tensor Tomography.
    
    Reconstructs a restricted process tensor (state-injection comb) under the chosen
    intervention set. Measures outputs at each time step and applies system-only
    interventions to preserve environment correlations.

    Args:
        operator: System evolution MPO.
        sim_params: Simulation parameters.
        timesteps: List of time durations for each evolution segment.
        num_trajectories: Number of trajectories to average.
        mode: Intervention mode - "selective" (trajectory-resolved, preferred) or
              "pure_env_approx" (pure-state approximation of mixed environment).

    Returns:
        ProcessTensor object representing the final-time map conditioned on preparation sequence.
    """
    import itertools
    from tqdm import tqdm

    print("Starting Multi-Step Process Tensor Tomography...")
    print(f"Time steps: {timesteps}")
    print(f"Trajectories per config: {num_trajectories}")

    # 1. Prepare Basis and Duals
    basis_set = _get_basis_states()
    basis_rhos = [b[2] for b in basis_set]
    duals = _calculate_dual_frame(basis_rhos)
    
    num_steps = len(timesteps)
    basis_indices = list(range(len(basis_set)))
    sequences = list(itertools.product(basis_indices, repeat=num_steps))
    
    print(f"Total configurations to simulate: {len(sequences)}")
    
    # Storage: sequence -> list of output states at each time
    # outputs[seq][time_idx] = averaged density matrix
    sequence_outputs = {}

    for seq in tqdm(sequences, desc="Simulating Sequences"):
        # Storage for this sequence: [time_step][trajectory] -> (x, y, z)
        time_outputs = [[[] for _ in range(num_trajectories)] for _ in range(num_steps + 1)]
        
        for traj_idx in range(num_trajectories):
            # One RNG per trajectory for consistent sampling discipline
            rng = np.random.default_rng(traj_idx)
            
            # Initialize fresh MPS
            mps = MPS(length=operator.length, state="zeros")
            
            # Prepare initial state
            initial_idx = seq[0]
            _, psi_0, _ = basis_set[initial_idx]
            mps.tensors[0] = np.expand_dims(psi_0, axis=(1, 2)).astype(np.complex128)
            
            # Measure output at t=0 (initial state)
            x0 = mps.expect(Observable(X(), sites=[0]))
            y0 = mps.expect(Observable(Y(), sites=[0]))
            z0 = mps.expect(Observable(Z(), sites=[0]))
            time_outputs[0][traj_idx] = (x0, y0, z0)
            
            # Evolution loop
            for step_i, duration in enumerate(timesteps):
                # Evolve
                step_params = copy.deepcopy(sim_params)
                step_params.elapsed_time = duration
                step_params.dt = sim_params.dt
                step_params.num_traj = 1  # Crucial: evolve single trajectory for selective intervention
                # Use linspace to avoid floating-point overshoot
                n_steps = int(np.round(duration / step_params.dt))
                step_params.times = np.linspace(0, n_steps * step_params.dt, n_steps + 1)
                step_params.observables = []
                step_params.get_state = True  # Ensure we get the state back
                
                simulator.run(mps, operator, step_params, noise_model=noise_model)
                
                # Update MPS with evolved state
                # In serial execution (forced by num_traj=1), step_params.output_state is populated.
                if step_params.output_state is not None:
                    mps = step_params.output_state
                else:
                    raise RuntimeError("Simulator did not return output state. Ensure get_state=True and serial execution.")
                
                # Measure output AFTER evolution
                x_out = mps.expect(Observable(X(), sites=[0]))
                y_out = mps.expect(Observable(Y(), sites=[0]))
                z_out = mps.expect(Observable(Z(), sites=[0]))
                time_outputs[step_i + 1][traj_idx] = (x_out, y_out, z_out)
                
                # Intervention: reprepare site 0 while preserving environment
                if step_i < num_steps - 1:
                    next_idx = seq[step_i + 1]
                    _, psi_next, _ = basis_set[next_idx]
                    
                    # Use bond-consistent reprepare
                    if mode == "selective":
                        _reprepare_site_zero_selective(mps, psi_next, rng)
                    elif mode == "pure_env_approx":
                        _reprepare_site_zero_pure_env_approx(mps, psi_next)
                    else:
                        raise ValueError(f"Unknown mode: {mode}")
        
        # Average over trajectories for each time
        avg_outputs = []
        for time_idx in range(num_steps + 1):
            x_vals = [time_outputs[time_idx][t][0] for t in range(num_trajectories)]
            y_vals = [time_outputs[time_idx][t][1] for t in range(num_trajectories)]
            z_vals = [time_outputs[time_idx][t][2] for t in range(num_trajectories)]
            
            avg_x = np.mean(x_vals)
            avg_y = np.mean(y_vals)
            avg_z = np.mean(z_vals)
            
            rho = _reconstruct_state({"x": avg_x, "y": avg_y, "z": avg_z})
            avg_outputs.append(rho)
        
        sequence_outputs[seq] = avg_outputs
        
        # Debug: print final output
        final_rho = avg_outputs[-1]
        final_z = np.trace(final_rho @ Z().matrix).real
        print(f"Seq={seq} Final Z={final_z:.3f}", flush=True)

    # 2. Construct Process Tensor
    # Structure: (out_final, in_frame_0, in_frame_1, ...)
    # Output is in 4D operator basis (vectorized density matrix)
    # Inputs are in 6-state frame basis (the actual measurement basis)
    
    num_frame_states = len(basis_set)  # 6 for Pauli frame
    tensor_shape = [4] + [num_frame_states] * num_steps  # [Out_final, In_frame0, In_frame1, ...]
    process_tensor = np.zeros(tensor_shape, dtype=complex)
    
    for seq, outputs in sequence_outputs.items():
        # Use FINAL output
        rho_final = outputs[-1]
        
        # Get duals for input sequence
        seq_duals = [duals[idx] for idx in seq]
        
        # Build term using matrix outer products (not tensor products!)
        # For each dual, we do: rho_vec @ dual_vec.conj().T
        # This gives a (4,4) matrix for each dual
        # We need to combine them into a single (4,) output vector
        
        # Start with rho_final as output
        rho_vec = rho_final.reshape(-1)  # Shape: (4,)
        
        # For multi-step PT, we need to compute:
        # T[out, in0, in1, ...] = Tr(D_in0^† D_in1^† ... ρ_in) ρ_out[out]
        # 
        # But we're building it differently: we measure with each input state
        # and get an output, so we just store: T[:, in0, in1, ...] = rho_out
        
        # Index into process_tensor at the frame indices for this sequence
        idx = (slice(None),) + tuple(seq)  # (all outputs, frame_idx_0, frame_idx_1, ...)
        process_tensor[idx] = rho_vec
    
    print("Process Tensor Reconstruction Complete.")
    
    # Create ProcessTensor object
    pt = ProcessTensor(process_tensor, timesteps)
    pt.data = process_tensor  # Store for analysis
    
    return pt
