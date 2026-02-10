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
from qiskit.quantum_info import DensityMatrix

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.simulator import run as sim_run

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from mqt.yaqs.core.data_structures.networks import MPO
    from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


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

    # Convert to density matrices and flatten
    basis_set = []
    for name, psi in states:
        rho = np.outer(psi, psi.conj())
        basis_set.append((name, rho))
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
        basis_matrices: List of density matrices (2x2) forming the frame.

    Returns:
        List of dual matrices D_k.
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


def _reconstruct_state(expectations: dict[str, float]) -> NDArray[np.complex128]:
    """Reconstructs single-qubit density matrix from Pauli expectations.

    rho = 0.5 * (I + <X>X + <Y>Y + <Z>Z)
    """
    I = np.eye(2, dtype=complex)
    X = GateLibrary.x().matrix
    Y = GateLibrary.y().matrix
    Z = GateLibrary.z().matrix
    
    rho = 0.5 * (I + expectations["x"] * X + expectations["y"] * Y + expectations["z"] * Z)
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
    basis_rhos = [b[1] for b in basis_set]
    
    # 2. Calculate Dual Frame
    duals = _calculate_dual_frame(basis_rhos)
    
    output_rhos = []

    # 3. Evolution Loop
    for i, (name, _) in enumerate(basis_set):
        print(f"Simulating basis state {i+1}/6: {name}")
        
        # Prepare MPS: Site 0 is basis state, others |0>
        # We use the state name for init if supported, or manually set tensor
        mps = MPS(length=length, state="zeros")
        
        # For mixed states or specific superpositions not in standard init, we might need manual tensor setting.
        # But our basis states are pure states well-defined.
        # However, MPS init strings are "zeros", "ones", "x+", "x-", "y+", "y-".
        # These match our basis_set names exactly.
        # But MPS constructor sets *all* sites to that state. We only want site 0.
        
        # Initialize all to |0> ("zeros"), then fix site 0.
        # We need the vector for the specific state.
        # The basis_set has density matrices, let's get the vectors again or store them.
        # Actually _get_basis_states constructs them from vectors. 
        # Let's just create a temporary MPS of length 1 with the desired state and copy the tensor.
        
        target_state_mps = MPS(length=1, state=name)
        # Copy the first tensor to our full MPS
        # Note: MPS tensors have shape (phys, left, right).
        # Site 0 of full MPS: (2, 1, 1). target_state_mps tensor: (2, 1, 1).
        mps.tensors[0] = target_state_mps.tensors[0]
        
        # Set up observables for X, Y, Z on site 0
        tomo_params = copy.deepcopy(sim_params)
        tomo_params.observables = [
            Observable(GateLibrary.x(), sites=[0]),
            Observable(GateLibrary.y(), sites=[0]),
            Observable(GateLibrary.z(), sites=[0]),
        ]
        # Sync sorted_observables (all on site 0, so order doesn't strictly matter for correctness of simulation, but must be present)
        tomo_params.sorted_observables = tomo_params.observables
        # Ensure we sort/prep them as the simulator expects (simulator usually does this, but good to be safe)
        # Actually simulator.run calls aggregate_trajectories which needs proper setup.
        # We just pass the params to run.
        
        # Run Simulation
        sim_run(mps, operator, tomo_params)
        
        # Collect Results (expectations at final time)
        res_x = tomo_params.observables[0].results[-1]
        res_y = tomo_params.observables[1].results[-1]
        res_z = tomo_params.observables[2].results[-1]
        
        expectations = {"x": res_x, "y": res_y, "z": res_z}
        rho_out = _reconstruct_state(expectations)
        output_rhos.append(rho_out)

    # 4. Construct Process Tensor
    # Lambda = sum_k rho_out_k (x) D_k^*  (using outer product vectorization convention)
    # The user asked for "tensor product of the density and the duals".
    # Typically this means the Choi matrix represented as a tensor.
    # Let's return the Lambda tensor of shape (2, 2, 2, 2) acting on rho.
    # E(rho)_{ij} = sum_{mn} Lambda_{ijmn} rho_{mn}
    # This corresponds to constructing the superoperator matrix.
    
    # Let's build the superoperator matrix S first, shape (4, 4).
    # S = sum_k |rho_out_k>> <<D_k|
    
    dim = 2
    superop = np.zeros((4, 4), dtype=complex)
    for rho_out, dual in zip(output_rhos, duals):
        rho_vec = rho_out.reshape(-1, 1) # column vector
        dual_vec = dual.reshape(-1, 1)
        # Outer product |rho_out>> <<dual_conj| ?
        # Formula: E(rho) = sum_k Tr(D_k^dag rho) rho_out_k
        # Tr(A^dag B) is inner product <<A|B>>.
        # So E(rho) = sum_k <<D_k|rho>> |rho_out_k>>
        #           = sum_k |rho_out_k>> <<D_k| |rho>>
        # So S = sum_k |rho_out_k>> <<D_k|
        # Note: duals calculated earlier were such that sum_k |F_k>> <<D_k| = I.
        # D_k in _calculate_dual_frame corresponds to the "dual vector".
        # The inner product uses conjugate of D_k.
        # <<D_k| is D_k.conj().T (as a vector).
        
        # dual_vec is the vector form of the dual matrix.
        superop += rho_vec @ dual_vec.conj().T
        
    print("Process Tomography Complete.")
    
    # 5. Simple Verification (Log fidelity with Identity if nearly identity)
    # Or test evolution of a random state as requested.
    
    # Test with a random state (not in basis)
    test_psi = np.array([np.cos(0.4), np.exp(1j*0.2)*np.sin(0.4)])
    test_rho_in = np.outer(test_psi, test_psi.conj())
    
    # Predict using Process Tensor
    # Vectorize input
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
        Observable(GateLibrary.x(), sites=[0]),
        Observable(GateLibrary.y(), sites=[0]),
        Observable(GateLibrary.z(), sites=[0]),
    ]
    test_params.sorted_observables = test_params.observables
    sim_run(mps_test, operator, test_params)
    
    rx = test_params.observables[0].results[-1]
    ry = test_params.observables[1].results[-1]
    rz = test_params.observables[2].results[-1]
    actual_rho = _reconstruct_state({"x": rx, "y": ry, "z": rz})
    
    
    # Compute Frobenius distance
    diff = np.linalg.norm(predicted_rho - actual_rho)
    print(f"Verification Frobenius Distance: {diff:.6e}")
    
    # Return the process tensor reshaped to (2, 2, 2, 2)
    # Indices: out_row, out_col, in_row, in_col
    # S maps (in_row, in_col) -> (out_row, out_col)
    # S has shape (4, 4). element S_{IJ} where I = 2*r_out + c_out, J = 2*r_in + c_in.
    return superop.reshape(2, 2, 2, 2)
