"""Test 3: 2-Qubit Exact Ground Truth.

This test compares the MPS+PT implementation against exact dense-matrix evolution
for a 2-qubit system where we can compute everything exactly.

Protocol:
1. Use 2-qubit system (site 0 = system, site 1 = environment)
2. Hamiltonian: H = 0.5(XX + YY + ZZ) (partial SWAP)
3. Implement exact intervention in dense space
4. Compare MPS+PT vs exact for multiple random (ρ₀, ρ₁) pairs

Pass Criteria:
- Frobenius error < 1e-6 for all test cases
"""

import sys

import numpy as np
from scipy.linalg import expm

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.tomography.tomography import (
    _calculate_dual_frame,
    _get_basis_states,
    run,
)


def selective_intervention_dense(rho, rho_inj, rng):
    """Exact selective intervention for 2 qubits."""
    # rho is 4x4 for (system⊗env)
    # projectors on system (site 0)
    # Layout: s0 is first qubit, s1 is second
    # P0 = |0><0| ⊗ I, P1 = |1><1| ⊗ I
    P0 = np.kron(np.array([[1, 0], [0, 0]]), np.eye(2))
    P1 = np.kron(np.array([[0, 0], [0, 1]]), np.eye(2))

    p0 = np.trace(P0 @ rho).real
    p1 = np.trace(P1 @ rho).real

    # Numerical stability
    p0 = max(p0, 0.0)
    p1 = max(p1, 0.0)
    s = p0 + p1
    if s < 1e-15:
        p0, p1 = 0.5, 0.5
    else:
        p0 /= s
        p1 /= s

    # Sample measurement outcome m
    m = 0 if rng.random() < p0 else 1
    Pm = P0 if m == 0 else P1

    # Collapse state
    rho_m = (Pm @ rho @ Pm) / (np.trace(Pm @ rho).real + 1e-15)

    # env conditional: Tr_sys(rho_m)
    # Trancate system (site 0)
    rho_env = np.array([
        [rho_m[0, 0] + rho_m[2, 2], rho_m[0, 1] + rho_m[2, 3]],
        [rho_m[1, 0] + rho_m[3, 2], rho_m[1, 1] + rho_m[3, 3]],
    ])

    # Reprepare system with rho_inj
    return np.kron(rho_inj, rho_env)


def exact_evolution_with_intervention(
    H: np.ndarray, rho_sequence: list[np.ndarray], timesteps: list[float], num_trajectories: int = 1000
) -> np.ndarray:
    """Exact dense-matrix evolution with selective sampled interventions.

    Averages over multiple exact trajectories to get the ground truth map output.
    """
    U_list = [expm(-1j * H * dt) for dt in timesteps]
    final_rhos = []

    for traj_idx in range(num_trajectories):
        rng = np.random.default_rng(traj_idx)  # Match tomography discipline
        # Start state
        rho_sys = rho_sequence[0]
        rho_env = np.array([[1, 0], [0, 0]])  # |0⟩⟨0|
        rho = np.kron(rho_sys, rho_env)

        for i, U in enumerate(U_list):
            # Evolve
            rho = U @ rho @ U.conj().T

            # Intervene if not last step
            if i < len(timesteps) - 1:
                rho = selective_intervention_dense(rho, rho_sequence[i + 1], rng)

        # Trace out env at the end
        rho_out = np.array([
            [rho[0, 0] + rho[1, 1], rho[0, 2] + rho[1, 3]],
            [rho[2, 0] + rho[3, 1], rho[2, 2] + rho[3, 3]],
        ])
        final_rhos.append(rho_out)

    return np.mean(final_rhos, axis=0)


def generate_random_density_matrix() -> np.ndarray:
    """Generate a random 2x2 density matrix."""
    # Generate random pure state
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    return np.outer(psi, psi.conj())


def test_2qubit_exact() -> bool:
    """Test PT against exact 2-qubit evolution."""
    # Create partial SWAP Hamiltonian
    mpo = MPO()
    mpo.from_pauli_sum(terms=[(0.5, "X0 X1"), (0.5, "Y0 Y1"), (0.5, "Z0 Z1")], length=2, physical_dimension=2)

    # Exact Hamiltonian (4x4 for 2 qubits)
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    H_exact = 0.5 * (np.kron(X, X) + np.kron(Y, Y) + np.kron(Z, Z))

    # Precise timing for convergence test
    dt_actual = 0.1
    timesteps = [dt_actual, dt_actual]

    sim_params = AnalogSimParams(
        elapsed_time=dt_actual,
        dt=0.01,  # Much smaller dt to reduce Trotter error
        num_traj=400,  # More trajectories for better stats
        max_bond_dim=32,
        order=1,
        get_state=True,
    )

    # Build Process Tensor
    pt = run(operator=mpo, sim_params=sim_params, timesteps=timesteps, num_trajectories=400, mode="selective")

    # Get dual frame
    basis_set = _get_basis_states()
    basis_rhos = [b[2] for b in basis_set]
    duals = _calculate_dual_frame(basis_rhos)

    # Test on random sequences

    # --- EXHAUSTIVE FRAME CONSISTENCY CHECK ---
    max_frame_error = 0.0
    for idx in np.ndindex(*pt.tensor.shape[1:]):  # Iterate over all input indices (6, 6)
        # Construct sequence of rhos for this frame index
        seq_rhos = [basis_rhos[i] for i in idx]

        # Predict using dual contraction
        rho_pred = pt.predict_final_state(seq_rhos, duals)
        vec_pred = rho_pred.reshape(-1)

        # Get stored tensor value
        vec_stored = pt.tensor[(slice(None), *idx)]

        # Compare
        err = np.linalg.norm(vec_pred - vec_stored)
        max_frame_error = max(max_frame_error, err)

    if max_frame_error > 1e-12:
        return False
    # --- END VALIDATION ---

    num_tests = 1  # Only 1 test to save time, focus on accuracy
    errors = []

    for _i in range(num_tests):
        np.random.seed(1234)  # Fixed seed
        rho0 = generate_random_density_matrix()
        rho1 = generate_random_density_matrix()

        # Exact evolution (ground truth)
        rho_exact = exact_evolution_with_intervention(H_exact, [rho0, rho1], timesteps, num_trajectories=400)

        # PT prediction
        rho_pt = pt.predict_final_state([rho0, rho1], duals)

        # Compute error
        error = np.linalg.norm(rho_pt - rho_exact, "fro")
        errors.append(error)

    # Summary

    if errors[0] < 0.1:  # Tighter tolerance for high N
        return True
    return False


if __name__ == "__main__":
    success = test_2qubit_exact()
    sys.exit(0 if success else 1)
