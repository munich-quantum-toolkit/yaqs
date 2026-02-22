"""Test 1: Held-Out Prediction (Generalization Test).

This test validates that the Process Tensor correctly predicts final states
for input sequences NOT used during tomography.

Protocol:
1. Build PT using 6-state Pauli frame
2. Generate random held-out states (θ, φ) not in frame
3. Run simulator with held-out sequence → ground truth
4. Predict using PT + dual coefficients → prediction
5. Compare Frobenius distance

Pass Criteria:
- Error < 0.1 for reasonable parameters
- Error decreases with more trajectories, smaller dt, higher χ
"""

import copy
import sys

import numpy as np

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.tomography.tomography import (
    _calculate_dual_frame,
    _get_basis_states,
    _reprepare_site_zero_selective,
    run,
)


def generate_random_state(theta: float, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate a random pure state |ψ(θ,φ)⟩ and its density matrix.

    |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ) sin(θ/2)|1⟩
    """
    psi = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    rho = np.outer(psi, psi.conj())
    return psi, rho


def run_simulator_with_intervention(
    mpo: MPO,
    sim_params: AnalogSimParams,
    psi_sequence: list[np.ndarray],
    timesteps: list[float],
    num_trajectories: int = 100,
) -> np.ndarray:
    """Run simulator with selective interventions to get ground truth.

    This simulates the actual process with state injections at each timestep,
    averaging over stochastic trajectories to match the PT definition.
    """
    length = mpo.length
    num_steps = len(timesteps)

    # Accumulators for final state tomography
    results_x = []
    results_y = []
    results_z = []

    for traj_idx in range(num_trajectories):
        rng = np.random.default_rng(traj_idx)

        # Start with first state
        mps = MPS(length=length, state="zeros")
        psi0 = psi_sequence[0]
        # Initialize site 0 with psi0
        mps.tensors[0] = psi0.reshape(2, 1, 1).astype(np.complex128)

        # Evolve and intervene
        for step_i, duration in enumerate(timesteps):
            # Evolve
            step_params = copy.deepcopy(sim_params)
            step_params.elapsed_time = duration
            step_params.dt = sim_params.dt
            step_params.num_traj = 1
            step_params.observables = []

            simulator.run(mps, mpo, step_params)

            # Intervene if not last step
            if step_i < num_steps - 1:
                psi_next = psi_sequence[step_i + 1]
                _reprepare_site_zero_selective(mps, psi_next, rng)

        # Measure final state
        rx = mps.expect(Observable(X(), sites=[0]))
        ry = mps.expect(Observable(Y(), sites=[0]))
        rz = mps.expect(Observable(Z(), sites=[0]))

        results_x.append(rx)
        results_y.append(ry)
        results_z.append(rz)

    # Reconstruct averaged density matrix
    avg_x = np.mean(results_x)
    avg_y = np.mean(results_y)
    avg_z = np.mean(results_z)

    return 0.5 * np.array([[1 + avg_z, avg_x - 1j * avg_y], [avg_x + 1j * avg_y, 1 - avg_z]])


def test_held_out_prediction() -> bool:
    """Test PT prediction on held-out state sequences."""
    # Create SWAP Hamiltonian for non-trivial memory
    mpo = MPO()
    mpo.from_pauli_sum(terms=[(0.5, "X0 X1"), (0.5, "Y0 Y1"), (0.5, "Z0 Z1")], length=2, physical_dimension=2)

    dt = np.pi / 4  # Quarter SWAP time
    sim_params = AnalogSimParams(
        elapsed_time=dt,
        dt=0.01,  # Match high precision
        num_traj=100,
        max_bond_dim=32,
        order=1,
        get_state=True,
    )

    dt_actual = 0.1  # Match discrete steps if needed, but let's stick to params
    # Actually, let's use the exact same logic as Test 3 for consistency
    dt_target = np.pi / 4
    n_steps = int(np.round(dt_target / 0.05))
    dt_actual = n_steps * 0.05
    timesteps = [dt_actual, dt_actual]

    # Build Process Tensor
    pt = run(
        operator=mpo,
        sim_params=sim_params,
        timesteps=timesteps,
        num_trajectories=200,  # Sufficient for reasonable PT
        mode="selective",
    )

    # Get dual frame
    basis_set = _get_basis_states()
    basis_rhos = [b[2] for b in basis_set]
    duals = _calculate_dual_frame(basis_rhos)

    # Generate held-out test sequences
    num_tests = 5
    errors = []

    for _i in range(num_tests):
        # Generate random states NOT in the frame
        theta0, phi0 = np.random.uniform(0, np.pi), np.random.uniform(0, 2 * np.pi)
        theta1, phi1 = np.random.uniform(0, np.pi), np.random.uniform(0, 2 * np.pi)

        psi0, rho0 = generate_random_state(theta0, phi0)
        psi1, rho1 = generate_random_state(theta1, phi1)

        # Ground truth: run simulator with stats
        # Use more trajectories for ground truth to be the "truth"
        rho_true = run_simulator_with_intervention(mpo, sim_params, [psi0, psi1], timesteps, num_trajectories=500)

        # Prediction: use PT
        rho_pred = pt.predict_final_state([rho0, rho1], duals)

        # Compute error
        error = np.linalg.norm(rho_pred - rho_true, "fro")
        errors.append(error)

    # Summary
    mean_error = np.mean(errors)
    max_error = np.max(errors)

    return bool(mean_error < 0.1 and max_error < 0.2)


if __name__ == "__main__":
    success = test_held_out_prediction()
    sys.exit(0 if success else 1)
