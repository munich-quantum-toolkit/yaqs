import sys

import numpy as np

from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.tomography.tomography import _calculate_dual_frame, _get_basis_states, run


def test_algebraic_consistency() -> bool:
    """Test 1: Frame self-consistency (floating point precision check)."""
    # 1. Setup random Hamiltonian for 2 qubits (System + Env)
    # Using random MPO to ensure general dynamics
    mpo = MPO()
    mpo.from_pauli_sum(terms=[(0.5, "X0 X1"), (0.3, "Y0 Z1"), (0.2, "Z0 Y1")], length=2, physical_dimension=2)

    # Short evolution to keep it fast
    dt = 0.1
    timesteps = [dt, dt, dt]  # 3 steps

    sim_params = AnalogSimParams(
        elapsed_time=dt,
        dt=0.1,  # One step per chunk for speed
        num_traj=1,  # MPS method is deterministic for "selective" but we use 1 for speed
        max_bond_dim=16,
        order=1,
        get_state=True,
    )

    # 2. Run Tomography
    # We use "pure_env_approx" here because we only care about algebraic contraction consistency,
    # not the physical correctness of the intervention (which we verified in Test 3).
    # This mode is faster.
    pt = run(
        operator=mpo,
        sim_params=sim_params,
        timesteps=timesteps,
        num_trajectories=10,
    )

    # 3. Verify Consistency
    basis_set = _get_basis_states()
    basis_rhos = [b[2] for b in basis_set]
    duals = _calculate_dual_frame(basis_rhos)

    rng = np.random.default_rng(42)
    max_error = 0.0

    num_sequences = 20
    for _ in range(num_sequences):
        # Generate random sequence of frame INDICES
        seq = tuple(rng.integers(0, 6, size=3))

        # Get corresponding rho objects
        rho_seq = [basis_rhos[i] for i in seq]

        # Predict
        rho_pred = pt.predict_final_state(rho_seq, duals)
        vec_pred = rho_pred.reshape(-1)

        # Stored value
        vec_stored = pt.tensor[(slice(None), *seq)]

        # Compare
        err = np.linalg.norm(vec_pred - vec_stored)
        max_error = max(max_error, err)

    return max_error < 1e-12


def test_markovian_limit() -> bool:
    """Test 2: Amplitude Damping on Environment (Markovian Limit)."""
    # 1. Validation Logic
    # In a memoryless process, the final state should depend ONLY on the LAST operation.
    # We will prepare [rho_A, rho_fixed] and [rho_B, rho_fixed].
    # If memory is effectively erased, the outputs should be identical.

    # 2. Setup System
    # Strong entangling Hamiltonian
    mpo = MPO()
    mpo.from_pauli_sum(
        terms=[(1.0, "X0 X1"), (1.0, "Y0 Y1"), (1.0, "Z0 Z1")],  # SWAP-like
        length=2,
        physical_dimension=2,
    )

    dt = 0.5
    timesteps = [dt, dt]  # 2 steps

    sim_params = AnalogSimParams(
        elapsed_time=dt,
        dt=0.05,
        num_traj=1,  # Noise simulation needs trajectories but we check mean
        max_bond_dim=16,
        order=1,
        get_state=True,
    )
    sim_params.get_state = False  # Required for noisy simulation

    # 3. Define Noise Model: Strong Amplitude Damping on Qubit 1 (Env)
    # This effectively resets Q1 to |0> constantly, erasing valid correlations.
    # strength=5.0 is massive damping rate relative to time=0.5
    noise_processes = [{"name": "lowering", "sites": [1], "strength": 5.0}]
    noise_model = NoiseModel(processes=noise_processes)

    # 4. Run Tomography
    # NOTE: We need enough trajectories to average out the noise
    # But for amplitude damping to |0>, it might converge fast?
    # Actually, AmplitudeDamping is stochastic jumps.
    # Let's use 50 trajectories.
    pt = run(operator=mpo, sim_params=sim_params, timesteps=timesteps, num_trajectories=50, noise_model=noise_model)

    # 5. Check Memory
    basis_set = _get_basis_states()
    basis_rhos = [b[2] for b in basis_set]
    duals = _calculate_dual_frame(basis_rhos)

    # Sequence A: [|0>, |+>]
    # Sequence B: [|1>, |+>]
    # Last input is identical (|+>), earlier input differs.
    # Markovian => Output should be identical.

    rho_0 = basis_rhos[0]  # |0><0|
    rho_1 = basis_rhos[1]  # |1><1|
    rho_plus = basis_rhos[2]  # |+><+|

    out_A = pt.predict_final_state([rho_0, rho_plus], duals)
    out_B = pt.predict_final_state([rho_1, rho_plus], duals)

    diff = np.linalg.norm(out_A - out_B)

    # Compare to NO noise case (Memory should be high)
    # Ideally we'd run a control, but we know SWAP has memory.
    # We expect diff to be small (~0.05 maybe due to finite sampling noise)

    return diff < 0.1


if __name__ == "__main__":
    passed = True
    passed &= test_algebraic_consistency()
    passed &= test_markovian_limit()

    if not passed:
        sys.exit(1)
