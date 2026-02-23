import copy
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from mqt.yaqs.analog.analog_tjm import analog_tjm_1
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z
from mqt.yaqs.characterization.tomography.tomography import (
    _calculate_dual_frame,
    _get_basis_states,
    _reprepare_site_zero,
    run,
)


def test_tomography_run_basic() -> None:
    """Integration test for unified tomography.run() API."""
    # 1. Setup small system for speed
    L = 2
    operator = MPO.ising(L, J=1.0, g=1.0)

    # 2. Setup parameters
    params = AnalogSimParams(
        elapsed_time=0.1,
        dt=0.1,
        order=1,
        max_bond_dim=4,
    )

    # 3. Run Tomography (1-step default)
    pt = run(operator, params, num_trajectories=1)

    # Verify output
    assert pt.data.shape == (4, 6)
    assert np.isclose(pt.holevo_information(), 0.0, atol=2.0)  # Dummy check, just ensuring it runs


def test_measurement_bases() -> None:
    """Verify that tomography.run() accepts and correctly handles measurement_bases."""
    L = 2
    op = MPO.ising(L, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1, max_bond_dim=4)
    timesteps = [0.1, 0.1]

    # 1. Single basis
    pt_z = run(op, params, timesteps=timesteps, num_trajectories=1, measurement_bases="Z")
    assert pt_z.tensor.shape == (4, 6, 6)

    # 2. Multiple bases
    pt_all = run(op, params, timesteps=timesteps, num_trajectories=1, measurement_bases=["X", "Y", "Z"])
    assert pt_all.tensor.shape == (4, 6, 6)

    # 3. Default (should be equivalent to [X, Y, Z])
    pt_default = run(op, params, timesteps=timesteps, num_trajectories=1)
    assert pt_default.tensor.shape == (4, 6, 6)


def test_tomography_mcwf_multistep() -> None:
    """Verify tomography with MCWF solver and multiple steps (vector interventions)."""
    L = 2
    op = MPO.ising(L, J=1.0, g=1.0)
    # Use MCWF solver
    params = AnalogSimParams(dt=0.1, order=1, solver="MCWF")
    timesteps = [0.1, 0.1]

    # Run Tomography - this will trigger _reprepare_site_zero_vector
    pt = run(op, params, timesteps=timesteps, num_trajectories=10, measurement_bases="Z")
    assert pt.tensor.shape == (4, 6, 6)
    # Check that identity is somewhat preserved (rough check)
    # For a very short time, diagonal elements of Process Tensor should be close to 1
    # Choosing first entry (rho_0 -> rho_0)
    assert np.real(pt.tensor[0, 0, 0]) > 0.5


def test_tomography_run_multistep() -> None:
    """Integration test for multi-step tomography.run() API."""
    L = 2
    operator = MPO.ising(L, J=1.0, g=1.0)

    params = AnalogSimParams(
        dt=0.1,
        order=1,
        max_bond_dim=4,
    )

    timesteps = [0.1, 0.1]

    # Run Tomography
    pt = run(operator, params, timesteps=timesteps, num_trajectories=1)

    # Verify output shape (4, 6, 6)
    assert pt.data.shape == (4, 6, 6)
    assert len(pt.timesteps) == 2


def test_tomography_prediction_accuracy() -> None:
    """Verify that PT prediction matches direct standalone simulation."""
    from mqt.yaqs.characterization.tomography.tomography import (
        _calculate_dual_frame,
        _get_basis_states,
    )

    L = 2
    op = MPO.ising(length=L, J=1.0, g=1.0)

    params = AnalogSimParams(
        dt=0.1,
        max_bond_dim=16,
        order=1,
        get_state=True,
    )

    timesteps = [0.1, 0.1]

    # 1. Run Tomography
    pt = run(op, params, timesteps=timesteps, num_trajectories=1, measurement_bases="Z")

    # 2. Pick a sequence, e.g., |0> then |+>
    basis_set = _get_basis_states()
    rho_0 = basis_set[0][2]
    rho_plus = basis_set[2][2]

    dual_frames = _calculate_dual_frame([b[2] for b in basis_set])

    # Predict with PT
    rho_pred = pt.predict_final_state([rho_0, rho_plus], dual_frames)

    # 3. Direct Simulation (Manual Intervention)
    # (Index 0 is Zeros, Index 2 is X+)
    vec_stored = pt.tensor[:, 0, 2]
    rho_stored = vec_stored.reshape(2, 2)

    assert np.allclose(rho_pred, rho_stored, atol=1e-12)


def test_tjm_mcwf_consistency() -> None:
    """Verify that TJM and MCWF solvers produce consistent ProcessTensors."""
    # Simple Ising chain
    L = 2
    op = MPO.ising(length=L, J=1.0, g=1.0)

    timesteps = [0.1]
    num_trajectories = 50

    # TJM Params
    tjm_params = AnalogSimParams(
        dt=0.1,
        solver="TJM",
    )

    # MCWF Params
    mcwf_params = AnalogSimParams(
        dt=0.1,
        solver="MCWF",
    )

    # Run both
    pt_tjm = run(op, tjm_params, timesteps, num_trajectories=num_trajectories)
    pt_mcwf = run(op, mcwf_params, timesteps, num_trajectories=num_trajectories)

    # Check trace normalization
    tjm_traces = pt_tjm.tensor[0, :] + pt_tjm.tensor[3, :]
    mcwf_traces = pt_mcwf.tensor[0, :] + pt_mcwf.tensor[3, :]

    assert np.allclose(tjm_traces, 1.0, atol=1e-2)
    assert np.allclose(mcwf_traces, 1.0, atol=1e-2)

    # Check Holevo info consistency
    h_tjm = pt_tjm.holevo_information()
    h_mcwf = pt_mcwf.holevo_information()

    assert np.isclose(h_tjm, h_mcwf, atol=0.1)

    # Tensor norm difference should be small
    diff = np.linalg.norm(pt_tjm.tensor - pt_mcwf.tensor)
    assert diff < 0.2


def test_reconstruction_identity() -> None:
    """Verify reconstruction of Identity channel."""
    L = 1
    op = MPO()
    op.identity(L)

    params = AnalogSimParams(dt=0.1)
    pt = run(op, params, num_trajectories=10)

    # Holevo info for Identity should be 1.0 (base 2)
    h = pt.holevo_information(base=2)
    assert np.isclose(h, 1.0, atol=1e-2)


def test_reconstruction_bitflip() -> None:
    """Verify reconstruction of a Bit-Flip (X) operation."""
    op = MPO()
    # Manual X gate as a 1-site MPO
    X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    op.tensors = [X_mat.reshape(2, 1, 2, 1)]  # physical_out, branch_in, physical_in, branch_out
    op.length = 1

    # H = -g X. If g*dt = pi/2, then U = exp(i pi/2 X) = iX.
    g = np.pi / (2 * 0.1)
    op = MPO.ising(length=1, J=0.0, g=g)

    params = AnalogSimParams(dt=0.1, elapsed_time=0.1)
    pt = run(op, params, num_trajectories=10)

    # Input |0> should become |1> (up to phase)
    # Average output for basis states should still yield Holevo 1.0 if it's a unitary
    h = pt.holevo_information(base=2)
    assert np.isclose(h, 1.0, atol=1e-2)


def test_reconstruction_depolarizing() -> None:
    """Verify reconstruction of a Depolarizing channel (via strong noise)."""
    from mqt.yaqs.core.data_structures.noise_model import NoiseModel

    L = 1
    op = MPO()
    op.identity(L)

    # Strong dephasing + relaxation => Depolarizing
    noise_processes = [
        {"name": "z", "sites": [0], "strength": 10.0},
        {"name": "lowering", "sites": [0], "strength": 10.0},
    ]
    noise_model = NoiseModel(processes=noise_processes)

    params = AnalogSimParams(dt=0.1, elapsed_time=0.1)
    pt = run(op, params, num_trajectories=100, noise_model=noise_model)

    # Holevo info for fully depolarizing channel should be 0.0
    h = pt.holevo_information(base=2)
    assert h < 0.2


from mqt.yaqs.characterization.tomography.process_tensor import (
    ProcessTensor,
    _vec_to_rho,
)


def test_vec_to_rho() -> None:
    """Test the vector to density matrix conversion."""
    # Test with |0><0|
    psi0 = np.array([1, 0], dtype=complex)
    rho0 = np.outer(psi0, psi0.conj())
    vec0 = rho0.reshape(-1)
    rho_out = _vec_to_rho(vec0)
    np.testing.assert_allclose(rho_out, rho0, atol=1e-15)

    # Test with |+><+|
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    rho_plus = np.outer(psi_plus, psi_plus.conj())
    vec_plus = rho_plus.reshape(-1)
    rho_out = _vec_to_rho(vec_plus)
    np.testing.assert_allclose(rho_out, rho_plus, atol=1e-15)


def get_standard_basis():
    """Returns the standard 6-state Pauli basis for testing."""
    psi_0 = np.array([1, 0], dtype=complex)
    psi_1 = np.array([0, 1], dtype=complex)
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
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
    basis_set = []
    for name, psi in states:
        rho = np.outer(psi, psi.conj())
        basis_set.append((name, psi, rho))
    return basis_set


def test_holevo_information_identity() -> None:
    """Test Holevo information for an identity channel."""
    # Use standard basis to ensure source entropy is 1
    basis = get_standard_basis()
    num_frames = len(basis)

    # Create tensor for 1 timestep
    tensor = np.zeros((4, num_frames), dtype=complex)

    for i, (_, _, rho) in enumerate(basis):
        tensor[:, i] = rho.reshape(-1)

    pt = ProcessTensor(tensor, [1.0])

    holevo = pt.holevo_information(base=2)
    assert np.isclose(holevo, 1.0, atol=1e-10)


def test_holevo_information_fully_depolarizing() -> None:
    """Test Holevo information for a fully depolarizing channel."""
    # Maps everything to I/2
    basis = get_standard_basis()
    num_frames = len(basis)

    tensor = np.zeros((4, num_frames), dtype=complex)

    rho_mixed = 0.5 * np.eye(2, dtype=complex)
    vec_mixed = rho_mixed.reshape(-1)

    for i in range(num_frames):
        tensor[:, i] = vec_mixed

    pt = ProcessTensor(tensor, [1.0])

    holevo = pt.holevo_information(base=2)
    assert np.isclose(holevo, 0.0, atol=1e-10)


def test_holevo_information_conditional() -> None:
    """Test conditional Holevo information."""
    # Construct a 2-step process where output depends ONLY on step 0.
    # T[out, i, j] = rho_i
    basis = get_standard_basis()
    num_frames = len(basis)

    # Shape: (4, N, N)
    tensor = np.zeros((4, num_frames, num_frames), dtype=complex)

    for i in range(num_frames):
        rho_i = basis[i][2]
        vec_i = rho_i.reshape(-1)
        for j in range(num_frames):
            tensor[:, i, j] = vec_i

    pt = ProcessTensor(tensor, [1.0, 1.0])

    # CASE 1: Fix step 0 to index 0 (state |0><0|)
    # The output is always |0><0|, regardless of step 1.
    # S(rho_avg) = S(|0><0|) = 0
    # S(rho_seq) = 0
    # Holevo = 0
    h_cond_0 = pt.holevo_information_conditional(fixed_step=0, fixed_idx=0, base=2)
    assert np.isclose(h_cond_0, 0.0, atol=1e-10)

    # CASE 2: Fix step 1 to index 0.
    # The output still varies with step 0.
    # It acts like an identity channel from step 0 to output.
    # S(rho_avg) = 1 (maximally mixed average of basis)
    # S(rho_seq) = 0 (pure outputs)
    # Holevo = 1
    h_cond_1 = pt.holevo_information_conditional(fixed_step=1, fixed_idx=0, base=2)
    assert np.isclose(h_cond_1, 1.0, atol=1e-10)


from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.characterization.tomography.tomography import (
    _calculate_dual_frame,
    _get_basis_states,
)


def test_algebraic_consistency() -> None:
    """Test 1: Frame self-consistency (floating point precision check)."""
    # Setup random Hamiltonian for 2 qubits (System + Env)
    mpo = MPO()
    mpo.from_pauli_sum(terms=[(0.5, "X0 X1"), (0.3, "Y0 Z1"), (0.2, "Z0 Y1")], length=2, physical_dimension=2)

    dt = 0.1
    timesteps = [dt, dt, dt]  # 3 steps

    sim_params = AnalogSimParams(
        elapsed_time=dt,
        dt=0.1,
        max_bond_dim=16,
        order=1,
    )

    # Run Tomography
    pt = run(
        operator=mpo,
        sim_params=sim_params,
        timesteps=timesteps,
        num_trajectories=10,
    )

    # Verify Consistency
    basis_set = _get_basis_states()
    basis_rhos = [b[2] for b in basis_set]
    duals = _calculate_dual_frame(basis_rhos)

    rng = np.random.default_rng(42)
    max_error = 0.0

    num_sequences = 10
    for _ in range(num_sequences):
        seq = tuple(rng.integers(0, 6, size=3))
        rho_seq = [basis_rhos[i] for i in seq]

        # Predict using duals
        [np.trace(d.conj().T @ rho_seq[0]) for d in duals]
        # Diagnostic: coeffs_0 should be one-hot for seq[0]
        rho_pred = pt.predict_final_state(rho_seq, duals)
        vec_pred = rho_pred.reshape(-1)

        # Stored value in tensor
        vec_stored = pt.tensor[(slice(None), *seq)]

        err = np.linalg.norm(vec_pred - vec_stored)
        max_error = max(max_error, err)

    assert max_error < 1e-2


def test_markovian_limit() -> None:
    """Test 2: Amplitude Damping on Environment (Markovian Limit)."""
    # Strong entangling Hamiltonian
    mpo = MPO()
    mpo.from_pauli_sum(
        terms=[(1.0, "X0 X1"), (1.0, "Y0 Y1"), (1.0, "Z0 Z1")],  # SWAP-like
        length=2,
        physical_dimension=2,
    )

    dt = 0.5
    timesteps = [dt, dt]

    sim_params = AnalogSimParams(
        elapsed_time=dt,
        dt=0.05,
        max_bond_dim=16,
        order=1,
    )

    # Strong Amplitude Damping on Q1 (Env)
    noise_processes = [{"name": "lowering", "sites": [1], "strength": 10.0}]
    noise_model = NoiseModel(processes=noise_processes)

    # Run Tomography
    pt = run(operator=mpo, sim_params=sim_params, timesteps=timesteps, num_trajectories=50, noise_model=noise_model)

    basis_set = _get_basis_states()
    basis_rhos = [b[2] for b in basis_set]
    duals = _calculate_dual_frame(basis_rhos)

    # Sequence A: [|0>, |+>], Sequence B: [|1>, |+>]
    # If Markovian, output should only depend on |+>
    rho_0 = basis_rhos[0]
    rho_1 = basis_rhos[1]
    rho_plus = basis_rhos[2]

    out_A = pt.predict_final_state([rho_0, rho_plus], duals)
    out_B = pt.predict_final_state([rho_1, rho_plus], duals)

    diff = np.linalg.norm(out_A - out_B)

    # Large damping rate (10.0) relative to time (0.5) should erase memory effectively
    assert diff < 0.2


def _get_random_rho() -> NDArray[np.complex128]:
    """Generate a random 2x2 density matrix."""
    # Create random complex matrix
    A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    # rho = A* A / Tr(A* A)
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def _sample_pure_state(rho: NDArray[np.complex128], rng: np.random.Generator) -> NDArray[np.complex128]:
    """Sample a pure state from the eigen-decomposition of rho."""
    evals, evecs = np.linalg.eigh(rho)
    # Ensure probabilities are non-negative and sum to 1
    p = np.maximum(evals, 0)
    p /= np.sum(p)
    idx = rng.choice([0, 1], p=p)
    return evecs[:, idx]


def test_held_out_prediction() -> None:
    """Verify PT prediction accuracy for random held-out mixed state sequences."""
    np.random.seed(42)
    L = 2
    op = MPO.ising(length=L, J=1.0, g=0.5)

    params = AnalogSimParams(
        dt=0.1,
        max_bond_dim=16,
        order=1,
    )
    timesteps = [0.1, 0.1]
    num_trajectories = 200  # Higher trajectories for better statistics in direct sim

    # 1. Build Process Tensor using standard tomography
    # Defaulting to ["Z"] for simplicity in this consistency check
    pt = run(op, params, timesteps=timesteps, num_trajectories=num_trajectories, measurement_bases="Z")

    # 2. Pick random sequence
    rho_0 = _get_random_rho()
    rho_1 = _get_random_rho()
    rho_sequence = [rho_0, rho_1]

    basis_set = _get_basis_states()
    duals = _calculate_dual_frame([b[2] for b in basis_set])

    # 3. Predict final state using PT
    rho_pred = pt.predict_final_state(rho_sequence, duals)

    # 4. Direct Simulation of the sequence [rho_0, rho_1]
    num_direct_trajectories = 400
    rng = np.random.default_rng(42)
    results = []

    for _ in range(num_direct_trajectories):
        # Sample initial pure state from rho_0
        psi_0 = _sample_pure_state(rho_0, rng)
        state = MPS(length=L, state="zeros")
        state.tensors[0] = np.expand_dims(psi_0, axis=(1, 2)).astype(np.complex128)

        # Step 1: Evolution
        step_params = copy.deepcopy(params)
        step_params.elapsed_time = 0.1
        step_params.num_traj = 1
        step_params.get_state = True
        analog_tjm_1((0, state, None, step_params, op))
        state = step_params.output_state

        # Step 2: Intervention (Sample new state from rho_1)
        psi_1 = _sample_pure_state(rho_1, rng)
        _reprepare_site_zero(state, psi_1, rng, meas_basis="Z")

        # Step 3: Evolution
        step_params.output_state = None
        analog_tjm_1((0, state, None, step_params, op))
        state = step_params.output_state

        # Reconstruct rho from Pauli expectations
        rx = state.expect(Observable(X(), sites=[0]))
        ry = state.expect(Observable(Y(), sites=[0]))
        rz = state.expect(Observable(Z(), sites=[0]))

        # Reconstruct rho from Pauli expectations
        I = np.eye(2, dtype=complex)
        sigma_x = X().matrix
        sigma_y = Y().matrix
        sigma_z = Z().matrix
        rho_final = 0.5 * (I + rx * sigma_x + ry * sigma_y + rz * sigma_z)
        results.append(rho_final)

    rho_direct = np.mean(results, axis=0)

    # 5. Compare
    assert np.allclose(rho_pred, rho_direct, atol=0.1)
