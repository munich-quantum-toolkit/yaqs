# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Integration tests for tomography module."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.analog.analog_tjm import analog_tjm_1
from mqt.yaqs.characterization.tomography.process_tensor import ProcessTensor
from mqt.yaqs.characterization.tomography.tomography import (
    _reprepare_site_zero,  # noqa: PLC2701
    calculate_dual_frame,
    get_basis_states,
    run,
)
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test_tomography_run_basic() -> None:
    """Integration test for basic 1-step tomography.run() API."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1, elapsed_time=0.1)

    # Run Tomography
    pt = run(op, params, num_trajectories=10)

    assert pt.tensor.shape == (4, 6)
    # Check that identity is somewhat preserved (rough check)
    assert np.real(pt.tensor[0, 0]) > 0.5


def test_tomography_run_defaults() -> None:
    """Verify tomography run with default timesteps."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1, elapsed_time=0.2)
    timesteps = [0.1, 0.1]

    pt = run(op, params, timesteps=timesteps, num_trajectories=1)
    assert pt.tensor.shape == (4, 6, 6)

    pt_default = run(op, params, timesteps=timesteps, num_trajectories=1)
    assert pt_default.tensor.shape == (4, 6, 6)


def test_tomography_mcwf_multistep() -> None:
    """Verify tomography with MCWF solver and multiple steps (vector interventions)."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    # Use MCWF solver
    params = AnalogSimParams(dt=0.1, order=1, solver="MCWF")
    timesteps = [0.1, 0.1]

    # Run Tomography - this will trigger _reprepare_site_zero_vector
    pt = run(op, params, timesteps=timesteps, num_trajectories=10, measurement_bases="Z")
    assert pt.tensor.shape == (4, 6, 6)
    assert np.real(pt.tensor[0, 0, 0]) > 0.5


def test_tomography_run_multistep() -> None:
    """Integration test for multi-step tomography.run() API."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1)
    timesteps = [0.1, 0.1]

    # Run Tomography
    pt = run(op, params, timesteps=timesteps, num_trajectories=10)

    assert pt.tensor.shape == (4, 6, 6)
    # Check that identity is somewhat preserved (rough check)
    assert np.real(pt.tensor[0, 0, 0]) > 0.5


def test_tomography_measurement_bases() -> None:
    """Verify tomography with multiple measurement bases."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1)
    timesteps = [0.1, 0.1]

    # Run with X and Z bases
    pt = run(op, params, timesteps=timesteps, num_trajectories=5, measurement_bases=["X", "Z"])
    assert pt.tensor.shape == (4, 6, 6)


def test_holevo_information() -> None:
    """Verify Holevo information calculation on a ProcessTensor."""
    # Create a 1-step PT that is an identity channel
    tensor = np.zeros((4, 6), dtype=complex)
    # In 1-step, pt[out, in] stores the output density matrix (vectorized)
    # for each of the 6 Pauli basis inputs.
    # Basis: |0>, |1>, |+>, |->, |i+>, |i->
    basis_set = get_basis_states()
    for i in range(6):
        _, _, rho = basis_set[i]
        tensor[:, i] = rho.reshape(-1)

    pt = ProcessTensor(tensor, [0.1])

    # Identity channel on 1 qubit should have 1 bit of Holevo information
    h = pt.holevo_information(base=2)
    assert np.isclose(h, 1.0, atol=1e-2)


def test_reconstruction_x_gate() -> None:
    """Verify tomography correctly reconstructs an X gate."""
    op = MPO()
    # Manual X gate as a 1-site MPO
    x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    op.tensors = [x_mat.reshape(2, 1, 2, 1)]  # physical_out, branch_in, physical_in, branch_out
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
    l_size = 1
    op = MPO()
    op.identity(l_size)

    # Strong dephasing + relaxation => Depolarizing
    noise_processes = [
        {"name": "z", "sites": [0], "strength": 10.0},
        {"name": "lowering", "sites": [0], "strength": 10.0},
    ]
    noise_model = NoiseModel(processes=noise_processes)
    params = AnalogSimParams(dt=0.1, elapsed_time=1.0)

    pt = run(op, params, num_trajectories=100, noise_model=noise_model)

    # Fully depolarized state is 0.5*I, Holevo information should be 0.0
    h = pt.holevo_information(base=2)
    assert np.isclose(h, 0.0, atol=0.1)


def _reconstruct_state(expectations: dict[str, float]) -> NDArray[np.complex128]:
    """Reconstructs single-qubit density matrix from Pauli expectations.

    Returns:
        NDArray[np.complex128]: The reconstructed single-qubit density matrix.
    """
    eye = np.eye(2, dtype=complex)
    x_matrix = X().matrix
    y_matrix = Y().matrix
    z_matrix = Z().matrix

    return 0.5 * (eye + expectations["x"] * x_matrix + expectations["y"] * y_matrix + expectations["z"] * z_matrix)


def test_pt_prediction_consistency() -> None:
    """Verify ProcessTensor.predict_final_state consistency with direct simulation."""
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    params = AnalogSimParams(dt=0.1, order=1, elapsed_time=0.1)

    timesteps = [0.1, 0.1]
    pt = run(op, params, timesteps=timesteps, num_trajectories=10)

    basis_set = get_basis_states()
    basis_rhos = [b[2] for b in basis_set]
    duals = calculate_dual_frame(basis_rhos)

    # Predicting for a sequence that was used in tomography (should match averaged data)
    # Sequence [|0>, |0>]
    rho_seq = [basis_rhos[0], basis_rhos[0]]
    rho_pred = pt.predict_final_state(rho_seq, duals)

    # Stored value in tensor
    vec_stored = pt.tensor[:, 0, 0]
    rho_stored = vec_stored.reshape(2, 2)

    np.testing.assert_allclose(rho_pred, rho_stored, atol=1e-6)


def get_standard_basis() -> list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]]:
    """Returns the standard 6-state Pauli basis for testing.

    Returns:
        list[tuple[str, NDArray[np.complex128], NDArray[np.complex128]]]: Standard 6-state basis.
    """
    psi_0 = np.array([1, 0], dtype=complex)
    psi_1 = np.array([0, 1], dtype=complex)
    psi_plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    psi_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    psi_i_plus = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    psi_i_minus = np.array([1, -1j], dtype=complex) / np.sqrt(2)

    basis = [
        ("zeros", psi_0),
        ("ones", psi_1),
        ("x+", psi_plus),
        ("x-", psi_minus),
        ("y+", psi_i_plus),
        ("y-", psi_i_minus),
    ]
    return [(name, psi, np.outer(psi, psi.conj())) for name, psi in basis]


def test_holevo_information_conditional() -> None:
    """Test conditional Holevo information."""
    # Construct a 2-step process where output depends ONLY on step 0.
    basis = get_standard_basis()
    num_frames = len(basis)

    tensor = np.zeros((4, num_frames, num_frames), dtype=complex)

    for i in range(num_frames):
        rho_i = basis[i][2]
        vec_i = rho_i.reshape(-1)
        for j in range(num_frames):
            tensor[:, i, j] = vec_i

    pt = ProcessTensor(tensor, [1.0, 1.0])

    # CASE 1: Fix step 0 to index 0.
    # All sequences with step 0 = index 0 have the SAME output rho_0.
    # S(rho_avg) = S(|0><0|) = 0
    # S(rho_seq) = 0
    h_cond_0 = pt.holevo_information_conditional(fixed_step=0, fixed_idx=0, base=2)
    assert np.isclose(h_cond_0, 0.0, atol=1e-10)

    # CASE 2: Fix step 1 to index 0.
    # The output still varies with step 0.
    # It acts like an identity channel from step 0 to output.
    # S(rho_avg) = 1 (maximally mixed average of basis)
    # S(rho_seq) = 0 (pure outputs)
    h_cond_1 = pt.holevo_information_conditional(fixed_step=1, fixed_idx=0, base=2)
    assert np.isclose(h_cond_1, 1.0, atol=1e-10)


def test_algebraic_consistency() -> None:
    """Check algebraic consistency for various operators."""
    rng = np.random.default_rng(42)
    # 0. Handle default timesteps
    l_size = 2
    op = MPO.ising(l_size, J=1.0, g=1.0)
    # Reconstruct Process Tensor
    timesteps = [0.1]
    pt = run(op, AnalogSimParams(dt=0.05, order=2), timesteps)

    # Test prediction vs dual contraction
    basis_set = get_basis_states()
    basis_rhos = [b[2] for b in basis_set]
    duals = calculate_dual_frame(basis_rhos)

    max_error = 0.0

    num_sequences = 10
    for _ in range(num_sequences):
        seq = tuple(rng.integers(0, 6, size=1))  # Only 1 timestep for this test
        rho_seq = [basis_rhos[i] for i in seq]

        # Predict using duals
        rho_pred = pt.predict_final_state(rho_seq, duals)
        vec_pred = rho_pred.reshape(-1)

        # Stored value in tensor
        vec_stored = pt.tensor[(slice(None), *seq)]

        err = np.linalg.norm(vec_pred - vec_stored)
        max_error = max(max_error, err)

    assert max_error < 1e-10  # Should be very precise for 1 timestep


def _get_random_rho(rng: np.random.Generator) -> NDArray[np.complex128]:
    """Generate a random 2x2 density matrix.

    Returns:
        NDArray[np.complex128]: A random 2x2 density matrix.
    """
    # Create random complex matrix
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def _sample_pure_state(rho: NDArray[np.complex128], rng: np.random.Generator) -> NDArray[np.complex128]:
    """Sample a pure state from the eigen-decomposition of rho.

    Returns:
        NDArray[np.complex128]: A sampled pure state.
    """
    evals, evecs = np.linalg.eigh(rho)
    # Ensure probabilities are non-negative and sum to 1
    evals = np.maximum(evals, 0)
    evals /= np.sum(evals)
    idx = rng.choice(len(evals), p=evals)
    return evecs[:, idx]


def test_held_out_prediction() -> None:
    """Verify PT prediction accuracy for random held-out mixed state sequences."""
    rng = np.random.default_rng(42)
    l_size = 2
    op = MPO.ising(length=l_size, J=1.0, g=0.5)

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
    rho_0 = _get_random_rho(rng)
    rho_1 = _get_random_rho(rng)
    rho_sequence = [rho_0, rho_1]

    basis_set = get_basis_states()
    duals = calculate_dual_frame([b[2] for b in basis_set])

    # 3. Predict final state using PT
    rho_pred = pt.predict_final_state(rho_sequence, duals)

    # 4. Direct Simulation of the sequence [rho_0, rho_1]
    num_direct_trajectories = 400
    results = []

    for _ in range(num_direct_trajectories):
        # Sample initial pure state from rho_0
        psi_0 = _sample_pure_state(rho_0, rng)
        state = MPS(length=l_size, state="zeros")
        state.tensors[0] = np.expand_dims(psi_0, axis=(1, 2)).astype(np.complex128)

        # Step 1: Evolution
        step_params = copy.deepcopy(params)
        step_params.elapsed_time = 0.1
        step_params.num_traj = 1
        step_params.get_state = True

        analog_tjm_1((0, state, None, step_params, op))
        assert step_params.output_state is not None
        state = step_params.output_state

        # Step 2: Intervention (Sample new state from rho_1)
        psi_1 = _sample_pure_state(rho_1, rng)
        _reprepare_site_zero(state, psi_1, rng, meas_basis="Z")

        # Step 3: Evolution
        step_params.output_state = None
        analog_tjm_1((0, state, None, step_params, op))
        assert step_params.output_state is not None
        state = step_params.output_state

        # Reconstruct rho from Pauli expectations
        expectations = {
            "x": state.expect(Observable(X(), sites=[0])),
            "y": state.expect(Observable(Y(), sites=[0])),
            "z": state.expect(Observable(Z(), sites=[0])),
        }
        rho_final = _reconstruct_state(expectations)
        results.append(rho_final)

    rho_direct = np.mean(results, axis=0)

    # 5. Compare
    assert np.allclose(rho_pred, rho_direct, atol=0.1)
