# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the simulator module in YAQS.

This module verifies the functionality of the simulator by testing both analog (Hamiltonian)
and circuit simulation branches. It includes tests for identity circuits, two-qubit operations,
long-range gate handling, circuit observable and shot modes, and error cases such as mismatched
qubit counts.
"""

# ignore non-lowercase variable names for physics notation
# ruff:file-ignore[non-lowercase-variable-in-function]

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs import (
    MPO,
    MPS,
    AnalogSimParams,
    DigitalSimParams,
    Hamiltonian,
    NoiseModel,
    Observable,
    Result,
    Simulator,
    State,
    simulator,
)
from mqt.yaqs.analog.analog_tjm import analog_tjm_2
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import XX, YY, ZZ, X, Z
from mqt.yaqs.core.random_utils import make_sample_rng
from tests.conftest import (
    LARGE_QASM2_STRING,
    SAMPLE_QASM3_STRING,
    YAQS_TEST_SEED,
    requires_qasm3_import,
    write_qasm_file,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_simulator_defaults() -> None:
    """Simulator() initializes with sensible defaults (parallel=True, auto mp_context)."""
    sim = Simulator()
    assert sim.parallel is True
    assert sim.show_progress is True
    assert sim.mp_context == "auto"
    assert sim.max_workers >= 1
    assert sim.max_retries == 10
    assert isinstance(sim.retry_exceptions, tuple)
    assert all(issubclass(exc, BaseException) for exc in sim.retry_exceptions)


def test_simulator_max_workers_resolution() -> None:
    """An explicit ``max_workers`` is preserved as-is and can be cleared."""
    sim = Simulator(max_workers=3)
    assert sim.max_workers == 3
    sim.max_workers = None
    assert sim.max_workers == Simulator().max_workers


def test_simulator_retry_exceptions_setter() -> None:
    """retry_exceptions can be reconfigured after construction."""
    sim = Simulator()
    sim.retry_exceptions = (ValueError,)
    assert sim.retry_exceptions == (ValueError,)


def test_simulator_parallel_serial_equivalence() -> None:
    """Parallel and serial execution yield identical results for deterministic runs."""
    length = 2
    state = State(length, initial="zeros")
    H = Hamiltonian.ising(length, J=1.0, g=0.5)

    def _build_params() -> AnalogSimParams:
        return AnalogSimParams(
            observables=[Observable(Z(), site) for site in range(length)],
            elapsed_time=0.4,
            dt=0.1,
            num_traj=4,
            max_bond_dim=4,
            svd_threshold=1e-9,
            order=1,
            sample_timesteps=False,
            random_seed=YAQS_TEST_SEED,
        )

    params_serial = _build_params()
    result_serial = Simulator(parallel=False, show_progress=False).run(state, H, params_serial)

    params_parallel = _build_params()
    result_parallel = Simulator(parallel=True, max_workers=2, show_progress=False).run(state, H, params_parallel)

    for serial_vals, parallel_vals in zip(
        result_serial.expectation_values, result_parallel.expectation_values, strict=False
    ):
        assert serial_vals is not None
        assert parallel_vals is not None
        np.testing.assert_allclose(serial_vals, parallel_vals, atol=1e-10)


def test_simulator_show_progress_disabled(capsys: pytest.CaptureFixture[str]) -> None:
    """``show_progress=False`` suppresses the tqdm bar."""
    num_qubits = 2
    state = State(num_qubits, initial="zeros")
    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=1)
    circuit.measure_all()
    sim_params = DigitalSimParams(shots=2, max_bond_dim=4)

    Simulator(parallel=False, show_progress=False).run(state, circuit, sim_params, None)

    captured = capsys.readouterr()
    assert "Running trajectories" not in captured.err
    assert "Running trajectories" not in captured.out


def test_simulator_run_returns_result() -> None:
    """:meth:`Simulator.run` returns a :class:`Result` holding all simulation outputs."""
    length = 2
    state = State(length, initial="zeros")
    H = Hamiltonian.ising(length, J=1.0, g=0.5)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=1,
        sample_timesteps=False,
    )

    result = Simulator(parallel=False, show_progress=False).run(state, H, sim_params)

    assert isinstance(result, Result)
    assert result.sim_params is sim_params


def test_simulator_module_does_not_export_run() -> None:
    """The free ``simulator.run`` function has been removed in favour of :class:`Simulator`."""
    assert not hasattr(simulator, "run"), "simulator.run should be removed; use Simulator.run instead."


def test_analog_simulation() -> None:
    """Test the branch for Hamiltonian simulation (analog simulation) using AnalogSimParams.

    This test creates an MPS of length 5 initialized to the "zeros" state and an Ising MPO operator.
    It also creates a NoiseModel with two processes ("lowering" and "pauli_z") and corresponding strengths.
    With AnalogSimParams configured for a two-site evolution (order=2) and sample_timesteps False,
    Simulator.run is called. The test then verifies that for each observable the results and trajectories have been
    correctly initialized and that the measurement results are approximately as expected.
    """
    length = 5
    initial_state = State(length, initial="zeros")

    H = Hamiltonian.ising(length, J=1, g=0.5)

    sim_params = AnalogSimParams(
        observables=[Observable(Z(), site) for site in range(length)],
        elapsed_time=1,
        dt=0.1,
        num_traj=10,
        max_bond_dim=4,
        svd_threshold=1e-6,
        order=2,
        sample_timesteps=False,
        random_seed=YAQS_TEST_SEED,
    )
    gamma = 0.1
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(length) for name in ["lowering", "pauli_z"]
    ])

    result = Simulator(show_progress=False).run(initial_state, H, sim_params, noise_model)

    expected_z = [
        0.748947146695782,
        0.8720515025769692,
        0.8652609567462763,
        0.8673233347433466,
        0.6872036335377433,
    ]
    for i in range(len(result.observables)):
        assert result.expectation_values[i] is not None, "Results was not initialized for AnalogSimParams."
        assert result.trajectories[i] is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(result.trajectories[i]) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(result.expectation_values[i]) == 1, "Results was not initialized for AnalogSimParams."
        # Noisy analog observable simulation can drift slightly across platforms / minimum dependency sets
        # due to floating-point reduction order and BLAS/LAPACK differences.
        assert np.isclose(np.real(result.expectation_values[i][0]), expected_z[i], atol=2e-4)


def test_analog_simulation_parallel_off() -> None:
    """Test the branch for Hamiltonian simulation (analog simulation) using AnalogSimParams, parallelization off.

    This test creates an MPS of length 5 initialized to the "zeros" state and an Ising MPO operator.
    It also creates a NoiseModel with two processes ("lowering" and "pauli_z") and corresponding strengths.
    With AnalogSimParams configured for a two-site evolution (order=2) and sample_timesteps False,
    Simulator.run is called. The test then verifies that for each observable the results and trajectories have been
    correctly initialized and that the measurement results are approximately as expected.

    Additionally, this tests that single-site observables can be initialized with a list of a single int for usability.
    """
    length = 5
    initial_state = State(length, initial="zeros")

    H = Hamiltonian.ising(length, J=1, g=0.5)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), site) for site in range(length)],
        elapsed_time=1,
        dt=0.1,
        num_traj=10,
        max_bond_dim=4,
        svd_threshold=1e-6,
        order=2,
        sample_timesteps=False,
        random_seed=YAQS_TEST_SEED,
    )
    gamma = 0.1
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(length) for name in ["lowering", "pauli_z"]
    ])

    result = Simulator(parallel=False, show_progress=False).run(initial_state, H, sim_params, noise_model)

    expected_z = [
        0.748947146695782,
        0.8720515025769692,
        0.8652609567462763,
        0.8673233347433466,
        0.6872036335377433,
    ]
    for i in range(len(result.observables)):
        assert result.expectation_values[i] is not None, "Results was not initialized for AnalogSimParams."
        assert result.trajectories[i] is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(result.trajectories[i]) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(result.expectation_values[i]) == 1, "Results was not initialized for AnalogSimParams."
        # Noisy analog observable simulation can drift slightly across platforms / minimum dependency sets
        # due to floating-point reduction order and BLAS/LAPACK differences.
        assert np.isclose(np.real(result.expectation_values[i][0]), expected_z[i], atol=2e-4)


def test_analog_simulation_get_state() -> None:
    """Test the Hamiltonian simulation (analog simulation) using AnalogSimParams without noise to get a statevector.

    This test creates an MPS of length 2 initialized to the "zeros" state and an Ising MPO operator.
    With sample_timesteps set to False, the test verifies for two-site (order=2) and single-site (order=1) that the
    resulting output statevector is correct.
    """
    for order in [1, 2]:
        length = 2
        initial_state = State(length, initial="zeros")

        H = Hamiltonian.ising(length, J=1, g=0.5)

        sim_params = AnalogSimParams(
            observables=[Observable(X(), length // 2)],
            elapsed_time=1,
            dt=0.1,
            num_traj=1,
            max_bond_dim=4,
            svd_threshold=1e-6,
            order=order,
            get_state=True,
            sample_timesteps=False,
        )

        result = Simulator(show_progress=False).run(initial_state, H, sim_params)
        assert result.output_state is not None
        assert isinstance(result.output_state, State)
        sv = result.output_state.mps.to_vec()

        expected = [
            3.48123000e-01 + 0.76996349j,
            0.00000000e00 + 0.349228j,
            0.00000000e00 + 0.349228j,
            -1.92179306e-01 - 0.07150749j,
        ]
        fidelity = np.abs(np.vdot(sv, expected)) ** 2
        np.testing.assert_allclose(1, fidelity)


def test_trapped_ion_position_grid_vector_and_mps_simulation_agree() -> None:
    """Noiseless vector and MPS evolution agree for a displaced ion in a static harmonic well."""
    initial_displacement = 1.0
    omega = 1.0
    half_period = np.pi / omega

    positions = np.linspace(-8.0, 8.0, 33, dtype=np.float64)
    grid_dim = len(positions)
    initial_grid_state = np.exp(-0.5 * (positions - initial_displacement) ** 2).astype(np.complex128)
    initial_grid_state /= np.linalg.norm(initial_grid_state)

    hamiltonian = Hamiltonian.from_mpo(MPO.trapped_ion(positions, masses=[1.0], omega=omega))
    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=half_period,
        dt=half_period / 16,
        num_traj=1,
        max_bond_dim=None,
        svd_threshold=1e-12,
        krylov_tol=1e-12,
        order=2,
        preset="exact",
        get_state=True,
        sample_timesteps=False,
    )

    vector_state = State(length=1, vector=initial_grid_state, physical_dimensions=[grid_dim])
    mps_state = State(length=1, tensors=[initial_grid_state.reshape(grid_dim, 1, 1)], physical_dimensions=[grid_dim])

    vector_result = Simulator(parallel=False, show_progress=False).run(vector_state, hamiltonian, sim_params, None)
    mps_result = Simulator(parallel=False, show_progress=False).run(mps_state, hamiltonian, sim_params, None)

    assert vector_result.output_state is not None
    assert mps_result.output_state is not None
    vector_final = vector_result.output_state.vector
    mps_final = mps_result.output_state.mps.to_vec()
    overlap = np.vdot(vector_final, mps_final)

    np.testing.assert_allclose(np.abs(overlap) ** 2, 1.0, atol=1e-12)
    # A displaced harmonic-oscillator ground state reaches the opposite turning point
    # after half a trap period. The tolerance accounts for the finite grid/discretized kinetic operator.
    np.testing.assert_allclose(
        float(np.sum(positions * np.abs(vector_final) ** 2)),
        -initial_displacement,
        atol=3e-2,
    )


def test_density_matrix_get_state() -> None:
    """density_matrix evolution returns the final density matrix when get_state=True."""
    psi = State(2, initial="zeros", representation="density_matrix")
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        get_state=True,
    )
    result = Simulator(show_progress=False).run(psi, h, sim_params, None)
    assert result.output_state is not None
    assert result.output_state.representation == "density_matrix"
    rho = result.output_state.density_matrix
    assert rho.shape == (4, 4)
    assert np.isclose(np.trace(rho), 1.0)


def test_density_matrix_get_state_noisy() -> None:
    """Noisy Lindblad evolution still returns the exact ensemble-averaged density matrix."""
    n_sites = 1
    initial_state = State(n_sites, initial="ones", representation="density_matrix")
    hamiltonian = Hamiltonian.ising(n_sites, J=0.0, g=0.0)
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    gamma = 1.0
    t = 1.0
    noise_model = NoiseModel(
        processes=[{"name": "destroy", "sites": [0], "strength": gamma, "matrix": sigma_minus}],
    )
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=t,
        dt=0.1,
        get_state=True,
    )
    result = Simulator(show_progress=False).run(initial_state, hamiltonian, sim_params, noise_model)
    assert result.output_state is not None
    rho = result.output_state.density_matrix
    expected = np.array(
        [[1.0 - np.exp(-gamma * t), 0.0], [0.0, np.exp(-gamma * t)]],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(rho, expected, atol=1e-4)
    assert np.isclose(np.trace(rho), 1.0)
    assert np.allclose(rho.imag, 0.0, atol=1e-10)


def test_density_matrix_non_qubit_physical_dimension() -> None:
    """Lindblad density-matrix evolution supports non-qubit local dimensions."""
    physical_dimension = 3
    rho_initial = np.zeros((physical_dimension, physical_dimension), dtype=np.complex128)
    rho_initial[2, 2] = 1.0
    initial_state = State(length=1, density_matrix=rho_initial, physical_dimensions=[physical_dimension])
    hamiltonian = Hamiltonian(
        sparse_matrix=scipy.sparse.csr_matrix((physical_dimension, physical_dimension), dtype=np.complex128),
        length=1,
        physical_dimension=physical_dimension,
    )

    lowering_21 = np.zeros((physical_dimension, physical_dimension), dtype=np.complex128)
    lowering_21[1, 2] = 1.0
    gamma = 0.7
    elapsed_time = 0.4
    noise_model = NoiseModel(
        processes=[{"name": "qutrit_decay_2_to_1", "sites": [0], "strength": gamma, "matrix": lowering_21}],
    )
    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=elapsed_time,
        dt=0.1,
        get_state=True,
    )

    result = Simulator(show_progress=False).run(initial_state, hamiltonian, sim_params, noise_model)

    assert result.output_state is not None
    assert result.output_state.length == 1
    assert result.output_state.physical_dimensions == [physical_dimension]
    rho = result.output_state.density_matrix
    expected = np.zeros_like(rho)
    expected[1, 1] = 1.0 - np.exp(-gamma * elapsed_time)
    expected[2, 2] = np.exp(-gamma * elapsed_time)
    np.testing.assert_allclose(rho, expected, atol=1e-4)


def test_density_matrix_get_state_at_elapsed_time() -> None:
    """get_state returns rho at elapsed_time, matching the fixed-dt final grid point."""
    n_sites = 1
    initial_state = State(n_sites, initial="ones", representation="density_matrix")
    hamiltonian = Hamiltonian.ising(n_sites, J=0.0, g=0.0)
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    gamma = 1.0
    elapsed_time = 0.3
    noise_model = NoiseModel(
        processes=[{"name": "destroy", "sites": [0], "strength": gamma, "matrix": sigma_minus}],
    )
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=elapsed_time,
        dt=0.1,
        get_state=True,
        sample_timesteps=False,
    )
    assert sim_params.times[-1] == pytest.approx(elapsed_time)
    result = Simulator(show_progress=False).run(initial_state, hamiltonian, sim_params, noise_model)
    assert result.output_state is not None
    rho = result.output_state.density_matrix
    expected = np.array(
        [[1.0 - np.exp(-gamma * elapsed_time), 0.0], [0.0, np.exp(-gamma * elapsed_time)]],
        dtype=np.complex128,
    )
    np.testing.assert_allclose(rho, expected, atol=1e-4)


def test_density_matrix_get_state_preserves_metadata() -> None:
    """Lindblad ``get_state`` copies lattice metadata onto ``result.output_state``."""
    pdim = 2
    initial_state = State(2, initial="zeros", representation="density_matrix", physical_dimensions=[pdim, pdim])
    hamiltonian = Hamiltonian.ising(2, J=0.0, g=0.0)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        get_state=True,
    )
    result = Simulator(show_progress=False).run(initial_state, hamiltonian, sim_params, None)
    assert result.output_state is not None
    assert result.output_state.length == 2
    assert result.output_state.physical_dimensions == [pdim, pdim]
    assert result.output_state.representation == "density_matrix"


def test_density_matrix_without_get_state_leaves_output_state_empty() -> None:
    """No ``output_state`` is stored when ``get_state`` is false for Lindblad runs."""
    initial_state = State(1, initial="ones", representation="density_matrix")
    hamiltonian = Hamiltonian.ising(1, J=0.0, g=0.0)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        get_state=False,
    )
    result = Simulator(show_progress=False).run(initial_state, hamiltonian, sim_params, None)
    assert result.output_state is None


@pytest.mark.parametrize(
    "state",
    [
        State(2, initial="zeros", representation="vector"),
        State(2, initial="zeros", representation="density_matrix"),
        State(vector=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)),
    ],
)
def test_circuit_run_rejects_non_mps_state(state: State) -> None:
    """Circuit simulation requires State.representation='mps'."""
    circuit = QuantumCircuit(2)
    sim_params = DigitalSimParams(observables=[Observable(Z(), 0)])
    with pytest.raises(ValueError, match=r"Circuit simulation requires State\.representation='mps'"):
        Simulator(show_progress=False).run(state, circuit, sim_params, None)


def test_digital_observables() -> None:
    """Test the circuit-based simulation branch using DigitalSimParams.

    This test constructs an MPS of length 5 (initialized to "zeros") and an Ising circuit with a CX gate.
    It configures DigitalSimParams with specified simulation parameters and a noise model (non-None).
    Simulator.run is then called, and the test verifies that the observables' results and trajectories
    are initialized correctly. Expected measurement outcomes are compared approximately to pre-defined values.
    """
    num_qubits = 5
    state = State(num_qubits, initial="zeros")

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()

    sim_params = DigitalSimParams(
        observables=[Observable(Z(), site) for site in range(num_qubits)],
        num_traj=10,
        max_bond_dim=4,
        krylov_tol=1e-12,
        random_seed=YAQS_TEST_SEED,
    )
    # Use a noise model that is not None so that sim_params.num_traj remains unchanged.
    gamma = 1e-3
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(num_qubits) for name in ["lowering", "pauli_z"]
    ])

    result = Simulator(show_progress=False).run(state, circuit, sim_params, noise_model)

    expected_z = [
        0.6733214071546825,
        0.8502664720526317,
        0.8709639049732125,
        0.8628627940961556,
        0.6730350827430835,
    ]
    for i in range(len(result.observables)):
        assert result.expectation_values[i] is not None, "Results was not initialized for AnalogSimParams."
        assert result.trajectories[i] is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(result.trajectories[i]) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(result.expectation_values[i]) == 1, "Results was not initialized for AnalogSimParams."
        # Noisy digital observable simulation can drift slightly across platforms / minimum dependency sets.
        assert np.isclose(np.real(result.expectation_values[i][0]), expected_z[i], atol=2e-4)


def test_digital_observables_no_noise() -> None:
    """Test the circuit-based simulation using DigitalSimParams without noise to get a statevector.

    This test constructs a 2-site Ising circuit and compares the output statevector with known values from qiskit.
    """
    num_qubits = 2
    circ = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circ.measure_all()

    state = State(length=num_qubits)

    sim_params = DigitalSimParams(observables=[Observable(Z(), 0)], max_bond_dim=16, get_state=True)

    result = Simulator(show_progress=False).run(state, circ, sim_params)
    assert result.output_state is not None
    assert isinstance(result.output_state, State)
    sv = result.output_state.mps.to_vec()

    expected = [0.34870601 + 0.7690227j, 0.03494528 + 0.34828721j, 0.03494528 + 0.34828721j, -0.19159629 - 0.07244828j]
    fidelity = np.abs(np.vdot(sv, expected)) ** 2
    np.testing.assert_allclose(1, fidelity)


def test_digital_observables_parallel_off() -> None:
    """Test the circuit-based simulation branch using DigitalSimParams, parallelization off.

    This test constructs an MPS of length 5 (initialized to "zeros") and an Ising circuit with a CX gate.
    It configures DigitalSimParams with specified simulation parameters and a noise model (non-None).
    Simulator.run is then called, and the test verifies that the observables' results and trajectories
    are initialized correctly. Expected measurement outcomes are compared approximately to pre-defined values.
    """
    num_qubits = 5
    state = State(num_qubits, initial="zeros")

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()

    sim_params = DigitalSimParams(
        observables=[Observable(Z(), site) for site in range(num_qubits)],
        num_traj=10,
        max_bond_dim=4,
        krylov_tol=1e-12,
        random_seed=YAQS_TEST_SEED,
    )
    # Use a noise model that is not None so that sim_params.num_traj remains unchanged.
    gamma = 1e-3
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(num_qubits) for name in ["lowering", "pauli_z"]
    ])

    result = Simulator(parallel=False, show_progress=False).run(state, circuit, sim_params, noise_model)

    expected_z = [
        0.6733214071546825,
        0.8502664720526317,
        0.8709639049732125,
        0.8628627940961556,
        0.6730350827430835,
    ]
    for i in range(len(result.observables)):
        assert result.expectation_values[i] is not None, "Results was not initialized for AnalogSimParams."
        assert result.trajectories[i] is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(result.trajectories[i]) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(result.expectation_values[i]) == 1, "Results was not initialized for AnalogSimParams."
        # Noisy digital observable simulation can drift slightly across platforms / minimum dependency sets.
        assert np.isclose(np.real(result.expectation_values[i][0]), expected_z[i], atol=2e-4)


def test_digital_shots_noise() -> None:
    """Test shot-based digital simulation with a non-None noise model.

    This test creates an MPS and an Ising circuit (with measurement) for a 5-qubit system.
    It sets up DigitalSimParams with a sufficient number of shots for statistical verification, max bond dimension,
    threshold, and window size, and a noise model with small strengths. After running Simulator.run, the test
    verifies that result.measurements holds one measurement dictionary per shot trajectory,
    and that the total number of shots recorded in result.counts equals the expected number.
    """
    num_qubits = 5
    initial_state = State(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=1)
    circuit.measure_all()

    sim_params = DigitalSimParams(shots=32, max_bond_dim=4, random_seed=YAQS_TEST_SEED)

    gamma = 1e-3
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(num_qubits) for name in ["lowering", "pauli_z"]
    ])

    result = Simulator(show_progress=False).run(initial_state, circuit, sim_params, noise_model)

    assert len(result.measurements) == sim_params.shots
    for measurement in result.measurements:
        assert isinstance(measurement, dict)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots, "Wrong number of shots in DigitalSimParams."


def test_digital_shots_no_noise() -> None:
    """Test shot-based digital simulation when the noise model is None.

    This test creates an MPS and an Ising circuit (with measurement) for a 5-qubit system,
    and configures DigitalSimParams with a sufficient number of shots. When noise_model is None,
    the noiseless shots path uses one trajectory and records one measurement dictionary. The test
    verifies that the measurements and results are consistent with this behavior.
    """
    num_qubits = 5
    initial_state = State(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=1)
    circuit.measure_all()
    sim_params = DigitalSimParams(shots=64, max_bond_dim=4)

    noise_model = None

    result = Simulator(show_progress=False).run(initial_state, circuit, sim_params, noise_model)

    assert len(result.measurements) == 1
    assert isinstance(result.measurements[0], dict), (
        "There should be only one measurement dict when noise-free shot simulation runs in one batch."
    )
    assert result.counts is not None
    max_value = max(result.counts.values())
    assert result.counts[0] == max_value, "Key 0 does not have the highest value."
    assert sum(result.counts.values()) == sim_params.shots, "Wrong number of shots in DigitalSimParams."


def test_digital_combined_observables_and_shots() -> None:
    """Observables and shots together fill expectations and counts without changing obs-only results."""
    num_qubits = 2
    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=2)
    circuit.measure_all()
    obs = [Observable(Z(), i) for i in range(num_qubits)]
    sim = Simulator(parallel=False, show_progress=False)

    obs_only = sim.run(
        State(num_qubits, initial="zeros"),
        circuit,
        DigitalSimParams(observables=obs, max_bond_dim=16, random_seed=YAQS_TEST_SEED),
    )
    combined = sim.run(
        State(num_qubits, initial="zeros"),
        circuit,
        DigitalSimParams(observables=obs, shots=64, max_bond_dim=16, random_seed=YAQS_TEST_SEED),
    )

    assert len(combined.expectation_values) == num_qubits
    for i in range(num_qubits):
        np.testing.assert_allclose(combined.expectation_values[i], obs_only.expectation_values[i])
    assert combined.counts is not None
    assert sum(combined.counts.values()) == 64
    assert combined.max_bond is not None


def test_digital_combined_observables_and_shots_noisy() -> None:
    """Noisy combined runs keep obs-only expectations and distribute shots across trajs."""
    num_qubits = 2
    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=2)
    circuit.measure_all()
    obs = [Observable(Z(), i) for i in range(num_qubits)]
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1e-3}])
    num_traj = 4
    shots = 10
    sim = Simulator(parallel=False, show_progress=False)

    obs_only = sim.run(
        State(num_qubits, initial="zeros"),
        circuit,
        DigitalSimParams(observables=obs, num_traj=num_traj, max_bond_dim=16, random_seed=YAQS_TEST_SEED),
        noise_model,
    )
    combined = sim.run(
        State(num_qubits, initial="zeros"),
        circuit,
        DigitalSimParams(
            observables=obs,
            shots=shots,
            num_traj=num_traj,
            max_bond_dim=16,
            random_seed=YAQS_TEST_SEED,
        ),
        noise_model,
    )

    assert len(combined.expectation_values) == num_qubits
    for i in range(num_qubits):
        np.testing.assert_allclose(combined.expectation_values[i], obs_only.expectation_values[i])
    assert combined.counts is not None
    assert sum(combined.counts.values()) == shots
    assert len(combined.measurements) == num_traj
    assert combined.max_bond is not None


@pytest.mark.parametrize("parallel", [False, True])
def test_digital_combined_noisy_shots_less_than_num_traj(*, parallel: bool) -> None:
    """When shots < num_traj, zero-shot trajs contribute no counts but still feed observables."""
    num_qubits = 2
    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=2)
    circuit.measure_all()
    obs = [Observable(Z(), i) for i in range(num_qubits)]
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [0], "strength": 1e-3}])
    num_traj = 4
    shots = 2
    sim = Simulator(parallel=parallel, max_workers=2 if parallel else None, show_progress=False)

    result = sim.run(
        State(num_qubits, initial="zeros"),
        circuit,
        DigitalSimParams(
            observables=obs,
            shots=shots,
            num_traj=num_traj,
            max_bond_dim=16,
            random_seed=YAQS_TEST_SEED,
        ),
        noise_model,
    )

    assert result.counts is not None
    assert sum(result.counts.values()) == shots
    assert len(result.measurements) == num_traj
    # Two trajectories get one shot each; the rest must be empty or None (never a phantom shot).
    per_traj_totals = [0 if m is None else sum(m.values()) for m in result.measurements]
    assert sorted(per_traj_totals) == [0, 0, 1, 1]
    assert len(result.expectation_values) == num_qubits
    for values in result.expectation_values:
        assert values is not None
        assert np.size(values) > 0
    assert result.max_bond is not None


def test_digital_shots_get_state() -> None:
    """Test the circuit-based simulation using DigitalSimParams without noise to get a statevector.

    This test constructs a 2-site Ising circuit and compares the output statevector with known values from qiskit.
    """
    num_qubits = 2
    initial_state = State(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()
    sim_params = DigitalSimParams(shots=1, max_bond_dim=4, get_state=True)
    noise_model = None

    result = Simulator(show_progress=False).run(initial_state, circuit, sim_params, noise_model)
    assert result.output_state is not None
    assert isinstance(result.output_state, State)
    sv = result.output_state.mps.to_vec()

    expected = [0.34870601 + 0.7690227j, 0.03494528 + 0.34828721j, 0.03494528 + 0.34828721j, -0.19159629 - 0.07244828j]
    fidelity = np.abs(np.vdot(sv, expected)) ** 2
    np.testing.assert_allclose(1, fidelity)


def test_digital_shots_get_state_noise() -> None:
    """Test the circuit-based simulation using DigitalSimParams noise to get a statevector.

    This test constructs a 2-site Ising circuit and configures the DigitalSimParams to include a noise model and
    return the final state. Since the noisy simulation cannot return the statevector, an exception should be raised.
    """
    num_qubits = 2
    initial_state = State(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()
    sim_params = DigitalSimParams(shots=1, max_bond_dim=4, get_state=True)

    gamma = 1e-3
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(num_qubits) for name in ["lowering", "pauli_z"]
    ])

    with pytest.raises(ValueError, match=r"Cannot return state in noisy circuit simulation due to stochastics."):
        Simulator(show_progress=False).run(initial_state, circuit, sim_params, noise_model)


def test_mismatch() -> None:
    """Test that Simulator.run raises ValueError when state and circuit qubit counts mismatch.

    This test creates an MPS of length 5 and a circuit with length 4 (one fewer qubits),
    and verifies that an AssertionError with the appropriate message is raised.
    """
    num_qubits = 5
    initial_state = State(num_qubits)

    circuit = create_ising_circuit(L=num_qubits - 1, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()

    sim_params = DigitalSimParams(shots=1024, max_bond_dim=4)

    noise_model = None

    with pytest.raises(ValueError, match=r"qubit counts do not match"):
        Simulator(show_progress=False).run(initial_state, circuit, sim_params, noise_model)


def test_two_site_correlator_left_boundary() -> None:
    """Tests the expectation value of a two-site correlator in analog simulation at the left boundary.

    This test initializes an MPS in the |0> state and computes the expectation value of a two-site correlator
    at the left boundary.
    """
    L = 4
    J = 1
    g = 0.1
    H_0 = Hamiltonian.ising(L, J, g)

    state = State(L, initial="zeros")

    sim_params = AnalogSimParams(
        observables=[Observable(XX(), [0, 1]), Observable(YY(), [0, 1]), Observable(ZZ(), [0, 1])],
        elapsed_time=2.0,
        dt=0.1,
        max_bond_dim=4,
        sample_timesteps=True,
    )

    result = Simulator(show_progress=False).run(state, H_0, sim_params)

    expected_xx = np.array([
        0.00000000e00,
        6.66452664e-07,
        1.05502765e-05,
        5.26491078e-05,
        1.63138073e-04,
        3.88308907e-04,
        7.80632988e-04,
        1.39421223e-03,
        2.27990558e-03,
        3.48041964e-03,
        5.02562186e-03,
        6.92830295e-03,
        9.18066634e-03,
        1.17517711e-02,
        1.45861768e-02,
        1.76040037e-02,
        2.07025856e-02,
        2.37597698e-02,
        2.66388096e-02,
        2.91946781e-02,
        3.12814428e-02,
    ])

    expected_yy = np.array([
        0.00000000e00,
        3.93976077e-04,
        1.50510612e-03,
        3.13171916e-03,
        4.97179669e-03,
        6.66857157e-03,
        7.86413999e-03,
        8.25285998e-03,
        7.62641119e-03,
        5.90377710e-03,
        3.14185693e-03,
        -4.74449274e-04,
        -4.66068042e-03,
        -9.07484179e-03,
        -1.33660570e-02,
        -1.72219763e-02,
        -2.04075098e-02,
        -2.27889737e-02,
        -2.43403132e-02,
        -2.51311316e-02,
        -2.52992067e-02,
    ])

    expected_zz = np.array([
        1.00000000e00,
        9.99603371e-01,
        9.98453198e-01,
        9.96663218e-01,
        9.94405804e-01,
        9.91888962e-01,
        9.89329205e-01,
        9.86924424e-01,
        9.84830791e-01,
        9.83147041e-01,
        9.81908295e-01,
        9.81089938e-01,
        9.80620593e-01,
        9.80401653e-01,
        9.80329971e-01,
        9.80319743e-01,
        9.80319851e-01,
        9.80323822e-01,
        9.80370747e-01,
        9.80537040e-01,
        9.80920548e-01,
    ])

    results_xx = result.expectation_values[0]
    assert results_xx is not None
    np.testing.assert_allclose(results_xx, expected_xx, atol=1e-3)

    results_yy = result.expectation_values[1]
    assert results_yy is not None
    np.testing.assert_allclose(results_yy, expected_yy, atol=1e-3)

    results_zz = result.expectation_values[2]
    assert results_zz is not None
    np.testing.assert_allclose(results_zz, expected_zz, atol=1e-3)


def test_two_site_correlator_center() -> None:
    """Tests the expectation value of a two-site correlator in analog simulation at the center site.

    This test initializes an MPS in the |0> state and computes the expectation value of a two-site correlator
    at the center of the chain.
    """
    L = 4
    J = 1
    g = 0.1
    H_0 = Hamiltonian.ising(L, J, g)

    state = State(L, initial="zeros")

    sim_params = AnalogSimParams(
        observables=[
            Observable(XX(), [L // 2, L // 2 + 1]),
            Observable(YY(), [L // 2, L // 2 + 1]),
            Observable(ZZ(), [L // 2, L // 2 + 1]),
        ],
        elapsed_time=2.0,
        dt=0.1,
        max_bond_dim=4,
        sample_timesteps=True,
    )

    result = Simulator(show_progress=False).run(state, H_0, sim_params)

    expected_xx = np.array([
        0.00000000e00,
        6.66452664e-07,
        1.05502765e-05,
        5.26491078e-05,
        1.63138073e-04,
        3.88308907e-04,
        7.80632988e-04,
        1.39421223e-03,
        2.27990558e-03,
        3.48041964e-03,
        5.02562186e-03,
        6.92830295e-03,
        9.18066634e-03,
        1.17517711e-02,
        1.45861768e-02,
        1.76040037e-02,
        2.07025856e-02,
        2.37597698e-02,
        2.66388096e-02,
        2.91946781e-02,
        3.12814428e-02,
    ])

    expected_yy = np.array([
        0.00000000e00,
        3.93976077e-04,
        1.50510612e-03,
        3.13171916e-03,
        4.97179669e-03,
        6.66857157e-03,
        7.86413999e-03,
        8.25285998e-03,
        7.62641119e-03,
        5.90377710e-03,
        3.14185693e-03,
        -4.74449274e-04,
        -4.66068042e-03,
        -9.07484179e-03,
        -1.33660570e-02,
        -1.72219763e-02,
        -2.04075098e-02,
        -2.27889737e-02,
        -2.43403132e-02,
        -2.51311316e-02,
        -2.52992067e-02,
    ])

    expected_zz = np.array([
        1.00000000e00,
        9.99603371e-01,
        9.98453198e-01,
        9.96663218e-01,
        9.94405804e-01,
        9.91888962e-01,
        9.89329205e-01,
        9.86924424e-01,
        9.84830791e-01,
        9.83147041e-01,
        9.81908295e-01,
        9.81089938e-01,
        9.80620593e-01,
        9.80401653e-01,
        9.80329971e-01,
        9.80319743e-01,
        9.80319851e-01,
        9.80323822e-01,
        9.80370747e-01,
        9.80537040e-01,
        9.80920548e-01,
    ])

    results_xx = result.expectation_values[0]
    assert results_xx is not None
    np.testing.assert_allclose(results_xx, expected_xx, atol=1e-3)

    results_yy = result.expectation_values[1]
    assert results_yy is not None
    np.testing.assert_allclose(results_yy, expected_yy, atol=1e-3)

    results_zz = result.expectation_values[2]
    assert results_zz is not None
    np.testing.assert_allclose(results_zz, expected_zz, atol=1e-3)


def test_two_site_correlator_right_boundary() -> None:
    """Tests the expectation value of a two-site correlator in analog simulation at the right boundary.

    This test initializes an MPS in the |0> state and computes the expectation value of a two-site correlator
    at the right boundary.
    """
    L = 4
    J = 1
    g = 0.1
    H_0 = Hamiltonian.ising(L, J, g)

    state = State(L, initial="zeros")

    sim_params = AnalogSimParams(
        observables=[
            Observable(XX(), [L - 2, L - 1]),
            Observable(YY(), [L - 2, L - 1]),
            Observable(ZZ(), [L - 2, L - 1]),
        ],
        elapsed_time=2.0,
        dt=0.1,
        max_bond_dim=4,
        sample_timesteps=True,
    )
    result = Simulator(show_progress=False).run(state, H_0, sim_params)

    expected_xx = np.array([
        0.00000000e00,
        6.66452664e-07,
        1.05502765e-05,
        5.26491078e-05,
        1.63138073e-04,
        3.88308907e-04,
        7.80632988e-04,
        1.39421223e-03,
        2.27990558e-03,
        3.48041964e-03,
        5.02562186e-03,
        6.92830295e-03,
        9.18066634e-03,
        1.17517711e-02,
        1.45861768e-02,
        1.76040037e-02,
        2.07025856e-02,
        2.37597698e-02,
        2.66388096e-02,
        2.91946781e-02,
        3.12814428e-02,
    ])

    expected_yy = np.array([
        0.00000000e00,
        3.93976077e-04,
        1.50510612e-03,
        3.13171916e-03,
        4.97179669e-03,
        6.66857157e-03,
        7.86413999e-03,
        8.25285998e-03,
        7.62641119e-03,
        5.90377710e-03,
        3.14185693e-03,
        -4.74449274e-04,
        -4.66068042e-03,
        -9.07484179e-03,
        -1.33660570e-02,
        -1.72219763e-02,
        -2.04075098e-02,
        -2.27889737e-02,
        -2.43403132e-02,
        -2.51311316e-02,
        -2.52992067e-02,
    ])

    expected_zz = np.array([
        1.00000000e00,
        9.99603371e-01,
        9.98453198e-01,
        9.96663218e-01,
        9.94405804e-01,
        9.91888962e-01,
        9.89329205e-01,
        9.86924424e-01,
        9.84830791e-01,
        9.83147041e-01,
        9.81908295e-01,
        9.81089938e-01,
        9.80620593e-01,
        9.80401653e-01,
        9.80329971e-01,
        9.80319743e-01,
        9.80319851e-01,
        9.80323822e-01,
        9.80370747e-01,
        9.80537040e-01,
        9.80920548e-01,
    ])

    results_xx = result.expectation_values[0]
    assert results_xx is not None
    np.testing.assert_allclose(results_xx, expected_xx, atol=1e-3)

    results_yy = result.expectation_values[1]
    assert results_yy is not None
    np.testing.assert_allclose(results_yy, expected_yy, atol=1e-3)

    results_zz = result.expectation_values[2]
    assert results_zz is not None
    np.testing.assert_allclose(results_zz, expected_zz, atol=1e-3)


def test_two_site_correlator_center_circuit() -> None:
    """Tests the expectation value of a two-site correlator in circuit simulation at the center site.

    This test initializes an MPS in the |0> state and computes the expectation value of a two-site correlator
    at the center of the chain.
    """
    L = 4
    J = 1
    g = 0.1
    circ = create_ising_circuit(L=L, J=J, g=g, dt=0.1, timesteps=20)
    state = State(L, initial="zeros")

    sim_params = DigitalSimParams(
        observables=[
            Observable(XX(), [L // 2, L // 2 + 1]),
            Observable(YY(), [L // 2, L // 2 + 1]),
            Observable(ZZ(), [L // 2, L // 2 + 1]),
        ],
        max_bond_dim=4,
    )

    result = Simulator(show_progress=False).run(state, circ, sim_params)

    expected_xx = np.array([3.12811457e-02])
    expected_yy = np.array([-2.52988868e-02])
    expected_zz = np.array([9.80920787e-01])

    results_xx = result.expectation_values[0]
    assert results_xx is not None
    np.testing.assert_allclose(results_xx, expected_xx, atol=2e-3)

    results_yy = result.expectation_values[1]
    assert results_yy is not None
    np.testing.assert_allclose(results_yy, expected_yy, atol=2e-3)

    results_zz = result.expectation_values[2]
    assert results_zz is not None
    np.testing.assert_allclose(results_zz, expected_zz, atol=2e-3)


def test_transmon_simulation() -> None:
    """Tests if a SWAP gate is implemented correctly.

    This test creates a mixed-dimensional coupled transmon system and implements a SWAP gate.
    """
    length = 3  # Qubit - resonator - qubit
    qubit_dim = 3
    resonator_dim = 3
    w_q = 4 / (2 * np.pi)
    w_r = 4 / (2 * np.pi)
    alpha = -0.3 / (2 * np.pi)
    g = 0.2 / (2 * np.pi)

    H_0 = Hamiltonian.coupled_transmon(
        length=length,
        qubit_dim=qubit_dim,
        resonator_dim=resonator_dim,
        qubit_freq=w_q,
        resonator_freq=w_r,
        anharmonicity=alpha,
        coupling=g,
    )

    state = State(
        length, initial="basis", basis_string="100", physical_dimensions=[qubit_dim, resonator_dim, qubit_dim]
    )
    T_swap = np.pi / (np.sqrt(2) * g)

    sim_params = AnalogSimParams(
        observables=[Observable(bitstring) for bitstring in ["000", "001", "010", "011", "100", "101", "110", "111"]],
        elapsed_time=T_swap,
        dt=T_swap / 100,
        sample_timesteps=False,
    )
    result = Simulator(show_progress=False).run(state, H_0, sim_params)

    res0 = result.expectation_values[0]
    assert res0 is not None, "Expected results to be set by Simulator.run"
    # Initialize leakage as a numpy array of ones:
    leakage = np.ones_like(res0)

    for meas, res in zip(result.observables, result.expectation_values, strict=True):
        assert hasattr(meas.gate, "bitstring")
        assert res is not None, f"No results for bitstring {meas.gate.bitstring!r}"

        # subtract elementwise
        leakage -= res

        # use meas.bitstring, not meas.gate.bitstring
        if meas.gate.bitstring == "111":
            # small pop in 111
            np.testing.assert_array_less(np.max(res), 1e-2)
        elif meas.gate.bitstring == "100":
            np.testing.assert_allclose(res[-1], 0, atol=5e-2)
        elif meas.gate.bitstring == "001":
            np.testing.assert_allclose(res[-1], 1, atol=1e-1)
        elif meas.gate.bitstring == "010":
            np.testing.assert_allclose(res[-1], 0, atol=5e-2)

    # finally check total leakage
    np.testing.assert_array_less(leakage, 5e-2)


def test_analog_result_observables_preserve_user_order() -> None:
    """Analog runs must preserve user observable order on Result."""
    state = State(2, initial="zeros")
    H = Hamiltonian.ising(2, J=1.0, g=0.7)
    requested = [Observable(Z(), 1), Observable(X(), 0), Observable(Z(), 0)]
    sim_params = AnalogSimParams(
        observables=requested,
        elapsed_time=0.1,
        dt=0.1,
        num_traj=1,
        get_state=True,
        sample_timesteps=False,
        preset="exact",
    )

    result = Simulator(parallel=False, show_progress=False).run(state, H, sim_params)

    assert result.output_state is not None
    vec = result.output_state.mps.to_vec()
    n = int(np.log2(vec.size))

    assert len(result.observables) == len(requested)
    for i, (got_obs, req_obs) in enumerate(zip(result.observables, requested, strict=True)):
        assert got_obs.gate.name == req_obs.gate.name
        assert got_obs.sites == req_obs.sites

        label = ["I"] * n
        site = got_obs.sites[0] if isinstance(got_obs.sites, list) else got_obs.sites
        assert isinstance(site, int)
        label[n - 1 - site] = got_obs.gate.name.upper()
        expected = float(np.real(Statevector(vec).expectation_value(Pauli("".join(label)))))
        got = float(np.real(result.expectation_values[i][-1]))
        assert got == pytest.approx(expected, abs=1e-10)


def test_scheduled_jump_single_site() -> None:
    """Tests a scheduled Pauli-X flip on a single qubit."""
    L = 1
    T = 1.0
    dt = 0.1
    jump_time = 0.5

    # Initial state |0>
    state = State(L, initial="zeros")

    # Scheduled X jump at t=0.5
    scheduled_jumps = [{"time": jump_time, "sites": [0], "name": "x"}]
    noise_model = NoiseModel(scheduled_jumps=scheduled_jumps)

    # Measure Z on site 0
    z_obs = Observable(Z(), sites=0)
    sim_params = AnalogSimParams(
        elapsed_time=T,
        dt=dt,
        num_traj=1,
        observables=[z_obs],
    )

    # Use a vacuum Hamiltonian (all zeros) for pure jump dynamics
    hamiltonian = Hamiltonian.ising(L, 0.0, 0.0)

    result = Simulator(show_progress=False).run(state, hamiltonian, sim_params, noise_model=noise_model)

    results = result.expectation_values[0]
    assert results is not None

    np.testing.assert_allclose(results[:5], 1.0, atol=1e-10)
    np.testing.assert_allclose(results[5:], -1.0, atol=1e-10)


def test_scheduled_jump_two_site() -> None:
    """Tests a scheduled XX jump on two qubits."""
    L = 2
    T = 0.4
    dt = 0.1
    jump_time = 0.2

    # Initial state |00>
    state = State(L, initial="zeros")

    # Scheduled XX jump at t=0.2
    scheduled_jumps = [{"time": jump_time, "sites": [0, 1], "name": "crosstalk_xx"}]
    noise_model = NoiseModel(scheduled_jumps=scheduled_jumps)

    # Measure ZZ on site 0, 1
    zz_obs = Observable(ZZ(), sites=[0, 1])
    sim_params = AnalogSimParams(
        elapsed_time=T,
        dt=dt,
        num_traj=1,
        observables=[zz_obs],
    )

    # Vacuum Hamiltonian
    hamiltonian = Hamiltonian.ising(L, 0.0, 0.0)

    result = Simulator(show_progress=False).run(state, hamiltonian, sim_params, noise_model=noise_model)

    results = result.expectation_values[0]
    assert results is not None

    # Reset state for second run to verify dynamics again with a different observable
    state = State(L, initial="zeros")

    sim_params = AnalogSimParams(
        observables=[Observable(Z(), sites=0)],
        elapsed_time=T,
        dt=dt,
        num_traj=1,
    )
    result = Simulator(show_progress=False).run(state, hamiltonian, sim_params, noise_model=noise_model)

    results = result.expectation_values[0]
    assert results is not None
    # t=0.0 (0), 0.1 (1), 0.2 (2) -> flip.
    np.testing.assert_allclose(results[:2], 1.0, atol=1e-10)
    np.testing.assert_allclose(results[2:], -1.0, atol=1e-10)


def test_run_vector_preset_without_materialized_mps() -> None:
    """Analog run with vector representation uses encoded dense state, not MPS."""
    length = 3
    state = State(length, initial="zeros", representation="vector")
    with pytest.raises(RuntimeError, match="MPS is not available"):
        _ = state.mps
    hamiltonian = Hamiltonian.ising(length, 1.0, 0.5)
    obs = Observable("z", sites=[0])
    params = AnalogSimParams(
        observables=[obs],
        elapsed_time=0.1,
        dt=0.1,
    )
    result = Simulator(show_progress=False).run(state, hamiltonian, params, None)
    assert result.expectation_values[0] is not None
    assert state.representation == "vector"
    with pytest.raises(RuntimeError, match="MPS is not available"):
        _ = state.mps


def test_run_density_matrix_preset_without_materialized_mps() -> None:
    """Analog run with density_matrix representation uses encoded rho, not MPS."""
    length = 3
    state = State(length, initial="zeros", representation="density_matrix")
    with pytest.raises(RuntimeError, match="MPS is not available"):
        _ = state.mps
    hamiltonian = Hamiltonian.ising(length, 1.0, 0.5)
    obs = Observable("z", sites=[0])
    params = AnalogSimParams(
        observables=[obs],
        elapsed_time=0.1,
        dt=0.1,
    )
    result = Simulator(show_progress=False).run(state, hamiltonian, params, None)
    assert result.expectation_values[0] is not None
    assert state.representation == "density_matrix"
    with pytest.raises(RuntimeError, match="MPS is not available"):
        _ = state.mps


def test_analog_run_rejects_mpo_operator() -> None:
    """Legacy MPO operators are not accepted by Simulator.run."""
    state = State(2, initial="zeros")
    mpo = MPO.ising(2, J=1.0, g=0.5)
    params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.1,
        dt=0.1,
    )
    sim = Simulator(show_progress=False)
    with pytest.raises(TypeError, match="Analog simulation requires a Hamiltonian operator"):
        sim.run(state, cast(Any, mpo), params, None)  # ruff:ignore[runtime-cast-value]  # cast is required to exercise the runtime TypeError guard for non-Hamiltonian operators


def test_analog_run_rejects_non_state_initial_state() -> None:
    """Analog simulation requires initial_state to be State or list[State]."""
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.1,
        dt=0.1,
    )
    sim = Simulator(show_progress=False)
    with pytest.raises(TypeError, match="Analog simulation requires initial_state to be a list or State"):
        sim.run(cast(Any, MPS(2, state="zeros")), h, params, None)  # ruff:ignore[runtime-cast-value]  # cast is required to exercise the runtime TypeError guard for non-State initial states


def test_analog_run_accepts_matrix_hamiltonian_with_mps_state() -> None:
    """TJM auto-converts dense Hamiltonians to MPO."""
    state = State(2, initial="zeros")
    h = Hamiltonian(matrix=np.eye(4, dtype=np.complex128))
    params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.1,
        dt=0.1,
    )
    result = Simulator(show_progress=False).run(state, h, params, None)
    assert result.expectation_values[0].shape[0] >= 1


def test_no_output_error() -> None:
    """Verify that no-output configurations are rejected."""
    num_qubits = 2
    state = State(num_qubits, initial="zeros")
    H = Hamiltonian.ising(num_qubits, J=1, g=0.5)
    sim = Simulator(show_progress=False)

    # 1. AnalogSimParams (No observables, get_state=False)
    sim_params_analog = AnalogSimParams(
        observables=[],
        elapsed_time=0.1,
        dt=0.1,
        get_state=False,
    )
    with pytest.raises(ValueError, match=r"No output specified: either observables or get_state must be set."):
        sim.run(state, H, sim_params_analog)

    # 2. DigitalSimParams (No observables, get_state=False) rejected by standalone run
    sim_params_digital = DigitalSimParams(
        observables=[],
        get_state=False,
    )
    circuit = QuantumCircuit(num_qubits)
    with pytest.raises(ValueError, match=r"No output specified: set observables, shots, and/or get_state."):
        sim.run(state, circuit, sim_params_digital)


def test_simulator_rejects_initial_state_list_with_non_state_elements() -> None:
    """``initial_state=[...]`` must contain only :class:`State` instances."""
    H = Hamiltonian.ising(2, J=1.0, g=0.5)
    params = AnalogSimParams(observables=[Observable(Z(), 0)], elapsed_time=0.1, dt=0.1)
    sim = Simulator(show_progress=False)
    bad_list = cast("Any", [State(2, initial="zeros"), MPS(2, state="zeros")])
    with pytest.raises(TypeError, match="initial_state list must contain only State objects"):
        sim.run(bad_list, H, params, None)


def test_circuit_simulation_rejects_state_list() -> None:
    """Circuit simulation does not support ``list[State]`` initial states."""
    circuit = create_ising_circuit(L=2, J=1.0, g=0.5, dt=0.1, timesteps=1)
    params = DigitalSimParams(observables=[Observable(Z(), 0)])
    states = [State(2, initial="zeros"), State(2, initial="ones")]
    with pytest.raises(TypeError, match="Circuit simulation requires a single State initial_state"):
        Simulator(show_progress=False).run(states, circuit, params, None)


def test_circuit_simulation_rejects_non_circuit_operator() -> None:
    """Circuit simulation requires a :class:`QuantumCircuit`."""
    state = State(2, initial="zeros")
    params = DigitalSimParams(observables=[Observable(Z(), 0)])
    bad_operator = cast("Any", Hamiltonian.ising(2, J=1.0, g=0.5))
    with pytest.raises(TypeError, match="Circuit simulation requires a QuantumCircuit operator"):
        Simulator(show_progress=False).run(state, bad_operator, params, None)


def test_circuit_simulation_rejects_non_state_initial_state() -> None:
    """Circuit simulation requires a :class:`State` initial state."""
    circuit = create_ising_circuit(L=2, J=1.0, g=0.5, dt=0.1, timesteps=1)
    params = DigitalSimParams(observables=[Observable(Z(), 0)])
    bad_state = cast("Any", MPS(2, state="zeros"))
    with pytest.raises(TypeError, match="Circuit simulation requires a State initial_state"):
        Simulator(show_progress=False).run(bad_state, circuit, params, None)


def test_digital_shots_parallel_returns_counts() -> None:
    """Parallel shot simulation aggregates per-shot counts via the worker pool."""
    num_qubits = 2
    state = State(num_qubits, initial="zeros")
    circuit = create_ising_circuit(L=num_qubits, J=1.0, g=0.5, dt=0.1, timesteps=1)
    circuit.measure_all()
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [i], "strength": 1e-3} for i in range(num_qubits)])
    sim_params = DigitalSimParams(shots=4, max_bond_dim=4, random_seed=YAQS_TEST_SEED)
    result = Simulator(parallel=True, max_workers=2, show_progress=False).run(state, circuit, sim_params, noise_model)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


def test_digital_observables_parallel_records_final_mps() -> None:
    """Noiseless digital observable simulation with ``get_state=True`` returns the output MPS.

    Noiseless digital observable runs always use one trajectory, so ``parallel=True``
    takes the serial execution fallback rather than the worker pool.
    """
    num_qubits = 2
    state = State(num_qubits, initial="zeros")
    circuit = create_ising_circuit(L=num_qubits, J=1.0, g=0.5, dt=0.1, timesteps=2)
    circuit.measure_all()
    sim_params = DigitalSimParams(
        observables=[Observable(Z(), 0)],
        num_traj=1,
        max_bond_dim=4,
        get_state=True,
    )
    result = Simulator(parallel=True, max_workers=2, show_progress=False).run(state, circuit, sim_params, None)
    assert result.output_state is not None
    assert isinstance(result.output_state, State)


def test_analog_simulation_vector_serial_get_state() -> None:
    """Deterministic vector MCWF runs return the final state vector through the serial path."""
    n_sites = 1
    state = State(n_sites, initial="zeros", representation="vector")
    hamiltonian = Hamiltonian.ising(n_sites, J=0.0, g=-1.0)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=1,
        get_state=True,
    )
    result = Simulator(parallel=False, show_progress=False).run(state, hamiltonian, sim_params, None)
    assert result.output_state is not None
    assert result.output_state.representation == "vector"


def test_analog_simulation_parallel_observables_no_state() -> None:
    """Noisy parallel analog runs aggregate trajectory observables without ``get_state``."""
    length = 2
    state = State(length, initial="zeros")
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
    noise = NoiseModel([{"name": "pauli_z", "sites": [i], "strength": 0.05} for i in range(length)])
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=2,
        max_bond_dim=4,
        random_seed=YAQS_TEST_SEED,
    )
    result = Simulator(parallel=True, max_workers=2, show_progress=False).run(state, hamiltonian, sim_params, noise)
    assert result.expectation_values[0] is not None
    assert result.runtime_cost is not None


def test_simulator_run_accepts_qasm2_path_object(tmp_path: Path) -> None:
    """Verify that Simulator.run accepts a QASM 2 file passed as a Path object."""
    qasm_file = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    state = State(6, initial="zeros")
    sim_params = DigitalSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_file, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


def test_simulator_run_accepts_qasm2_str_path(tmp_path: Path) -> None:
    """Verify that Simulator.run accepts a QASM 2 file passed as a str path."""
    qasm_file = str(write_qasm_file(tmp_path, LARGE_QASM2_STRING))
    state = State(6, initial="zeros")
    sim_params = DigitalSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_file, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


def test_simulator_run_accepts_qasm2_raw_string() -> None:
    """Verify that Simulator.run accepts a raw QASM 2 string (not a file path)."""
    state = State(6, initial="zeros")
    sim_params = DigitalSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, LARGE_QASM2_STRING, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


@requires_qasm3_import
def test_simulator_run_accepts_qasm3_path_object(tmp_path: Path) -> None:
    """Verify that Simulator.run accepts a QASM 3 file passed as a Path object."""
    qasm_file = write_qasm_file(tmp_path, SAMPLE_QASM3_STRING, filename="circuit3.qasm")
    state = State(2, initial="zeros")
    sim_params = DigitalSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_file, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


@requires_qasm3_import
def test_simulator_run_accepts_qasm3_str_path(tmp_path: Path) -> None:
    """Verify that Simulator.run accepts a QASM 3 file passed as a str path."""
    qasm_file = str(write_qasm_file(tmp_path, SAMPLE_QASM3_STRING, filename="circuit3.qasm"))
    state = State(2, initial="zeros")
    sim_params = DigitalSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_file, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


def test_simulator_run_observables_accepts_qasm_path(tmp_path: Path) -> None:
    """Verify that Simulator.run with DigitalSimParams accepts a QASM file passed as a Path."""
    qasm_file = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    state = State(6, initial="zeros")
    sim_params = DigitalSimParams(observables=[Observable(Z(), 0)], num_traj=1, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_file, sim_params)
    assert result.expectation_values[0] is not None


def test_simulator_run_observables_accepts_qasm_string(tmp_path: Path) -> None:
    """Verify that Simulator.run with DigitalSimParams accepts a QASM file passed as a str path."""
    qasm_string = str(write_qasm_file(tmp_path, LARGE_QASM2_STRING))
    state = State(6, initial="zeros")
    sim_params = DigitalSimParams(observables=[Observable(Z(), 0)], num_traj=1, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_string, sim_params)
    assert result.expectation_values[0] is not None


@requires_qasm3_import
def test_simulator_run_observables_accepts_qasm3_raw_string() -> None:
    """Verify that Simulator.run with DigitalSimParams accepts a raw OpenQASM 3 string."""
    state = State(2, initial="zeros")
    sim_params = DigitalSimParams(observables=[Observable(Z(), 0)], num_traj=1, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, SAMPLE_QASM3_STRING, sim_params)
    assert result.expectation_values[0] is not None


def test_simulator_run_analog_rejects_str_operator() -> None:
    """Analog simulation with a str operator requires a Hamiltonian, not OpenQASM."""
    state = State(2, initial="zeros")
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        elapsed_time=0.1,
        dt=0.1,
        num_traj=1,
        sample_timesteps=False,
    )
    with pytest.raises(TypeError, match="Hamiltonian"):
        Simulator(parallel=False, show_progress=False).run(state, "not-a-path.qasm", sim_params)


@requires_qasm3_import
def test_simulator_run_accepts_qasm3_raw_string_shots() -> None:
    """Verify that Simulator.run with DigitalSimParams accepts a raw OpenQASM 3 string."""
    state = State(2, initial="zeros")
    sim_params = DigitalSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, SAMPLE_QASM3_STRING, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


def test_simulator_run_qasm_path_and_string_observables_match(tmp_path: Path) -> None:
    """Digital observable simulation with fixed seed agrees for path and raw OpenQASM inputs."""
    qasm_path = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    state = State(6, initial="zeros")
    sim_params = DigitalSimParams(
        observables=[Observable(Z(), 0)],
        num_traj=1,
        max_bond_dim=4,
        random_seed=YAQS_TEST_SEED,
    )
    path_result = Simulator(parallel=False, show_progress=False).run(state, qasm_path, sim_params)
    string_result = Simulator(parallel=False, show_progress=False).run(state, LARGE_QASM2_STRING, sim_params)
    np.testing.assert_array_equal(path_result.expectation_values[0], string_result.expectation_values[0])


def test_scheduled_jumps_rejected_for_mcwf_and_lindblad() -> None:
    """Scheduled jumps are unsupported outside single-State analog MPS TJM."""
    hamiltonian = Hamiltonian.ising(2, J=1.0, g=0.5)
    noise = NoiseModel(scheduled_jumps=[{"time": 0.0, "sites": [0], "name": "x"}])
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        dt=0.1,
        elapsed_time=0.1,
        num_traj=1,
    )
    for representation in ("vector", "density_matrix"):
        state = State(2, representation=representation)
        with pytest.raises(ValueError, match="scheduled_jumps"):
            Simulator(show_progress=False).run(state, hamiltonian, sim_params, noise)


def test_scheduled_jumps_rejected_for_ensemble() -> None:
    """list[State] unitary ensemble rejects scheduled jumps."""
    hamiltonian = Hamiltonian.ising(2, J=1.0, g=0.5)
    noise = NoiseModel(scheduled_jumps=[{"time": 0.0, "sites": [0], "name": "x"}])
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        dt=0.1,
        elapsed_time=0.1,
        num_traj=1,
    )
    states = [State(2), State(2)]
    with pytest.raises(ValueError, match="scheduled_jumps"):
        Simulator(show_progress=False).run(states, hamiltonian, sim_params, noise)


def test_scheduled_jump_off_grid_rejected() -> None:
    """Scheduled jump times must lie on sim_params.times."""
    hamiltonian = Hamiltonian.ising(2, J=1.0, g=0.5)
    noise = NoiseModel(scheduled_jumps=[{"time": 0.05, "sites": [0], "name": "x"}])
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        dt=0.1,
        elapsed_time=0.2,
        num_traj=1,
    )
    with pytest.raises(ValueError, match="time grid"):
        Simulator(show_progress=False).run(State(2), hamiltonian, sim_params, noise)


def test_scheduled_jumps_rejected_for_order_2() -> None:
    """Scheduled jumps with order=2 are rejected (incorrect dual application)."""
    hamiltonian = Hamiltonian(matrix=np.zeros((2, 2), dtype=complex))
    noise = NoiseModel(scheduled_jumps=[{"time": 0.1, "sites": [0], "name": "x"}])
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        dt=0.1,
        elapsed_time=0.3,
        num_traj=1,
        order=2,
    )
    with pytest.raises(ValueError, match="order=1"):
        Simulator(show_progress=False).run(State(1), hamiltonian, sim_params, noise)


def test_scheduled_jump_at_t0_order_1_flips_z() -> None:
    """Order-1 scheduled X jump at t=0 is applied before the initial sample."""
    hamiltonian = Hamiltonian(matrix=np.zeros((2, 2), dtype=complex))
    noise = NoiseModel(scheduled_jumps=[{"time": 0.0, "sites": [0], "name": "x"}])
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        dt=0.1,
        elapsed_time=0.3,
        num_traj=1,
        order=1,
        get_state=True,
    )
    result = Simulator(show_progress=False).run(State(1, initial="zeros"), hamiltonian, sim_params, noise)
    z = np.asarray(result.expectation_values[0], dtype=float)
    np.testing.assert_allclose(z, -1.0, atol=1e-10)
    # Observables and final state agree after the t=0 jump.
    assert result.output_state is not None
    final_z = float(result.output_state.mps.expect(Observable(Z(), 0)))
    assert final_z == pytest.approx(-1.0)


def test_scheduled_jump_at_t0_final_only_elapsed_zero() -> None:
    """Final-only elapsed_time=0 still records observables after a t=0 scheduled jump."""
    hamiltonian = Hamiltonian(matrix=np.zeros((2, 2), dtype=complex))
    noise = NoiseModel(scheduled_jumps=[{"time": 0.0, "sites": [0], "name": "x"}])
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        dt=0.1,
        elapsed_time=0.0,
        num_traj=1,
        order=1,
        sample_timesteps=False,
        get_state=True,
    )
    result = Simulator(show_progress=False).run(State(1, initial="zeros"), hamiltonian, sim_params, noise)
    z = float(np.asarray(result.expectation_values[0], dtype=complex).reshape(-1)[0].real)
    assert result.output_state is not None
    final_z = float(result.output_state.mps.expect(Observable(Z(), 0)))
    assert z == pytest.approx(-1.0)
    assert final_z == pytest.approx(-1.0)
    assert z == pytest.approx(final_z)


@pytest.mark.parametrize(
    ("elapsed_time", "sample_timesteps"),
    [
        (0.0, True),
        (0.0, False),
        (0.1, False),
    ],
)
def test_order_2_short_runs_return_observables_and_state(elapsed_time: float, *, sample_timesteps: bool) -> None:
    """Order-2 TJM handles elapsed_time in {0, dt} without IndexError or empty results."""
    hamiltonian = Hamiltonian.ising(2, J=1.0, g=0.5)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        dt=0.1,
        elapsed_time=elapsed_time,
        num_traj=1,
        order=2,
        sample_timesteps=sample_timesteps,
        get_state=True,
        random_seed=0,
    )
    result = Simulator(show_progress=False).run(State(2, initial="zeros"), hamiltonian, sim_params)
    z = np.asarray(result.expectation_values[0], dtype=complex).reshape(-1)
    assert result.output_state is not None
    assert np.isfinite(z.real).all()
    # |0> has Z=+1 at t=0; short unitary evolution keeps |Z| near 1.
    assert np.all(np.abs(z.real) > 0.5)

    if elapsed_time == pytest.approx(0.1) and not sample_timesteps:
        sampled = Simulator(show_progress=False).run(
            State(2, initial="zeros"),
            hamiltonian,
            AnalogSimParams(
                observables=[Observable(Z(), 0)],
                dt=0.1,
                elapsed_time=0.1,
                num_traj=1,
                order=2,
                sample_timesteps=True,
                get_state=True,
                random_seed=0,
            ),
        )
        z_sampled_final = float(np.asarray(sampled.expectation_values[0], dtype=complex).reshape(-1)[-1].real)
        assert float(z.real[0]) == pytest.approx(z_sampled_final, abs=1e-10)


def test_order_2_zero_duration_final_only_skips_noise() -> None:
    """Final-only elapsed_time=0 must not apply F0 noise before measuring."""
    hamiltonian = Hamiltonian(matrix=np.zeros((2, 2), dtype=complex))
    # Lowering on |+>: unwanted F0 half-step dissipation shifts Z away from 0 (~0.025).
    noise = NoiseModel([{"name": "lowering", "sites": [0], "strength": 1.0}])
    sampled = Simulator(show_progress=False).run(
        State(1, initial="x+"),
        hamiltonian,
        AnalogSimParams(
            observables=[Observable(Z(), 0)],
            dt=0.1,
            elapsed_time=0.0,
            num_traj=1,
            order=2,
            sample_timesteps=True,
            random_seed=0,
        ),
        noise,
    )
    final_only = Simulator(show_progress=False).run(
        State(1, initial="x+"),
        hamiltonian,
        AnalogSimParams(
            observables=[Observable(Z(), 0)],
            dt=0.1,
            elapsed_time=0.0,
            num_traj=1,
            order=2,
            sample_timesteps=False,
            random_seed=0,
        ),
        noise,
    )
    z_sampled = float(np.asarray(sampled.expectation_values[0], dtype=complex).reshape(-1)[0].real)
    z_final = float(np.asarray(final_only.expectation_values[0], dtype=complex).reshape(-1)[0].real)
    assert z_sampled == pytest.approx(0.0, abs=1e-10)
    assert z_final == pytest.approx(0.0, abs=1e-10)
    assert z_final == pytest.approx(z_sampled, abs=1e-10)


def test_order_2_sample_rng_is_per_timestep_not_sequential() -> None:
    """Each measurement copy must get a fresh sample RNG keyed by timestep index."""
    hamiltonian = MPO.from_matrix(np.zeros((2, 2), dtype=complex), d=2)
    noise = NoiseModel([{"name": "pauli_z", "sites": [0], "strength": 1.0}])
    state = State(1, initial="x+").ensure_encoded("mps").mps
    # Intended three-point grid [0, dt, 2*dt]; sample calls at indices 1 and 2.
    dt = 0.1
    elapsed_time = 0.2

    class _ScriptedRng:
        """Deterministic jump draws: ``random()`` then optional ``choice()``."""

        def __init__(self, randoms: list[float], choices: list[int] | None = None) -> None:
            self._randoms = list(randoms)
            self._choices = list(choices or [])
            self._ri = 0
            self._ci = 0

        def random(self) -> float:
            value = self._randoms[self._ri]
            self._ri += 1
            return value

        def choice(self, a: int, p: list[float] | None = None) -> int:
            _ = (a, p)
            value = self._choices[self._ci] if self._ci < len(self._choices) else 0
            self._ci += 1
            return value

    def _run(*, sample_timesteps: bool) -> list[int]:
        timesteps: list[int] = []

        def _tracking_sample_rng(
            traj_idx: int,
            *,
            base_seed: int | None,
            timestep: int,
            stream_id: int | None = None,
        ) -> np.random.Generator:
            timesteps.append(timestep)
            return make_sample_rng(traj_idx, base_seed=base_seed, timestep=timestep, stream_id=stream_id)

        sim_params = AnalogSimParams(
            observables=[Observable(X(), 0)],
            dt=dt,
            elapsed_time=elapsed_time,
            num_traj=1,
            order=2,
            sample_timesteps=sample_timesteps,
            random_seed=0,
        )
        assert np.allclose(sim_params.times, [0.0, 0.1, 0.2])
        with patch("mqt.yaqs.analog.analog_tjm.make_sample_rng", side_effect=_tracking_sample_rng):
            analog_tjm_2((0, state, noise, sim_params, hamiltonian))
        return timesteps

    sampled_steps = _run(sample_timesteps=True)
    final_steps = _run(sample_timesteps=False)
    assert sampled_steps == [1, 2]
    assert final_steps == [2]

    # Shared sequential RNG: first sample consumes the jump draw; final sample then skips.
    # Final-only starts at the same script and jumps. Outcomes must differ.
    def _shared_factory(randoms: list[float], choices: list[int]) -> object:
        shared = _ScriptedRng(randoms, choices)

        def _factory(traj_idx: int, *, base_seed: int | None, timestep: int) -> _ScriptedRng:
            _ = (traj_idx, base_seed, timestep)
            return shared

        return _factory

    with patch(
        "mqt.yaqs.analog.analog_tjm.make_sample_rng",
        side_effect=_shared_factory([0.0, 0.999], [0]),
    ):
        sampled_bug = analog_tjm_2((
            0,
            state,
            noise,
            AnalogSimParams(
                observables=[Observable(X(), 0)],
                dt=dt,
                elapsed_time=elapsed_time,
                num_traj=1,
                order=2,
                sample_timesteps=True,
                random_seed=0,
            ),
            hamiltonian,
        ))
    with patch(
        "mqt.yaqs.analog.analog_tjm.make_sample_rng",
        side_effect=_shared_factory([0.0, 0.999], [0]),
    ):
        final_bug = analog_tjm_2((
            0,
            state,
            noise,
            AnalogSimParams(
                observables=[Observable(X(), 0)],
                dt=dt,
                elapsed_time=elapsed_time,
                num_traj=1,
                order=2,
                sample_timesteps=False,
                random_seed=0,
            ),
            hamiltonian,
        ))
    x_sampled = float(np.asarray(sampled_bug[0][0], dtype=complex).reshape(-1)[-1].real)
    x_final = float(np.asarray(final_bug[0][0], dtype=complex).reshape(-1)[0].real)
    assert x_sampled != pytest.approx(x_final, abs=1e-12)


def test_digital_rejects_nonadjacent_noise_matching_and_nonmatching() -> None:
    """Digital TJM rejects non-adjacent factorized noise even if a gate shares endpoints."""
    circuit = QuantumCircuit(3)
    circuit.cx(0, 2)  # matching endpoints for sites [0, 2]
    noise = NoiseModel([
        {"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 0.01},
    ])
    sim_params = DigitalSimParams(observables=[Observable(Z(), 0)], num_traj=1, max_bond_dim=8)
    with pytest.raises(ValueError, match="Digital TJM does not support non-adjacent"):
        Simulator(show_progress=False).run(State(3), circuit, sim_params, noise)

    circuit2 = QuantumCircuit(3)
    circuit2.cx(0, 1)  # non-matching endpoints
    with pytest.raises(ValueError, match="Digital TJM does not support non-adjacent"):
        Simulator(show_progress=False).run(State(3), circuit2, sim_params, noise)


def test_analog_longrange_crosstalk_xy_mps_runs() -> None:
    """Documented longrange_crosstalk_xy works on analog MPS TJM."""
    hamiltonian = Hamiltonian.ising(3, J=1.0, g=0.5)
    noise = NoiseModel([
        {"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 0.05},
    ])
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        dt=0.1,
        elapsed_time=0.2,
        num_traj=2,
        random_seed=0,
    )
    result = Simulator(show_progress=False).run(State(3), hamiltonian, sim_params, noise)
    assert result.expectation_values[0].shape[0] >= 1
