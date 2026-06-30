# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the simulator module in YAQS.

This module verifies the functionality of the simulator by testing both analog (Hamiltonian)
and circuit simulation branches. It includes tests for identity circuits, two-qubit operations,
long-range gate handling, weak and strong simulation modes, and error cases such as mismatched
qubit counts.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806, PLC2701

from __future__ import annotations

import importlib
import multiprocessing
import os
import sys
from typing import TYPE_CHECKING, Any, cast

import numba
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli, Statevector

from mqt.yaqs import Result, Simulator, simulator
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import (
    AnalogSimParams,
    Observable,
    StrongSimParams,
    WeakSimParams,
)
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import XX, YY, ZZ, X, Z
from mqt.yaqs.simulator import _expect_shot_counts, _get_parallel_context, worker_init
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
    """An explicit ``max_workers`` is preserved as-is."""
    sim = Simulator(max_workers=3)
    assert sim.max_workers == 3


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
    sim_params = WeakSimParams(shots=2, max_bond_dim=4)

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


def test_available_cpus_without_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path 1: SLURM_CPUS_ON_NODE *not* set.

    Should return multiprocessing.cpu_count().
    """
    # Ensure the env vars are absent
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)

    assert simulator.available_cpus() == multiprocessing.cpu_count()


def test_available_cpus_with_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path 2: SLURM_CPUS_ON_NODE is set.

    Should return that exact value.
    """
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.delenv("YAQS_MAX_WORKERS", raising=False)

    # Reload the module only if available_cpus caches anything at import;
    # here it's not necessary, but harmless:
    importlib.reload(simulator)

    assert simulator.available_cpus() == 8


def test_available_cpus_yaqs_max_workers_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``YAQS_MAX_WORKERS`` env var takes priority over xdist/SLURM/affinity."""
    monkeypatch.setenv("YAQS_MAX_WORKERS", "4")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    assert simulator.available_cpus() == 4


def test_available_cpus_yaqs_max_workers_malformed_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed ``YAQS_MAX_WORKERS`` is ignored; later detection logic runs."""
    monkeypatch.setenv("YAQS_MAX_WORKERS", "not-a-number")
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    assert simulator.available_cpus() == 1


def test_available_cpus_xdist_worker_returns_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Running inside an xdist worker pins ``available_cpus`` to 1."""
    monkeypatch.delenv("YAQS_MAX_WORKERS", raising=False)
    monkeypatch.setenv("PYTEST_XDIST_WORKER", "gw0")
    assert simulator.available_cpus() == 1


def test_available_cpus_slurm_malformed_falls_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed SLURM_* values are ignored; the function falls back to affinity/cpu_count."""
    monkeypatch.delenv("YAQS_MAX_WORKERS", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "0")
    assert simulator.available_cpus() >= 1


def test_threading_config() -> None:
    """Verify correct multiprocessing context and Numba threading configuration."""
    # 1. Context Selection
    ctx = _get_parallel_context()
    if sys.platform == "linux":
        # On Linux, we expect fork
        assert ctx.get_start_method() == "fork"
    else:
        # On Windows (win32) and macOS (darwin), we expect spawn
        assert ctx.get_start_method() == "spawn"

    # 2. Worker Initialization Logic
    # Verify _worker_init caps Numba threads

    # Save current state
    original_numba_threads = numba.get_num_threads()
    # Save environment variables that _limit_worker_threads modified
    env_snapshot = os.environ.copy()

    try:
        # Simulate worker init with strict thread cap
        worker_init({}, n_threads=1)

        # Check if Numba threads are set to 1
        assert numba.get_num_threads() == 1
        # Check if env var is set (best effort)
        assert os.environ.get("NUMBA_NUM_THREADS") == "1"

    finally:
        # Restore state
        numba.set_num_threads(original_numba_threads)

        # Restore environment variables
        # 1. Remove keys that were added
        for key in list(os.environ):
            if key not in env_snapshot:
                del os.environ[key]

        # 2. Restore keys that were modified/deleted
        for key, value in env_snapshot.items():
            if os.environ.get(key) != value:
                os.environ[key] = value


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
        0.6939175883763173,
        0.8723190598293048,
        0.8774367798552517,
        0.8642160639619357,
        0.6873260499377838,
    ]
    for i in range(len(result.observables)):
        assert result.expectation_values[i] is not None, "Results was not initialized for AnalogSimParams."
        assert result.trajectories[i] is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(result.trajectories[i]) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(result.expectation_values[i]) == 1, "Results was not initialized for AnalogSimParams."
        # Noisy strong simulation can drift slightly across platforms / minimum dependency sets
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
        0.6939175883763173,
        0.8723190598293048,
        0.8774367798552517,
        0.8642160639619357,
        0.6873260499377838,
    ]
    for i in range(len(result.observables)):
        assert result.expectation_values[i] is not None, "Results was not initialized for AnalogSimParams."
        assert result.trajectories[i] is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(result.trajectories[i]) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(result.expectation_values[i]) == 1, "Results was not initialized for AnalogSimParams."
        # Noisy strong simulation can drift slightly across platforms / minimum dependency sets
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


def test_density_matrix_get_state_rejected() -> None:
    """density_matrix evolution does not support returning an output state."""
    psi = State(2, initial="zeros", representation="density_matrix")
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        get_state=True,
    )
    with pytest.raises(ValueError, match=r"get_state=True is not supported for State\.representation='density_matrix'"):
        Simulator(show_progress=False).run(psi, h, sim_params, None)


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
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)])
    with pytest.raises(ValueError, match=r"Circuit simulation requires State\.representation='mps'"):
        Simulator(show_progress=False).run(state, circuit, sim_params, None)


def test_strong_simulation() -> None:
    """Test the circuit-based simulation branch using StrongSimParams.

    This test constructs an MPS of length 5 (initialized to "zeros") and an Ising circuit with a CX gate.
    It configures StrongSimParams with specified simulation parameters and a noise model (non-None).
    Simulator.run is then called, and the test verifies that the observables' results and trajectories
    are initialized correctly. Expected measurement outcomes are compared approximately to pre-defined values.
    """
    num_qubits = 5
    state = State(num_qubits, initial="zeros")

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()

    sim_params = StrongSimParams(
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
        0.6731226288088834,
        0.8628191799824898,
        0.8686777017191668,
        0.862819175965271,
        0.6731226287649416,
    ]
    for i in range(len(result.observables)):
        assert result.expectation_values[i] is not None, "Results was not initialized for AnalogSimParams."
        assert result.trajectories[i] is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(result.trajectories[i]) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(result.expectation_values[i]) == 1, "Results was not initialized for AnalogSimParams."
        # Noisy strong simulation can drift slightly across platforms / minimum dependency sets.
        assert np.isclose(np.real(result.expectation_values[i][0]), expected_z[i], atol=2e-4)


def test_strong_simulation_no_noise() -> None:
    """Test the circuit-based simulation using StrongSimParams without noise to get a statevector.

    This test constructs a 2-site Ising circuit and compares the output statevector with known values from qiskit.
    """
    num_qubits = 2
    circ = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circ.measure_all()

    state = State(length=num_qubits)

    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], max_bond_dim=16, get_state=True)

    result = Simulator(show_progress=False).run(state, circ, sim_params)
    assert result.output_state is not None
    assert isinstance(result.output_state, State)
    sv = result.output_state.mps.to_vec()

    expected = [0.34870601 + 0.7690227j, 0.03494528 + 0.34828721j, 0.03494528 + 0.34828721j, -0.19159629 - 0.07244828j]
    fidelity = np.abs(np.vdot(sv, expected)) ** 2
    np.testing.assert_allclose(1, fidelity)


def test_strong_simulation_parallel_off() -> None:
    """Test the circuit-based simulation branch using StrongSimParams, parallelization off.

    This test constructs an MPS of length 5 (initialized to "zeros") and an Ising circuit with a CX gate.
    It configures StrongSimParams with specified simulation parameters and a noise model (non-None).
    Simulator.run is then called, and the test verifies that the observables' results and trajectories
    are initialized correctly. Expected measurement outcomes are compared approximately to pre-defined values.
    """
    num_qubits = 5
    state = State(num_qubits, initial="zeros")

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()

    sim_params = StrongSimParams(
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
        0.6731226288088834,
        0.8628191799824898,
        0.8686777017191668,
        0.862819175965271,
        0.6731226287649416,
    ]
    for i in range(len(result.observables)):
        assert result.expectation_values[i] is not None, "Results was not initialized for AnalogSimParams."
        assert result.trajectories[i] is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(result.trajectories[i]) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(result.expectation_values[i]) == 1, "Results was not initialized for AnalogSimParams."
        # Noisy strong simulation can drift slightly across platforms / minimum dependency sets.
        assert np.isclose(np.real(result.expectation_values[i][0]), expected_z[i], atol=2e-4)


def test_weak_simulation_noise() -> None:
    """Test the weak simulation branch with a non-None noise model.

    This test creates an MPS and an Ising circuit (with measurement) for a 5-qubit system.
    It sets up WeakSimParams with a sufficient number of shots for statistical verification, max bond dimension,
    threshold, and window size, and a noise model with small strengths. After running Simulator.run, the test
    verifies that sim_params.num_traj equals the number of shots, that each measurement is a dictionary,
    and that the total number of shots recorded in result.counts equals the expected number.
    """
    num_qubits = 5
    initial_state = State(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=1)
    circuit.measure_all()

    sim_params = WeakSimParams(shots=32, max_bond_dim=4, random_seed=YAQS_TEST_SEED)

    gamma = 1e-3
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(num_qubits) for name in ["lowering", "pauli_z"]
    ])

    result = Simulator(show_progress=False).run(initial_state, circuit, sim_params, noise_model)

    assert len(result.measurements) == sim_params.shots
    for measurement in result.measurements:
        assert isinstance(measurement, dict)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots, "Wrong number of shots in WeakSimParams."


def test_weak_simulation_no_noise() -> None:
    """Test the weak simulation branch when the noise model is None.

    This test creates an MPS and an Ising circuit (with measurement) for a 5-qubit system,
    and configures WeakSimParams with a sufficient number of shots. When noise_model is None,
    the simulation should set sim_params.num_traj to 1. The test verifies that the measurements and results
    are consistent with this behavior.
    """
    num_qubits = 5
    initial_state = State(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=1)
    circuit.measure_all()
    sim_params = WeakSimParams(shots=64, max_bond_dim=4)

    noise_model = None

    result = Simulator(show_progress=False).run(initial_state, circuit, sim_params, noise_model)

    assert len(result.measurements) == 1
    assert isinstance(result.measurements[0], dict), (
        "There should be only one measurement dict when noise-free weak simulation runs in one batch."
    )
    assert result.counts is not None
    max_value = max(result.counts.values())
    assert result.counts[0] == max_value, "Key 0 does not have the highest value."
    assert sum(result.counts.values()) == sim_params.shots, "Wrong number of shots in WeakSimParams."


def test_weak_simulation_get_state() -> None:
    """Test the circuit-based simulation using WeakSimParams without noise to get a statevector.

    This test constructs a 2-site Ising circuit and compares the output statevector with known values from qiskit.
    """
    num_qubits = 2
    initial_state = State(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()
    sim_params = WeakSimParams(shots=1, max_bond_dim=4, get_state=True)
    noise_model = None

    result = Simulator(show_progress=False).run(initial_state, circuit, sim_params, noise_model)
    assert result.output_state is not None
    assert isinstance(result.output_state, State)
    sv = result.output_state.mps.to_vec()

    expected = [0.34870601 + 0.7690227j, 0.03494528 + 0.34828721j, 0.03494528 + 0.34828721j, -0.19159629 - 0.07244828j]
    fidelity = np.abs(np.vdot(sv, expected)) ** 2
    np.testing.assert_allclose(1, fidelity)


def test_weak_simulation_get_state_noise() -> None:
    """Test the circuit-based simulation using WeakSimParams noise to get a statevector.

    This test constructs a 2-site Ising circuit and configures the WeakSimParams to include a noise model and
    return the final state. Since the noisy simulation cannot return the statevector, an exception should be raised.
    """
    num_qubits = 2
    initial_state = State(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()
    sim_params = WeakSimParams(shots=1, max_bond_dim=4, get_state=True)

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

    sim_params = WeakSimParams(shots=1024, max_bond_dim=4)

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

    sim_params = StrongSimParams(
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
        sim.run(state, cast(Any, mpo), params, None)  # noqa: TC006  # cast is required to exercise the runtime TypeError guard for non-Hamiltonian operators


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
        sim.run(cast(Any, MPS(2, state="zeros")), h, params, None)  # noqa: TC006  # cast is required to exercise the runtime TypeError guard for non-State initial states


def test_analog_run_rejects_matrix_hamiltonian_with_mps_state() -> None:
    """TJM requires Hamiltonian.representation='mpo'."""
    state = State(2, initial="zeros")
    h = Hamiltonian(matrix=np.eye(4, dtype=np.complex128))
    params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.1,
        dt=0.1,
    )
    with pytest.raises(ValueError, match=r"TJM simulation requires Hamiltonian\.representation='mpo'"):
        Simulator(show_progress=False).run(state, h, params, None)


def test_no_output_error() -> None:
    """Verify that Simulator.run raises AssertionError when no output is specified."""
    num_qubits = 2
    state = State(num_qubits, initial="zeros")
    circ = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=1)
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

    # 2. StrongSimParams (No observables, get_state=False)
    sim_params_strong = StrongSimParams(
        observables=[],
        get_state=False,
    )
    with pytest.raises(ValueError, match=r"No output specified: either observables or get_state must be set."):
        sim.run(state, circ, sim_params_strong)


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
    params = StrongSimParams(observables=[Observable(Z(), 0)])
    states = [State(2, initial="zeros"), State(2, initial="ones")]
    with pytest.raises(TypeError, match="Circuit simulation requires a single State initial_state"):
        Simulator(show_progress=False).run(states, circuit, params, None)


def test_circuit_simulation_rejects_non_circuit_operator() -> None:
    """Circuit simulation requires a :class:`QuantumCircuit`."""
    state = State(2, initial="zeros")
    params = StrongSimParams(observables=[Observable(Z(), 0)])
    bad_operator = cast("Any", Hamiltonian.ising(2, J=1.0, g=0.5))
    with pytest.raises(TypeError, match="Circuit simulation requires a QuantumCircuit operator"):
        Simulator(show_progress=False).run(state, bad_operator, params, None)


def test_circuit_simulation_rejects_non_state_initial_state() -> None:
    """Circuit simulation requires a :class:`State` initial state."""
    circuit = create_ising_circuit(L=2, J=1.0, g=0.5, dt=0.1, timesteps=1)
    params = StrongSimParams(observables=[Observable(Z(), 0)])
    bad_state = cast("Any", MPS(2, state="zeros"))
    with pytest.raises(TypeError, match="Circuit simulation requires a State initial_state"):
        Simulator(show_progress=False).run(bad_state, circuit, params, None)


def test_get_parallel_context_explicit_fork_and_spawn() -> None:
    """Explicit ``mp_context`` overrides platform auto-detection.

    ``spawn`` is available on all supported platforms. ``fork`` is only
    registered where the interpreter exposes it (e.g. Linux); on Windows
    :func:`multiprocessing.get_context` raises ``ValueError``.
    """
    spawn_ctx = _get_parallel_context("spawn")
    assert spawn_ctx.get_start_method() == "spawn"

    try:
        multiprocessing.get_context("fork")
    except ValueError:
        with pytest.raises(ValueError, match="cannot find context"):
            _get_parallel_context("fork")
    else:
        fork_ctx = _get_parallel_context("fork")
        assert fork_ctx.get_start_method() == "fork"


def test_expect_shot_counts_rejects_non_dict() -> None:
    """``_expect_shot_counts`` raises ``TypeError`` for non-dict payloads."""
    with pytest.raises(TypeError, match="Expected measurement result to be dict"):
        _expect_shot_counts(np.zeros(2, dtype=np.float64))


def test_weak_simulation_parallel_returns_counts() -> None:
    """Parallel weak simulation aggregates per-shot counts via the worker pool."""
    num_qubits = 2
    state = State(num_qubits, initial="zeros")
    circuit = create_ising_circuit(L=num_qubits, J=1.0, g=0.5, dt=0.1, timesteps=1)
    circuit.measure_all()
    noise_model = NoiseModel([{"name": "pauli_x", "sites": [i], "strength": 1e-3} for i in range(num_qubits)])
    sim_params = WeakSimParams(shots=4, max_bond_dim=4, random_seed=YAQS_TEST_SEED)
    result = Simulator(parallel=True, max_workers=2, show_progress=False).run(state, circuit, sim_params, noise_model)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


def test_strong_simulation_parallel_records_final_mps() -> None:
    """Noiseless parallel strong simulation with ``get_state=True`` returns the output MPS."""
    num_qubits = 2
    state = State(num_qubits, initial="zeros")
    circuit = create_ising_circuit(L=num_qubits, J=1.0, g=0.5, dt=0.1, timesteps=2)
    circuit.measure_all()
    sim_params = StrongSimParams(
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


def test_simulator_run_dispatches_ditqasm_to_qudit_loader() -> None:
    """Simulator.run detects DITQASM source and loads it as an mqt.qudits circuit.

    This branch does not yet have a qudit simulation backend wired up (see
    ``qudit_simulation_backend``), so the run is expected to fail downstream with
    the generic "not a QuantumCircuit" error -- but only *after* successfully
    loading the DITQASM source into a real qudit circuit object, proving the
    DITQASM/OpenQASM dispatch in ``Simulator.run`` itself works correctly.
    """
    pytest.importorskip("mqt.qudits")
    ditqasm_text = "DITQASM 2.0;\n\nqreg field [2][2,3];\n\nh field[0];\ncx field[0], field[1];\n"
    state = State(2, physical_dimensions=[2, 3], initial="zeros")
    sim_params = WeakSimParams(shots=4, max_bond_dim=4)
    with pytest.raises(TypeError, match="Circuit simulation requires a QuantumCircuit operator"):
        Simulator(parallel=False, show_progress=False).run(state, ditqasm_text, sim_params)


def test_simulator_run_accepts_qasm2_path_object(tmp_path: Path) -> None:
    """Verify that Simulator.run accepts a QASM 2 file passed as a Path object."""
    qasm_file = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    state = State(6, initial="zeros")
    sim_params = WeakSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_file, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


def test_simulator_run_accepts_qasm2_str_path(tmp_path: Path) -> None:
    """Verify that Simulator.run accepts a QASM 2 file passed as a str path."""
    qasm_file = str(write_qasm_file(tmp_path, LARGE_QASM2_STRING))
    state = State(6, initial="zeros")
    sim_params = WeakSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_file, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


def test_simulator_run_accepts_qasm2_raw_string() -> None:
    """Verify that Simulator.run accepts a raw QASM 2 string (not a file path)."""
    state = State(6, initial="zeros")
    sim_params = WeakSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, LARGE_QASM2_STRING, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


@requires_qasm3_import
def test_simulator_run_accepts_qasm3_path_object(tmp_path: Path) -> None:
    """Verify that Simulator.run accepts a QASM 3 file passed as a Path object."""
    qasm_file = write_qasm_file(tmp_path, SAMPLE_QASM3_STRING, filename="circuit3.qasm")
    state = State(2, initial="zeros")
    sim_params = WeakSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_file, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


@requires_qasm3_import
def test_simulator_run_accepts_qasm3_str_path(tmp_path: Path) -> None:
    """Verify that Simulator.run accepts a QASM 3 file passed as a str path."""
    qasm_file = str(write_qasm_file(tmp_path, SAMPLE_QASM3_STRING, filename="circuit3.qasm"))
    state = State(2, initial="zeros")
    sim_params = WeakSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_file, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


def test_simulator_run_strong_accepts_qasm_path(tmp_path: Path) -> None:
    """Verify that Simulator.run with StrongSimParams accepts a QASM file passed as a Path."""
    qasm_file = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    state = State(6, initial="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], num_traj=1, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_file, sim_params)
    assert result.expectation_values[0] is not None


def test_simulator_run_strong_accepts_qasm_string(tmp_path: Path) -> None:
    """Verify that Simulator.run with StrongSimParams accepts a QASM file passed as a str path."""
    qasm_string = str(write_qasm_file(tmp_path, LARGE_QASM2_STRING))
    state = State(6, initial="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], num_traj=1, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, qasm_string, sim_params)
    assert result.expectation_values[0] is not None


@requires_qasm3_import
def test_simulator_run_strong_accepts_qasm3_raw_string() -> None:
    """Verify that Simulator.run with StrongSimParams accepts a raw OpenQASM 3 string."""
    state = State(2, initial="zeros")
    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], num_traj=1, max_bond_dim=4)
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
def test_simulator_run_accepts_qasm3_raw_string_weak() -> None:
    """Verify that Simulator.run with WeakSimParams accepts a raw OpenQASM 3 string."""
    state = State(2, initial="zeros")
    sim_params = WeakSimParams(shots=4, max_bond_dim=4)
    result = Simulator(parallel=False, show_progress=False).run(state, SAMPLE_QASM3_STRING, sim_params)
    assert result.counts is not None
    assert sum(result.counts.values()) == sim_params.shots


def test_simulator_run_qasm_path_and_string_strong_match(tmp_path: Path) -> None:
    """Strong simulation with fixed seed agrees for path and raw OpenQASM inputs."""
    qasm_path = write_qasm_file(tmp_path, LARGE_QASM2_STRING)
    state = State(6, initial="zeros")
    sim_params = StrongSimParams(
        observables=[Observable(Z(), 0)],
        num_traj=1,
        max_bond_dim=4,
        random_seed=YAQS_TEST_SEED,
    )
    path_result = Simulator(parallel=False, show_progress=False).run(state, qasm_path, sim_params)
    string_result = Simulator(parallel=False, show_progress=False).run(state, LARGE_QASM2_STRING, sim_params)
    assert path_result.expectation_values[0] == string_result.expectation_values[0]
