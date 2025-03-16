# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the simulator module in YAQS.

This module verifies the functionality of the simulator by testing both physics (Hamiltonian)
and circuit simulation branches. It includes tests for identity circuits, two-qubit operations,
long-range gate handling, weak and strong simulation modes, and error cases such as mismatched
qubit counts.
"""

# ignore non-lowercase variable names for physics notation
# ruff: noqa: N806

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import (
    Observable,
    PhysicsSimParams,
    StrongSimParams,
    WeakSimParams,
)
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import X, Z


def test_physics_simulation() -> None:
    """Test the branch for Hamiltonian simulation (physics simulation) using PhysicsSimParams.

    This test creates an MPS of length 5 initialized to the "zeros" state and an Ising MPO operator.
    It also creates a NoiseModel with two processes ("relaxation" and "dephasing") and corresponding strengths.
    With PhysicsSimParams configured for a two-site evolution (order=2) and sample_timesteps False,
    simulator.run is called. The test then verifies that for each observable the results and trajectories have been
    correctly initialized and that the measurement results are approximately as expected.
    """
    length = 5
    initial_state = MPS(length, state="zeros")

    H = MPO()
    H.init_ising(length, J=1, g=0.5)
    elapsed_time = 1
    dt = 0.1
    sample_timesteps = False
    num_traj = 10
    max_bond_dim = 4
    threshold = 1e-6
    order = 2

    measurements = [Observable(Z(), site) for site in range(length)]
    sim_params = PhysicsSimParams(
        measurements, elapsed_time, dt, num_traj, max_bond_dim, threshold, order, sample_timesteps=sample_timesteps
    )
    gamma = 0.1
    noise_model = NoiseModel(["relaxation", "dephasing"], [gamma, gamma])

    simulator.run(initial_state, H, sim_params, noise_model)

    for i, observable in enumerate(sim_params.observables):
        assert observable.results is not None, "Results was not initialized for PhysicsSimParams."
        assert observable.trajectories is not None, "Trajectories was not initialized for PhysicsSimParams 1."
        assert len(observable.trajectories) == num_traj, "Trajectories was not initialized for PhysicsSimParams 2."
        assert len(observable.results) == 1, "Results was not initialized for PhysicsSimParams."
        if i == 0:
            assert np.isclose(observable.results[0], 0.70, atol=1e-1)
        elif i == 1:
            assert np.isclose(observable.results[0], 0.87, atol=1e-1)
        elif i == 2:
            assert np.isclose(observable.results[0], 0.86, atol=1e-1)
        elif i == 3:
            assert np.isclose(observable.results[0], 0.87, atol=1e-1)
        elif i == 4:
            assert np.isclose(observable.results[0], 0.70, atol=1e-1)


def test_physics_simulation_parallel_off() -> None:
    """Test the branch for Hamiltonian simulation (physics simulation) using PhysicsSimParams, parallelization off.

    This test creates an MPS of length 5 initialized to the "zeros" state and an Ising MPO operator.
    It also creates a NoiseModel with two processes ("relaxation" and "dephasing") and corresponding strengths.
    With PhysicsSimParams configured for a two-site evolution (order=2) and sample_timesteps False,
    simulator.run is called. The test then verifies that for each observable the results and trajectories have been
    correctly initialized and that the measurement results are approximately as expected.
    """
    length = 5
    initial_state = MPS(length, state="zeros")

    H = MPO()
    H.init_ising(length, J=1, g=0.5)
    elapsed_time = 1
    dt = 0.1
    sample_timesteps = False
    num_traj = 10
    max_bond_dim = 4
    threshold = 1e-6
    order = 2

    measurements = [Observable(Z(), site) for site in range(length)]
    sim_params = PhysicsSimParams(
        measurements, elapsed_time, dt, num_traj, max_bond_dim, threshold, order, sample_timesteps=sample_timesteps
    )
    gamma = 0.1
    noise_model = NoiseModel(["relaxation", "dephasing"], [gamma, gamma])

    simulator.run(initial_state, H, sim_params, noise_model, parallel=False)

    for i, observable in enumerate(sim_params.observables):
        assert observable.results is not None, "Results was not initialized for PhysicsSimParams."
        assert observable.trajectories is not None, "Trajectories was not initialized for PhysicsSimParams 1."
        assert len(observable.trajectories) == num_traj, "Trajectories was not initialized for PhysicsSimParams 2."
        assert len(observable.results) == 1, "Results was not initialized for PhysicsSimParams."
        if i == 0:
            assert np.isclose(observable.results[0], 0.70, atol=1e-1)
        elif i == 1:
            assert np.isclose(observable.results[0], 0.87, atol=1e-1)
        elif i == 2:
            assert np.isclose(observable.results[0], 0.86, atol=1e-1)
        elif i == 3:
            assert np.isclose(observable.results[0], 0.87, atol=1e-1)
        elif i == 4:
            assert np.isclose(observable.results[0], 0.70, atol=1e-1)


def test_strong_simulation() -> None:
    """Test the circuit-based simulation branch using StrongSimParams.

    This test constructs an MPS of length 5 (initialized to "zeros") and an Ising circuit with a CX gate.
    It configures StrongSimParams with specified simulation parameters and a noise model (non-None).
    simulator.run is then called, and the test verifies that the observables' results and trajectories
    are initialized correctly. Expected measurement outcomes are compared approximately to pre-defined values.
    """
    num_qubits = 5
    state = MPS(num_qubits, state="zeros")

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()

    num_traj = 10
    max_bond_dim = 4
    threshold = 1e-12
    window_size = 0

    measurements = [Observable(Z(), site) for site in range(num_qubits)]
    sim_params = StrongSimParams(measurements, num_traj, max_bond_dim, threshold, window_size)
    # Use a noise model that is not None so that sim_params.num_traj remains unchanged.
    gamma = 1e-3
    noise_model = NoiseModel(["relaxation", "dephasing"], [gamma, gamma])

    simulator.run(state, circuit, sim_params, noise_model)

    for i, observable in enumerate(sim_params.observables):
        assert observable.results is not None, "Results was not initialized for PhysicsSimParams."
        assert observable.trajectories is not None, "Trajectories was not initialized for PhysicsSimParams 1."
        assert len(observable.trajectories) == num_traj, "Trajectories was not initialized for PhysicsSimParams 2."
        assert len(observable.results) == 1, "Results was not initialized for PhysicsSimParams."
        if i == 0:
            assert np.isclose(observable.results[0], 0.70, atol=1e-1)
        elif i == 1:
            assert np.isclose(observable.results[0], 0.87, atol=1e-1)
        elif i == 2:
            assert np.isclose(observable.results[0], 0.86, atol=1e-1)
        elif i == 3:
            assert np.isclose(observable.results[0], 0.87, atol=1e-1)
        elif i == 4:
            assert np.isclose(observable.results[0], 0.70, atol=1e-1)


def test_strong_simulation_no_noise() -> None:
    """Test the circuit-based simulation using StrongSimParams without noise to get a statevector.

    This test constructs a 2-site Ising circuit and compares the output statevector with known values from qiskit.
    """
    num_qubits = 2
    circ = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circ.measure_all()

    state = MPS(length=num_qubits)
    measurements = [Observable(X(), num_qubits // 2)]
    sim_params = StrongSimParams(
        measurements, num_traj=1, max_bond_dim=16, threshold=1e-6, window_size=0, get_state=True
    )
    simulator.run(state, circ, sim_params, noise_model=None)
    assert sim_params.output_state is not None
    assert isinstance(sim_params.output_state, MPS)
    sv = sim_params.output_state.to_vec()

    expected = [0.34870601 + 0.7690227j, 0.03494528 + 0.34828721j, 0.03494528 + 0.34828721j, -0.19159629 - 0.07244828j]
    fidelity = np.abs(np.vdot(sv, expected)) ** 2
    np.testing.assert_allclose(1, fidelity)


def test_strong_simulation_parallel_off() -> None:
    """Test the circuit-based simulation branch using StrongSimParams, parallelization off.

    This test constructs an MPS of length 5 (initialized to "zeros") and an Ising circuit with a CX gate.
    It configures StrongSimParams with specified simulation parameters and a noise model (non-None).
    simulator.run is then called, and the test verifies that the observables' results and trajectories
    are initialized correctly. Expected measurement outcomes are compared approximately to pre-defined values.
    """
    num_qubits = 5
    state = MPS(num_qubits, state="zeros")

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()

    num_traj = 10
    max_bond_dim = 4
    threshold = 1e-12
    window_size = 0

    measurements = [Observable(Z(), site) for site in range(num_qubits)]
    sim_params = StrongSimParams(measurements, num_traj, max_bond_dim, threshold, window_size)
    # Use a noise model that is not None so that sim_params.num_traj remains unchanged.
    gamma = 1e-3
    noise_model = NoiseModel(["relaxation", "dephasing"], [gamma, gamma])

    simulator.run(state, circuit, sim_params, noise_model, parallel=False)

    for i, observable in enumerate(sim_params.observables):
        assert observable.results is not None, "Results was not initialized for PhysicsSimParams."
        assert observable.trajectories is not None, "Trajectories was not initialized for PhysicsSimParams 1."
        assert len(observable.trajectories) == num_traj, "Trajectories was not initialized for PhysicsSimParams 2."
        assert len(observable.results) == 1, "Results was not initialized for PhysicsSimParams."
        if i == 0:
            assert np.isclose(observable.results[0], 0.70, atol=1e-1)
        elif i == 1:
            assert np.isclose(observable.results[0], 0.87, atol=1e-1)
        elif i == 2:
            assert np.isclose(observable.results[0], 0.86, atol=1e-1)
        elif i == 3:
            assert np.isclose(observable.results[0], 0.87, atol=1e-1)
        elif i == 4:
            assert np.isclose(observable.results[0], 0.70, atol=1e-1)


def test_weak_simulation_noise() -> None:
    """Test the weak simulation branch with a non-None noise model.

    This test creates an MPS and an Ising circuit (with measurement) for a 5-qubit system.
    It sets up WeakSimParams with a specified number of shots, max bond dimension, threshold, and window size,
    and a noise model with small strengths. After running simulator.run, the test verifies that sim_params.num_traj
    equals the number of shots, that each measurement is a dictionary, and that the total number of shots
    recorded in sim_params.results equals the expected number.
    """
    num_qubits = 5
    initial_state = MPS(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=1)
    circuit.measure_all()
    shots = 1024
    max_bond_dim = 4
    threshold = 1e-6
    window_size = 0
    sim_params = WeakSimParams(shots, max_bond_dim, threshold, window_size)

    gamma = 1e-3
    noise_model = NoiseModel(["relaxation", "dephasing"], [gamma, gamma])

    simulator.run(initial_state, circuit, sim_params, noise_model)

    assert shots == sim_params.num_traj, "sim_params.num_traj should be number of shots."
    for measurement in sim_params.measurements:
        assert isinstance(measurement, dict)
    assert sum(sim_params.results.values()) == shots, "Wrong number of shots in WeakSimParams."


def test_weak_simulation_no_noise() -> None:
    """Test the weak simulation branch when the noise model is None.

    This test creates an MPS and an Ising circuit (with measurement) for a 5-qubit system,
    and configures WeakSimParams with a specified number of shots. When noise_model is None,
    the simulation should set sim_params.num_traj to 1. The test verifies that the measurements and results
    are consistent with this behavior.
    """
    num_qubits = 5
    initial_state = MPS(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=1)
    circuit.measure_all()
    shots = 1024
    max_bond_dim = 4
    threshold = 1e-6
    window_size = 0
    sim_params = WeakSimParams(shots, max_bond_dim, threshold, window_size)

    noise_model = None

    simulator.run(initial_state, circuit, sim_params, noise_model)

    assert sim_params.num_traj == 1, "sim_params.num_traj should be 1 when noise model strengths are all zero."
    assert isinstance(sim_params.measurements[0], dict), (
        "There should be only one measurement when noise model strengths are zero. 1"
    )
    assert sim_params.measurements[1] is None, (
        "There should be only one measurement when noise model strengths are zero. 2"
    )
    max_value = max(sim_params.results.values())
    assert sim_params.results[0] == max_value, "Key 0 does not have the highest value."
    assert sum(sim_params.results.values()) == shots, "Wrong number of shots in WeakSimParams."


def test_mismatch() -> None:
    """Test that simulator.run raises an AssertionError when the state and circuit qubit counts mismatch.

    This test creates an MPS of length 5 and a circuit with length 4 (one fewer qubits),
    and verifies that an AssertionError with the appropriate message is raised.
    """
    num_qubits = 5
    initial_state = MPS(num_qubits)

    circuit = create_ising_circuit(L=num_qubits - 1, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()
    shots = 1024
    max_bond_dim = 4
    threshold = 1e-6
    window_size = 0
    sim_params = WeakSimParams(shots, max_bond_dim, threshold, window_size)

    noise_model = None

    with pytest.raises(AssertionError, match=r"State and circuit qubit counts do not match."):
        simulator.run(initial_state, circuit, sim_params, noise_model)
