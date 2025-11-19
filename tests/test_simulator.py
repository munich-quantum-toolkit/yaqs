# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
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
# ruff: noqa: N806

from __future__ import annotations

import importlib
import multiprocessing

import numpy as np
import pytest

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import (
    AnalogSimParams,
    Observable,
    StrongSimParams,
    WeakSimParams,
)
from mqt.yaqs.core.libraries.circuit_library import create_ising_circuit
from mqt.yaqs.core.libraries.gate_library import XX, YY, ZZ, X, Z


def test_available_cpus_without_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path 1: SLURM_CPUS_ON_NODE *not* set.

    Should return multiprocessing.cpu_count().
    """
    # Ensure the env var is absent
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)

    assert simulator.available_cpus() == multiprocessing.cpu_count()


def test_available_cpus_with_slurm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path 2: SLURM_CPUS_ON_NODE is set.

    Should return that exact value.
    """
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "8")

    # Reload the module only if available_cpus caches anything at import;
    # here it's not necessary, but harmless:
    importlib.reload(simulator)

    assert simulator.available_cpus() == 8


def test_analog_simulation() -> None:
    """Test the branch for Hamiltonian simulation (analog simulation) using AnalogSimParams.

    This test creates an MPS of length 5 initialized to the "zeros" state and an Ising MPO operator.
    It also creates a NoiseModel with two processes ("lowering" and "pauli_z") and corresponding strengths.
    With AnalogSimParams configured for a two-site evolution (order=2) and sample_timesteps False,
    simulator.run is called. The test then verifies that for each observable the results and trajectories have been
    correctly initialized and that the measurement results are approximately as expected.
    """
    length = 5
    initial_state = MPS(length, state="zeros")

    H = MPO()
    H.init_ising(length, J=1, g=0.5)

    sim_params = AnalogSimParams(
        observables=[Observable(Z(), site) for site in range(length)],
        elapsed_time=1,
        dt=0.1,
        num_traj=10,
        max_bond_dim=4,
        threshold=1e-6,
        order=2,
        sample_timesteps=False,
        show_progress=False,
    )
    gamma = 0.1
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(length) for name in ["lowering", "pauli_z"]
    ])

    simulator.run(initial_state, H, sim_params, noise_model)

    for i, observable in enumerate(sim_params.observables):
        assert observable.results is not None, "Results was not initialized for AnalogSimParams."
        assert observable.trajectories is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(observable.trajectories) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(observable.results) == 1, "Results was not initialized for AnalogSimParams."
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


def test_analog_simulation_parallel_off() -> None:
    """Test the branch for Hamiltonian simulation (analog simulation) using AnalogSimParams, parallelization off.

    This test creates an MPS of length 5 initialized to the "zeros" state and an Ising MPO operator.
    It also creates a NoiseModel with two processes ("lowering" and "pauli_z") and corresponding strengths.
    With AnalogSimParams configured for a two-site evolution (order=2) and sample_timesteps False,
    simulator.run is called. The test then verifies that for each observable the results and trajectories have been
    correctly initialized and that the measurement results are approximately as expected.

    Additionally, this tests that single-site observables can be initialized with a list of a single int for usability.
    """
    length = 5
    initial_state = MPS(length, state="zeros")

    H = MPO()
    H.init_ising(length, J=1, g=0.5)
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), site) for site in range(length)],
        elapsed_time=1,
        dt=0.1,
        num_traj=10,
        max_bond_dim=4,
        threshold=1e-6,
        order=2,
        sample_timesteps=False,
        show_progress=False,
    )
    gamma = 0.1
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(length) for name in ["lowering", "pauli_z"]
    ])

    simulator.run(initial_state, H, sim_params, noise_model, parallel=False)

    for i, observable in enumerate(sim_params.observables):
        assert observable.results is not None, "Results was not initialized for AnalogSimParams."
        assert observable.trajectories is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(observable.trajectories) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(observable.results) == 1, "Results was not initialized for AnalogSimParams."
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


def test_analog_simulation_get_state() -> None:
    """Test the Hamiltonian simulation (analog simulation) using AnalogSimParams without noise to get a statevector.

    This test creates an MPS of length 2 initialized to the "zeros" state and an Ising MPO operator.
    With sample_timesteps set to False, the test verifies for two-site (order=2) and single-site (order=1) that the
    resulting output statevector is correct.
    """
    for order in [1, 2]:
        length = 2
        initial_state = MPS(length, state="zeros")

        H = MPO()
        H.init_ising(length, J=1, g=0.5)

        sim_params = AnalogSimParams(
            observables=[Observable(X(), length // 2)],
            elapsed_time=1,
            dt=0.1,
            num_traj=1,
            max_bond_dim=4,
            threshold=1e-6,
            order=order,
            get_state=True,
            sample_timesteps=False,
            show_progress=False,
        )

        simulator.run(initial_state, H, sim_params)
        assert sim_params.output_state is not None
        assert isinstance(sim_params.output_state, MPS)
        sv = sim_params.output_state.to_vec()

        expected = [
            3.48123000e-01 + 0.76996349j,
            0.00000000e00 + 0.349228j,
            0.00000000e00 + 0.349228j,
            -1.92179306e-01 - 0.07150749j,
        ]
        fidelity = np.abs(np.vdot(sv, expected)) ** 2
        np.testing.assert_allclose(1, fidelity)


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

    sim_params = StrongSimParams(
        observables=[Observable(Z(), site) for site in range(num_qubits)],
        num_traj=10,
        max_bond_dim=4,
        show_progress=False,
    )
    # Use a noise model that is not None so that sim_params.num_traj remains unchanged.
    gamma = 1e-3
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(num_qubits) for name in ["lowering", "pauli_z"]
    ])

    simulator.run(state, circuit, sim_params, noise_model)

    for i, observable in enumerate(sim_params.observables):
        assert observable.results is not None, "Results was not initialized for AnalogSimParams."
        assert observable.trajectories is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(observable.trajectories) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(observable.results) == 1, "Results was not initialized for AnalogSimParams."
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

    sim_params = StrongSimParams(observables=[Observable(Z(), 0)], max_bond_dim=16, get_state=True, show_progress=False)

    simulator.run(state, circ, sim_params)
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

    sim_params = StrongSimParams(
        observables=[Observable(Z(), site) for site in range(num_qubits)],
        num_traj=10,
        max_bond_dim=4,
        show_progress=False,
    )
    # Use a noise model that is not None so that sim_params.num_traj remains unchanged.
    gamma = 1e-3
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(num_qubits) for name in ["lowering", "pauli_z"]
    ])

    simulator.run(state, circuit, sim_params, noise_model, parallel=False)

    for i, observable in enumerate(sim_params.observables):
        assert observable.results is not None, "Results was not initialized for AnalogSimParams."
        assert observable.trajectories is not None, "Trajectories was not initialized for AnalogSimParams 1."
        assert len(observable.trajectories) == sim_params.num_traj, (
            "Trajectories was not initialized for AnalogSimParams 2."
        )
        assert len(observable.results) == 1, "Results was not initialized for AnalogSimParams."
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

    sim_params = WeakSimParams(shots=1024, max_bond_dim=4, show_progress=False)

    gamma = 1e-3
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(num_qubits) for name in ["lowering", "pauli_z"]
    ])

    simulator.run(initial_state, circuit, sim_params, noise_model)

    assert sim_params.shots == sim_params.num_traj, "sim_params.num_traj should be number of shots."
    for measurement in sim_params.measurements:
        assert isinstance(measurement, dict)
    assert sum(sim_params.results.values()) == sim_params.shots, "Wrong number of shots in WeakSimParams."


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
    sim_params = WeakSimParams(shots=1024, max_bond_dim=4, show_progress=False)

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
    assert sum(sim_params.results.values()) == sim_params.shots, "Wrong number of shots in WeakSimParams."


def test_weak_simulation_get_state() -> None:
    """Test the circuit-based simulation using WeakSimParams without noise to get a statevector.

    This test constructs a 2-site Ising circuit and compares the output statevector with known values from qiskit.
    """
    num_qubits = 2
    initial_state = MPS(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()
    sim_params = WeakSimParams(shots=1, max_bond_dim=4, get_state=True, show_progress=False)
    noise_model = None

    simulator.run(initial_state, circuit, sim_params, noise_model)
    assert sim_params.output_state is not None
    assert isinstance(sim_params.output_state, MPS)
    sv = sim_params.output_state.to_vec()

    expected = [0.34870601 + 0.7690227j, 0.03494528 + 0.34828721j, 0.03494528 + 0.34828721j, -0.19159629 - 0.07244828j]
    fidelity = np.abs(np.vdot(sv, expected)) ** 2
    np.testing.assert_allclose(1, fidelity)


def test_weak_simulation_get_state_noise() -> None:
    """Test the circuit-based simulation using WeakSimParams noise to get a statevector.

    This test constructs a 2-site Ising circuit and configures the WeakSimParams to include a noise model and
    return the final state. Since the noisy simulation cannot return the statevector, an exception should be raised.
    """
    num_qubits = 2
    initial_state = MPS(num_qubits)

    circuit = create_ising_circuit(L=num_qubits, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()
    sim_params = WeakSimParams(shots=1, max_bond_dim=4, get_state=True, show_progress=False)

    gamma = 1e-3
    noise_model = NoiseModel([
        {"name": name, "sites": [i], "strength": gamma} for i in range(num_qubits) for name in ["lowering", "pauli_z"]
    ])

    with pytest.raises(AssertionError, match=r"Cannot return state in noisy circuit simulation due to stochastics."):
        simulator.run(initial_state, circuit, sim_params, noise_model)
    assert sim_params.output_state is None


def test_mismatch() -> None:
    """Test that simulator.run raises an AssertionError when the state and circuit qubit counts mismatch.

    This test creates an MPS of length 5 and a circuit with length 4 (one fewer qubits),
    and verifies that an AssertionError with the appropriate message is raised.
    """
    num_qubits = 5
    initial_state = MPS(num_qubits)

    circuit = create_ising_circuit(L=num_qubits - 1, J=1, g=0.5, dt=0.1, timesteps=10)
    circuit.measure_all()

    sim_params = WeakSimParams(shots=1024, max_bond_dim=4, show_progress=False)

    noise_model = None

    with pytest.raises(AssertionError, match=r"State and circuit qubit counts do not match."):
        simulator.run(initial_state, circuit, sim_params, noise_model)


def test_two_site_correlator_left_boundary() -> None:
    """Tests the expectation value of a two-site correlator in analog simulation at the left boundary.

    This test initializes an MPS in the |0> state and computes the expectation value of a two-site correlator
    at the left boundary.
    """
    L = 4
    J = 1
    g = 0.1
    H_0 = MPO()
    H_0.init_ising(L, J, g)

    state = MPS(L, state="zeros")

    sim_params = AnalogSimParams(
        observables=[Observable(XX(), [0, 1]), Observable(YY(), [0, 1]), Observable(ZZ(), [0, 1])],
        elapsed_time=2.0,
        dt=0.1,
        max_bond_dim=4,
        sample_timesteps=True,
        show_progress=False,
    )

    simulator.run(state, H_0, sim_params)

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

    results_xx = sim_params.observables[0].results
    assert results_xx is not None
    np.testing.assert_allclose(results_xx, expected_xx, atol=1e-3)

    results_yy = sim_params.observables[1].results
    assert results_yy is not None
    np.testing.assert_allclose(results_yy, expected_yy, atol=1e-3)

    results_zz = sim_params.observables[2].results
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
    H_0 = MPO()
    H_0.init_ising(L, J, g)

    state = MPS(L, state="zeros")

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
        show_progress=False,
    )

    simulator.run(state, H_0, sim_params)

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

    results_xx = sim_params.observables[0].results
    assert results_xx is not None
    np.testing.assert_allclose(results_xx, expected_xx, atol=1e-3)

    results_yy = sim_params.observables[1].results
    assert results_yy is not None
    np.testing.assert_allclose(results_yy, expected_yy, atol=1e-3)

    results_zz = sim_params.observables[2].results
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
    H_0 = MPO()
    H_0.init_ising(L, J, g)

    state = MPS(L, state="zeros")

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
        show_progress=False,
    )
    simulator.run(state, H_0, sim_params)

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

    results_xx = sim_params.observables[0].results
    assert results_xx is not None
    np.testing.assert_allclose(results_xx, expected_xx, atol=1e-3)

    results_yy = sim_params.observables[1].results
    assert results_yy is not None
    np.testing.assert_allclose(results_yy, expected_yy, atol=1e-3)

    results_zz = sim_params.observables[2].results
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
    state = MPS(L, state="zeros")

    sim_params = StrongSimParams(
        observables=[
            Observable(XX(), [L // 2, L // 2 + 1]),
            Observable(YY(), [L // 2, L // 2 + 1]),
            Observable(ZZ(), [L // 2, L // 2 + 1]),
        ],
        max_bond_dim=4,
        show_progress=False,
    )

    simulator.run(state, circ, sim_params)

    expected_xx = np.array([3.12811457e-02])
    expected_yy = np.array([-2.52988868e-02])
    expected_zz = np.array([9.80920787e-01])

    results_xx = sim_params.observables[0].results
    assert results_xx is not None
    np.testing.assert_allclose(results_xx, expected_xx, atol=2e-3)

    results_yy = sim_params.observables[1].results
    assert results_yy is not None
    np.testing.assert_allclose(results_yy, expected_yy, atol=2e-3)

    results_zz = sim_params.observables[2].results
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

    H_0 = MPO()
    H_0.init_coupled_transmon(
        length=length,
        qubit_dim=qubit_dim,
        resonator_dim=resonator_dim,
        qubit_freq=w_q,
        resonator_freq=w_r,
        anharmonicity=alpha,
        coupling=g,
    )

    state = MPS(length, state="basis", basis_string="100", physical_dimensions=[qubit_dim, resonator_dim, qubit_dim])
    T_swap = np.pi / (np.sqrt(2) * g)

    sim_params = AnalogSimParams(
        observables=[Observable(bitstring) for bitstring in ["000", "001", "010", "011", "100", "101", "110", "111"]],
        elapsed_time=T_swap,
        dt=T_swap / 1000,
        sample_timesteps=False,
        show_progress=False,
    )
    simulator.run(state, H_0, sim_params)

    res0 = sim_params.observables[0].results
    assert res0 is not None, "Expected results to be set by simulator.run"
    # Initialize leakage as a numpy array of ones:
    leakage = np.ones_like(res0)

    for meas in sim_params.observables:
        # Narrow results from Optional[...] to actual array
        res = meas.results
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
