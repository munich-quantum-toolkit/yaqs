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
    L = 8
    J = 1
    g = 0.1
    H_0 = MPO()
    H_0.init_ising(L, J, g)

    state = MPS(L, state="zeros")

    sim_params = AnalogSimParams(
        observables=[Observable(XX(), [0, 1]), Observable(YY(), [0, 1]), Observable(ZZ(), [0, 1])],
        elapsed_time=4,
        dt=0.1,
        max_bond_dim=4,
        sample_timesteps=True,
        show_progress=False,
    )

    simulator.run(state, H_0, sim_params)

    # Expected results from qutip
    expected_xx = np.array([
        0.00000000e00,
        6.70568225e-07,
        1.05437580e-05,
        5.26285769e-05,
        1.63101302e-04,
        3.88280622e-04,
        7.80607767e-04,
        1.39420314e-03,
        2.27992357e-03,
        3.48047460e-03,
        5.02574409e-03,
        6.92850878e-03,
        9.18100678e-03,
        1.17523221e-02,
        1.45870647e-02,
        1.76054393e-02,
        2.07048646e-02,
        2.37633224e-02,
        2.66441901e-02,
        2.92025390e-02,
        3.12924427e-02,
        3.27751483e-02,
        3.35269039e-02,
        3.34471213e-02,
        3.24649047e-02,
        3.05452186e-02,
        2.76927833e-02,
        2.39548164e-02,
        1.94214578e-02,
        1.42245430e-02,
        8.53406521e-03,
        2.55295374e-03,
        -3.49025773e-03,
        -9.35026310e-03,
        -1.47750779e-02,
        -1.95171039e-02,
        -2.33445139e-02,
        -2.60523190e-02,
        -2.74732118e-02,
        -2.74860419e-02,
        -2.60237952e-02,
    ])

    expected_yy = np.array([
        0.0,
        0.00039397,
        0.00150512,
        0.00313176,
        0.00497188,
        0.00666869,
        0.00786427,
        0.00825299,
        0.00762649,
        0.00590379,
        0.00314175,
        -0.00047467,
        -0.00466105,
        -0.00907538,
        -0.01336679,
        -0.01722302,
        -0.02040903,
        -0.02279125,
        -0.02434377,
        -0.02513633,
        -0.0253068,
        -0.02502368,
        -0.02444473,
        -0.02368054,
        -0.02276801,
        -0.02166023,
        -0.02023337,
        -0.01831075,
        -0.01569962,
        -0.01223515,
        -0.00782373,
        -0.00247869,
        0.00365826,
        0.01031386,
        0.01709963,
        0.02354635,
        0.0291519,
        0.03343598,
        0.03599466,
        0.0365457,
        0.0349607,
    ])

    expected_zz = np.array([
        1.0,
        0.99960337,
        0.99845319,
        0.99666319,
        0.99440574,
        0.99188883,
        0.98932904,
        0.98692419,
        0.98483052,
        0.98314673,
        0.981908,
        0.98108965,
        0.98062034,
        0.98040144,
        0.98032984,
        0.98031972,
        0.98031997,
        0.98032419,
        0.98037151,
        0.9805384,
        0.98092281,
        0.98162311,
        0.98271625,
        0.98423763,
        0.98616765,
        0.98842601,
        0.99087614,
        0.99333842,
        0.99561099,
        0.99749466,
        0.99881849,
        0.99946193,
        0.99937048,
        0.9985623,
        0.99712498,
        0.99520315,
        0.99297874,
        0.99064714,
        0.98839271,
        0.98636745,
        0.98467569,
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
    L = 8
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
        elapsed_time=4,
        dt=0.1,
        max_bond_dim=4,
        sample_timesteps=True,
        show_progress=False,
    )

    simulator.run(state, H_0, sim_params)

    # Expected results from qutip
    expected_xx = np.array([
        0.00000000e00,
        5.27882988e-06,
        8.16775334e-05,
        3.91271926e-04,
        1.14353077e-03,
        2.52027765e-03,
        4.59858875e-03,
        7.29201713e-03,
        1.03264342e-02,
        1.32605082e-02,
        1.55504504e-02,
        1.66496909e-02,
        1.61250393e-02,
        1.37666123e-02,
        9.66709585e-03,
        4.25077097e-03,
        -1.75975195e-03,
        -7.43921368e-03,
        -1.18054355e-02,
        -1.39919239e-02,
        -1.34164490e-02,
        -9.91355665e-03,
        -3.80214086e-03,
        4.12801008e-03,
        1.27154960e-02,
        2.05991459e-02,
        2.64369848e-02,
        2.91362522e-02,
        2.80547806e-02,
        2.31371756e-02,
        1.49583835e-02,
        4.66232970e-03,
        -6.19959970e-03,
        -1.59073323e-02,
        -2.28507568e-02,
        -2.58025533e-02,
        -2.41379146e-02,
        -1.79613903e-02,
        -8.11670218e-03,
        3.92816742e-03,
        1.63020588e-02,
    ])

    expected_yy = np.array([
        0.0,
        0.00038936,
        0.00143395,
        0.00279279,
        0.00398982,
        0.00453141,
        0.0040334,
        0.00232963,
        -0.00046308,
        -0.00393995,
        -0.00746776,
        -0.01029989,
        -0.01172614,
        -0.01122817,
        -0.00860864,
        -0.00406617,
        0.0018031,
        0.00808257,
        0.0136811,
        0.01752416,
        0.01875647,
        0.01691872,
        0.01206241,
        0.00477919,
        -0.00386737,
        -0.01249924,
        -0.01964786,
        -0.02400689,
        -0.0246702,
        -0.02131163,
        -0.014272,
        -0.00453379,
        0.00641691,
        0.0168216,
        0.02493683,
        0.02933051,
        0.02913578,
        0.0242173,
        0.01521943,
        0.00348226,
        -0.0091608,
    ])

    expected_zz = np.array([
        1.0,
        0.99960536,
        0.99848433,
        0.99681549,
        0.9948644,
        0.99294091,
        0.99134953,
        0.99034066,
        0.99007094,
        0.99057858,
        0.99177804,
        0.99347447,
        0.99539566,
        0.9972361,
        0.99870602,
        0.99957743,
        0.9997198,
        0.99911974,
        0.99788166,
        0.9962096,
        0.99437337,
        0.99266458,
        0.99134996,
        0.99062924,
        0.9906047,
        0.99126656,
        0.99249679,
        0.99408996,
        0.99578778,
        0.99732123,
        0.99845358,
        0.99901717,
        0.99893832,
        0.99824666,
        0.9970676,
        0.99559986,
        0.99408181,
        0.99275253,
        0.9918138,
        0.99139937,
        0.99155592,
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
    L = 8
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
        elapsed_time=4,
        dt=0.1,
        max_bond_dim=4,
        sample_timesteps=True,
        show_progress=False,
    )
    simulator.run(state, H_0, sim_params)

    # Expected results from qutip
    expected_xx = np.array([
        0.00000000e00,
        6.70568225e-07,
        1.05437580e-05,
        5.26285769e-05,
        1.63101302e-04,
        3.88280622e-04,
        7.80607767e-04,
        1.39420314e-03,
        2.27992357e-03,
        3.48047460e-03,
        5.02574409e-03,
        6.92850878e-03,
        9.18100678e-03,
        1.17523221e-02,
        1.45870647e-02,
        1.76054393e-02,
        2.07048646e-02,
        2.37633224e-02,
        2.66441901e-02,
        2.92025390e-02,
        3.12924427e-02,
        3.27751483e-02,
        3.35269039e-02,
        3.34471213e-02,
        3.24649047e-02,
        3.05452186e-02,
        2.76927833e-02,
        2.39548164e-02,
        1.94214578e-02,
        1.42245430e-02,
        8.53406521e-03,
        2.55295374e-03,
        -3.49025773e-03,
        -9.35026310e-03,
        -1.47750779e-02,
        -1.95171039e-02,
        -2.33445139e-02,
        -2.60523190e-02,
        -2.74732118e-02,
        -2.74860419e-02,
        -2.60237952e-02,
    ])

    expected_yy = np.array([
        0.0,
        0.00039397,
        0.00150512,
        0.00313176,
        0.00497188,
        0.00666869,
        0.00786427,
        0.00825299,
        0.00762649,
        0.00590379,
        0.00314175,
        -0.00047467,
        -0.00466105,
        -0.00907538,
        -0.01336679,
        -0.01722302,
        -0.02040903,
        -0.02279125,
        -0.02434377,
        -0.02513633,
        -0.0253068,
        -0.02502368,
        -0.02444473,
        -0.02368054,
        -0.02276801,
        -0.02166023,
        -0.02023337,
        -0.01831075,
        -0.01569962,
        -0.01223515,
        -0.00782373,
        -0.00247869,
        0.00365826,
        0.01031386,
        0.01709963,
        0.02354635,
        0.0291519,
        0.03343598,
        0.03599466,
        0.0365457,
        0.0349607,
    ])

    expected_zz = np.array([
        1.0,
        0.99960337,
        0.99845319,
        0.99666319,
        0.99440574,
        0.99188883,
        0.98932904,
        0.98692419,
        0.98483052,
        0.98314673,
        0.981908,
        0.98108965,
        0.98062034,
        0.98040144,
        0.98032984,
        0.98031972,
        0.98031997,
        0.98032419,
        0.98037151,
        0.9805384,
        0.98092281,
        0.98162311,
        0.98271625,
        0.98423763,
        0.98616765,
        0.98842601,
        0.99087614,
        0.99333842,
        0.99561099,
        0.99749466,
        0.99881849,
        0.99946193,
        0.99937048,
        0.9985623,
        0.99712498,
        0.99520315,
        0.99297874,
        0.99064714,
        0.98839271,
        0.98636745,
        0.98467569,
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
    L = 8
    J = 1
    g = 0.1
    circ = create_ising_circuit(L=L, J=J, g=g, dt=0.1, timesteps=10)
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

    # Expected results from qutip
    expected_xx = np.array([1.63020588e-02])
    expected_yy = np.array([-0.0091608])
    expected_zz = np.array([0.99155592])

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
