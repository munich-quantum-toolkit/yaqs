import pytest
import numpy as np

from yaqs.core.data_structures.networks import MPO, MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams, StrongSimParams, WeakSimParams
from yaqs.core.libraries.circuit_library import create_Ising_circuit
from yaqs import Simulator


def test_physics_simulation():
    """
    Test the branch for Hamiltonian simulation or circuit simulation with StrongSimParams.
    Here we use a DummyMPO operator and a DummyStrongSimParams.
    """
    length = 5
    initial_state = MPS(length, state='zeros')

    H = MPO()  # operator is an instance of MPO
    H.init_Ising(length, J=1, g=0.5)
    T = 1
    dt = 0.1
    sample_timesteps = False
    N = 10
    max_bond_dim = 4
    threshold = 1e-6
    order = 2

    measurements = [Observable('z', site) for site in range(length)]
    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)
    gamma = 0.1
    noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma, gamma])

    Simulator.run(initial_state, H, sim_params, noise_model, parallel=False)

    assert np.isclose(sim_params.observables[0].results[0], 0.70, atol=1e-1)
    assert np.isclose(sim_params.observables[1].results[0], 0.87, atol=1e-1)
    assert np.isclose(sim_params.observables[2].results[0], 0.86, atol=1e-1)
    assert np.isclose(sim_params.observables[3].results[0], 0.87, atol=1e-1)
    assert np.isclose(sim_params.observables[4].results[0], 0.70, atol=1e-1)

    assert len(sim_params.observables[0].trajectories) == N, "Trajectories was not initialized for PhysicsSimParams."

    # Verify that aggregate_trajectories was called.
    assert len(sim_params.observables[0].results == 1), "Results was not initialized for PhysicsSimParams."


def test_strong_simulation():
    num_qubits = 5
    state = MPS(num_qubits, state='zeros')

    model = {'name': 'Ising', 'L': num_qubits, 'J': 1, 'g': 0.5}
    circuit = create_Ising_circuit(model, dt=0.1, timesteps=10)
    circuit.measure_all()

    N = 10
    max_bond_dim = 4
    threshold = 1e-12
    window_size = 0

    measurements = [Observable('z', site) for site in range(num_qubits)]
    sim_params = StrongSimParams(measurements, N, max_bond_dim, threshold, window_size)
    # Use a noise model that is None so that the branch sets sim_params.N = 1.
    gamma = 1e-3
    noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma, gamma])

    Simulator.run(state, circuit, sim_params, noise_model)

    assert np.isclose(sim_params.observables[0].results[0], 0.70, atol=2e-1)
    assert np.isclose(sim_params.observables[1].results[0], 0.87, atol=2e-1)
    assert np.isclose(sim_params.observables[2].results[0], 0.86, atol=2e-1)
    assert np.isclose(sim_params.observables[3].results[0], 0.87, atol=2e-1)
    assert np.isclose(sim_params.observables[4].results[0], 0.70, atol=2e-1)

    assert len(sim_params.observables[0].trajectories) == N, "Trajectories was not initialized for StrongimParams."

    # Verify that aggregate_trajectories was called.
    assert len(sim_params.observables[0].results == 1), "Results was not initialized for StrongimParams."


def test_weak_simulation_noise():
    num_qubits = 5
    initial_state = MPS(num_qubits)

    model = {'name': 'Ising', 'L': num_qubits, 'J': 1, 'g': 0.5}
    circuit = create_Ising_circuit(model, dt=0.1, timesteps=1)
    circuit.measure_all()
    shots = 1024
    max_bond_dim = 4
    threshold = 1e-6
    window_size = 0
    sim_params = WeakSimParams(shots, max_bond_dim, threshold, window_size)

    gamma = 1e-3
    noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma, gamma])

    Simulator.run(initial_state, circuit, sim_params, noise_model)

    assert sim_params.N == shots, "sim_params.N should be number of shots."
    for measurement in sim_params.measurements:
        assert isinstance(measurement, dict)

    assert sum(sim_params.results.values()) == shots, "Wrong number of shots in WeakSimParams."


def test_weak_simulation_no_noise():
    num_qubits = 5
    initial_state = MPS(num_qubits)

    model = {'name': 'Ising', 'L': num_qubits, 'J': 1, 'g': 0.5}
    circuit = create_Ising_circuit(model, dt=0.1, timesteps=1)
    circuit.measure_all()
    shots = 1024
    max_bond_dim = 4
    threshold = 1e-6
    window_size = 0
    sim_params = WeakSimParams(shots, max_bond_dim, threshold, window_size)

    noise_model = None

    Simulator.run(initial_state, circuit, sim_params, noise_model)

    assert sim_params.N == 1, "sim_params.N should be 1 when noise model strengths are all zero."
    assert isinstance(sim_params.measurements[0], dict) and sim_params.measurements[1] == None, "There should be only one measurement when noise model strengths are zero."

    max_value = max(sim_params.results.values())
    assert sim_params.results[0] == max_value, "Key 0 does not have the highest value."

    assert sum(sim_params.results.values()) == shots, "Wrong number of shots in WeakSimParams."


def test_mismatch():
    num_qubits = 5
    initial_state = MPS(num_qubits)

    model = {'name': 'Ising', 'L': num_qubits-1, 'J': 1, 'g': 0.5}
    circuit = create_Ising_circuit(model, dt=0.1, timesteps=10)
    circuit.measure_all()
    shots = 1024
    max_bond_dim = 4
    threshold = 1e-6
    window_size = 0
    sim_params = WeakSimParams(shots, max_bond_dim, threshold, window_size)

    noise_model = None

    with pytest.raises(AssertionError, match="State and circuit qubit counts do not match."):
        Simulator.run(initial_state, circuit, sim_params, noise_model)
