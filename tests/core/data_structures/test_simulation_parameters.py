# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for simulation parameters classes.

This module contains unit tests for the Observable and AnalogSimParams classes used in
quantum simulation. It verifies that:
  - An Observable is correctly initialized with valid parameters and that invalid parameters
    raise an appropriate error.
  - AnalogSimParams instances are created with the correct attributes (such as elapsed_time, dt, times,
    sample_timesteps, and num_traj) both with explicit and default values.
  - The Observable.initialize method properly sets up the results and trajectories arrays
    depending on whether sample_timesteps is True or False.
"""

# ignore non-lowercase variable names for physics notation

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable, StrongSimParams
from mqt.yaqs.core.libraries.gate_library import GateLibrary, X


def test_observable_creation_valid() -> None:
    """Test that an Observable is created correctly with valid parameters.

    This test constructs an Observable with the name "x" on site 0 and verifies that its attributes
    (name, site, results, and trajectories) are correctly initialized.
    """
    gate = X()
    site = 0
    obs = Observable(gate, site)

    assert np.array_equal(obs.gate.matrix, np.array([[0, 1], [1, 0]]))
    assert obs.sites == site
    assert obs.results is None
    assert obs.trajectories is None


def test_analog_simparams_basic() -> None:
    """Test that AnalogSimParams is initialized with correct parameters.

    This test creates a AnalogSimParams instance with a single observable, total time elapsed_time, time step dt,
    sample_timesteps flag set to True, and a specified number of trajectories num_traj. It then verifies that the
    observables, elapsed_time, dt, times array, sample_timesteps flag, and num_traj are set correctly.
    """
    obs_list = [Observable(X(), 0)]
    elapsed_time = 1.0
    dt = 0.2
    params = AnalogSimParams(observables=obs_list, elapsed_time=elapsed_time, dt=dt, num_traj=50)

    assert params.observables == obs_list
    assert params.elapsed_time == elapsed_time
    assert params.dt == dt
    expected_times = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(params.times, expected_times), "Times array should match numpy.arange(0, elapsed_time+dt, dt)."
    assert params.sample_timesteps is True
    assert params.num_traj == 50


def test_analog_simparams_defaults() -> None:
    """Test the default parameters for AnalogSimParams.

    This test constructs a AnalogSimParams instance with an empty observable list and total time elapsed_time,
    and verifies that default values for dt, sample_timesteps, number of trajectories (num_traj), max_bond_dim,
    threshold, and order are correctly assigned.
    """
    obs_list: list[Observable] = []
    elapsed_time = 2.0
    params = AnalogSimParams(observables=obs_list, elapsed_time=elapsed_time)

    assert params.observables == obs_list
    assert params.elapsed_time == 2.0
    assert params.dt == 0.1
    assert params.sample_timesteps is True
    # times should be np.arange(0, elapsed_time+dt, dt)
    assert np.isclose(params.times[-1], 2.0)
    assert params.num_traj == 1000
    assert params.max_bond_dim == 4096
    assert params.threshold == 1e-9
    assert params.order == 1


def test_observable_initialize_with_sample_timesteps() -> None:
    """Test that Observable.initialize sets up results and trajectories correctly when sample_timesteps is True.

    This test creates an Observable on site 1 and a AnalogSimParams instance with sample_timesteps=True.
    It verifies that the results array has shape equal to the length of the times array and that the
    trajectories array has shape (num_traj, len(times)).
    """
    obs = Observable(X(), 1)
    sim_params = AnalogSimParams(
        observables=[obs], elapsed_time=1.0, dt=0.5, num_traj=10, sample_timesteps=True, show_progress=False
    )
    # sim_params.times => [0.0, 0.5, 1.0]

    obs.initialize(sim_params)
    assert obs.results is not None
    assert obs.trajectories is not None
    assert obs.results.shape == (3,), "results should match len(sim_params.times)."
    assert obs.trajectories.shape == (sim_params.num_traj, 3), "trajectories should have shape (num_traj, len(times))."


def test_observable_initialize_without_sample_timesteps() -> None:
    """Test that Observable.initialize sets up results and trajectories correctly when sample_timesteps is False.

    This test creates an Observable on site 0 and a AnalogSimParams instance with sample_timesteps=False.
    It verifies that the results array has shape equal to the length of the times array, the trajectories array
    has shape (num_traj, 1), and that the observable's times attribute is set to elapsed_time.
    """
    obs = Observable(X(), 0)
    sim_params = AnalogSimParams(
        observables=[obs], elapsed_time=1.0, dt=0.25, num_traj=5, sample_timesteps=False, show_progress=False
    )
    # times => [0.0, 0.25, 0.5, 0.75, 1.0]

    obs.initialize(sim_params)
    assert obs.results is not None
    assert obs.trajectories is not None
    assert obs.results.shape == (len(sim_params.times),)
    assert obs.trajectories.shape == (sim_params.num_traj, 1)
    assert obs.times == 1.0, "If sample_timesteps=False, obs.times should be equal to elapsed_time."


def test_observable_from_string_runtime_cost() -> None:
    """Constructor maps 'runtime_cost' string to the runtime_cost diagnostic gate."""
    obs = Observable("runtime_cost", sites=0)
    assert obs.gate.name == "runtime_cost"
    # placeholder identity backing for diagnostics
    assert obs.gate.matrix.shape == (2, 2)
    assert np.allclose(obs.gate.matrix, np.eye(2))


def test_observable_from_string_max_total_bond() -> None:
    """Constructor maps 'max_bond' and 'total_bond' to their diagnostic gates."""
    obs_max = Observable("max_bond", sites=1)
    obs_tot = Observable("total_bond", sites=2)

    assert obs_max.gate.name == "max_bond"
    assert obs_tot.gate.name == "total_bond"
    assert np.allclose(obs_max.gate.matrix, np.eye(2))
    assert np.allclose(obs_tot.gate.matrix, np.eye(2))


def test_observable_from_string_entropy_and_spectrum_with_list_sites() -> None:
    """Constructor maps 'entropy' and 'schmidt_spectrum' and accepts list[int] sites."""
    cut = [3, 4]
    obs_ent = Observable("entropy", sites=cut)
    obs_ssp = Observable("schmidt_spectrum", sites=cut)

    assert obs_ent.gate.name == "entropy"
    assert obs_ssp.gate.name == "schmidt_spectrum"
    # meta-observables use identity placeholders for BaseGate compatibility
    assert np.allclose(obs_ent.gate.matrix, np.eye(2))
    assert np.allclose(obs_ssp.gate.matrix, np.eye(2))
    assert obs_ent.sites == cut
    assert obs_ssp.sites == cut


def test_observable_from_string_falls_back_to_pvm() -> None:
    """Any other string is interpreted as a PVM bitstring; gate must store that bitstring."""
    bitstring = "10101"
    obs = Observable(bitstring, sites=None)
    assert obs.gate.name == "pvm"
    # gate must expose the queried bitstring
    assert hasattr(obs.gate, "bitstring")
    assert obs.gate.bitstring == bitstring
    # PVM uses identity placeholder matrix for compatibility in your implementation
    assert np.allclose(obs.gate.matrix, np.eye(2))


def test_observable_from_gate_instance_keeps_gate_and_sites_int() -> None:
    """Passing a concrete BaseGate instance should be preserved and sites can be an int."""
    x_gate = GateLibrary.x()
    obs = Observable(x_gate, sites=5)
    # same object semantics not required; equality via matrix is sufficient
    assert obs.gate.name == "x"
    assert np.allclose(obs.gate.matrix, x_gate.matrix)
    assert obs.sites == 5


def test_observable_from_gate_instance_with_list_sites() -> None:
    """Gate instance + list[int] sites should preserve the list (for two-site ops)."""
    cz_gate = GateLibrary.cz()
    obs = Observable(cz_gate, sites=[1, 3])
    assert obs.gate.name == "cz"
    assert obs.sites == [1, 3]


def test_aggregate_trajectories_regular_observable_mean() -> None:
    """Regular observables: results = mean(trajectories, axis=0).

    We create a single-site Z observable with a (num_traj x T) trajectory array and
    verify that `results` equals the columnwise mean.
    """
    # Observable to aggregate
    z_obs = Observable(GateLibrary.z(), sites=0)

    # Two trajectories across 3 time steps → mean is easy to verify
    traj = np.array(
        [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
        dtype=np.float64,
    )
    z_obs.trajectories = traj

    # Params (no PVM mixing, so just this observable)
    sim = AnalogSimParams(observables=[z_obs], elapsed_time=0.2, dt=0.1, num_traj=2, show_progress=False)

    sim.aggregate_trajectories()

    expected = traj.mean(axis=0)
    assert isinstance(z_obs.results, np.ndarray)
    np.testing.assert_allclose(z_obs.results, expected)


def test_aggregate_trajectories_schmidt_concatenation() -> None:
    """Schmidt spectrum: results = concatenation of raveled arrays from list entries.

    Provide a list of arrays with different shapes (1D/2D) to confirm `.ravel()` and
    `np.concatenate` behavior.
    """
    ss_obs = Observable(GateLibrary.schmidt_spectrum(), sites=[1, 2])

    # List of arrays (the method requires a list, not a single ndarray)
    a = np.array([0.8, 0.6], dtype=np.float64)
    b = np.array([0.4, 0.3], dtype=np.float64)  # will ravel to [0.4, 0.3]
    c = np.array([0.2, 0.1], dtype=np.float64)  # will ravel to [0.2, 0.1]
    ss_obs.trajectories = np.array([a, b, c])

    sim = AnalogSimParams(observables=[ss_obs], elapsed_time=0.1, dt=0.1, num_traj=3, show_progress=False)

    sim.aggregate_trajectories()

    assert isinstance(ss_obs.results, np.ndarray)
    np.testing.assert_allclose(ss_obs.results, np.array([0.8, 0.6, 0.4, 0.3, 0.2, 0.1], dtype=np.float64))


def test_aggregate_trajectories_mixed_regular_and_schmidt() -> None:
    """Combination: both regular and Schmidt observables are updated correctly."""
    # Regular observable with 3 trajectories x 2 time steps
    x_obs = Observable(GateLibrary.x(), sites=2)
    x_obs.trajectories = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float64)

    # Schmidt spectrum list
    ss_obs = Observable(GateLibrary.schmidt_spectrum(), sites=[0, 1])
    ss_obs.trajectories = np.array([np.array([1.0, 0.5], dtype=np.float64), np.array([0.5, 0.25], dtype=np.float64)])

    sim = AnalogSimParams(observables=[x_obs, ss_obs], elapsed_time=0.2, dt=0.1, num_traj=3, show_progress=False)

    sim.aggregate_trajectories()

    # Regular → column-wise mean over axis=0
    np.testing.assert_allclose(x_obs.results, np.array([1.0, 1.0], dtype=np.float64))

    # Schmidt → concatenation
    np.testing.assert_allclose(ss_obs.results, np.array([1.0, 0.5, 0.5, 0.25], dtype=np.float64))


def test_aggregate_trajectories_schmidt_requires_array() -> None:
    """For Schmidt spectrum, trajectories must be a *array*; list should raise AssertionError."""
    ss_obs = Observable(GateLibrary.schmidt_spectrum(), sites=[2, 3])

    # Wrong type: a single ndarray (method expects list[...] and asserts)
    ss_obs.trajectories = [0.9, 0.1]

    sim = AnalogSimParams(observables=[ss_obs], elapsed_time=0.1, dt=0.1, show_progress=False)

    with pytest.raises(AssertionError):
        sim.aggregate_trajectories()


def test_strong_params_sorting_and_fields() -> None:
    """Constructor sorts non-diagnostic observables by site, diagnostics appended.

    Sortable: gates NOT in {pvm, runtime_cost, max_bond, total_bond, schmidt_spectrum}
    Unsorted tail: the listed diagnostics/meta that keep their relative order.
    """
    # Sortable by site:
    obs_z3 = Observable(GateLibrary.z(), sites=3)
    obs_x2 = Observable(GateLibrary.x(), sites=2)
    obs_y1 = Observable(GateLibrary.y(), sites=1)
    # Unsorted block (diagnostics/meta) — keep insertion order:
    obs_cost = Observable(GateLibrary.runtime_cost(), sites=0)
    obs_tot = Observable(GateLibrary.total_bond(), sites=0)
    obs_ssp = Observable(GateLibrary.schmidt_spectrum(), sites=[1, 2])

    params = StrongSimParams(
        observables=[obs_z3, obs_x2, obs_y1, obs_cost, obs_tot, obs_ssp],
        num_traj=7,
        max_bond_dim=128,
        get_state=True,
        sample_layers=True,
        num_mid_measurements=2,
        show_progress=False,
    )

    # Expect sortable by site: y@1, x@2, z@3 then diagnostics/meta in given order
    for j, o in enumerate(params.sorted_observables):
        if j == 0:
            assert o is obs_y1
        elif j == 1:
            assert o is obs_ssp
        elif j == 2:
            assert o is obs_x2
        elif j == 3:
            assert o is obs_z3
        elif j == 4:
            assert o is obs_cost
        elif j == 5:
            assert o is obs_tot

    # Parameter fields are retained
    assert params.num_traj == 7
    assert params.max_bond_dim == 128
    assert np.isclose(params.threshold, 1e-10)
    assert params.get_state is True
    assert params.sample_layers is True
    assert params.num_mid_measurements == 2


def test_strong_params_rejects_mixed_pvm_with_non_pvm() -> None:
    """Constructor must assert when mixing PVM with non-PVM observables."""
    pvm = Observable(GateLibrary.pvm("101"), sites=None)
    z0 = Observable(GateLibrary.z(), sites=0)
    with pytest.raises(AssertionError):
        _ = StrongSimParams(observables=[pvm, z0])


def test_strong_params_accepts_all_pvm_or_all_non_pvm() -> None:
    """Constructor allows all-PVM and all-non-PVM sets."""
    # All PVM
    p1 = Observable(GateLibrary.pvm("0"), sites=None)
    p2 = Observable(GateLibrary.pvm("1"), sites=None)
    _ = StrongSimParams(observables=[p1, p2])  # should not raise

    # All non-PVM
    z0 = Observable(GateLibrary.z(), sites=0)
    x1 = Observable(GateLibrary.x(), sites=1)
    _ = StrongSimParams(observables=[z0, x1])  # should not raise


def test_strong_aggregate_regular_mean() -> None:
    """Regular observables: results = mean(trajectories, axis=0)."""
    x = Observable(GateLibrary.x(), sites=2)
    traj = np.array(
        [[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    x.trajectories = traj

    params = StrongSimParams(observables=[x], num_traj=3, show_progress=False)
    params.aggregate_trajectories()

    assert isinstance(x.results, np.ndarray)
    np.testing.assert_allclose(x.results, traj.mean(axis=0))


def test_strong_aggregate_schmidt_concat() -> None:
    """Schmidt spectrum: concatenation of raveled list entries."""
    ssp = Observable(GateLibrary.schmidt_spectrum(), sites=[0, 1])
    ssp.trajectories = np.array([
        np.array([0.9, 0.8], dtype=np.float64),
        np.array([0.6, 0.4], dtype=np.float64),  # ravel -> [0.6, 0.4]
        np.array([0.2, 0.1], dtype=np.float64),  # ravel -> [0.2, 0.1]
    ])

    params = StrongSimParams(observables=[ssp], num_traj=3)
    params.aggregate_trajectories()

    assert isinstance(ssp.results, np.ndarray)
    np.testing.assert_allclose(ssp.results, np.array([0.9, 0.8, 0.6, 0.4, 0.2, 0.1], dtype=np.float64))


def test_strong_aggregate_mixed_regular_and_schmidt() -> None:
    """Combination case: regular and Schmidt updated correctly in one call."""
    z = Observable(GateLibrary.z(), sites=0)
    z.trajectories = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)  # mean -> [2.0, 3.0]

    ssp = Observable(GateLibrary.schmidt_spectrum(), sites=[1, 2])
    ssp.trajectories = np.array([np.array([1.0, 0.5], dtype=np.float64), np.array([0.5, 0.25], dtype=np.float64)])

    params = StrongSimParams(observables=[z, ssp], num_traj=2)
    params.aggregate_trajectories()

    np.testing.assert_allclose(z.results, np.array([2.0, 3.0], dtype=np.float64))
    np.testing.assert_allclose(ssp.results, np.array([1.0, 0.5, 0.5, 0.25], dtype=np.float64))


def test_strong_aggregate_schmidt_requires_array() -> None:
    """Schmidt branch must assert if trajectories is not an array."""
    ssp = Observable(GateLibrary.schmidt_spectrum(), sites=[0, 1])
    ssp.trajectories = [0.9, 0.1]

    params = StrongSimParams(observables=[ssp], num_traj=1)

    with pytest.raises(AssertionError):
        params.aggregate_trajectories()
