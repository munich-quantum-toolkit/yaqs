# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the NoiseModel class.

This module provides unit tests for the NoiseModel class.
It verifies that a NoiseModel is created correctly when valid processes and strengths are provided,
raises an AssertionError when the lengths of the processes and strengths lists differ,
and handles empty noise models appropriately.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import Z
from mqt.yaqs.core.libraries.noise_library import PauliX, PauliY, PauliZ
from mqt.yaqs.simulator import run


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    return np.allclose(a, b, atol=1e-12)


def test_noise_model_creation() -> None:
    """Test that NoiseModel is created correctly with valid process dicts.

    This test constructs a NoiseModel with two single-site processes
    ("lowering" and "pauli_z") and corresponding strengths.
    It verifies that:
      - Each process is stored as a dictionary with correct fields.
      - The number of processes is correct.
      - Each process contains a jump_operator with the expected shape (2x2).
    """
    processes: list[dict[str, Any]] = [
        {"name": "lowering", "sites": [0], "strength": 0.1},
        {"name": "pauli_z", "sites": [1], "strength": 0.05},
    ]

    model = NoiseModel(processes)

    assert len(model.processes) == 2
    assert model.processes[0]["name"] == "lowering"
    assert model.processes[1]["name"] == "pauli_z"
    assert model.processes[0]["strength"] == 0.1
    assert model.processes[1]["strength"] == 0.05
    assert model.processes[0]["sites"] == [0]
    assert model.processes[1]["sites"] == [1]
    assert model.processes[0]["matrix"].shape == (2, 2)
    assert model.processes[1]["matrix"].shape == (2, 2)


def test_noise_model_assertion() -> None:
    """Test that NoiseModel raises an AssertionError when a process dict is missing required fields.

    This test constructs a process list where one entry is missing the 'strength' field,
    which should cause the NoiseModel initialization to fail.
    """
    # Missing 'strength' in the second dict
    processes: list[dict[str, Any]] = [
        {"name": "lowering", "sites": [0], "strength": 0.1},
        {"name": "pauli_z", "sites": [1]},  # Missing strength
    ]

    with pytest.raises(AssertionError):
        _ = NoiseModel(processes)


def test_noise_model_empty() -> None:
    """Test that NoiseModel handles an empty list of processes without error.

    This test initializes a NoiseModel with an empty list of process dictionaries and verifies that the resulting
    model has empty `processes` and `jump_operators` lists.
    """
    model = NoiseModel()

    assert model.processes == []


def test_noise_model_none() -> None:
    """Test that NoiseModel handles a None input without error.

    This test initializes a NoiseModel with `None` and verifies that the resulting
    model has no processes.
    """
    model = NoiseModel(None)

    assert model.processes == []


def test_one_site_matrix_auto() -> None:
    """Test that one-site processes auto-fill a 2x2 'matrix'.

    This verifies that providing name/sites/strength for a single-site process
    produces a process with a 2x2 operator populated from the library.
    """
    nm = NoiseModel([{"name": "pauli_x", "sites": [1], "strength": 0.1}])
    assert len(nm.processes) == 1
    p = nm.processes[0]
    assert "matrix" in p, "1-site process should have matrix auto-filled"
    assert p["matrix"].shape == (2, 2)
    assert _allclose(p["matrix"], PauliX.matrix)


def test_adjacent_two_site_matrix_auto() -> None:
    """Test that adjacent two-site processes auto-fill a 4x4 'matrix'.

    This checks that nearest-neighbor crosstalk uses the library matrix (kron)
    and requires no explicit operator in the process dict.
    """
    nm = NoiseModel([{"name": "crosstalk_xz", "sites": [1, 2], "strength": 0.2}])
    p = nm.processes[0]
    assert "matrix" in p, "Adjacent 2-site process should have matrix auto-filled"
    assert p["matrix"].shape == (4, 4)
    expected = np.kron(PauliX.matrix, PauliZ.matrix)
    assert _allclose(p["matrix"], expected)


def test_longrange_two_site_factors_auto() -> None:
    """Test that long-range two-site processes auto-fill 'factors' only.

    Using the canonical 'longrange_crosstalk_{ab}' name, the model should attach
    per-site 2x2 factors (A,B) and omit any large Kronecker 'matrix'.
    """
    nm = NoiseModel([{"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 0.3}])
    p = nm.processes[0]
    assert "factors" in p, "Long-range 2-site process should have factors auto-filled"
    a_op, b_op = p["factors"]
    assert a_op.shape == (2, 2)
    assert b_op.shape == (2, 2)
    assert _allclose(a_op, PauliX.matrix)
    assert _allclose(b_op, PauliY.matrix)
    assert "matrix" not in p, "Long-range processes should not attach a full matrix"


def test_longrange_two_site_factors_explicit() -> None:
    """Test that explicit 'factors' for long-range are accepted and sites normalize.

    Supplying (A,B) and unsorted endpoints should result in stored ascending sites,
    preserving factors and omitting a full 'matrix'.
    """
    nm = NoiseModel([
        {
            "name": "custom_longrange_xy",
            "sites": [3, 1],  # intentionally unsorted
            "strength": 0.3,
            "factors": (PauliX.matrix, PauliY.matrix),
        }
    ])
    p = nm.processes[0]
    # Sites must be normalized to ascending order
    assert p["sites"] == [1, 3]
    assert "factors" in p
    assert len(p["factors"]) == 2
    a_op, b_op = p["factors"]
    assert _allclose(a_op, PauliX.matrix)
    assert _allclose(b_op, PauliY.matrix)
    assert "matrix" not in p


def test_longrange_unknown_label_without_factors_raises() -> None:
    """Test that unknown long-range labels without 'factors' raise.

    If the name is not 'longrange_crosstalk_{ab}' and no factors are provided,
    initialization must fail to avoid guessing operators.

    Raises:
        AssertionError: If the model accepts an unknown long-range label without factors.
    """
    try:
        # Name is not a recognized non-adjacent 'crosstalk_{ab}' and no factors provided
        _ = NoiseModel([{"name": "foo_bar", "sites": [0, 2], "strength": 0.1}])
    except AssertionError:
        return
    msg = "Expected AssertionError for unknown long-range label without factors."
    raise AssertionError(msg)


def test_noise_distribution_integration() -> None:
    """Test that running a simulation with a distributed noise model works.

    The noise model should be sampled once per run, meaning all trajectories
    share the same (randomly sampled) noise strengths.
    """
    num_qubits = 2
    # Define Hamiltonian: Ising model
    hamiltonian = MPO.ising(num_qubits, J=1.0, g=0.5)

    # Define noise model with distribution
    processes = [
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "normal", "mean": 0.1, "std": 0.01},
        }
    ]
    noise_model = NoiseModel(processes)

    # Initial state
    initial_state = MPS(num_qubits)

    # Simulation parameters
    sim_params = AnalogSimParams(
        observables=[Observable(Z(), 0)],
        dt=0.1,
        elapsed_time=1.0,
        num_traj=10,  # Run multiple trajectories to confirm it runs
        sample_timesteps=False,
    )

    # Run simulation
    run(initial_state, hamiltonian, sim_params, noise_model)

    # Verify that the noise model was sampled and stored
    assert sim_params.noise_model is not None, "Simulation parameters should store the sampled noise model."
    assert len(sim_params.noise_model.processes) == 1, "Sampled noise model should have one process."
    assert isinstance(sim_params.noise_model.processes[0]["strength"], float), "Process strength should be a float."


def test_static_noise_strength() -> None:
    """Test that static float strengths are preserved."""
    processes = [{"name": "pauli_x", "sites": [0], "strength": 0.5}]
    nm = NoiseModel(processes)
    rng = np.random.default_rng(42)
    sampled_nm = nm.sample(rng=rng)
    assert len(sampled_nm.processes) == 1
    assert sampled_nm.processes[0]["strength"] == 0.5


def test_normal_distribution_sampling() -> None:
    """Test sampling from a normal distribution."""
    mean = 0.5
    std = 0.1
    processes = [
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "normal", "mean": mean, "std": std},
        }
    ]
    nm = NoiseModel(processes)

    # Sample multiple times and check statistics
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    samples = []
    for _ in range(1000):
        sampled_nm = nm.sample(rng=rng)
        samples.append(sampled_nm.processes[0]["strength"])

    assert np.isclose(np.mean(samples), mean, atol=0.02)
    assert np.isclose(np.std(samples), std, atol=0.02)

    # Check that individual samples are floats
    assert isinstance(samples[0], float)


def test_normal_clamping_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Test that a warning is logged when normal distribution samples are clamped."""
    # Mean is negative, so most samples will be negative and clamped to 0
    mean = -0.5
    std = 0.1
    processes = [
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "normal", "mean": mean, "std": std},
        }
    ]
    nm = NoiseModel(processes)

    # Use a fixed seed that is known to produce a negative value for mean=-0.5
    rng = np.random.default_rng(42)
    with caplog.at_level(logging.WARNING):
        sampled_nm = nm.sample(rng=rng)
        # Ensure at least one sample was clamped (likely, given mean=-0.5)
        # We can force it by seeding if needed, but mean=-0.5 is robust enough.

    assert "was negative and clamped to 0.0" in caplog.text
    # usage of max(0.0, ...) implies strength should be 0.0
    assert sampled_nm.processes[0]["strength"] == 0.0


def test_truncated_normal_sampling() -> None:
    """Test sampling from a truncated normal distribution (lower bound 0)."""
    # Mean near 0, std large enough that normal would produce negatives
    mean = 0.0
    std = 1.0
    processes = [
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "truncated_normal", "mean": mean, "std": std},
        }
    ]
    nm = NoiseModel(processes)

    rng = np.random.default_rng(42)
    samples = []
    for _ in range(2000):
        sampled_nm = nm.sample(rng=rng)
        # Truncated normal (a=0) should strictly be >= 0
        s = sampled_nm.processes[0]["strength"]
        assert s >= 0
        samples.append(s)

    # For standard half-normal (mean=0, sigma=1, lower=0),
    expected_mean = std * np.sqrt(2 / np.pi)
    assert np.isclose(np.mean(samples), expected_mean, atol=0.05)


def test_truncated_normal_zero_std() -> None:
    """Test that truncated normal sampling with zero std returns the mean."""
    mean = 0.5
    std = 0.0
    processes = [
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "truncated_normal", "mean": mean, "std": std},
        }
    ]
    nm = NoiseModel(processes)
    rng = np.random.default_rng(42)
    sampled_nm = nm.sample(rng=rng)
    assert sampled_nm.processes[0]["strength"] == mean


def test_lognormal_distribution_sampling() -> None:
    """Test sampling from a log-normal distribution."""
    # Parameters for the underlying normal distribution
    mean = 0.0
    std = 0.1
    processes = [
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "lognormal", "mean": mean, "std": std},
        }
    ]
    nm = NoiseModel(processes)

    rng = np.random.default_rng(42)
    # Sample multiple times and check statistics
    samples = []
    for _ in range(5000):
        sampled_nm = nm.sample(rng=rng)
        samples.append(sampled_nm.processes[0]["strength"])

    # Expected mean and std for lognormal distribution
    expected_mean = np.exp(mean + (std**2) / 2)
    expected_var = (np.exp(std**2) - 1) * np.exp(2 * mean + std**2)
    expected_std = np.sqrt(expected_var)

    assert np.isclose(np.mean(samples), expected_mean, atol=0.05)
    assert np.isclose(np.std(samples), expected_std, atol=0.05)
    assert all(s > 0 for s in samples)


def test_mixed_static_and_distribution() -> None:
    """Test mixing static and distributed strengths."""
    processes = [
        {"name": "pauli_x", "sites": [0], "strength": 0.5},
        {
            "name": "pauli_z",
            "sites": [1],
            "strength": {"distribution": "lognormal", "mean": 0.0, "std": 0.1},
        },
    ]
    nm = NoiseModel(processes)
    rng = np.random.default_rng(42)
    sampled_nm = nm.sample(rng=rng)

    assert len(sampled_nm.processes) == 2
    assert sampled_nm.processes[0]["strength"] == 0.5
    assert isinstance(sampled_nm.processes[1]["strength"], float)
    assert sampled_nm.processes[1]["strength"] > 0.0


def test_invalid_distribution_type() -> None:
    """Test that invalid distribution types raise ValueError."""
    processes = [
        {
            "name": "pauli_x",
            "sites": [0],
            "strength": {"distribution": "unknown", "mean": 0.5, "std": 0.1},
        }
    ]
    nm = NoiseModel(processes)
    with pytest.raises(ValueError, match="Unsupported distribution type: unknown"):
        nm.sample()


def test_independent_site_sampling() -> None:
    """Test that multiple sites get independently sampled strengths."""
    # Define 10 sites with the same distribution parameters
    processes = [
        {
            "name": "pauli_x",
            "sites": [i],
            "strength": {"distribution": "lognormal", "mean": 0.0, "std": 0.1},
        }
        for i in range(10)
    ]
    nm = NoiseModel(processes)

    rng = np.random.default_rng(42)
    # Sample once
    sampled_nm = nm.sample(rng=rng)

    # Extract strengths
    strengths = [proc["strength"] for proc in sampled_nm.processes]

    # Check that we have 10 strengths
    assert len(strengths) == 10

    # Check that they are float values
    assert all(isinstance(s, float) for s in strengths)

    # Check that they are not all identical (extremely unlikely if independent)
    # Convert to set to check for uniqueness
    unique_strengths = set(strengths)
    assert len(unique_strengths) > 1, (
        "All sampled strengths were identical, implying they were not sampled independently."
    )

    # Check that they are positive (lognormal range)
    assert all(s > 0 for s in strengths)
