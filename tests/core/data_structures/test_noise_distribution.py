# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for noise distribution support in NoiseModel."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.noise_model import NoiseModel


def test_static_noise_strength() -> None:
    """Test that static float strengths are preserved."""
    processes = [{"name": "pauli_x", "sites": [0], "strength": 0.5}]
    nm = NoiseModel(processes)
    sampled_nm = nm.sample()
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
    # expected mean = sigma * sqrt(2/pi) approx 0.798
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
    # No need for specific RNG here as it's deterministic, but good practice
    sampled_nm = nm.sample()
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
    # E[X] = exp(mu + sigma^2/2)
    # Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
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
