# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

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
    samples = []
    for _ in range(1000):
        sampled_nm = nm.sample()
        samples.append(sampled_nm.processes[0]["strength"])

    assert np.isclose(np.mean(samples), mean, atol=0.02)
    assert np.isclose(np.std(samples), std, atol=0.02)

    # Check that individual samples are floats
    assert isinstance(samples[0], float)


def test_mixed_static_and_distribution() -> None:
    """Test mixing static and distributed strengths."""
    processes = [
        {"name": "pauli_x", "sites": [0], "strength": 0.5},
        {
            "name": "pauli_z",
            "sites": [1],
            "strength": {"distribution": "normal", "mean": 0.2, "std": 0.01},
        },
    ]
    nm = NoiseModel(processes)
    sampled_nm = nm.sample()

    assert len(sampled_nm.processes) == 2
    assert sampled_nm.processes[0]["strength"] == 0.5
    assert isinstance(sampled_nm.processes[1]["strength"], float)
    assert sampled_nm.processes[1]["strength"] != 0.2  # Unlikely to be exactly mean


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
            "strength": {"distribution": "normal", "mean": 0.5, "std": 0.2},
        }
        for i in range(10)
    ]
    nm = NoiseModel(processes)

    # Sample once
    sampled_nm = nm.sample()

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

    # Optional: check that they are somewhat distributed (rough range check)
    # With mean 0.5 and std 0.2, most values should be within [-0.1, 1.1]
    assert all(-0.5 < s < 1.5 for s in strengths)
