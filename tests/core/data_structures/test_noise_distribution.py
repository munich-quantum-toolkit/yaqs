from __future__ import annotations

import numpy as np
import pytest
from mqt.yaqs.core.data_structures.noise_model import NoiseModel

def test_static_noise_strength():
    """Test that static float strengths are preserved."""
    processes = [{"name": "pauli_x", "sites": [0], "strength": 0.5}]
    nm = NoiseModel(processes)
    sampled_nm = nm.sample()
    assert len(sampled_nm.processes) == 1
    assert sampled_nm.processes[0]["strength"] == 0.5

def test_normal_distribution_sampling():
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

def test_mixed_static_and_distribution():
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
    assert sampled_nm.processes[1]["strength"] != 0.2 # Unlikely to be exactly mean

def test_invalid_distribution_type():
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
