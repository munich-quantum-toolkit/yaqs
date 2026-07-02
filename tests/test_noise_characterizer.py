# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :class:`~mqt.yaqs.noise_characterizer.NoiseCharacterizer`."""

from __future__ import annotations

from concurrent.futures import CancelledError

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.noise_characterizer import NoiseCharacterizer
from tests.characterization.noise.fixtures import NoiseTestConfig, build_propagator


@pytest.fixture
def noise_test_config() -> NoiseTestConfig:
    """Default open-system geometry for facade smoke tests.

    Returns:
        Single-site Lindblad test configuration.
    """
    return NoiseTestConfig()


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_characterize_smoke(noise_test_config: NoiseTestConfig) -> None:
    """One-shot characterize reduces trajectory error on a tiny problem."""
    hamiltonian, init_state, observables, sim_params, reference_model, _ = build_propagator(noise_test_config)
    init_guess = CompactNoiseModel([
        {"name": "pauli_x", "sites": list(range(noise_test_config.sites)), "strength": 0.2},
        {"name": "pauli_y", "sites": list(range(noise_test_config.sites)), "strength": 0.08},
        {"name": "pauli_z", "sites": list(range(noise_test_config.sites)), "strength": 0.05},
    ])
    nc = NoiseCharacterizer(show_progress=False)
    result = nc.characterize(
        hamiltonian,
        sim_params,
        init_state=init_state,
        init_guess=init_guess,
        observables=observables,
        reference_model=reference_model,
        x_low=np.zeros(3),
        x_up=np.full(3, 0.5),
        max_iter=3,
        popsize=4,
        sigma0=0.05,
        seed=1,
    )

    assert nc.resolved_representation == "density_matrix"
    assert result.best_loss >= 0.0
    assert result.ref_traj is not None
    assert result.fit_traj is not None
    assert result.trajectory_rmse() >= 0.0


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_characterize_ref_expectations_path(noise_test_config: NoiseTestConfig) -> None:
    """Characterize accepts precomputed experimental trajectories."""
    hamiltonian, init_state, observables, sim_params, reference_model, propagator = build_propagator(noise_test_config)
    propagator.run(reference_model)
    experimental = np.asarray(propagator.obs_array, dtype=float)
    init_guess = CompactNoiseModel([
        {"name": "pauli_x", "sites": list(range(noise_test_config.sites)), "strength": 0.2},
        {"name": "pauli_y", "sites": list(range(noise_test_config.sites)), "strength": 0.08},
        {"name": "pauli_z", "sites": list(range(noise_test_config.sites)), "strength": 0.05},
    ])
    result = NoiseCharacterizer(show_progress=False).characterize(
        hamiltonian,
        sim_params,
        init_state=init_state,
        init_guess=init_guess,
        observables=observables,
        ref_expectations=experimental,
        x_low=np.zeros(3),
        x_up=np.full(3, 0.5),
        max_iter=2,
        popsize=4,
        sigma0=0.05,
        seed=2,
    )
    np.testing.assert_allclose(np.asarray(result.ref_traj, dtype=float), experimental)


def test_execution_config_properties() -> None:
    """Facade exposes execution settings like MemoryCharacterizer."""
    nc = NoiseCharacterizer(
        parallel=False,
        show_progress=False,
        mp_context="fork",
        max_retries=3,
    )
    assert nc.parallel is False
    assert nc.show_progress is False
    assert nc.max_workers >= 1
    assert nc.mp_context == "fork"
    assert nc.max_retries == 3
    assert CancelledError in nc.retry_exceptions


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_characterize_reference_model_path(noise_test_config: NoiseTestConfig) -> None:
    """Characterize accepts reference_model as a benchmark shortcut."""
    hamiltonian, init_state, observables, sim_params, reference_model, _ = build_propagator(noise_test_config)
    init_guess = CompactNoiseModel([
        {"name": "pauli_x", "sites": list(range(noise_test_config.sites)), "strength": 0.2},
        {"name": "pauli_y", "sites": list(range(noise_test_config.sites)), "strength": 0.08},
        {"name": "pauli_z", "sites": list(range(noise_test_config.sites)), "strength": 0.05},
    ])
    result = NoiseCharacterizer(show_progress=False).characterize(
        hamiltonian,
        sim_params,
        init_state=init_state,
        init_guess=init_guess,
        observables=observables,
        reference_model=reference_model,
        x_low=np.zeros(3),
        x_up=np.full(3, 0.5),
        max_iter=2,
        popsize=4,
        sigma0=0.05,
        seed=3,
    )
    assert result.ref_traj is not None
    assert result.fit_traj is not None


def test_characterize_requires_reference_source(noise_test_config: NoiseTestConfig) -> None:
    """Characterize rejects calls that omit both reference sources."""
    hamiltonian, init_state, observables, sim_params, _, _ = build_propagator(noise_test_config)
    init_guess = CompactNoiseModel([
        {"name": "pauli_x", "sites": list(range(noise_test_config.sites)), "strength": 0.2},
        {"name": "pauli_y", "sites": list(range(noise_test_config.sites)), "strength": 0.08},
        {"name": "pauli_z", "sites": list(range(noise_test_config.sites)), "strength": 0.05},
    ])
    with pytest.raises(ValueError, match="exactly one"):
        NoiseCharacterizer(show_progress=False).characterize(
            hamiltonian,
            sim_params,
            init_state=init_state,
            init_guess=init_guess,
            observables=observables,
            x_low=np.zeros(3),
            x_up=np.full(3, 0.5),
        )
