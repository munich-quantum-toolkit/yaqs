# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the noise characterizer facade."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.noise_characterizer import NoiseCharacterizer

from .conftest import NoiseTestConfig, build_propagator


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_noise_characterizer_cma_smoke(noise_test_config: NoiseTestConfig) -> None:
    """CMA-ES can reduce trajectory error on a tiny reference problem."""
    hamiltonian, init_state, observables, sim_params, reference_model, _ = build_propagator(noise_test_config)
    init_guess = CompactNoiseModel([
        {"name": "pauli_x", "sites": list(range(noise_test_config.sites)), "strength": 0.2},
        {"name": "pauli_y", "sites": list(range(noise_test_config.sites)), "strength": 0.08},
        {"name": "pauli_z", "sites": list(range(noise_test_config.sites)), "strength": 0.05},
    ])
    characterizer = NoiseCharacterizer.from_reference(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        reference_model=reference_model,
        init_guess=init_guess,
        observables=observables,
    )

    x_low = np.array([0.0, 0.0, 0.0])
    x_up = np.array([0.5, 0.5, 0.5])
    result = characterizer.optimize(x_low=x_low, x_up=x_up, max_iter=3, popsize=4, sigma0=0.05)

    assert result.best_loss >= 0.0
    assert len(result.parameter_history) >= 1
    assert isinstance(result.optimal_model, CompactNoiseModel)
    assert characterizer.propagator is characterizer.loss.propagator
    assert characterizer.loss.num_traj(0) == sim_params.num_traj
