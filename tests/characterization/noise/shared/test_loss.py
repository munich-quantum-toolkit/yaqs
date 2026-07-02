# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for trajectory-matching loss."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.noise.shared.loss import TrajectoryLoss
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel

from ..fixtures import NoiseTestConfig, build_propagator


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_trajectory_loss_call_evaluates_propagation(noise_test_config: NoiseTestConfig) -> None:
    """Loss evaluation runs propagation and returns a scalar objective."""
    _hamiltonian, _state, _observables, _sim_params, noise_model, propagator = build_propagator(noise_test_config)
    propagator.run(noise_model)
    ref = np.asarray(propagator.obs_array, dtype=float)
    loss = TrajectoryLoss(ref_expectations=ref, propagator=propagator)
    value = loss(noise_model.strength_list)
    assert value >= 0.0


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_trajectory_loss_honors_num_traj(noise_test_config: NoiseTestConfig) -> None:
    """Loss propagation uses ``AnalogSimParams.num_traj`` from the wired propagator."""
    _hamiltonian, _state, _observables, sim_params, noise_model, propagator = build_propagator(noise_test_config)
    propagator.run(noise_model)
    ref = np.asarray(propagator.obs_array, dtype=float)
    loss = TrajectoryLoss(ref_expectations=ref, propagator=propagator)
    loss(noise_model.strength_list)
    assert loss.propagator.sim_params.num_traj == sim_params.num_traj


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_trajectory_loss_wrong_parameter_length(noise_test_config: NoiseTestConfig) -> None:
    """Loss rejects parameter vectors with the wrong length."""
    _hamiltonian, _state, _observables, _sim_params, noise_model, propagator = build_propagator(noise_test_config)
    propagator.run(noise_model)
    ref = np.asarray(propagator.obs_array, dtype=float)
    loss = TrajectoryLoss(ref_expectations=ref, propagator=propagator)
    with pytest.raises(ValueError, match="Input array must have length"):
        loss(np.array([0.1]))


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_x_to_noise_model_updates_strengths(noise_test_config: NoiseTestConfig) -> None:
    """Strength vector maps back to a compact noise model."""
    _hamiltonian, _state, _observables, _sim_params, noise_model, propagator = build_propagator(noise_test_config)
    propagator.run(noise_model)
    ref = np.asarray(propagator.obs_array, dtype=float)
    loss = TrajectoryLoss(ref_expectations=ref, propagator=propagator)
    updated = loss.x_to_noise_model(np.array([0.11, 0.12, 0.13]))
    assert isinstance(updated, CompactNoiseModel)
    assert updated.strength_list == pytest.approx([0.11, 0.12, 0.13])
