# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for noise characterization propagation."""

from __future__ import annotations

import re

import numpy as np
import pytest

from mqt.yaqs import Observable
from mqt.yaqs.characterization.noise.shared.propagation import Propagator
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.libraries.gate_library import Z

from ..fixtures import NoiseTestConfig, build_propagator


def test_propagator_runs(noise_test_config: NoiseTestConfig) -> None:
    """Propagation returns observable trajectories with the expected shape."""
    _hamiltonian, _state, _observables, _sim_params, noise_model, propagator = build_propagator(noise_test_config)
    propagator.run(noise_model)

    assert isinstance(propagator.times, np.ndarray)
    assert isinstance(propagator.obs_array, np.ndarray)
    assert propagator.times.shape == (noise_test_config.n_t,)
    assert propagator.obs_array.shape == (noise_test_config.n_obs, noise_test_config.n_t)


def test_propagator_validation_errors(noise_test_config: NoiseTestConfig) -> None:
    """Propagator rejects invalid site indices and topology changes."""
    hamiltonian, init_state, observables, sim_params, noise_model, _ = build_propagator(noise_test_config)

    exceed_noise = CompactNoiseModel([
        {
            "name": "pauli_x",
            "sites": list(range(noise_test_config.sites + 1)),
            "strength": noise_test_config.gamma_x,
        },
        {"name": "pauli_y", "sites": list(range(noise_test_config.sites)), "strength": noise_test_config.gamma_y},
        {"name": "pauli_z", "sites": list(range(noise_test_config.sites)), "strength": noise_test_config.gamma_z},
    ])
    with pytest.raises(ValueError, match=re.escape("Noise site index exceeds number of sites in the Hamiltonian.")):
        Propagator(
            sim_params=sim_params,
            hamiltonian=hamiltonian,
            compact_noise_model=exceed_noise,
            init_state=init_state,
        )

    exceed_observables = [*observables, Observable(Z(), noise_test_config.sites)]
    propagator = Propagator(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        compact_noise_model=noise_model,
        init_state=init_state,
    )
    obs_err = "Observable site index exceeds number of sites in the Hamiltonian."
    with pytest.raises(ValueError, match=re.escape(obs_err)):
        propagator.set_observable_list(exceed_observables)

    propagator = Propagator(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        compact_noise_model=noise_model,
        init_state=init_state,
    )
    with pytest.raises(ValueError, match=re.escape("Observable list not set. Call set_observable_list first.")):
        propagator.run(noise_model)

    wrong_noise = CompactNoiseModel([
        {"name": "pauli_x", "sites": list(range(noise_test_config.sites)), "strength": noise_test_config.gamma_x},
        {"name": "pauli_y", "sites": list(range(noise_test_config.sites)), "strength": noise_test_config.gamma_y},
        {"name": "pauli_x", "sites": list(range(noise_test_config.sites)), "strength": noise_test_config.gamma_z},
    ])
    propagator = Propagator(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        compact_noise_model=noise_model,
        init_state=init_state,
    )
    propagator.set_observable_list(observables)
    topo_err = "Noise model topology does not match the initialized compact model."
    with pytest.raises(ValueError, match=re.escape(topo_err)):
        propagator.run(wrong_noise)
