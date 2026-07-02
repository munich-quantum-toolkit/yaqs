# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for noise-characterization forward-model selection."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from mqt.yaqs.characterization.noise.optimization.trajectories import build_trajectory_loss
from mqt.yaqs.characterization.noise.shared.representation import (
    DEFAULT_LINDBLAD_MAX_QUBITS,
    DEFAULT_VECTOR_MAX_QUBITS,
    NoiseRepresentation,
    prepare_state_for_representation,
    resolve_noise_representation,
)
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.noise_characterizer import NoiseCharacterizer
from mqt.yaqs.simulator import Simulator

from ..fixtures import NoiseTestConfig, build_propagator


def test_resolve_explicit_representations() -> None:
    """Explicit representation strings pass through unchanged."""
    assert resolve_noise_representation(4, "density_matrix") == "density_matrix"
    assert resolve_noise_representation(4, "vector") == "vector"
    assert resolve_noise_representation(4, "mps") == "mps"


def test_resolve_invalid_representation_raises() -> None:
    """Unknown representation strings raise ValueError."""
    with pytest.raises(ValueError, match="representation must be"):
        resolve_noise_representation(2, cast("NoiseRepresentation", "invalid"))


def test_resolve_auto_lindblad_first() -> None:
    """Auto mode prefers Lindblad, then MCWF, then TJM by chain length."""
    assert resolve_noise_representation(1, "auto") == "density_matrix"
    assert (
        resolve_noise_representation(
            DEFAULT_LINDBLAD_MAX_QUBITS + 1,
            "auto",
            lindblad_max_qubits=DEFAULT_LINDBLAD_MAX_QUBITS,
            vector_max_qubits=DEFAULT_VECTOR_MAX_QUBITS,
        )
        == "vector"
    )
    assert (
        resolve_noise_representation(
            DEFAULT_VECTOR_MAX_QUBITS + 1,
            "auto",
            lindblad_max_qubits=DEFAULT_LINDBLAD_MAX_QUBITS,
            vector_max_qubits=DEFAULT_VECTOR_MAX_QUBITS,
        )
        == "mps"
    )


def test_prepare_state_for_representation_density_matrix() -> None:
    """Preset states can be encoded for Lindblad propagation."""
    prepared = prepare_state_for_representation(State(1, initial="zeros"), "density_matrix")
    assert prepared.representation == "density_matrix"
    assert prepared.density_matrix.shape == (2, 2)


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_lindblad_loss_is_deterministic() -> None:
    """Repeated loss evaluations at the same rates return identical values under Lindblad."""
    test = NoiseTestConfig(sites=1, ntraj=4)
    hamiltonian, init_state, observables, sim_params, reference_model, propagator = build_propagator(test)
    propagator.run(reference_model)
    ref = np.asarray(propagator.obs_array, dtype=float)
    init_guess = NoiseModel([{"name": "pauli_y", "sites": [0], "strength": 0.1}])
    loss, _ = build_trajectory_loss(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        init_guess=init_guess,
        observables=observables,
        ref_expectations=ref,
        simulator=Simulator(show_progress=False),
        representation="density_matrix",
        lindblad_max_qubits=DEFAULT_LINDBLAD_MAX_QUBITS,
        vector_max_qubits=DEFAULT_VECTOR_MAX_QUBITS,
    )
    x = np.array([proc["strength"] for proc in init_guess.processes], dtype=float)
    loss_a = loss(x)
    loss_b = loss(x)
    assert loss_a == pytest.approx(loss_b)


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_mcwf_and_tjm_smoke() -> None:
    """Explicit vector and mps representations still run through characterize."""
    test = NoiseTestConfig(sites=1, ntraj=2, max_bond_dim=4)
    hamiltonian, init_state, observables, sim_params, reference_model, _ = build_propagator(test)
    init_guess = NoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.2},
        {"name": "pauli_y", "sites": [0], "strength": 0.08},
        {"name": "pauli_z", "sites": [0], "strength": 0.05},
    ])
    x_low = np.zeros(3)
    x_up = np.full(3, 0.5)

    mcwf_result = NoiseCharacterizer(show_progress=False, representation="vector").characterize(
        hamiltonian,
        sim_params,
        init_state=init_state,
        init_guess=init_guess,
        observables=observables,
        reference_model=reference_model,
        x_low=x_low,
        x_up=x_up,
        max_iter=1,
        popsize=4,
        sigma0=0.05,
        seed=1,
    )
    assert mcwf_result.best_loss >= 0.0

    tjm_result = NoiseCharacterizer(show_progress=False, representation="mps").characterize(
        hamiltonian,
        sim_params,
        init_state=init_state,
        init_guess=init_guess,
        observables=observables,
        reference_model=reference_model,
        x_low=x_low,
        x_up=x_up,
        max_iter=1,
        popsize=4,
        sigma0=0.05,
        seed=2,
    )
    assert tjm_result.best_loss >= 0.0
