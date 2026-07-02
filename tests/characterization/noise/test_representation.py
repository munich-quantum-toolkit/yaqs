# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for noise-characterization forward-model selection."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.noise.shared.representation import (
    DEFAULT_LINDBLAD_MAX_QUBITS,
    DEFAULT_VECTOR_MAX_QUBITS,
    prepare_state_for_representation,
    resolve_noise_representation,
)
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.noise_characterizer import NoiseCharacterizer

from .conftest import NoiseTestConfig, build_propagator


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
    hamiltonian, init_state, observables, sim_params, reference_model, _ = build_propagator(test)
    init_guess = CompactNoiseModel([{"name": "pauli_y", "sites": [0], "strength": 0.1}])
    characterizer = NoiseCharacterizer.from_reference(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        reference_model=reference_model,
        init_guess=init_guess,
        observables=observables,
        representation="density_matrix",
    )
    x = characterizer.init_x.copy()
    loss_a, _, _ = characterizer.loss(x)
    loss_b, _, _ = characterizer.loss(x)
    assert loss_a == pytest.approx(loss_b)


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_from_reference_sets_resolved_representation() -> None:
    """The characterizer records the resolved forward backend."""
    test = NoiseTestConfig(sites=1)
    hamiltonian, init_state, observables, sim_params, reference_model, _ = build_propagator(test)
    init_guess = CompactNoiseModel([{"name": "pauli_y", "sites": [0], "strength": 0.1}])
    characterizer = NoiseCharacterizer.from_reference(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        reference_model=reference_model,
        init_guess=init_guess,
        observables=observables,
    )
    assert characterizer.resolved_representation == "density_matrix"
    assert characterizer.propagator.representation == "density_matrix"


@pytest.mark.filterwarnings("ignore:.*special injected samples.*:UserWarning")
def test_mcwf_and_tjm_smoke() -> None:
    """Explicit vector and mps representations still run through the characterizer."""
    test = NoiseTestConfig(sites=1, ntraj=2, max_bond_dim=4)
    hamiltonian, init_state, observables, sim_params, reference_model, _ = build_propagator(test)
    init_guess = CompactNoiseModel([
        {"name": "pauli_x", "sites": [0], "strength": 0.2},
        {"name": "pauli_y", "sites": [0], "strength": 0.08},
        {"name": "pauli_z", "sites": [0], "strength": 0.05},
    ])
    x_low = np.zeros(3)
    x_up = np.full(3, 0.5)

    mcwf_characterizer = NoiseCharacterizer.from_reference(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        reference_model=reference_model,
        init_guess=init_guess,
        observables=observables,
        representation="vector",
    )
    mcwf_result = mcwf_characterizer.optimize(
        x_low=x_low,
        x_up=x_up,
        max_iter=1,
        popsize=4,
        sigma0=0.05,
        seed=1,
    )
    assert mcwf_characterizer.resolved_representation == "vector"
    assert mcwf_result.best_loss >= 0.0

    tjm_characterizer = NoiseCharacterizer.from_reference(
        sim_params=sim_params,
        hamiltonian=hamiltonian,
        init_state=init_state,
        reference_model=reference_model,
        init_guess=init_guess,
        observables=observables,
        representation="mps",
    )
    tjm_result = tjm_characterizer.optimize(
        x_low=x_low,
        x_up=x_up,
        max_iter=1,
        popsize=4,
        sigma0=0.05,
        seed=2,
    )
    assert tjm_characterizer.resolved_representation == "mps"
    assert tjm_result.best_loss >= 0.0
