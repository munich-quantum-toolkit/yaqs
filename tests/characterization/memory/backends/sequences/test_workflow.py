# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for comb-schedule sequence simulation."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.sequences.workflow import simulate_sequences
from mqt.yaqs.characterization.memory.shared.utils import make_mcwf_static_context
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams
from mqt.yaqs.core.parallel_utils import ExecutionConfig


def test_simulate_sequences_input_validation_errors() -> None:
    """simulate_sequences validates comb schedule and trace feature inputs."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)

    with pytest.raises(ValueError, match="intervention_steps_list and initial_psis must have equal length"):
        simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.1],
            intervention_steps_list=[],
            initial_psis=[np.array([1.0, 0.0], dtype=np.complex128)],
            static_ctx=None,
            parallel=False,
        )

    with pytest.raises(ValueError, match="record_step_states=True requires e_features_rows"):
        simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.1, 0.1],
            intervention_steps_list=[[(np.array([1.0, 0.0]), np.array([1.0, 0.0]))]],
            initial_psis=[np.array([1.0, 0.0], dtype=np.complex128)],
            static_ctx=None,
            parallel=False,
            record_step_states=True,
            e_features_rows=None,
        )

    with pytest.raises(ValueError, match="e_features_rows is only used when record_step_states=True"):
        simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.1, 0.1],
            intervention_steps_list=[[(np.array([1.0, 0.0]), np.array([1.0, 0.0]))]],
            initial_psis=[np.array([1.0, 0.0], dtype=np.complex128)],
            static_ctx=None,
            parallel=False,
            record_step_states=False,
            e_features_rows=[np.zeros((1, 32), dtype=np.float32)],
        )

    with pytest.raises(ValueError, match="context_vec is only used when record_step_states=True"):
        simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.1, 0.1],
            intervention_steps_list=[[(np.array([1.0, 0.0]), np.array([1.0, 0.0]))]],
            initial_psis=[np.array([1.0, 0.0], dtype=np.complex128)],
            static_ctx=None,
            parallel=False,
            record_step_states=False,
            context_vec=np.zeros(4, dtype=np.float32),
        )


def test_simulate_sequences_mcwf_final_states_and_traces_smoke() -> None:
    """MCWF simulation returns final packed states or sequence traces."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)

    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    intervention_steps_list = [[(psi0, psi0)]]
    initial_psis = [psi0.copy()]
    timesteps = [0.0, 0.0]

    finals = simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=timesteps,
        intervention_steps_list=intervention_steps_list,
        initial_psis=initial_psis,
        static_ctx=static_ctx,
        parallel=False,
        show_progress=False,
        record_step_states=False,
    )
    assert isinstance(finals, np.ndarray)
    assert finals.shape == (1, 8)

    samples = simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=timesteps,
        intervention_steps_list=intervention_steps_list,
        initial_psis=initial_psis,
        static_ctx=static_ctx,
        parallel=False,
        show_progress=False,
        record_step_states=True,
        e_features_rows=[np.zeros((1, 32), dtype=np.float32)],
    )
    assert isinstance(samples, list)
    assert len(samples) == 1
    s0 = samples[0]
    assert s0.rho_0.shape == (8,)
    assert s0.E_features.shape == (1, 32)
    assert s0.rho_seq.shape == (1, 8)


def test_simulate_sequences_traced_incompatible_with_record_step_states() -> None:
    """Traced mode cannot be combined with per-step trace recording."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    with pytest.raises(ValueError, match="incompatible"):
        simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.0, 0.0],
            intervention_steps_list=[[(psi0, psi0)]],
            initial_psis=[psi0.copy()],
            static_ctx=static_ctx,
            parallel=False,
            traced=True,
            record_step_states=True,
            e_features_rows=[np.zeros((1, 32), dtype=np.float32)],
        )


def test_simulate_sequences_e_features_rows_length_mismatch() -> None:
    """Per-sequence Choi rows must align with the number of sequences."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    with pytest.raises(ValueError, match="e_features_rows length"):
        simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.0, 0.0],
            intervention_steps_list=[[(psi0, psi0)], [(psi0, psi0)]],
            initial_psis=[psi0.copy(), psi0.copy()],
            static_ctx=static_ctx,
            parallel=False,
            record_step_states=True,
            e_features_rows=[np.zeros((1, 32), dtype=np.float32)],
        )


def test_simulate_sequences_parallel_smoke() -> None:
    """Parallel MCWF sequence simulation completes for a tiny batch."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    intervention_steps_list = [[(psi0, psi0)], [(psi0, psi0)]]
    initial_psis = [psi0.copy(), psi0.copy()]
    cfg = ExecutionConfig(parallel=True, max_workers=2, show_progress=False)

    finals = simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=[0.0, 0.0],
        intervention_steps_list=intervention_steps_list,
        initial_psis=initial_psis,
        static_ctx=static_ctx,
        record_step_states=False,
        _execution=cfg,
    )
    assert isinstance(finals, np.ndarray)
    assert finals.shape == (2, 8)


def test_simulate_sequences_empty_workload_returns_defined_results() -> None:
    """Empty sequence batches return empty arrays or trace lists instead of failing."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)

    finals = simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=[0.1],
        intervention_steps_list=[],
        initial_psis=[],
        static_ctx=None,
        parallel=False,
        record_step_states=False,
    )
    assert isinstance(finals, np.ndarray)
    assert finals.shape == (0, 8)

    packed, traces = simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=[0.1],
        intervention_steps_list=[],
        initial_psis=[],
        static_ctx=None,
        parallel=False,
        record_step_states=False,
        traced=True,
    )
    assert isinstance(packed, np.ndarray)
    assert packed.shape == (0, 8)
    assert traces == []


def test_simulate_sequences_empty_workload_rejects_invalid_mode_options() -> None:
    """Empty batches still run mode validation before returning early."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)

    with pytest.raises(ValueError, match="context_vec is only used when record_step_states=True"):
        simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.1],
            intervention_steps_list=[],
            initial_psis=[],
            static_ctx=None,
            parallel=False,
            record_step_states=False,
            context_vec=np.zeros(4, dtype=np.float32),
        )

    with pytest.raises(ValueError, match="e_features_rows is only used when record_step_states=True"):
        simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.1],
            intervention_steps_list=[],
            initial_psis=[],
            static_ctx=None,
            parallel=False,
            record_step_states=False,
            e_features_rows=[np.zeros((1, 32), dtype=np.float32)],
        )
