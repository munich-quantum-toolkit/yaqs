# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: PLC2701 -- white-box validation of comb-schedule worker internals

"""Tests for comb-schedule worker validation and traced simulation."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.sequences.workers import (
    _comb_durations_ops_ctx,
    _get_times_cached,
    _reshape_choi_feature_rows,
    _validate_comb_sequence_inputs,
)
from mqt.yaqs.characterization.memory.backends.sequences.workflow import simulate_sequences
from mqt.yaqs.characterization.memory.shared.encoding import unpack_rho8
from mqt.yaqs.characterization.memory.shared.utils import make_mcwf_static_context
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_get_times_cached_zero_and_distinct_durations() -> None:
    """Duration-based cache keys distinguish zero-length and non-aligned segments."""
    cache: dict[tuple[float, float], np.ndarray] = {}
    zero = _get_times_cached(cache, dt=0.1, duration=0.0)
    np.testing.assert_allclose(zero, np.array([0.0]))
    short = _get_times_cached(cache, dt=0.1, duration=0.1)
    long = _get_times_cached(cache, dt=0.1, duration=0.2)
    assert short[-1] == pytest.approx(0.1)
    assert long[-1] == pytest.approx(0.2)
    assert len(cache) == 3
    with pytest.raises(ValueError, match="integer multiple"):
        _get_times_cached(cache, dt=0.1, duration=0.15)


def test_reshape_choi_feature_rows_rejects_malformed_inputs() -> None:
    """Malformed Choi feature storage raises before silent reshaping."""
    with pytest.raises(ValueError, match="divisible"):
        _reshape_choi_feature_rows(np.arange(5, dtype=np.float32), num_steps=2)
    with pytest.raises(ValueError, match="num_steps"):
        _reshape_choi_feature_rows(np.ones((3, 4), dtype=np.float32), num_steps=2)


def test_validate_comb_sequence_inputs_timesteps_length() -> None:
    """Comb schedule requires timesteps of length k+1."""
    psi_pairs = [[(np.array([1.0, 0.0]), np.array([1.0, 0.0]))] * 2]
    with pytest.raises(ValueError, match=r"timesteps.*k\+1"):
        _validate_comb_sequence_inputs(
            psi_pairs_list=psi_pairs,
            timesteps=[0.1],
            timesteps_rows=None,
            operators_list=None,
            static_ctx_list=None,
        )
    _validate_comb_sequence_inputs(
        psi_pairs_list=psi_pairs,
        timesteps=[0.0, 0.0, 0.0],
        timesteps_rows=None,
        operators_list=None,
        static_ctx_list=None,
    )


def test_validate_comb_sequence_inputs_mismatched_k_without_rows() -> None:
    """Sequences must share k when timesteps_rows is omitted."""
    with pytest.raises(ValueError, match="share the same k"):
        _validate_comb_sequence_inputs(
            psi_pairs_list=[
                [(np.array([1.0, 0.0]), np.array([1.0, 0.0]))],
                [(np.array([1.0, 0.0]), np.array([1.0, 0.0]))] * 2,
            ],
            timesteps=[0.0, 0.0],
            timesteps_rows=None,
            operators_list=None,
            static_ctx_list=None,
        )


def test_simulate_sequences_traced_returns_diagnostics() -> None:
    """traced=True returns finals and per-sequence trace dicts (exact backend path)."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    psi_pairs_list = [[(psi0, psi0)], [(psi0, psi0)]]
    initial_psis = [psi0.copy(), psi0.copy()]

    result = simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=[0.0, 0.0],
        psi_pairs_list=psi_pairs_list,
        initial_psis=initial_psis,
        static_ctx=static_ctx,
        parallel=False,
        show_progress=False,
        record_step_states=False,
        traced=True,
    )
    assert isinstance(result, tuple)
    finals, traces = result
    assert isinstance(finals, np.ndarray)
    assert isinstance(traces, list)
    assert finals.shape == (2, 8)
    assert len(traces) == 2
    for trace in traces:
        assert isinstance(trace, dict)
        assert "step_probs" in trace
        assert "cumulative_weight_final" in trace
        assert "terminated_early" in trace


def test_validate_comb_sequence_inputs_per_sequence_schedules() -> None:
    """Per-sequence timesteps, operators, and MCWF contexts must align with k."""
    psi_pairs = [[(np.array([1.0, 0.0]), np.array([1.0, 0.0]))]]
    op = MPO.ising(length=1, J=0.0, g=0.0)
    static_ctx = make_mcwf_static_context(op, AnalogSimParams(dt=0.1), noise_model=None)

    with pytest.raises(ValueError, match="timesteps_rows"):
        _validate_comb_sequence_inputs(
            psi_pairs_list=psi_pairs,
            timesteps=[0.0, 0.0],
            timesteps_rows=[[0.0]],
            operators_list=None,
            static_ctx_list=None,
        )
    with pytest.raises(ValueError, match="length must match number of sequences"):
        _validate_comb_sequence_inputs(
            psi_pairs_list=[psi_pairs[0], psi_pairs[0]],
            timesteps=[0.0, 0.0],
            timesteps_rows=[[0.0, 0.0]],
            operators_list=None,
            static_ctx_list=None,
        )
    with pytest.raises(ValueError, match="operators_list"):
        _validate_comb_sequence_inputs(
            psi_pairs_list=psi_pairs,
            timesteps=[0.0, 0.0],
            timesteps_rows=None,
            operators_list=[[op]],
            static_ctx_list=None,
        )
    with pytest.raises(ValueError, match="operators_list` length"):
        _validate_comb_sequence_inputs(
            psi_pairs_list=[psi_pairs[0], psi_pairs[0]],
            timesteps=[0.0, 0.0],
            timesteps_rows=None,
            operators_list=[[op, op]],
            static_ctx_list=None,
        )
    with pytest.raises(ValueError, match="static_ctx_list"):
        _validate_comb_sequence_inputs(
            psi_pairs_list=psi_pairs,
            timesteps=[0.0, 0.0],
            timesteps_rows=None,
            operators_list=None,
            static_ctx_list=[[static_ctx]],
        )
    with pytest.raises(ValueError, match="static_ctx_list` length"):
        _validate_comb_sequence_inputs(
            psi_pairs_list=[psi_pairs[0], psi_pairs[0]],
            timesteps=[0.0, 0.0],
            timesteps_rows=None,
            operators_list=None,
            static_ctx_list=[[static_ctx, static_ctx]],
        )


def test_comb_durations_ops_ctx_uses_per_sequence_rows() -> None:
    """Per-sequence duration rows override the shared comb schedule."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    durs, ops, ctxs = _comb_durations_ops_ctx(
        sequence_idx=0,
        k=1,
        timesteps=[0.1, 0.2],
        timesteps_rows=[[0.3, 0.4]],
        hamiltonian=op,
        operators_list=None,
        mcwf_static_ctx=None,
        mcwf_static_ctx_list=None,
    )
    assert durs == [0.3, 0.4]
    assert len(ops) == 2
    assert ctxs == [None, None]


def test_simulate_sequences_dict_step_types() -> None:
    """Comb workers accept structured dict intervention steps."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    x = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    steps = [
        {"type": "unitary", "U": u},
        {"type": "measure_only", "psi_meas": z, "psi_reset": z},
        {"type": "prepare_only", "psi_prep": x},
        {"type": "reset_only", "psi_reset": z},
    ]
    finals = simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=[0.0] * 5,
        psi_pairs_list=[steps],
        initial_psis=[z.copy()],
        static_ctx=static_ctx,
        parallel=False,
        show_progress=False,
        record_step_states=False,
    )
    assert isinstance(finals, np.ndarray)
    assert finals.shape == (1, 8)


def test_prepare_only_unconditional_from_non_zero_state() -> None:
    """prepare_only resets site 0 without conditioning on |0>."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    initial = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    steps = [{"type": "prepare_only", "psi_prep": plus}]
    finals = simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=[0.0, 0.0],
        psi_pairs_list=[steps],
        initial_psis=[initial],
        static_ctx=static_ctx,
        parallel=False,
        show_progress=False,
        record_step_states=False,
    )
    assert isinstance(finals, np.ndarray)
    rho2 = unpack_rho8(np.asarray(finals[0], dtype=np.float64))
    target = np.outer(plus, plus.conj())
    np.testing.assert_allclose(rho2, target, atol=1e-10)


def test_simulate_sequences_trace_worker_early_termination_fill() -> None:
    """Trace worker pads remaining steps when branch weight vanishes."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    x = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
    # Project onto |1> while state is |0> → zero weight, early stop
    steps = [{"type": "measure_only", "psi_meas": x, "psi_reset": z}]
    samples = simulate_sequences(
        operator=op,
        sim_params=params,
        timesteps=[0.0, 0.0],
        psi_pairs_list=[steps],
        initial_psis=[z.copy()],
        static_ctx=static_ctx,
        parallel=False,
        show_progress=False,
        record_step_states=True,
        e_features_rows=[np.zeros((1, 32), dtype=np.float32)],
    )
    assert len(samples) == 1
    assert samples[0].rho_seq.shape == (1, 8)


def test_reshape_choi_feature_rows_rejects_high_dim() -> None:
    """Choi feature rows must be 1D or 2D."""
    with pytest.raises(ValueError, match="1D or 2D"):
        _reshape_choi_feature_rows(np.ones((2, 2, 2), dtype=np.float32), num_steps=2)


def test_simulate_sequences_rejects_unsupported_dict_step() -> None:
    """Unknown structured step types fail fast in the worker core."""
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1)
    static_ctx = make_mcwf_static_context(op, params, noise_model=None)
    z = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
    with pytest.raises(ValueError, match="Unsupported step type"):
        simulate_sequences(
            operator=op,
            sim_params=params,
            timesteps=[0.0, 0.0],
            psi_pairs_list=[[{"type": "bogus"}]],
            initial_psis=[z.copy()],
            static_ctx=static_ctx,
            parallel=False,
            show_progress=False,
            record_step_states=False,
        )
