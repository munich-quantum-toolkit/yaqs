# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for comb-schedule worker validation and traced simulation."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.sequences.workers import (  # noqa: PLC2701
    _validate_comb_sequence_inputs,
)
from mqt.yaqs.characterization.memory.backends.sequences.workflow import simulate_sequences
from mqt.yaqs.characterization.memory.shared.utils import make_mcwf_static_context
from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_validate_comb_sequence_inputs_timesteps_length() -> None:
    """Comb schedule requires timesteps of length k+1."""
    psi_pairs = [[(np.array([1.0, 0.0]), np.array([1.0, 0.0]))] * 2]
    with pytest.raises(ValueError, match="timesteps.*k\\+1"):
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

    finals, traces = simulate_sequences(
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
    assert isinstance(finals, np.ndarray)
    assert finals.shape == (2, 8)
    assert len(traces) == 2
    for trace in traces:
        assert "step_probs" in trace
        assert "cumulative_weight_final" in trace
        assert "terminated_early" in trace
