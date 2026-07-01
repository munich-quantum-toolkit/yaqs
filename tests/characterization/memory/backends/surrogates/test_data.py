# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for surrogate sequence-record batching."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.surrogates.data import (
    SequenceRecord,
    stack_sequence_records,
)


def test_stack_sequence_records_shapes() -> None:
    """stack_sequence_records batches rho_0, features, and per-step states with correct ranks."""
    s1 = SequenceRecord(
        rho_0=np.ones(8, dtype=np.float32),
        E_features=np.ones((2, 4), dtype=np.float32),
        rho_seq=np.ones((2, 8), dtype=np.float32),
        context=None,
        weight=1.0,
    )
    s2 = SequenceRecord(
        rho_0=np.full(8, 2.0, dtype=np.float32),
        E_features=np.full((2, 4), 2.0, dtype=np.float32),
        rho_seq=np.full((2, 8), 2.0, dtype=np.float32),
        context=None,
        weight=0.5,
    )
    rho0, e_features, rho_seq, ctx = stack_sequence_records([s1, s2])
    assert rho0.shape == (2, 8)
    assert e_features.shape == (2, 2, 4)
    assert rho_seq.shape == (2, 2, 8)
    assert ctx is None


def test_stack_sequence_records_raises_on_empty() -> None:
    """stack_sequence_records rejects an empty sample list."""
    with pytest.raises(ValueError, match="stack_sequence_records requires at least one"):
        stack_sequence_records([])


def test_stack_sequence_records_appends_context_to_features() -> None:
    """append_context_to_features concatenates context onto every step row."""
    s = SequenceRecord(
        rho_0=np.zeros(8, dtype=np.float32),
        E_features=np.zeros((1, 4), dtype=np.float32),
        rho_seq=np.zeros((1, 8), dtype=np.float32),
        context=np.array([1.0, 2.0], dtype=np.float32),
        weight=1.0,
    )
    rho0, e_features, rho_seq, ctx = stack_sequence_records([s], append_context_to_features=True)
    assert e_features.shape == (1, 1, 6)
    assert ctx is None
    assert rho0.shape == (1, 8)
    assert rho_seq.shape == (1, 1, 8)


def test_stack_sequence_records_rejects_mixed_context_samples() -> None:
    """Mixed presence of context across samples is rejected."""
    base = SequenceRecord(
        rho_0=np.zeros(8, dtype=np.float32),
        E_features=np.zeros((1, 4), dtype=np.float32),
        rho_seq=np.zeros((1, 8), dtype=np.float32),
        context=None,
        weight=1.0,
    )
    with_ctx = SequenceRecord(
        rho_0=np.zeros(8, dtype=np.float32),
        E_features=np.zeros((1, 4), dtype=np.float32),
        rho_seq=np.zeros((1, 8), dtype=np.float32),
        context=np.array([1.0], dtype=np.float32),
        weight=1.0,
    )
    with pytest.raises(ValueError, match=r"SequenceRecord\.context must be present for all"):
        stack_sequence_records([base, with_ctx])
