# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for surrogate rollout sample stacking."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.surrogates.data import (
    SeqTrace,
    stack_traces,
)


def test_stack_traces_shapes() -> None:
    """stack_traces batches rho_0, features, and per-step states with correct ranks."""
    s1 = SeqTrace(
        rho_0=np.zeros(8, dtype=np.float32),
        E_features=np.zeros((2, 32), dtype=np.float32),
        rho_seq=np.zeros((2, 8), dtype=np.float32),
        context=None,
        weight=1.0,
    )
    s2 = SeqTrace(
        rho_0=np.ones(8, dtype=np.float32),
        E_features=np.ones((2, 32), dtype=np.float32),
        rho_seq=np.ones((2, 8), dtype=np.float32),
        context=None,
        weight=2.0,
    )
    rho0, e_features, rho_seq, ctx = stack_traces([s1, s2])
    assert rho0.shape == (2, 8)
    assert e_features.shape == (2, 2, 32)
    assert rho_seq.shape == (2, 2, 8)
    assert ctx is None


def test_stack_traces_raises_on_empty() -> None:
    """stack_traces rejects an empty sample list."""
    with pytest.raises(ValueError, match="stack_traces requires at least one"):
        stack_traces([])


def test_stack_traces_appends_context_to_features() -> None:
    """append_context_to_features concatenates context vectors onto feature rows."""
    s = SeqTrace(
        rho_0=np.zeros(8, dtype=np.float32),
        E_features=np.zeros((2, 32), dtype=np.float32),
        rho_seq=np.zeros((2, 8), dtype=np.float32),
        context=np.ones(5, dtype=np.float32),
        weight=1.0,
    )
    rho0, e_features, rho_seq, ctx = stack_traces([s], append_context_to_features=True)
    assert rho0.shape == (1, 8)
    assert e_features.shape == (1, 2, 32 + 5)
    assert rho_seq.shape == (1, 2, 8)
    assert ctx is None
