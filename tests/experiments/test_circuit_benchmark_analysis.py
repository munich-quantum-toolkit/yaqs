# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused tests for reliability and rank-certificate analysis."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.circuit_benchmarks.analyze import _samples_through, _schmidt_tail_bound


def test_contiguous_samples_require_every_preceding_step() -> None:
    """A missing earlier sample cannot be repaired by a later re-entry."""
    complete = {"rows": [{"step": step} for step in range(4)]}
    missing = {"rows": [{"step": 0}, {"step": 1}, {"step": 3}]}
    assert _samples_through(complete, 3) == complete["rows"]
    assert _samples_through(missing, 3) is None


def test_schmidt_tail_certifies_the_best_rank_one_infidelity() -> None:
    """The Bell-state rank-one lower bound is exactly one half."""
    bell = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    tail, cut, exact_rank = _schmidt_tail_bound(bell, n_qubits=2, chi=1)
    assert tail == pytest.approx(0.5)
    assert cut == 1
    assert exact_rank == 2


def test_product_state_has_zero_tail_above_rank_one() -> None:
    """A product state has no rank-one approximation obstruction."""
    product = np.zeros(16, dtype=np.complex128)
    product[0] = 1.0
    tail, _, exact_rank = _schmidt_tail_bound(product, n_qubits=4, chi=1)
    assert tail == pytest.approx(0.0)
    assert exact_rank == 1
