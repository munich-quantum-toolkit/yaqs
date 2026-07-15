# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for process-tensor reconstruction error metrics."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.shared.encoding import pack_rho8
from mqt.yaqs.characterization.memory.shared.metrics import (
    compute_rel_fro_error,
    compute_trace_distance,
    mean_frobenius_mse_rho8,
    mean_trace_distance_rho8,
)


def test_rel_fro_error_zero_for_equal_matrices() -> None:
    """Relative Frobenius error vanishes for identical matrices."""
    mat = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.complex128)
    assert compute_rel_fro_error(mat, mat.copy()) == pytest.approx(0.0)


def test_rel_fro_error_scaling() -> None:
    """Relative Frobenius error scales predictably under scalar multiplication."""
    mat = np.eye(2, dtype=np.complex128)
    scaled = 2.0 * mat
    # ||A - 2A||_F / ||2A||_F = ||A||_F / (2||A||_F) = 0.5
    assert np.isclose(compute_rel_fro_error(mat, scaled), 0.5)


def test_trace_distance_basic_pure_states() -> None:
    """Trace distance between orthogonal pure states equals one."""
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    dist = compute_trace_distance(rho0, rho1)
    assert np.isclose(dist, 1.0)


def test_rel_fro_error_rejects_shape_mismatch() -> None:
    """Mismatched matrix shapes are rejected instead of broadcasting."""
    a = np.eye(2, dtype=np.complex128)
    b = np.eye(3, dtype=np.complex128)
    with pytest.raises(ValueError, match="must share the same shape"):
        compute_rel_fro_error(a, b)


def test_trace_distance_rejects_non_square_inputs() -> None:
    """Non-square density matrices are rejected."""
    rho = np.zeros((2, 3), dtype=np.complex128)
    sigma = np.zeros((2, 3), dtype=np.complex128)
    with pytest.raises(ValueError, match="must be square matrices"):
        compute_trace_distance(rho, sigma)


def test_rho8_metrics_zero_when_equal() -> None:
    """Packed-state metrics vanish when comparing identical inputs."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    y = pack_rho8(rho)[None, :]
    assert mean_trace_distance_rho8(y, y) == pytest.approx(0.0)
    assert mean_frobenius_mse_rho8(y, y) == pytest.approx(0.0)


def test_rho8_metrics_positive_for_different_states() -> None:
    """Packed-state metrics are positive for distinct quantum states."""
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    y0 = pack_rho8(rho0)[None, :]
    y1 = pack_rho8(rho1)[None, :]
    assert mean_trace_distance_rho8(y0, y1) > 0.9
    assert mean_frobenius_mse_rho8(y0, y1) > 0.0


def test_rho8_metrics_shape_mismatch_raises() -> None:
    """Packed-state metrics reject mismatched batch shapes."""
    y = pack_rho8(np.eye(2, dtype=np.complex128))[None, :]
    with pytest.raises(ValueError, match="must share shape"):
        mean_trace_distance_rho8(y, y[:0])
    with pytest.raises(ValueError, match="must share shape"):
        mean_frobenius_mse_rho8(y, np.zeros((2, 8), dtype=np.float32))


def test_rho8_metrics_reject_empty_batch() -> None:
    """Packed-state metrics reject zero-length batches."""
    empty = np.zeros((0, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="non-zero batch"):
        mean_trace_distance_rho8(empty, empty)
    with pytest.raises(ValueError, match="non-zero batch"):
        mean_frobenius_mse_rho8(empty, empty)
