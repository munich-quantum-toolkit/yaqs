# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for experiment data normalization helpers."""

from __future__ import annotations

import numpy as np

from experiments.pt_entropy_figure.data import (
    meaningful_weights,
    modes_to_display,
    normalize_singular_weights,
    resolved_mask,
)


def test_normalize_singular_weights_sums_to_one() -> None:
    s = np.array([3.0, 1.0, 0.1], dtype=np.float64)
    p = normalize_singular_weights(s)
    assert abs(float(np.sum(p)) - 1.0) < 1e-10
    assert np.all(p[:-1] >= p[1:])


def test_meaningful_weights_drop_unresolved_tail() -> None:
    s = np.array([1.0, 1e-3, 1e-20], dtype=np.float64)
    p = normalize_singular_weights(s)
    s_keep, p_keep = meaningful_weights(s, p)
    assert s_keep.size == 2
    assert p_keep.size == 2
    assert p_keep[-1] > 1e-8


def test_modes_to_display_respects_cap_and_floor() -> None:
    p = np.array([0.7, 0.2, 0.05, 0.03, 0.01, 0.005, 0.003, 0.001, 1e-6], dtype=np.float64)
    p /= np.sum(p)
    assert modes_to_display(p) == 8
    assert resolved_mask(np.sqrt(p)).sum() >= 1
