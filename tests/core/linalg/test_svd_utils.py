# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for singular-value truncation helpers."""

from __future__ import annotations

import numpy as np
from typing import Any, cast

import pytest

from mqt.yaqs.core import linalg


def test_truncate_discarded_weight_gte() -> None:
    """``discarded_cmp='gte'`` accumulates from smallest until cumulative >= threshold."""
    s = np.array([10.0, 3.0, 1.0, 0.5], dtype=np.float64)
    # Discarded from smallest: 0.5^2 + 1^2 = 1.25 < 10 -> need more; + 3^2 = 10.25 >= 10
    keep = linalg.truncate(s, mode="discarded_weight", threshold=10.0, discarded_cmp="gte", min_keep=1)
    assert keep == 2  # keep 10 and 3


def test_truncate_discarded_weight_gt_tdvp_style() -> None:
    """``discarded_cmp='gt'`` matches TDVP strict inequality on the next partial sum."""
    s = np.array([5.0, 4.0, 3.0], dtype=np.float64)
    # reversed: 3^2=9, next 9+16=25 > 10 -> keep max(3-1, 1)=2
    keep = linalg.truncate(s, mode="discarded_weight", threshold=10.0, discarded_cmp="gt", min_keep=1)
    assert keep == 2


def test_truncate_relative_and_zero_smax() -> None:
    """Relative mode uses ratio to largest singular value."""
    s = np.array([2.0, 1.0, 0.4], dtype=np.float64)
    keep = linalg.truncate(s, mode="relative", threshold=0.45, min_keep=1)
    assert keep == 2  # 1.0 and 0.4/2=0.2 < 0.45 for third? 0.4/2=0.2 < 0.45, 1.0/2=0.5 >= 0.45 -> 2 values
    s0 = np.array([0.0, 1.0], dtype=np.float64)
    keep0 = linalg.truncate(s0, mode="relative", threshold=0.1, min_keep=1)
    assert keep0 == 1  # smax==0 -> keep 0 then min_keep


def test_truncate_hard_cutoff_and_caps() -> None:
    """Hard cutoff counts ``s > threshold``; ``max_bond_dim`` and ``min_keep`` apply."""
    s = np.array([5.0, 2.0, 0.5, 0.1], dtype=np.float64)
    keep = linalg.truncate(s, mode="hard_cutoff", threshold=0.2, min_keep=1)
    assert keep == 3
    keep_capped = linalg.truncate(s, mode="hard_cutoff", threshold=0.2, max_bond_dim=2, min_keep=1)
    assert keep_capped == 2


def test_truncate_unknown_mode_raises() -> None:
    """Invalid ``mode`` should raise ``ValueError``."""
    s = np.ones(3, dtype=np.float64)
    with pytest.raises(ValueError, match="Unknown truncation mode"):
        linalg.truncate(s, mode=cast(Any, "invalid"), threshold=1.0)
