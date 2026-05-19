# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Singular-value truncation helpers shared across tensor-network code."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

TruncMode = Literal["discarded_weight", "relative", "hard_cutoff"]
DiscardedCmp = Literal["gt", "gte"]


def truncate(
    s_vec: NDArray[Any],
    *,
    mode: TruncMode,
    threshold: float,
    max_bond_dim: int | None = None,
    min_keep: int = 1,
    discarded_cmp: DiscardedCmp = "gte",
) -> int:
    """Return how many leading singular values to keep after truncation.

    Args:
        s_vec: Singular values in non-increasing order (as returned by SVD).
        mode:
            - ``discarded_weight``: accumulate squared singular values from the
              smallest upward until the comparison with ``threshold`` fires.
            - ``relative``: keep those with ``s / s[0] >= threshold`` (unless
              ``s[0] == 0``, then keep is ``0`` before caps).
            - ``hard_cutoff``: count singular values strictly greater than
              ``threshold``, then apply ``min_keep`` / ``max_bond_dim``.
        threshold: Mode-dependent cutoff (see above).
        max_bond_dim: Optional hard cap on the returned keep count.
        min_keep: Minimum number of singular values to retain (applied last).
        discarded_cmp: For ``discarded_weight`` only: ``\"gt\"`` matches TDVP
            (strict inequality on the *next* partial sum), ``\"gte\"`` matches
            two-site / truncated-right SVD (inequality after including the
            current singular value).

    Returns:
        Integer ``keep`` in ``[min_keep, len(s_vec)]`` (also capped by
        ``max_bond_dim`` when given).

    Raises:
        ValueError: If ``mode`` is not recognized.
    """
    n = int(s_vec.size)
    if n == 0:
        return 0

    if mode == "hard_cutoff":
        keep = int(np.sum(s_vec > threshold))
    elif mode == "relative":
        smax = float(s_vec[0])
        keep = 0 if smax == 0.0 else int(np.sum((s_vec / smax) >= threshold))
    elif mode == "discarded_weight":
        keep = n
        discard = 0.0
        for idx, s in enumerate(reversed(s_vec)):
            next_discard = discard + float(s) ** 2
            if discarded_cmp == "gt":
                if next_discard > threshold:
                    keep = max(n - idx, min_keep)
                    break
                discard = next_discard
            else:
                discard = next_discard
                if discard >= threshold:
                    keep = max(n - idx, min_keep)
                    break
    else:
        msg = f"Unknown truncation mode: {mode!r}"
        raise ValueError(msg)

    if max_bond_dim is not None:
        keep = min(keep, max_bond_dim)
    keep = max(keep, min_keep)
    return min(keep, n)
