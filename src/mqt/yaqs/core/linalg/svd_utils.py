# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Singular-value truncation helpers shared across tensor-network code."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

TruncMode = Literal["discarded_weight", "relative", "hard_cutoff", "relative_discarded_weight"]


def truncate(
    s_vec: NDArray[Any],
    *,
    mode: TruncMode,
    threshold: float,
    max_bond_dim: int | None = None,
    min_keep: int = 1,
) -> int:
    r"""Return how many leading singular values to keep after truncation.

    Args:
        s_vec: Singular values in non-increasing order (as returned by SVD).
        mode:
            - ``discarded_weight``: accumulate squared singular values from the
              smallest upward and stop once the cumulative discarded weight is
              ``>= threshold``.
            - ``relative``: keep those with ``s / s[0] >= threshold`` (unless
              ``s[0] == 0``, then keep is ``0`` before caps).
            - ``hard_cutoff``: count singular values strictly greater than
              ``threshold``, then apply ``min_keep`` / ``max_bond_dim``.
            - ``relative_discarded_weight``: discard the largest trailing block
              whose relative discarded weight
              ``sum(discarded s**2) / sum(all s**2)`` stays ``<= threshold``.
              Invariant under nonzero common rescaling of ``s``. An all-zero
              spectrum yields keep ``0`` before caps.
        threshold: Mode-dependent cutoff (see above).
        max_bond_dim: Optional hard cap on the returned keep count.
        min_keep: Minimum number of singular values to retain (applied last).

    Returns:
        Integer ``keep`` in ``[min_keep, len(s_vec)]`` (also capped by
        ``max_bond_dim`` when given).

    Raises:
        ValueError: If ``mode`` is not recognized, or if ``max_bond_dim`` is
            smaller than ``min_keep`` (which would make the hard cap
            unsatisfiable).
    """
    if max_bond_dim is not None and max_bond_dim < min_keep:
        msg = f"max_bond_dim ({max_bond_dim}) must be >= min_keep ({min_keep})"
        raise ValueError(msg)

    n = int(s_vec.size)
    if n == 0:
        return 0

    if mode == "hard_cutoff":
        keep = int(np.sum(s_vec > threshold))
    elif mode == "relative":
        smax = float(s_vec[0])
        keep = 0 if smax <= 0.0 else int(np.sum((s_vec / smax) >= threshold))
    elif mode == "discarded_weight":
        keep = n
        discard = 0.0
        for idx, s in enumerate(reversed(s_vec)):
            discard += float(s) ** 2
            if discard >= threshold:
                keep = max(n - idx, min_keep)
                break
    elif mode == "relative_discarded_weight":
        weights = np.asarray(s_vec, dtype=np.float64) ** 2
        total = float(np.sum(weights))
        if total <= 0.0:
            keep = 0
        else:
            keep = n
            discard = 0.0
            for idx, w in enumerate(reversed(weights)):
                candidate = discard + float(w)
                if candidate / total <= threshold:
                    discard = candidate
                    keep = n - idx - 1
                else:
                    break
    else:
        msg = f"Unknown truncation mode: {mode!r}"
        raise ValueError(msg)

    if max_bond_dim is not None:
        keep = min(keep, max_bond_dim)
    keep = max(keep, min_keep)
    return min(keep, n)
