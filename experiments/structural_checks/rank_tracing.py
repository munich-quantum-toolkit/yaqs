# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Temporary instrumentation for retained ranks in a production TDVP update."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class RankTrace:
    """Retained bond ranks observed during the TDVP split."""

    retained_bonds: list[int] = field(default_factory=list)
    peak_retained_rank: int = 1

    def record(self, left: np.ndarray, right: np.ndarray) -> None:
        """Record and validate one retained bond."""
        bond = int(left.shape[2])
        assert bond == int(right.shape[1])
        self.retained_bonds.append(bond)
        self.peak_retained_rank = max(self.peak_retained_rank, bond)


@contextmanager
def trace_split_ranks() -> Iterator[RankTrace]:
    """Wrap the production split only for the duration of the check."""
    from mqt.yaqs.core.methods.tdvp import integrators

    trace = RankTrace()
    original = integrators.split_tdvp

    def wrapped(merged, sim_params, physical_dimensions, svd_distribution, *, dynamic):
        left, right = original(
            merged,
            sim_params,
            physical_dimensions,
            svd_distribution,
            dynamic=dynamic,
        )
        trace.record(left, right)
        return left, right

    integrators.split_tdvp = wrapped  # type: ignore[assignment]
    try:
        yield trace
    finally:
        integrators.split_tdvp = original
