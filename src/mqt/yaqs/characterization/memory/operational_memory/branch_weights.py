# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Analytic branch weights for operational memory."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mqt.yaqs.characterization.memory.shared.encoding import DEFAULT_INITIAL_RHO0
from mqt.yaqs.characterization.memory.shared.intervention_steps import (
    apply_intervention_to_rho,
    compute_intervention_probability,
)

from .grid import assemble_probe_sequence

if TYPE_CHECKING:
    from .samples import ProbeSet


def _compute_branch_weight_for_sequence(steps: list[Any], *, cut: int) -> float:
    """Compute analytic branch weight from step probabilities up to ``cut``.

    Args:
        steps: Full intervention sequence.
        cut: Causal cut index.

    Returns:
        Cumulative branch weight ``prod_t p_t`` for ``t < cut``.
    """
    rho = DEFAULT_INITIAL_RHO0.copy()
    weight = 1.0
    for t in range(min(int(cut), len(steps))):
        sp = compute_intervention_probability(rho, steps[t])
        weight *= sp
        if weight < 1e-15:
            return float(weight)
        rho = apply_intervention_to_rho(rho, steps[t])
    return float(weight)


def compute_branch_weights(probe_set: ProbeSet) -> np.ndarray:
    r"""Compute analytic branch weights :math:`w_{\alpha,m}` at the causal cut.

    Args:
        probe_set: Sampled split-cut probes.

    Returns:
        Array of shape ``(n_pasts, n_futures)`` constant across future columns per past.
    """
    n_pasts = len(probe_set.past_pairs)
    n_futures = len(probe_set.future_pairs)
    cut = probe_set.cut
    w = np.empty((n_pasts, n_futures), dtype=np.float64)
    for i in range(n_pasts):
        w_i = _compute_branch_weight_for_sequence(assemble_probe_sequence(probe_set, i, 0), cut=cut)
        w[i, :] = w_i
    return w
