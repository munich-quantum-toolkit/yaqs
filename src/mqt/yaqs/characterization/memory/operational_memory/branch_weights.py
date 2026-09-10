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


def _compute_branch_weight_for_sequence(steps: list[Any]) -> float:
    """Compute the analytic probability of all retained outcomes in a sequence.

    Args:
        steps: Full intervention sequence.

    Returns:
        Complete cumulative branch weight ``prod_t p_t``.
    """
    rho = DEFAULT_INITIAL_RHO0.copy()
    weight = 1.0
    for step in steps:
        sp = compute_intervention_probability(rho, step)
        weight *= sp
        if weight < 1e-15:
            return float(weight)
        rho = apply_intervention_to_rho(rho, step)
    return float(weight)


def compute_branch_weights(probe_set: ProbeSet) -> np.ndarray:
    """Compute isolated-probe complete-record weight diagnostics.

    This helper propagates only a single-qubit reference state through the probe
    interventions. It does not include target dynamics or environmental correlations and is
    therefore not used by the canonical response-matrix path.

    Args:
        probe_set: Sampled split-cut probes.

    Returns:
        Array of shape ``(n_pasts, n_futures)``. Entries can vary along both axes when future
        probe steps retain non-deterministic outcomes.
    """
    n_pasts = len(probe_set.past_pairs)
    n_futures = len(probe_set.future_pairs)
    w = np.empty((n_pasts, n_futures), dtype=np.float64)
    for i in range(n_pasts):
        for j in range(n_futures):
            w[i, j] = _compute_branch_weight_for_sequence(assemble_probe_sequence(probe_set, i, j))
    return w
