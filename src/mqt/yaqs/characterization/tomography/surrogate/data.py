# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Surrogate training data: one sequence rollout and batch stacking.

Used by :mod:`mqt.yaqs.characterization.tomography.surrogate.workflow` and benchmarks that roll out sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np


@dataclass(frozen=True)
class SequenceRolloutSample:
    """One **sequence rollout**: simulation along a fixed intervention **sequence** with per-step states.

    ``rho_seq[t]`` is the reduced state on site 0 **after** intervention ``t`` and the
    subsequent evolution segment (aligned with ``timesteps[t]``).

    ``E_features`` rows have length ``d_e`` (32 for the default single-qubit Choi flattening). This is
    not the same as a single stochastic
    **trajectory** when ``num_trajectories > 1`` under a noise model (see
    :func:`~mqt.yaqs.tomography.construct`), returning :class:`~mqt.yaqs.characterization.tomography.process_tensor.data.SequenceData`.
    """

    rho_0: np.ndarray  # shape (8,), float32 — packed 2x2 rho before first intervention
    E_features: np.ndarray  # shape (K, d_e), float32
    rho_seq: np.ndarray  # shape (K, 8), float32
    context: np.ndarray | None  # optional static features (e.g. dt, J, g), shape (d_ctx,)
    weight: float


def stack_rollouts(
    samples: list[SequenceRolloutSample],
    *,
    append_context_to_E: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Stack a batch of :class:`SequenceRolloutSample` into ``rho_0``, ``E``, ``rho_seq``, optional ``context``."""
    if not samples:
        msg = "stack_rollouts requires at least one SequenceRolloutSample."
        raise ValueError(msg)
    rho_0 = np.stack([s.rho_0 for s in samples], axis=0).astype(np.float32)
    E = np.stack([s.E_features for s in samples], axis=0).astype(np.float32)
    rho_seq = np.stack([s.rho_seq for s in samples], axis=0).astype(np.float32)
    ctx = None
    if samples[0].context is not None:
        ctx = np.stack([cast(np.ndarray, s.context) for s in samples], axis=0).astype(np.float32)
    if append_context_to_E and ctx is not None:
        k = E.shape[1]
        ctx_b = np.broadcast_to(ctx[:, None, :], (E.shape[0], k, ctx.shape[1])).astype(np.float32)
        E = np.concatenate([E, ctx_b], axis=-1)
        ctx = None
    return rho_0, E, rho_seq, ctx
