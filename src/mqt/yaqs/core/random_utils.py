# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utilities for reproducible stochastic simulation."""

from __future__ import annotations

import numpy as np


def make_trajectory_rng(traj_idx: int, *, base_seed: int | None) -> np.random.Generator:
    """Create a NumPy RNG for one stochastic trajectory.

    When ``base_seed`` is set, each trajectory index gets a distinct but reproducible stream
    via ``default_rng(base_seed + traj_idx)``. When ``base_seed`` is ``None``, returns an
    unseeded generator (non-deterministic across runs).

    Args:
        traj_idx: Trajectory index (0-based), typically the worker job id.
        base_seed: Optional run-level seed from simulation parameters.

    Returns:
        A NumPy random generator for quantum-jump decisions in that trajectory.
    """
    if base_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(base_seed + traj_idx)
