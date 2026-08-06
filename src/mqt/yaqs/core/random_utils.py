# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utilities for reproducible stochastic simulation."""

from __future__ import annotations

import numpy as np

# Distinct stream tags so (base_seed, traj_idx) coordinates never collapse via addition.
_STREAM_TRAJECTORY = 0x5452414A  # "TRAJ"
_STREAM_SAMPLE = 0x53414D50  # "SAMP"
_STREAM_DISORDER = 0x4449534F  # "DISO"


def make_trajectory_rng(traj_idx: int, *, base_seed: int | None) -> np.random.Generator:
    """Create a NumPy RNG for one stochastic trajectory.

    When ``base_seed`` is set, each ``(base_seed, traj_idx)`` pair gets a distinct
    reproducible stream via ``SeedSequence`` coordinates (not ``base_seed + traj_idx``,
    which would alias ``(0, 1)`` with ``(1, 0)``). When ``base_seed`` is ``None``,
    returns an unseeded generator (non-deterministic across runs).

    Args:
        traj_idx: Trajectory index (0-based), typically the worker job id.
        base_seed: Optional run-level seed from simulation parameters.

    Returns:
        A NumPy random generator for quantum-jump decisions in that trajectory.
    """
    if base_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([base_seed, traj_idx, _STREAM_TRAJECTORY]))


def make_sample_rng(
    traj_idx: int,
    *,
    base_seed: int | None,
    timestep: int,
    stream_id: int | None = None,
) -> np.random.Generator:
    """Create an RNG for one TJM-2 measurement copy at a given timestep.

    Intermediate ``sample()`` calls evolve a deep copy of the sampling MPS. Their jump
    draws must not advance the generator used for ``initialize`` / ``step_through``, and
    each measurement timestep must get its own stream so enabling ``sample_timesteps``
    does not change the final measurement draw relative to final-only mode.

    When ``base_seed`` is set, the stream is derived from separate ``SeedSequence``
    coordinates ``(base_seed, traj_idx, timestep, sample-tag)``. A caller that
    composes multiple bounded evolutions may additionally supply ``stream_id`` to
    keep equal local timestep indices in distinct streams. When ``base_seed`` is
    ``None``, returns an unseeded generator.

    Args:
        traj_idx: Trajectory index (0-based), typically the worker job id.
        base_seed: Optional run-level seed from simulation parameters.
        timestep: Index into ``sim_params.times`` for this measurement copy.
        stream_id: Optional bounded-evolution identifier. Omit it to preserve the
            standalone sampling stream.

    Returns:
        A NumPy random generator for jump decisions on that measurement copy only.
    """
    if base_seed is None:
        return np.random.default_rng()
    coordinates = [base_seed, traj_idx, timestep, _STREAM_SAMPLE]
    if stream_id is not None:
        coordinates.insert(2, stream_id)
    return np.random.default_rng(np.random.SeedSequence(coordinates))


def make_disorder_rng(*, base_seed: int | None) -> np.random.Generator:
    """Create an RNG for static noise-model disorder sampling.

    Uses a stream tag distinct from trajectory and measurement RNGs so disorder
    sampling does not collide with jump streams for the same ``base_seed``.

    Args:
        base_seed: Optional run-level seed from simulation parameters.

    Returns:
        A NumPy random generator for ``NoiseModel.sample``.
    """
    if base_seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([base_seed, _STREAM_DISORDER]))
