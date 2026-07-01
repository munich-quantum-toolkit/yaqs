# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Trajectory-matching loss for noise-parameter optimization."""

from __future__ import annotations

import copy
import time
from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from mqt.yaqs.characterization.noise.shared.propagation import Propagator


def default_num_traj(_evaluation: int) -> int:
    """Return a constant trajectory count for each loss evaluation."""
    return 1


class TrajectoryLoss:
    """Mean-squared trajectory mismatch used by gradient-free optimizers."""

    def __init__(
        self,
        *,
        ref_expectations: np.ndarray,
        propagator: Propagator,
        num_traj: Callable[[int], int] = default_num_traj,
    ) -> None:
        """Initialize the loss from a reference trajectory.

        Args:
            ref_expectations: Reference observable expectations with shape
                ``(n_obs, n_times)``.
            propagator: Forward model used to simulate candidate noise parameters.
            num_traj: Callable returning the trajectory count for each evaluation.
        """
        self.ref_traj_array = np.asarray(ref_expectations, dtype=float)
        self.propagator = copy.deepcopy(propagator)
        self.num_traj = num_traj
        self.n_eval = 0

        self.d = len(self.propagator.compact_noise_model.compact_processes)
        self.n_obs, self.n_t = self.ref_traj_array.shape
        self.loss_scale_factor = 1.0 / (self.n_obs * self.n_t)
        self.obs_array = np.empty_like(self.ref_traj_array)

    def x_to_noise_model(self, x: np.ndarray) -> CompactNoiseModel:
        """Map a flat strength vector back to a :class:`CompactNoiseModel`.

        Returns:
            Updated compact noise model.
        """
        processes = copy.deepcopy(self.propagator.compact_noise_model.compact_processes)
        for i in range(self.d):
            processes[i]["strength"] = float(x[i])
        return CompactNoiseModel(processes)

    def __call__(self, x: np.ndarray) -> tuple[float, np.ndarray, float]:
        """Evaluate the scaled mean-squared trajectory error.

        Args:
            x: Compact strength vector.

        Returns:
            Tuple of loss value, zero gradient placeholder, and wall-clock seconds.

        Raises:
            ValueError: If ``x`` has the wrong length.
        """
        if len(x) != self.d:
            msg = f"Input array must have length {self.d}, got {len(x)}"
            raise ValueError(msg)

        noise_model = self.x_to_noise_model(x)
        self.propagator.sim_params.num_traj = self.num_traj(self.n_eval)
        self.n_eval += 1

        start = time.time()
        self.propagator.run(noise_model)
        self.obs_array = np.asarray(self.propagator.obs_array, dtype=float)
        elapsed = time.time() - start

        diff = self.obs_array - self.ref_traj_array
        loss = float(np.sum(diff**2) * self.loss_scale_factor)
        return loss, np.zeros_like(x), elapsed
