# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Trajectory-matching loss for noise-parameter optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mqt.yaqs.core.data_structures.noise_model import NoiseModel

if TYPE_CHECKING:
    from mqt.yaqs.characterization.noise.shared.propagation import Propagator


class TrajectoryLoss:
    """Mean-squared trajectory mismatch used by gradient-free optimizers."""

    def __init__(
        self,
        *,
        ref_expectations: np.ndarray,
        propagator: Propagator,
    ) -> None:
        """Initialize the loss from a reference trajectory.

        Args:
            ref_expectations: Reference observable expectations with shape
                ``(n_obs, n_times)``.
            propagator: Forward model used to simulate candidate noise parameters.
        """
        self.ref_traj_array = np.asarray(ref_expectations, dtype=float)
        self.propagator = propagator

        self.d = len(self.propagator.noise_model.processes)
        self.n_obs, self.n_t = self.ref_traj_array.shape
        self.loss_scale_factor = 1.0 / (self.n_obs * self.n_t)

    def x_to_noise_model(self, x: np.ndarray) -> NoiseModel:
        """Map a flat strength vector back to a :class:`NoiseModel`.

        Args:
            x: Process strength vector with length ``self.d``.

        Returns:
            Updated noise model.
        """
        processes = [{**proc, "strength": float(x[i])} for i, proc in enumerate(self.propagator.noise_model.processes)]
        return NoiseModel(processes)

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate the scaled mean-squared trajectory error.

        Args:
            x: Compact strength vector.

        Returns:
            Scaled mean-squared trajectory mismatch.

        Raises:
            ValueError: If ``x`` has the wrong length.
        """
        if len(x) != self.d:
            msg = f"Input array must have length {self.d}, got {len(x)}"
            raise ValueError(msg)

        noise_model = self.x_to_noise_model(x)
        self.propagator.run(noise_model)
        obs_array = np.asarray(self.propagator.obs_array, dtype=float)

        diff = obs_array - self.ref_traj_array
        return float(np.sum(diff**2) * self.loss_scale_factor)
