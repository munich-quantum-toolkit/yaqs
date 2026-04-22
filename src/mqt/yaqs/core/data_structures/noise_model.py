# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Noise Models.

This module defines the NoiseModel class, which represents a noise model in a quantum system.
It stores a list of noise processes and their corresponding strengths, and automatically retrieves
the associated jump operator matrices from the GateLibrary. These jump operators are used to simulate
the effects of noise in quantum simulations.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import truncnorm

from ..libraries.gate_library import BaseGate, Crosstalk, GateLibrary


if TYPE_CHECKING:
    from numpy.typing import NDArray


logger = logging.getLogger(__name__)


class NoiseModel:
    """A class to represent a noise model with arbitrary-site jump operators.

     Attributes.
    ----------
    processes : list of str
        A list of noise processes affecting the system.
        Each process is a dict with:
            - name: process name or identifier
            - sites: indices of sites this process acts on
            - strength: noise strength
            - matrix: matrix representing the operator on those sites (for 1-site and adjacent 2-site processes)
            - factors: tuple of two 1-site operator matrices (for long-range 2-site processes)

    Methods:
    -------
    __init__:
        Initializes the NoiseModel with given processes.
    get_operator:
        Static method to retrieve the operator matrix for a given noise process name.
    """

    def __init__(
        self, processes: list[dict[str, Any]] | None = None, scheduled_jumps: list[dict[str, Any]] | None = None
    ) -> None:
        """Initialize the NoiseModel.

        Parameters
        ----------
        processes :
            A list of noise process dictionaries affecting the quantum system. Default is None.
        scheduled_jumps :
            A list of scheduled jumps to apply at specific times. Default is None.

        Note:
            Input validation is performed and assertion errors may be raised by
            internal helpers if inputs are malformed.
        """
        self.processes: list[dict[str, Any]] = []
        self.scheduled_jumps: list[dict[str, Any]] = []
        if scheduled_jumps is not None:
            for jump in scheduled_jumps:
                assert "time" in jump, "Each scheduled jump must have a 'time' key"
                assert "sites" in jump, "Each scheduled jump must have a 'sites' key"
                assert "name" in jump, "Each scheduled jump must have a 'name' key"
                assert len(jump["sites"]) <= 2, "Each scheduled jump must have at most 2 sites"
                jump_dict = dict(jump)  # Copy to avoid mutating caller's dict
                jump_op=NoiseModel.get_operator(jump_dict["name"])
                if "matrix" not in jump_dict:
                    jump_dict["matrix"] = jump_op.matrix
                self.scheduled_jumps.append(jump_dict)

        if processes is None:
            return

        filled_processes: list[dict[str, Any]] = []
        for original in processes:
            assert "name" in original, "Each process must have a 'name' key"
            assert "sites" in original, "Each process must have a 'sites' key"
            assert "strength" in original, "Each process must have a 'strength' key"
            assert len(original["sites"]) <= 2, "Each noise process must have at most 2 sites"

            proc = dict(original)
            name = proc["name"]
            sites = proc["sites"]

            name_op=NoiseModel.get_operator(name)

            if len(sites) == 1:
                proc["matrix"] = name_op.matrix

            else:  # Two-site: normalize site ordering
                sorted_sites = sorted(sites)
                swapped = sorted_sites != sites
                if swapped:
                    proc["sites"] = sorted_sites
                i, j = proc["sites"]

                if abs(j - i) == 1:  # Adjacent: store full matrix
                    if isinstance(name_op, Crosstalk):
                        proc["matrix"] = (
                            name_op.swapped_matrix if swapped else name_op.matrix
                        )
                    else:
                        proc["matrix"] = name_op.matrix

                else:  # Non-adjacent: store per-site factors
                    if isinstance(name_op, Crosstalk):
                        proc["factors"] = (name_op.matrix2, name_op.matrix1) if swapped else (name_op.matrix1, name_op.matrix2)
                    else:
                        assert "factors" in proc, (
                            "Non-adjacent 2-site processes must specify 'factors' unless a Crosstalk gate is provided."
                        )

            filled_processes.append(proc)

        self.processes = filled_processes

    def sample(self, rng: np.random.Generator | int | None = None) -> NoiseModel:
        """Sample a concrete NoiseModel from any distribution-based strengths.

        For each process:
            - If 'strength' is a float, it is kept as is.
            - If 'strength' is a dict describing a distribution, a value is sampled.

        Args:
            rng: The random number generator or seed to use for sampling.
                 If None, a new generator is created.

        Returns:
            NoiseModel: A new NoiseModel instance where all process strengths are concrete floats.
                        This sampled model represents the specific realization of static disorder used
                        for a simulation run.

        Raises:
            ValueError: If an unsupported distribution type is provided.
        """
        generator = np.random.default_rng(rng)
        new_processes: list[dict[str, Any]] = []
        for proc in self.processes:
            new_proc = copy.deepcopy(proc)
            strength_val = proc["strength"]

            if isinstance(strength_val, dict):
                if "distribution" not in strength_val:
                    msg = "Noise strength dict must contain 'distribution' key."
                    raise ValueError(msg)
                dist_type = strength_val["distribution"]
                mean = strength_val.get("mean", 0.0)
                std = strength_val.get("std", 0.0)

                if dist_type == "normal":
                    sampled_val = generator.normal(loc=mean, scale=std)
                    if sampled_val < 0:
                        logger.warning(
                            "Sampled noise strength %f using 'normal' distribution (mean=%f, std=%f) "
                            "was negative and clamped to 0.0.",
                            sampled_val,
                            mean,
                            std,
                        )
                    new_proc["strength"] = float(max(0.0, sampled_val))
                elif dist_type == "lognormal":
                    # For lognormal, mean/std refer to the underlying normal distribution parameters
                    sampled_val = generator.lognormal(mean=mean, sigma=std)
                    new_proc["strength"] = float(sampled_val)
                elif dist_type == "truncated_normal":
                    if math.isclose(std, 0.0, abs_tol=1e-8):
                        new_proc["strength"] = float(max(0.0, mean))
                    else:
                        # Truncate at 0 (a=0) and +inf (b=inf)
                        a, b = 0.0, np.inf
                        a_norm = (a - mean) / std
                        b_norm = (b - mean) / std
                        sampled_val = truncnorm.rvs(a_norm, b_norm, loc=mean, scale=std, random_state=generator)
                        new_proc["strength"] = float(sampled_val)
                else:
                    # Fallback or error for unknown distributions
                    msg = f"Unsupported distribution type: {dist_type}"
                    raise ValueError(msg)
            else:
                # Assume it's already a float/int
                new_proc["strength"] = float(strength_val)

            new_processes.append(new_proc)

        # Create new instance without re-validation
        new_model = object.__new__(NoiseModel)
        new_model.processes = new_processes
        new_model.scheduled_jumps = copy.deepcopy(self.scheduled_jumps)
        return new_model

    @staticmethod
    def get_operator(name: str | BaseGate) -> BaseGate:
        """Retrieve the operator from GateLibrary, possibly as a tensor product if needed.

        Args:
            name: Name of the noise process (e.g., 'xx', 'zz').

        Returns:
            BaseGate: The matrix representation of the operator.
        """
        
        if isinstance(name, BaseGate):
            return name

        operator_class = getattr(GateLibrary, name)

        return operator_class()
