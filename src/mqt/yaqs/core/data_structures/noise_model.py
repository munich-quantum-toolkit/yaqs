# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Noise Models.

This module defines the NoiseModel class, which represents a noise model in a quantum system.
It stores a list of noise processes and their corresponding strengths, and automatically retrieves
the associated jump operator matrices from the NoiseLibrary. These jump operators are used to simulate
the effects of noise in quantum simulations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..libraries.noise_library import NoiseLibrary

if TYPE_CHECKING:
    from numpy.typing import NDArray


class NoiseModel:
    """A class to represent a noise model with arbitrary-site jump operators.

    Each process is a dict with:
        - name (str): process name or identifier
        - sites (list of int): indices of sites this process acts on
        - strength (float): noise strength
        - jump_operator (np.ndarray): matrix representing the operator on those sites
    """

    def __init__(self, processes: list[dict[str, Any]] | None = None) -> None:
        """processes: list of dicts with keys 'name', 'sites', 'strength', and optionally 'jump_operator'."""
        self.processes: list[dict[str, Any]] = []
        if processes is not None:
            for proc in processes:
                assert "name" in proc, "Each process must have a 'name' key"
                assert "sites" in proc, "Each process must have a 'sites' key"
                assert "strength" in proc, "Each process must have a 'strength' key"
                # Try to look up the operator if not explicitly provided
                if "jump_operator" not in proc:
                    proc["jump_operator"] = self.get_operator(proc["name"])
                self.processes.append(proc)

    @staticmethod
    def get_operator(name: str) -> NDArray[Any]:
        """Retrieve the operator from NoiseLibrary, possibly as a tensor product if needed.

        Args:
            name (str): Name of the noise process (e.g., 'xx', 'zz').
            num_sites (int): Number of sites this operator acts on.

        Returns:
            np.ndarray: The matrix representation of the operator.
        """
        operator_class = getattr(NoiseLibrary, name)
        return operator_class().matrix
