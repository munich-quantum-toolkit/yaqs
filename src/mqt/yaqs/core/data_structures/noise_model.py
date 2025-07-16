# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
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

     Attributes.
    ----------
    processes : list of str
        A list of noise processes affecting the system.
        Each process is a dict with:
            - name: process name or identifier
            - sites: indices of sites this process acts on
            - strength: noise strength
            - matrix: matrix representing the operator on those sites

    Methods:
    -------
    __init__:
        Initializes the NoiseModel with given processes.
    get_operator:
        Static method to retrieve the operator matrix for a given noise process name.
    """

    def __init__(self, processes: list[dict[str, Any]] | None = None) -> None:
        """Initializes the NoiseModel.

        Parameters
        ----------
        processes :
            A dict of noise processes affecting the quantum system. Default is None.
        """
        self.processes: list[dict[str, Any]] = []
        if processes is not None:
            for proc in processes:
                assert "name" in proc, "Each process must have a 'name' key"
                assert "sites" in proc, "Each process must have a 'sites' key"
                assert "strength" in proc, "Each process must have a 'strength' key"
                # Try to look up the operator if not explicitly provided
                if "matrix" not in proc:
                    proc["matrix"] = self.get_operator(proc["name"])
                self.processes.append(proc)

    @staticmethod
    def get_operator(name: str) -> NDArray[Any]:
        """Retrieve the operator from NoiseLibrary, possibly as a tensor product if needed.

        Args:
            name: Name of the noise process (e.g., 'xx', 'zz').

        Returns:
            np.ndarray: The matrix representation of the operator.
        """
        operator_class = getattr(NoiseLibrary, name)
        return operator_class().matrix
