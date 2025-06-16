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
import numpy as np

from ..libraries.noise_library import NoiseLibrary


# class NoiseModel:
#     """A class to represent a noise model in a quantum system.

#     Attributes.
#     ----------
#     processes : list of str
#         A list of noise processes affecting the system.
#     strengths : list of float
#         A list of strengths corresponding to each noise process.
#     jump_operators : list
#         A list of jump operators corresponding to each noise process.

#     Methods:
#     -------
#     __init__(processes: list[str] | None = None, strengths: list[float] | None = None) -> None
#         Initializes the NoiseModel with given processes and strengths.
#     """

#     def __init__(self, processes: list[str] | None = None, strengths: list[float] | None = None) -> None:
#         """Initializes the NoiseModel.

#         Parameters
#         ----------
#         processes : list[str], optional
#             A list of noise processes affecting the quantum system. Default is an empty list.
#         strengths : list[float], optional
#             A list of strengths corresponding to each noise process. Default is an empty list.

#         Raises:
#         ------
#         AssertionError
#             If the lengths of 'processes' and 'strengths' lists do not match.
#         """
#         if strengths is None:
#             strengths = []
#         if processes is None:
#             processes = []
#         assert len(processes) == len(strengths)
#         self.processes = processes
#         self.strengths = strengths
#         self.jump_operators = []
#         # for process in processes:
#         #     self.jump_operators.append(getattr(NoiseLibrary, process)().matrix)

#         for site_processes in processes:
#             site_ops = []
#             for process in site_processes:
#                 new_op = getattr(NoiseLibrary, process)().matrix
#                 if not any(np.allclose(op, new_op) for op in site_ops):
#                     site_ops.append(new_op)
#             self.jump_operators.append(site_ops)



class NoiseModel:
    """
    A class to represent a noise model with arbitrary-site jump operators.

    Each process is a dict with:
        - name (str): process name or identifier
        - sites (list of int): indices of sites this process acts on
        - strength (float): noise strength
        - jump_operator (np.ndarray): matrix representing the operator on those sites
    """

    def __init__(self, processes: list[dict] = None) -> None:
        """
        processes: list of dicts with keys 'name', 'sites', 'strength', and optionally 'jump_operator'.
        """
        self.processes = []
        if processes is not None:
            for proc in processes:
                assert "name" in proc, "Each process must have a 'name' key"
                assert "sites" in proc, "Each process must have a 'sites' key"
                assert "strength" in proc, "Each process must have a 'strength' key"
                # Try to look up the operator if not explicitly provided
                if 'jump_operator' not in proc:
                    proc['jump_operator'] = self.get_operator(proc['name'], len(proc['sites']))
                self.processes.append(proc)

    @staticmethod
    def get_operator(name: str, num_sites: int):
        """Retrieve the operator from NoiseLibrary, possibly as a tensor product if needed."""
        # Example: for two-site process 'xx', call NoiseLibrary.xx().matrix, etc.
        # This logic should match your NoiseLibrary's API.
        # For a generic approach:
        operator_class = getattr(NoiseLibrary, name)
        op = operator_class().matrix
        # Optionally check op.shape matches 2^num_sites x 2^num_sites
        return op
