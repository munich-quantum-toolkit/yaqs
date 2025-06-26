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

from ..libraries.noise_library import NoiseLibrary


class NoiseModel:
    """A class to represent a noise model in a quantum system.

    Attributes.
    ----------
    processes : list of str
        A list of noise processes affecting the system.
    strengths : list of float
        A list of strengths corresponding to each noise process.
    jump_operators : list
        A list of jump operators corresponding to each noise process.

    Methods:
    -------
    __init__(processes: list[str] | None = None, strengths: list[float] | None = None) -> None
        Initializes the NoiseModel with given processes and strengths.
    """

    def __init__(self, processes: list[str] | None = None, strengths: list[float] | None = None) -> None:
        """Initializes the NoiseModel.

        Parameters
        ----------
        processes : list[str], optional
            A list of noise processes affecting the quantum system. Default is an empty list.
        strengths : list[float], optional
            A list of strengths corresponding to each noise process. Default is an empty list.

        Raises:
        ------
        AssertionError
            If the lengths of 'processes' and 'strengths' lists do not match.
        """
        if strengths is None:
            strengths = []
        if processes is None:
            processes = []
        assert len(processes) == len(strengths)
        self.processes = processes
        self.strengths = strengths
        self.jump_operators = []
        for process in processes:
            self.jump_operators.append(getattr(NoiseLibrary, process)().matrix)
