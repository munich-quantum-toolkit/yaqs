# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Library of noise processes.

This module defines noise operator classes for quantum systems.
It includes implementations for excitation, relaxation, and dephasing noise operators,
each represented as a 2x2 numpy array. The module also provides a NoiseLibrary class
that aggregates these noise operators for convenient access. Future improvements
may extend these implementations to d-level systems.
"""

from __future__ import annotations

import numpy as np


# TODO(Aaron): Extend to d-levels
class Excitation:
    """Class representing the excitation noise operator for a two-level system.

    Attributes:
        d (int): The dimension of the Hilbert space. Defaults to 2.
        matrix (np.ndarray): A 2x2 matrix representing the excitation operator.
            The matrix is constructed such that matrix[row, col] = 1 if row - col == 1, and 0 otherwise.

    Todo:
        Extend the implementation to d-level systems.
    """

    d = 2
    matrix = np.zeros((d, d))
    for row, array in enumerate(matrix):
        for col, _ in enumerate(array):
            if row - col == 1:
                matrix[row][col] = 1


class Relaxation:
    """Class representing the relaxation noise operator for a two-level system.

    Attributes:
        d (int): The dimension of the Hilbert space. Defaults to 2.
        matrix (np.ndarray): A 2x2 matrix representing the relaxation operator.
            The matrix is constructed such that matrix[row, col] = 1 if col - row == 1, and 0 otherwise.

    Todo:
        Extend the implementation to d-level systems.
    """

    d = 2
    matrix = np.zeros((d, d))
    for row, array in enumerate(matrix):
        for col, _ in enumerate(array):
            if col - row == 1:
                matrix[row][col] = 1


class Dephasing:
    """Class representing the dephasing noise operator for a two-level system.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the dephasing operator,
            defined as [[1, 0], [0, -1]].
    """

    matrix = np.array([[1, 0], [0, -1]])


class NoiseLibrary:
    """A library of noise operator classes.

    Attributes:
        excitation: Class representing the excitation noise operator.
        relaxation: Class representing the relaxation noise operator.
        dephasing: Class representing the dephasing noise operator.
    """

    excitation = Excitation
    relaxation = Relaxation
    dephasing = Dephasing
