# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
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


class Excitation:
    """Class representing excitation noise.

    Attributes:
        d: The dimension of the Hilbert space. Defaults to 2.
        matrix: A 2x2 matrix representing the excitation operator.
            The matrix is constructed such that matrix[row, col] = 1 if row - col == 1, and 0 otherwise.
    """

    d = 2
    matrix = np.zeros((d, d))
    for row, array in enumerate(matrix):
        for col, _ in enumerate(array):
            if row - col == 1:
                matrix[row][col] = 1


class Relaxation:
    """Class representing relaxation noise.

    Attributes:
        d (int): The dimension of the Hilbert space. Defaults to 2.
        matrix (np.ndarray): A 2x2 matrix representing the relaxation operator.
            The matrix is constructed such that matrix[row, col] = 1 if col - row == 1, and 0 otherwise.
    """

    d = 2
    matrix = np.zeros((d, d))
    for row, array in enumerate(matrix):
        for col, _ in enumerate(array):
            if col - row == 1:
                matrix[row][col] = 1


class Dephasing:
    """Class representing dephasing noise.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the dephasing operator,
            defined as [[1, 0], [0, -1]].
    """

    matrix = np.array([[1, 0], [0, -1]])


class BitFlip:
    """Class representing bitflip (Pauli-X) noise.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the Pauli-X operator,
            defined as [[0, 1], [1, 0]].
    """

    matrix = np.array([[0, 1], [1, 0]])


class BitPhaseFlip:
    """Class representing bit-phase flip (Pauli-Y) noise.

    Attributes:
        matrix (np.ndarray): A 2x2 matrix representing the Pauli-Y operator,
            defined as [[0, -1j], [1j, 0]].
    """

    matrix = np.array([[0, -1j], [1j, 0]])


class TwoSiteExcitation:
    """Class representing two-site excitation noise.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Excitation ⊗ Excitation.
    """

    matrix = np.kron(Excitation.matrix, Excitation.matrix)


class TwoSiteRelaxation:
    """Class representing two-site relaxation noise.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Relaxation ⊗ Relaxation.
    """

    matrix = np.kron(Relaxation.matrix, Relaxation.matrix)


class CrossTalk:
    """Class representing cross talk between neighboring sites along the z-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Dephasing ⊗ Dephasing.
    """

    matrix = np.kron(Dephasing.matrix, Dephasing.matrix)


class CrossTalkX:
    """Class representing cross talk between neighboring sites along the x-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product X ⊗ X.
    """

    matrix = np.kron(BitFlip.matrix, BitFlip.matrix)


class CrossTalkY:
    """Class representing cross talk between neighboring sites along the y-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Y ⊗ Y.
    """

    matrix = np.kron(BitPhaseFlip.matrix, BitPhaseFlip.matrix)


class NoiseLibrary:
    """A library of noise processes.

    Attributes:
        excitation: Excitation noise (0 --> 1).
        relaxation: Relaxation noise (1 --> 0).
        dephasing: Dephasing noise.
        bitflip: Bitflip (Pauli-X) noise (0 --> 1, 1 --> 0).
        bitphaseflip: Bit-phase flip (Pauli-Y) noise.
        excitation_two: Two-site excitation noise (00 --> 11).
        relaxation_two: Two-site relaxation noise (11 --> 00).
        crosstalk: Cross talk between neighboring sites along the z-axis.
        crosstalk_x: Cross talk between neighboring sites along the x-axis.
        crosstalk_y: Cross talk between neighboring sites along the y-axis.
    """

    excitation = Excitation
    relaxation = Relaxation
    dephasing = Dephasing
    bitflip = BitFlip
    bitphaseflip = BitPhaseFlip
    excitation_two = TwoSiteExcitation
    relaxation_two = TwoSiteRelaxation
    crosstalk = CrossTalk
    crosstalk_x = CrossTalkX
    crosstalk_y = CrossTalkY
