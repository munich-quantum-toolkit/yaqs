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
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Excitation x Excitation.
    """

    matrix = np.kron(Excitation.matrix, Excitation.matrix)


class TwoSiteRelaxation:
    """Class representing two-site relaxation noise.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Relaxation x Relaxation.
    """

    matrix = np.kron(Relaxation.matrix, Relaxation.matrix)


class CrossTalkZZ:
    """Class representing cross talk between neighboring sites along the z-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Dephasing x Dephasing.
    """

    matrix = np.kron(Dephasing.matrix, Dephasing.matrix)


class CrossTalkXX:
    """Class representing cross talk between neighboring sites along the x-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product X x X.
    """

    matrix = np.kron(BitFlip.matrix, BitFlip.matrix)


class CrossTalkYY:
    """Class representing cross talk between neighboring sites along the y-axis.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Y x Y.
    """

    matrix = np.kron(BitPhaseFlip.matrix, BitPhaseFlip.matrix)


class CrossTalkXY:
    """Class representing cross talk between neighboring sites with X x Y.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product X x Y.
    """

    matrix = np.kron(BitFlip.matrix, BitPhaseFlip.matrix)


class CrossTalkYX:
    """Class representing cross talk between neighboring sites with Y x X.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Y x X.
    """

    matrix = np.kron(BitPhaseFlip.matrix, BitFlip.matrix)


class CrossTalkZY:
    """Class representing cross talk between neighboring sites with Z x Y.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Z x Y.
    """

    matrix = np.kron(Dephasing.matrix, BitPhaseFlip.matrix)


class CrossTalkZX:
    """Class representing cross talk between neighboring sites with Z x X.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Z x X.
    """

    matrix = np.kron(Dephasing.matrix, BitFlip.matrix)


class CrossTalkYZ:
    """Class representing cross talk between neighboring sites with Y x Z.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product Y x Z.
    """

    matrix = np.kron(BitPhaseFlip.matrix, Dephasing.matrix)


class CrossTalkXZ:
    """Class representing cross talk between neighboring sites with X x Z.

    Attributes:
        matrix (np.ndarray): A 4x4 matrix representing the tensor product X x Z.
    """

    matrix = np.kron(BitFlip.matrix, Dephasing.matrix)


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
        crosstalk_zz: Cross talk between neighboring sites along the z-axis.
        crosstalk_xx: Cross talk between neighboring sites along the x-axis.
        crosstalk_y: Cross talk between neighboring sites along the y-axis.
        crosstalk_xy: Cross talk between neighboring sites with X x Y.
        crosstalk_yx: Cross talk between neighboring sites with Y x X.
        crosstalk_zy: Cross talk between neighboring sites with Z x Y.
        crosstalk_zx: Cross talk between neighboring sites with Z x X.
        crosstalk_yz: Cross talk between neighboring sites with Y x Z.
        crosstalk_xz: Cross talk between neighboring sites with X x Z.
    """

    excitation = Excitation
    relaxation = Relaxation
    dephasing = Dephasing
    bitflip = BitFlip
    bitphaseflip = BitPhaseFlip
    excitation_two = TwoSiteExcitation
    relaxation_two = TwoSiteRelaxation
    crosstalk_zz = CrossTalkZZ
    crosstalk_xx = CrossTalkXX
    crosstalk_yy = CrossTalkYY
    crosstalk_xy = CrossTalkXY
    crosstalk_yx = CrossTalkYX
    crosstalk_zy = CrossTalkZY
    crosstalk_zx = CrossTalkZX
    crosstalk_yz = CrossTalkYZ
    crosstalk_xz = CrossTalkXZ
