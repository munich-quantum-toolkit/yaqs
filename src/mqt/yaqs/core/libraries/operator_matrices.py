# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Small operator matrices shared by gate and observable definitions."""

from __future__ import annotations

import numpy as np

IDENTITY = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
PROJECTOR_ZERO = np.array([[1, 0], [0, 0]], dtype=np.complex128)
PROJECTOR_ONE = np.array([[0, 0], [0, 1]], dtype=np.complex128)
CONTROLLED_X = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128)
CONTROLLED_Z = np.diag(np.array([1, 1, 1, -1], dtype=np.complex128))
SWAP_MATRIX = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
