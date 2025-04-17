# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for quantum circuit generation.

This module provides helper routines used in the YAQS circuit library:
- `extract_u_parameters`: decompose a 2×2 SU(2) unitary matrix into U3 gate parameters (θ, φ, λ).
- `add_random_single_qubit_rotation`: append a random single-qubit U3 rotation to a `QuantumCircuit`, sampling the axis uniformly on the Bloch sphere.
"""

# ignore non-lowercase argument names for physics notation

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit


def extract_u_parameters(
    matrix: NDArray[np.complex128],
) -> tuple[float, float, float]:
    """Extract θ, φ, λ from a 2x2 SU(2) unitary `matrix`.

    This removes any global phase and then solves
    matrix = U3(θ,φ,λ) exactly.

    Args:
        matrix: 2x2 complex array (must be SU(2), det=1 up to phase).

    Returns:
        A tuple (θ, φ, λ) of real gate angles.
    """
    assert matrix.shape == (2, 2), "Input must be a 2x2 matrix."

    # strip global phase
    u: NDArray[np.complex128] = matrix.astype(np.complex128)
    u *= np.exp(-1j * np.angle(u[0, 0]))

    a, b = u[0, 0], u[0, 1]
    c, d = u[1, 0], u[1, 1]

    theta = 2 * np.arccos(np.clip(np.abs(a), -1.0, 1.0))
    sin_th2: float = float(np.sin(theta / 2))
    if np.isclose(sin_th2, 0.0):
        phi = 0.0
        lam = np.angle(d) - np.angle(a)
    else:
        phi = np.angle(c)
        lam = np.angle(-b)

    return float(theta), float(phi), float(lam)


def add_random_single_qubit_rotation(
    qc: QuantumCircuit,
    qubit: int,
    rng: Generator | None = None,
) -> None:
    """Append a random single-qubit rotation exp(-i θ n sigma) as a U3 gate.

    Samples:
      - θ ∈ [0, 2π)
      - axis n uniformly on the Bloch sphere

    Decomposes the resulting 2x2 into U3(θ,φ,λ) and does `qc.u(...)`.

    Args:
        qc: the circuit to modify.
        qubit: which wire to rotate.
        rng: if given, used instead of the global `np.random`.
    """
    sampler = rng if rng is not None else np.random

    # sample angles
    theta = sampler.uniform(0, 2 * np.pi)
    alpha = sampler.uniform(0, np.pi)
    phi = sampler.uniform(0, 2 * np.pi)

    # Bloch-sphere axis
    nx = np.sin(alpha) * np.cos(phi)
    ny = np.sin(alpha) * np.sin(phi)
    nz = np.cos(alpha)

    # Pauli matrices
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.array([[1, 0], [0, -1]])

    h = nx * x + ny * y + nz * z
    u_mat = expm(-1j * theta * h)

    th_u3, ph_u3, lam_u3 = extract_u_parameters(u_mat)
    qc.u(th_u3, ph_u3, lam_u3, qubit)
