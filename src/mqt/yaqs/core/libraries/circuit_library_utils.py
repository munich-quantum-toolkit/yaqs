# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Utility functions for quantum circuit generation.

Decompose single-qubit unitaries up to global phase and append random rotations
with uniformly sampled rotation and axis angles.
"""

# ignore non-lowercase argument names for physics notation

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .. import linalg

if TYPE_CHECKING:
    from numpy.random import Generator
    from numpy.typing import NDArray
    from qiskit.circuit import QuantumCircuit


def extract_u_parameters(
    matrix: NDArray[np.complex128],
) -> tuple[float, float, float]:
    """Extract U3 angles for a single-qubit unitary up to global phase.

    Args:
        matrix: A ``2 x 2`` unitary matrix. Unitarity is assumed, not checked.

    Returns:
        Real angles ``(theta, phi, lambda)`` whose U3 matrix equals ``matrix``
        up to global phase.

    Raises:
        ValueError: If ``matrix`` is not a 2x2 matrix.
    """
    if matrix.shape != (2, 2):
        msg = "Input must be a 2x2 matrix."
        raise ValueError(msg)

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
    """Append ``exp(-i theta n·sigma)`` as a U3 gate, up to global phase.

    Samples ``theta`` and the axis azimuth uniformly in ``[0, 2*pi)``, and the
    axis polar angle uniformly in ``[0, pi)``. The axes are not uniform on the
    Bloch sphere.

    Args:
        qc: Circuit to modify in place.
        qubit: Wire to rotate.
        rng: Sampling generator. Uses global ``np.random`` when omitted.
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
    u_mat = linalg.expm(-1j * theta * h)

    th_u3, ph_u3, lam_u3 = extract_u_parameters(u_mat)
    qc.u(th_u3, ph_u3, lam_u3, qubit)
