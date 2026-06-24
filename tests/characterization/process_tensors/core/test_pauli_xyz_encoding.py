# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Sanity checks for Pauli (x, y, z) Bloch features used in split-cut :math:`V`."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.process_tensors.core.encoding import rho_to_xyz


@pytest.mark.parametrize(
    ("psi", "expected_xyz"),
    [
        (np.array([1.0, 0.0], dtype=np.complex128), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 1.0], dtype=np.complex128), np.array([0.0, 0.0, -1.0])),
        (np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2), np.array([1.0, 0.0, 0.0])),
        (np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2), np.array([0.0, 1.0, 0.0])),
    ],
)
def test_rho_to_xyz_standard_bloch_states(psi: np.ndarray, expected_xyz: np.ndarray) -> None:
    """Standard single-qubit states map to the expected Bloch coordinates."""
    rho = np.outer(psi, psi.conj())
    tr = float(np.trace(rho).real)
    if tr > 1e-15:
        rho /= tr
    xyz = rho_to_xyz(rho)
    np.testing.assert_allclose(xyz, expected_xyz, atol=1e-10, rtol=0.0)
