# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for circuit utility functions in the YAQS project.

This module contains tests for verifying the correctness of the helper routines
`extract_u_parameters` and `add_random_single_qubit_rotation` from
`circuit_library_utils.py`.

The tests ensure:
- `extract_u_parameters` rejects invalid matrix shapes.
- `extract_u_parameters` returns (0,0,0) for identity and -identity.
- Round‑trip decomposition: building a U3 and extracting its parameters recovers the originals.
- `add_random_single_qubit_rotation` appends exactly one U‐gate with deterministic parameters
  when driven by a seeded `numpy.random.default_rng`.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.circuit import QuantumCircuit

from mqt.yaqs.core.libraries.circuit_library_utils import (
    add_random_single_qubit_rotation,
    extract_u_parameters,
)


def test_extract_u_parameters_invalid_shape() -> None:
    """extract_u_parameters must reject non-2x2 inputs."""
    with pytest.raises(AssertionError):
        extract_u_parameters(np.eye(3))


def test_extract_u_parameters_identity() -> None:
    """extract_u_parameters on I or -I returns (0,0,0)."""
    theta, phi, lam = extract_u_parameters(np.eye(2))
    assert theta == pytest.approx(0.0)
    assert phi == pytest.approx(0.0)
    assert lam == pytest.approx(0.0)

    theta, phi, lam = extract_u_parameters(-np.eye(2))
    assert theta == pytest.approx(0.0)
    assert phi == pytest.approx(0.0)
    assert lam == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("theta0", "phi0", "lam0"),
    [
        (0.1, 0.5, -1.0),
        (np.pi / 2, np.pi / 4, np.pi / 2),
        (1.0, 2.0, 3.0),
    ],
)
def test_extract_u_parameters_roundtrip(theta0: float, phi0: float, lam0: float) -> None:
    """Round-trip U3→matrix→extract_u_parameters recovers (θ,φ,λ)."""
    # build the standard U3(θ,φ,λ) matrix
    u = np.array(
        [
            [np.cos(theta0 / 2), -np.exp(1j * lam0) * np.sin(theta0 / 2)],
            [np.exp(1j * phi0) * np.sin(theta0 / 2), np.exp(1j * (phi0 + lam0)) * np.cos(theta0 / 2)],
        ],
        dtype=complex,
    )

    theta, phi, lam = extract_u_parameters(u)
    assert theta == pytest.approx(theta0, rel=1e-8)
    assert phi == pytest.approx(phi0, rel=1e-8)
    assert lam == pytest.approx(lam0, rel=1e-8)


def test_add_random_single_qubit_rotation_adds_u_gate() -> None:
    """add_random_single_qubit_rotation must append a U-gate with reproducible params."""
    qc = QuantumCircuit(2)
    # seed so we know exactly which gate params are generated
    rng = np.random.default_rng(0)
    add_random_single_qubit_rotation(qc, qubit=1, rng=rng)

    assert qc.num_qubits == 2
    assert len(qc.data) == 1
    instr = qc.data[0].operation
    assert instr.name == "u"
    theta, phi, lam = instr.params
    # these numbers match default_rng(0)+extract_u_parameters
    assert theta == pytest.approx(1.2091241975743714, rel=1e-7)
    assert phi == pytest.approx(-0.6574252905805019, rel=1e-7)
    assert lam == pytest.approx(1.9692788758507522, rel=1e-7)
