# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: PLC2701 -- white-box tests import private encoding helpers

"""Tests for process-tensor Choi and density-matrix encoding helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.shared.encoding import (
    _flatten_choi4,
    decode_packed_pauli_batch,
    decode_pauli_rho,
    encode_rho_pauli,
    normalize_backend_rho,
    pack_rho8,
    stack_choi_features,
    unpack_rho8,
)


def test_flatten_choi4_shape_and_dtype() -> None:
    """Flattened Choi features are float32 vectors of length 32."""
    j = np.eye(4, dtype=np.complex128)
    y = _flatten_choi4(j)
    assert y.shape == (32,)
    assert y.dtype == np.float32


def test_stack_choi_features_shape() -> None:
    """Feature table stacks one row per input Choi matrix."""
    mats = [np.eye(4, dtype=np.complex128) for _ in range(16)]
    table = stack_choi_features(mats)
    assert table.shape == (16, 32)
    assert table.dtype == np.float32


def test_pack_unpack_roundtrip_hermitianized() -> None:
    """pack_rho8 and unpack_rho8 preserve a physical single-qubit state."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    y = pack_rho8(rho)
    assert y.shape == (8,)
    rho2 = unpack_rho8(y)
    np.testing.assert_allclose(rho2, rho2.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho2).real, 1.0, atol=1e-12)


def test_normalize_backend_rho_returns_physical_dm() -> None:
    """Backend output normalization yields Hermitian, trace-one, PSD density matrix."""
    rho_raw = np.array([[2.0 + 0.0j, 1.0 + 2.0j], [0.0 + 0.0j, 0.1 + 0.0j]], dtype=np.complex128)
    rho = normalize_backend_rho(rho_raw)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho).real, 1.0, atol=1e-12)
    evals = np.linalg.eigvalsh(rho).real
    assert float(evals.min()) >= -1e-12


def test_pauli_tomography_roundtrip_and_identity_component() -> None:
    """Four-component Pauli tomography reconstructs rho; I expectation is unity."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    pauli = encode_rho_pauli(rho)
    assert pauli[0] == pytest.approx(1.0)
    recon = decode_pauli_rho(pauli)
    np.testing.assert_allclose(recon, rho, atol=1e-10)


@pytest.mark.parametrize(
    ("psi", "expected_xyz"),
    [
        (np.array([1.0, 0.0], dtype=np.complex128), np.array([0.0, 0.0, 1.0])),
        (np.array([0.0, 1.0], dtype=np.complex128), np.array([0.0, 0.0, -1.0])),
        (np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2), np.array([1.0, 0.0, 0.0])),
        (np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2), np.array([0.0, 1.0, 0.0])),
    ],
)
def test_rho_to_pauli_xyz_standard_bloch_states(psi: np.ndarray, expected_xyz: np.ndarray) -> None:
    """Standard single-qubit states map to the expected X,Y,Z Pauli expectations."""
    rho = np.outer(psi, psi.conj())
    tr = float(np.trace(rho).real)
    if tr > 1e-15:
        rho /= tr
    xyz = encode_rho_pauli(rho)[1:4]
    np.testing.assert_allclose(xyz, expected_xyz, atol=1e-10, rtol=0.0)


def test_normalize_backend_rho_zero_trace_returns_maximally_mixed() -> None:
    """Near-zero trace inputs fall back to a valid I/2 density matrix."""
    rho = normalize_backend_rho(np.zeros((2, 2), dtype=np.complex128))
    np.testing.assert_allclose(rho, 0.5 * np.eye(2, dtype=np.complex128), atol=1e-12)
    np.testing.assert_allclose(np.trace(rho).real, 1.0, atol=1e-12)


def test_packed_rho8_pauli_batch_shape_and_identity() -> None:
    """rho8 batch maps to (I,X,Y,Z) with I≈1 for normalized states."""
    packed = np.random.default_rng(1).standard_normal((3, 8)).astype(np.float32)
    full = decode_packed_pauli_batch(packed)
    assert full.shape == (3, 4)
    assert full[..., 0] == pytest.approx(1.0, abs=0.05)
