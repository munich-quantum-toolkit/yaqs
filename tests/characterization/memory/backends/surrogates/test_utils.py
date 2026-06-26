# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: PLC2701 -- white-box tests import private surrogate helpers

"""Tests for surrogate data-generation utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.surrogates.utils import (
    _initial_mcwf_state_from_rho0,
    sample_density_matrix,
    sample_intervention_sequence,
)


def test_sample_density_matrix_is_physical() -> None:
    """Random density matrices are Hermitian, trace-one, and PSD."""
    rng = np.random.default_rng(0)
    rho = sample_density_matrix(rng)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho).real, 1.0, atol=1e-12)
    evals = np.linalg.eigvalsh(rho).real
    assert float(evals.min()) >= -1e-12


def test_sample_intervention_sequence_shapes() -> None:
    """Intervention sequences return k maps and k float32 feature rows."""
    rng = np.random.default_rng(1)
    maps, rows = sample_intervention_sequence(3, rng)
    assert len(maps) == 3
    assert rows.shape == (3, 32)
    assert rows.dtype == np.float32


def test_initial_mcwf_state_from_rho0_invalid_shape_raises() -> None:
    """Non-2x2 density matrices are rejected."""
    with pytest.raises(ValueError, match="rho must be a 2x2"):
        _initial_mcwf_state_from_rho0(np.zeros((3, 3)), length=1)


def test_initial_mcwf_state_from_rho0_invalid_mode_raises() -> None:
    """Unknown init_mode values are rejected."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="init_mode must be"):
        _initial_mcwf_state_from_rho0(rho, length=1, init_mode="bad")  # type: ignore[arg-type]


def test_initial_mcwf_state_from_rho0_eigenstate_return_eig_sample() -> None:
    """Eigenstate mode can return the sampled eigenvector index and probability."""
    rng = np.random.default_rng(0)
    rho = np.array([[0.25, 0.0], [0.0, 0.75]], dtype=np.complex128)
    psi, idx, p = _initial_mcwf_state_from_rho0(
        rho,
        length=1,
        rng=rng,
        init_mode="eigenstate",
        return_eig_sample=True,
    )
    assert psi.shape == (2,)
    assert idx in {0, 1}
    assert 0.0 <= p <= 1.0
    np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-12)


def test_initial_mcwf_state_from_rho0_purified_length1_requires_pure_state() -> None:
    """Purified mode rejects mixed single-qubit inputs."""
    rho = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
    with pytest.raises(ValueError, match="purified init_mode requires a pure"):
        _initial_mcwf_state_from_rho0(rho, length=1, init_mode="purified")
    rho_pure = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    psi_out = _initial_mcwf_state_from_rho0(rho_pure, length=1, init_mode="purified")
    assert isinstance(psi_out, np.ndarray)
    psi = psi_out
    assert psi.shape == (2,)
    np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-12)


def test_initial_mcwf_state_from_rho0_branches_length_gt_1() -> None:
    """Multi-site chains are supported for both eigenstate and purified modes."""
    rng = np.random.default_rng(0)
    rho = np.array([[0.25, 0.0], [0.0, 0.75]], dtype=np.complex128)

    psi_eig_out = _initial_mcwf_state_from_rho0(rho, length=3, rng=rng, init_mode="eigenstate")
    assert isinstance(psi_eig_out, np.ndarray)
    psi_eig = psi_eig_out
    assert psi_eig.shape == (2**3,)
    np.testing.assert_allclose(np.linalg.norm(psi_eig), 1.0, atol=1e-12)

    psi_pur_out = _initial_mcwf_state_from_rho0(rho, length=3, init_mode="purified")
    assert isinstance(psi_pur_out, np.ndarray)
    psi_pur = psi_pur_out
    assert psi_pur.shape == (2**3,)
    np.testing.assert_allclose(np.linalg.norm(psi_pur), 1.0, atol=1e-12)

    psi_tensor = np.reshape(psi_pur, (2, 2, 2))
    rho_reduced = np.einsum("abc,dbc->ad", psi_tensor, psi_tensor.conj())
    np.testing.assert_allclose(rho_reduced, rho, atol=1e-10)
