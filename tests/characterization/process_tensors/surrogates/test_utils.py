from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.process_tensors.surrogates.utils import (
    _initial_mcwf_state_from_rho0,
    _random_density_matrix,
    _sample_random_intervention_sequence,
)


def test_random_density_matrix_is_physical() -> None:
    rng = np.random.default_rng(0)
    rho = _random_density_matrix(rng)
    np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)
    np.testing.assert_allclose(np.trace(rho).real, 1.0, atol=1e-12)
    evals = np.linalg.eigvalsh(rho).real
    assert float(evals.min()) >= -1e-12


def test_sample_random_intervention_sequence_shapes() -> None:
    rng = np.random.default_rng(1)
    maps, rows = _sample_random_intervention_sequence(3, rng)
    assert len(maps) == 3
    assert rows.shape == (3, 32)
    assert rows.dtype == np.float32


def test_initial_mcwf_state_from_rho0_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        _initial_mcwf_state_from_rho0(np.zeros((3, 3)), length=1)


def test_initial_mcwf_state_from_rho0_invalid_mode_raises() -> None:
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    with pytest.raises(ValueError):
        _initial_mcwf_state_from_rho0(rho, length=1, init_mode="bad")  # type: ignore[arg-type]


def test_initial_mcwf_state_from_rho0_eigenstate_return_eig_sample() -> None:
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
    assert idx in (0, 1)
    assert 0.0 <= p <= 1.0
    np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-12)


def test_initial_mcwf_state_from_rho0_purified_length1_returns_state() -> None:
    rho = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
    psi = _initial_mcwf_state_from_rho0(rho, length=1, init_mode="purified")
    assert psi.shape == (2,)
    np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-12)


def test_initial_mcwf_state_from_rho0_branches_length_gt_1() -> None:
    rng = np.random.default_rng(0)
    rho = np.array([[0.25, 0.0], [0.0, 0.75]], dtype=np.complex128)

    psi_eig = _initial_mcwf_state_from_rho0(rho, length=3, rng=rng, init_mode="eigenstate")
    assert psi_eig.shape == (2**3,)
    np.testing.assert_allclose(np.linalg.norm(psi_eig), 1.0, atol=1e-12)

    psi_pur = _initial_mcwf_state_from_rho0(rho, length=3, init_mode="purified")
    assert psi_pur.shape == (2**3,)
    np.testing.assert_allclose(np.linalg.norm(psi_pur), 1.0, atol=1e-12)
