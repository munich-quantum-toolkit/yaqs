# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared intervention step parsing and application."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.shared.intervention_steps import (
    apply_intervention_to_backend,
    apply_intervention_to_rho,
    build_intervention_operator,
    compute_born_probability,
    compute_intervention_probability,
)
from mqt.yaqs.characterization.memory.shared.interventions import InterventionMap

_PSI0 = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=np.complex128)
_PSI1 = np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128)
_PLUS = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)


def test_compute_born_probability_identity_state() -> None:
    """Born probability for |0⟩ on |0⟩⟨0| is unity."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    assert compute_born_probability(rho, _PSI0) == pytest.approx(1.0)


def test_compute_born_probability_orthogonal_state() -> None:
    """Born probability for |1⟩ on |0⟩⟨0| is zero."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    assert compute_born_probability(rho, _PSI1) == pytest.approx(0.0)


def test_parse_intervention_step_rejects_unknown_dict_type() -> None:
    """Unsupported structured step types raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported probe step type"):
        build_intervention_operator({"type": "unknown", "U": np.eye(2)})


def test_build_intervention_operator_unitary() -> None:
    """Unitary dict steps return a 2x2 matrix."""
    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    op = build_intervention_operator({"type": "unitary", "U": u})
    assert isinstance(op, np.ndarray)
    op_arr = np.asarray(op, dtype=np.complex128)
    np.testing.assert_allclose(op_arr, u)


def test_build_intervention_operator_cut_measurement() -> None:
    """cut_measurement steps build an intervention map."""
    op = build_intervention_operator(
        {"type": "cut_measurement", "psi_meas": _PSI0, "psi_reset": _PSI1},
    )
    assert isinstance(op, InterventionMap)


def test_build_intervention_operator_cut_preparation() -> None:
    """cut_preparation steps build an identity-effect intervention map."""
    op = build_intervention_operator({"type": "cut_preparation", "psi_prep": _PLUS})
    assert isinstance(op, InterventionMap)


def test_build_intervention_operator_measure_prepare_pair() -> None:
    """Legacy measure/prepare pairs build an intervention map."""
    op = build_intervention_operator((_PSI0, _PSI1))
    assert isinstance(op, InterventionMap)


def test_apply_intervention_to_rho_unitary() -> None:
    """Unitary steps rotate the density matrix."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    out = apply_intervention_to_rho(rho, {"type": "unitary", "U": u})
    np.testing.assert_allclose(out, u @ rho @ u.conj().T, atol=1e-12)


def test_compute_intervention_probability_unitary_is_one() -> None:
    """Unitary and cut_preparation steps have unit branch probability."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    assert compute_intervention_probability(rho, {"type": "unitary", "U": np.eye(2)}) == pytest.approx(1.0)
    assert compute_intervention_probability(rho, {"type": "cut_preparation", "psi_prep": _PSI0}) == pytest.approx(1.0)


def test_compute_intervention_probability_measurement() -> None:
    """Measurement steps return Born probabilities."""
    rho = np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.complex128)
    prob = compute_intervention_probability(rho, (_PSI0, _PSI1))
    assert prob == pytest.approx(0.5)


def test_apply_intervention_to_backend_mcwf_unitary() -> None:
    """Backend unitary steps preserve norm on dense MCWF states."""
    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    state_out, prob = apply_intervention_to_backend(
        _PSI0.copy(),
        {"type": "unitary", "U": u},
        solver="MCWF",
        chain_length=1,
    )
    assert prob == pytest.approx(1.0)
    state_out_arr = np.asarray(state_out, dtype=np.complex128)
    np.testing.assert_allclose(state_out_arr, u @ _PSI0, atol=1e-12)


def test_cut_measurement_without_reset_uses_measured_state() -> None:
    """cut_measurement without psi_reset defaults reset ket to the measurement ket."""
    op = build_intervention_operator({"type": "cut_measurement", "psi_meas": _PLUS})
    assert isinstance(op, InterventionMap)
    np.testing.assert_allclose(op.rho_prep, np.outer(_PLUS, _PLUS.conj()), atol=1e-12)
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    rho_out = apply_intervention_to_rho(rho, {"type": "cut_measurement", "psi_meas": _PLUS})
    np.testing.assert_allclose(rho_out, np.outer(_PLUS, _PLUS.conj()), atol=1e-12)
    prob = compute_intervention_probability(rho, {"type": "cut_measurement", "psi_meas": _PLUS})
    assert prob == pytest.approx(0.5)


def test_intervention_step_types_cover_dict_and_pair_forms() -> None:
    """Dict steps and measure/prepare pairs parse through the production helpers."""
    rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    u = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    dict_op = build_intervention_operator({"type": "unitary", "U": u})
    assert isinstance(dict_op, np.ndarray)
    assert compute_intervention_probability(rho, {"type": "unitary", "U": u}) == pytest.approx(1.0)

    pair_op = build_intervention_operator((_PSI1, _PSI0))
    assert isinstance(pair_op, InterventionMap)
    assert compute_intervention_probability(rho, (_PSI1, _PSI0)) == pytest.approx(0.0)
