# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for stochastic gate-local noise model definitions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mqt.yaqs import XBasisDissipativeNoiseModel, XYZPauliNoiseModel


@pytest.mark.parametrize("model_type", [XYZPauliNoiseModel, XBasisDissipativeNoiseModel])
def test_stochastic_noise_parameter_is_normalized_to_float(model_type: type) -> None:
    """Integer channel parameters are accepted and stored as floats."""
    assert model_type(0).p == pytest.approx(0.0)
    assert isinstance(model_type(0).p, float)


@pytest.mark.parametrize("bad", [True, "0.1", None])
@pytest.mark.parametrize("model_type", [XYZPauliNoiseModel, XBasisDissipativeNoiseModel])
def test_stochastic_noise_parameter_rejects_non_real_values(model_type: type, bad: object) -> None:
    """Channel parameters reject booleans and non-real objects."""
    with pytest.raises(TypeError, match="p must be a real number"):
        model_type(bad)


@pytest.mark.parametrize("bad", [-0.1, math.inf, -math.inf, math.nan])
@pytest.mark.parametrize("model_type", [XYZPauliNoiseModel, XBasisDissipativeNoiseModel])
def test_stochastic_noise_parameter_rejects_nonphysical_values(model_type: type, bad: float) -> None:
    """Channel parameters must be finite and lie in their physical domains."""
    with pytest.raises(ValueError, match="p must"):
        model_type(bad)


@pytest.mark.parametrize("model_type", [XYZPauliNoiseModel, XBasisDissipativeNoiseModel])
def test_stochastic_parameter_rejects_values_above_one(model_type: type) -> None:
    """Both stochastic channel parameters are bounded above by one."""
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model_type(1.0001)


@pytest.mark.parametrize("p", [0.0, 0.2, 1.0])
def test_xyz_pauli_probabilities_match_paper_convention(p: float) -> None:
    """One-qubit I/X/Y/Z probabilities are exact and normalized."""
    probabilities = XYZPauliNoiseModel(p).probabilities
    expected_identity = 1.0 - p
    expected_pauli = p / 3.0
    assert probabilities["I"] == pytest.approx(expected_identity)
    assert probabilities["X"] == pytest.approx(expected_pauli)
    assert probabilities["Y"] == pytest.approx(expected_pauli)
    assert probabilities["Z"] == pytest.approx(expected_pauli)
    assert math.fsum(probabilities.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("p", [0.0, 0.37, 1.0])
def test_x_basis_dissipative_kraus_pair_is_complete(p: float) -> None:
    """The paper Kraus pair satisfies sum K-dagger K equals identity."""
    k0, k1 = XBasisDissipativeNoiseModel(p).kraus_operators()
    completeness = k0.conj().T @ k0 + k1.conj().T @ k1
    np.testing.assert_allclose(completeness, np.eye(2), rtol=1e-13, atol=1e-13)


def test_x_basis_dissipative_general_p_matrices() -> None:
    """The computational-basis Kraus matrices match the X-basis definition."""
    p = 0.3
    s = math.sqrt(1.0 - p)
    expected_k0 = 0.5 * np.asarray([[1.0 + s, 1.0 - s], [1.0 - s, 1.0 + s]])
    expected_k1 = math.sqrt(p) / 2.0 * np.asarray([[1.0, -1.0], [1.0, -1.0]])

    k0, k1 = XBasisDissipativeNoiseModel(p).kraus_operators()

    np.testing.assert_allclose(k0, expected_k0, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(k1, expected_k1, rtol=1e-14, atol=1e-14)


def test_x_basis_dissipative_endpoint_matrices() -> None:
    """At p=0 the channel is identity; at p=1 it resets into the plus state."""
    k0_zero, k1_zero = XBasisDissipativeNoiseModel(0.0).kraus_operators()
    np.testing.assert_allclose(k0_zero, np.eye(2), atol=1e-15)
    np.testing.assert_allclose(k1_zero, np.zeros((2, 2)), atol=1e-15)

    k0_one, k1_one = XBasisDissipativeNoiseModel(1.0).kraus_operators()
    np.testing.assert_allclose(k0_one, np.asarray([[0.5, 0.5], [0.5, 0.5]]), atol=1e-15)
    np.testing.assert_allclose(k1_one, np.asarray([[0.5, -0.5], [0.5, -0.5]]), atol=1e-15)
