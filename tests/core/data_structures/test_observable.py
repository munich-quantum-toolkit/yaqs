# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for user-facing observable definitions."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.observable import Observable
from mqt.yaqs.core.libraries.gate_library import BaseGate


def test_named_observable_stores_operator_metadata() -> None:
    """A named observable stores operator metadata without a gate object."""
    observable = Observable("x", 0)

    np.testing.assert_array_equal(observable.matrix, np.array([[0, 1], [1, 0]]))
    assert observable.name == "x"
    assert observable.kind == "operator"
    assert observable.interaction == 1
    assert observable.sites == 0
    assert not hasattr(observable, "gate")


def test_custom_local_matrix_supports_non_qubit_dimension() -> None:
    """An integer site makes a custom matrix a one-site operator."""
    matrix = np.diag(np.array([-1.0, 0.25, 2.0]))

    observable = Observable(matrix, 0)

    assert observable.name == "local"
    assert observable.interaction == 1
    assert observable.matrix is not None
    np.testing.assert_allclose(observable.matrix, matrix)


def test_custom_matrix_site_list_sets_interaction() -> None:
    """A site list sets the support size for a custom qubit operator."""
    observable = Observable(np.eye(4), [0, 1])

    assert observable.interaction == 2
    assert observable.sites == [0, 1]


def test_position_observable_uses_supplied_basis() -> None:
    """Position observables are diagonal in the supplied local basis."""
    positions = np.array([-1.5, 0.0, 2.5])

    observable = Observable("position", 1, positions=positions)

    assert observable.name == "position"
    assert observable.matrix is not None
    np.testing.assert_allclose(observable.matrix, np.diag(positions))


def test_position_observable_requires_positions() -> None:
    """Position observables require their basis values."""
    with pytest.raises(TypeError, match="required keyword-only argument: 'positions'"):
        Observable("position", 0)


@pytest.mark.parametrize(
    ("name", "kwargs", "match"),
    [
        ("position", {"position_values": [0.0, 1.0]}, "unexpected keyword argument 'position_values'"),
        ("z", {"positions": [0.0, 1.0]}, "unexpected keyword argument 'positions'"),
    ],
)
def test_named_observable_rejects_unexpected_parameters(
    name: str,
    kwargs: dict[str, object],
    match: str,
) -> None:
    """Named factories reject misspelled or inapplicable parameters."""
    with pytest.raises(TypeError, match=match):
        Observable(name, 0, **kwargs)


def test_matrix_observable_rejects_named_parameters() -> None:
    """Matrix observables do not accept named-factory parameters."""
    with pytest.raises(TypeError, match="only supported for named observables"):
        Observable(np.eye(2), 0, positions=[0.0, 1.0])


def test_bitstring_observable_rejects_named_parameters() -> None:
    """Bitstring requests do not accept named-observable parameters."""
    with pytest.raises(TypeError, match="do not accept operator parameters"):
        Observable("01", positions=[0.0, 1.0])


@pytest.mark.parametrize(
    ("operator", "sites", "exception", "match"),
    [
        ("z", None, ValueError, "sites are required for named observables"),
        (np.eye(2), None, ValueError, "sites are required for matrix observables"),
        ("z", "0", TypeError, "sites must be an int or a list of ints"),
        ("z", True, TypeError, "sites must be an int or a list of ints"),
        ("z", [0, False], TypeError, "sites must be an int or a list of ints"),
        ("z", [], ValueError, "sites must not be empty"),
        ("zz", 0, ValueError, "acts on 2 site"),
        (np.eye(2), [0, 1], ValueError, "must have shape"),
    ],
)
def test_observable_rejects_invalid_site_definitions(
    operator: str | np.ndarray,
    sites: object,
    exception: type[Exception],
    match: str,
) -> None:
    """Observable construction rejects absent, malformed, or incompatible sites."""
    with pytest.raises(exception, match=match):
        Observable(operator, sites)  # ty: ignore[invalid-argument-type]  # exercise runtime validation


@pytest.mark.parametrize("name", ["pvm", "unknown"])
def test_observable_rejects_unknown_names(name: str) -> None:
    """Unknown names do not fall back to projectors."""
    with pytest.raises(ValueError, match=f"Unknown observable {name!r}"):
        Observable(name)


def test_observable_rejects_gate_instance() -> None:
    """Observable definitions do not accept gate objects."""
    with pytest.raises(TypeError, match="named observable or a numeric matrix"):
        Observable(BaseGate(np.eye(2)), 0)  # ty: ignore[invalid-argument-type]  # exercise runtime validation


@pytest.mark.parametrize("name", ["s", "t", "rx", "destroy"])
def test_observable_rejects_gate_only_names(name: str) -> None:
    """Gate-only names are not observable names."""
    with pytest.raises(ValueError, match="gate name, not a named observable"):
        Observable(name, 0)


def test_observable_rejects_non_hermitian_matrix() -> None:
    """Custom observable matrices must be Hermitian."""
    with pytest.raises(ValueError, match="must be Hermitian"):
        Observable(np.array([[0, 1], [0, 0]], dtype=np.complex128), 0)


def test_observable_copies_custom_matrix() -> None:
    """Changing caller-owned matrix data does not change an observable."""
    matrix = np.diag([1.0, -1.0])
    observable = Observable(matrix, 0)

    matrix[0, 0] = 5.0

    assert observable.matrix is not None
    assert observable.matrix[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("positions", [np.array([]), np.array([0.0, np.nan]), np.array([0.0, 1.0j])])
def test_position_observable_rejects_invalid_positions(positions: np.ndarray) -> None:
    """Position bases must be non-empty, finite, and real."""
    with pytest.raises(ValueError, match="positions must"):
        Observable("position", 0, positions=positions)


@pytest.mark.parametrize("matrix", [np.ones(3), np.ones((2, 3))])
def test_observable_rejects_invalid_matrix(matrix: np.ndarray) -> None:
    """Matrix observables must be two-dimensional and square."""
    with pytest.raises(ValueError, match="Observable matrix"):
        Observable(matrix, 0)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_observable_rejects_nonfinite_matrix(value: float) -> None:
    """Custom observable matrices must contain finite values."""
    with pytest.raises(ValueError, match="must contain only finite values"):
        Observable(np.diag([value, 1.0]), 0)


def test_diagnostics_have_no_operator_placeholder() -> None:
    """Entropy and Schmidt spectra are state diagnostics, not operators."""
    cut = [3, 4]

    for name in ("entropy", "schmidt_spectrum"):
        observable = Observable(name, cut)
        assert observable.name == name
        assert observable.kind == "diagnostic"
        assert observable.matrix is None
        assert observable.sites == cut


def test_binary_string_builds_bitstring_request() -> None:
    """A binary string requests its computational-basis probability."""
    observable = Observable("10101")

    assert observable.name == "pvm"
    assert observable.kind == "bitstring"
    assert observable.bitstring == "10101"
    assert observable.matrix is None
    assert observable.sites is None


def test_bitstring_observable_rejects_sites() -> None:
    """A bitstring defines its full support and does not accept sites."""
    with pytest.raises(TypeError, match="do not accept sites"):
        Observable("10101", sites=0)
