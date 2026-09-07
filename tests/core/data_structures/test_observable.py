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

from mqt.yaqs.core.data_structures.mpo import MPO
from mqt.yaqs.core.data_structures.observable import Observable, prepare_observables
from mqt.yaqs.core.libraries.gate_library import BaseGate


def _embed_dense_on_support(
    operator: np.ndarray,
    sites: list[int],
    physical_dimensions: list[int],
) -> np.ndarray:
    """Embed an operator by explicit computational-basis enumeration.

    Args:
        operator: Matrix in the tensor-factor order given by ``sites``.
        sites: Full-chain target sites.
        physical_dimensions: Full-chain local dimensions in site order.

    Returns:
        Dense matrix on the smallest contiguous interval containing ``sites``.
    """
    first_site = min(sites)
    last_site = max(sites)
    span_dimensions = physical_dimensions[first_site : last_site + 1]
    active_dimensions = [physical_dimensions[site] for site in sites]
    dimension = int(np.prod(span_dimensions))
    embedded = np.zeros((dimension, dimension), dtype=np.complex128)
    site_offsets = [site - first_site for site in sites]
    spectator_offsets = [offset for offset in range(len(span_dimensions)) if offset not in site_offsets]
    for row in range(dimension):
        row_digits = np.unravel_index(row, span_dimensions)
        active_row = np.ravel_multi_index(tuple(row_digits[offset] for offset in site_offsets), active_dimensions)
        for column in range(dimension):
            column_digits = np.unravel_index(column, span_dimensions)
            if any(row_digits[offset] != column_digits[offset] for offset in spectator_offsets):
                continue
            active_column = np.ravel_multi_index(
                tuple(column_digits[offset] for offset in site_offsets), active_dimensions
            )
            embedded[row, column] = operator[active_row, active_column]
    return embedded


def test_named_observable_stores_operator_metadata() -> None:
    """A named observable stores operator metadata without a gate object."""
    observable = Observable("x", 0)

    np.testing.assert_array_equal(observable.matrix, np.array([[0, 1], [1, 0]]))
    assert observable.name == "x"
    assert observable.type == "operator"
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


@pytest.mark.parametrize("name", ["pvm", "unknown", "i", "iden", "h", "cx", "cz", "swap"])
def test_observable_rejects_unknown_names(name: str) -> None:
    """Observable rejects each unsupported string name."""
    with pytest.raises(ValueError, match=f"Unknown observable {name!r}"):
        Observable(name)


def test_observable_rejects_gate_instance() -> None:
    """Observable definitions do not accept gate objects."""
    with pytest.raises(TypeError, match="named observable or a numeric matrix"):
        Observable(BaseGate(np.eye(2)), 0)  # ty: ignore[invalid-argument-type]  # exercise runtime validation


def test_observable_rejects_non_hermitian_matrix() -> None:
    """Custom observable matrices must be Hermitian."""
    with pytest.raises(ValueError, match="must be Hermitian"):
        Observable(np.array([[0, 1], [0, 0]], dtype=np.complex128), 0)


def test_matrix_hermiticity_uses_complete_frobenius_residual() -> None:
    """Several small matrix residuals cannot exceed the global tolerance."""
    within_tolerance = 0.2e-12j * np.eye(4)
    above_tolerance = 0.375e-12j * np.eye(4)

    Observable(within_tolerance, 0)
    with pytest.raises(ValueError, match="must be Hermitian"):
        Observable(above_tolerance, 0)


def test_observable_copies_custom_matrix() -> None:
    """Changing caller-owned matrix data does not change an observable."""
    matrix = np.diag([1.0, -1.0])
    observable = Observable(matrix, 0)

    matrix[0, 0] = 5.0

    assert observable.matrix is not None
    assert observable.matrix[0, 0] == pytest.approx(1.0)


def test_observable_copies_site_lists() -> None:
    """Changing a caller-owned site list does not change an observable."""
    sites = [0, 1]
    observable = Observable("zz", sites)

    sites[1] = 3

    assert observable.sites == [0, 1]


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
        assert observable.type == "diagnostic"
        assert observable.matrix is None
        assert observable.sites == cut


def test_binary_string_builds_bitstring_request() -> None:
    """A binary string requests its computational-basis probability."""
    observable = Observable("10101")

    assert observable.name == "pvm"
    assert observable.type == "bitstring"
    assert observable.bitstring == "10101"
    assert observable.matrix is None
    assert observable.sites is None


def test_bitstring_observable_rejects_sites() -> None:
    """A bitstring defines its full support and does not accept sites."""
    with pytest.raises(TypeError, match="do not accept sites"):
        Observable("10101", sites=0)


def test_prepare_builds_compact_nonadjacent_mpo() -> None:
    """A local matrix produces an MPO only on its contiguous support interval."""
    rng = np.random.default_rng(12)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    matrix = np.asarray(raw + raw.conj().T, dtype=np.complex128)
    dimensions = [2, 3, 2, 4]

    prepared = Observable(matrix, [0, 2]).prepare(4, dimensions)

    assert prepared.mpo is not None
    assert prepared.mpo_sites == (0, 1, 2)
    assert prepared.mpo.length == 3
    expected = _embed_dense_on_support(matrix, [0, 2], dimensions)
    np.testing.assert_allclose(prepared.mpo.to_matrix(), expected, atol=1e-12)


def test_prepare_keeps_long_range_pauli_product_bond_one() -> None:
    """Named Pauli products carry a bond-one identity channel across gaps."""
    prepared = Observable("zz", [0, 5]).prepare(6)

    assert prepared.mpo is not None
    assert prepared.mpo_sites == tuple(range(6))
    assert all(tensor.shape[2:] == (1, 1) for tensor in prepared.mpo.tensors)
    z_matrix = np.diag([1.0, -1.0])
    expected = np.kron(z_matrix, np.kron(np.eye(16), z_matrix))
    np.testing.assert_allclose(prepared.mpo.to_matrix(), expected, atol=1e-12)


def test_prepare_permuted_reversed_sites_with_mixed_dimensions() -> None:
    """Matrix factors follow the user site list and MPO tensors follow chain order."""
    rng = np.random.default_rng(13)
    raw = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    matrix = np.asarray(raw + raw.conj().T, dtype=np.complex128)
    dimensions = [3, 4, 2]

    observable = Observable(matrix, [2, 0])
    prepared = observable.prepare(3, dimensions)

    assert prepared.mpo is not None
    assert prepared.mpo_sites == (0, 1, 2)
    expected = _embed_dense_on_support(matrix, [2, 0], dimensions)
    np.testing.assert_allclose(prepared.mpo.to_matrix(), expected, atol=1e-12)
    assert observable.mpo is None
    assert observable.sites == [2, 0]


def test_prepare_rejects_invalid_support_and_dimensions() -> None:
    """Preparation validates duplicate sites, bounds, and matrix dimensions."""
    with pytest.raises(ValueError, match="must be distinct"):
        Observable(np.eye(4), [0, 0]).prepare(2)
    with pytest.raises(ValueError, match="outside the state"):
        Observable("z", 2).prepare(2)
    with pytest.raises(ValueError, match="does not match site dimensions 3 and 2"):
        Observable(np.eye(4), [0, 1]).prepare(2, [3, 2])
    with pytest.raises(ValueError, match="does not match site dimensions 2 and 2"):
        Observable(np.eye(2), [0, 1]).prepare(2)


def test_prepare_rejects_wrong_bitstring_length() -> None:
    """Bitstring length is checked against the state before execution."""
    with pytest.raises(ValueError, match="does not match state length"):
        Observable("01").prepare(3)


def test_observable_accepts_and_copies_full_chain_mpo() -> None:
    """A supplied MPO is copied and checked against the prepared state layout."""
    matrix = np.kron(np.diag([1.0, 0.0, -2.0]), np.array([[0.0, 1.0], [1.0, 0.0]]))
    source = MPO.from_matrix_with_dimensions(matrix, [3, 2])

    observable = Observable(source)
    source.tensors[0].fill(0)
    prepared = observable.prepare(2, [3, 2])

    assert prepared.mpo is not None
    assert prepared.mpo_sites == (0, 1)
    np.testing.assert_allclose(prepared.mpo.to_matrix(), matrix, atol=1e-12)


def test_full_chain_mpo_rejects_sites_and_state_mismatch() -> None:
    """Full-chain MPOs define their sites and physical dimensions themselves."""
    mpo = MPO.identity(2)
    with pytest.raises(TypeError, match="do not accept sites"):
        Observable(mpo, sites=[0, 1])

    observable = Observable(mpo)
    with pytest.raises(ValueError, match="length 2 does not match state length 3"):
        observable.prepare(3)
    with pytest.raises(ValueError, match="do not match state dimensions"):
        observable.prepare(2, [2, 3])


def test_observable_rejects_non_hermitian_mpo() -> None:
    """Complete-MPO Hermiticity is required at construction."""
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    with pytest.raises(ValueError, match="MPO must be Hermitian"):
        Observable(MPO.from_local_ops([lowering]))


def test_pauli_sum_is_exact_and_requires_hermiticity() -> None:
    """Pauli sums retain weak terms and reject complex non-Hermitian coefficients."""
    observable = Observable.from_pauli_sum(
        terms=[(1.0, "Z0 Z2"), (1e-18, "X1")],
        length=3,
    )
    prepared = observable.prepare(3)
    assert prepared.mpo is not None
    expected = np.kron(np.diag([1.0, -1.0]), np.kron(np.eye(2), np.diag([1.0, -1.0])))
    expected += 1e-18 * np.kron(np.eye(2), np.kron(np.array([[0.0, 1.0], [1.0, 0.0]]), np.eye(2)))
    np.testing.assert_allclose(prepared.mpo.to_matrix(), expected, rtol=0.0, atol=1e-30)

    with pytest.raises(ValueError, match="MPO must be Hermitian"):
        Observable.from_pauli_sum(terms=[(1j, "Z0")], length=1)


def test_pauli_sum_hermiticity_uses_the_contracted_operator() -> None:
    """Non-Hermitian terms can cancel to a Hermitian complete operator."""
    observable = Observable.from_pauli_sum(
        terms=[(1j, "Z0"), (-1j, "Z0")],
        length=1,
    )

    assert observable.mpo is not None
    np.testing.assert_array_equal(observable.mpo.to_matrix(), np.zeros((2, 2)))


def test_zero_pauli_sum_is_hermitian() -> None:
    """The empty Pauli sum creates a valid zero observable."""
    observable = Observable.from_pauli_sum(terms=[], length=2)
    prepared = observable.prepare(2)

    assert prepared.mpo is not None
    np.testing.assert_array_equal(prepared.mpo.to_matrix(), np.zeros((4, 4)))


def test_prepare_does_not_cache_state_layout_on_source() -> None:
    """One observable can be prepared for different compatible state layouts."""
    observable = Observable(np.eye(2), 0)

    first = observable.prepare(2, [2, 3])
    second = observable.prepare(3, [2, 4, 5])

    assert observable.prepared_length is None
    assert observable.mpo is None
    assert first.prepared_dimensions == (2, 3)
    assert second.prepared_dimensions == (2, 4, 5)
    assert first.mpo is not second.mpo


def test_prepare_observables_copies_compatible_prepared_inputs() -> None:
    """Worker preparation does not return a caller-owned prepared object."""
    prepared = Observable("zz", [0, 2]).prepare(3)

    worker_observable = prepare_observables([prepared], 3)[0]

    assert worker_observable is not prepared
    assert worker_observable.mpo is not None
    assert prepared.mpo is not None
    for worker_tensor, source_tensor in zip(worker_observable.mpo.tensors, prepared.mpo.tensors, strict=True):
        assert worker_tensor is not source_tensor
        np.testing.assert_array_equal(worker_tensor, source_tensor)

    worker_observable.mpo.tensors[0].fill(0)
    assert np.any(prepared.mpo.tensors[0])


def test_to_mpo_returns_an_independent_compact_support_mpo() -> None:
    """Local MPO conversion documents its offset through the observable support."""
    observable = Observable("zz", [1, 3])

    first = observable.to_mpo(5)
    second = observable.to_mpo(5)

    assert first.length == 3
    assert second.length == 3
    assert all(left is not right for left, right in zip(first.tensors, second.tensors, strict=True))
    first.tensors[0].fill(0)
    assert np.any(second.tensors[0])
