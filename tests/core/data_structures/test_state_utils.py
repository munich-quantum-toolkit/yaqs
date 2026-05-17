# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :mod:`mqt.yaqs.core.data_structures.state_utils`."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.data_structures.state_utils import (
    infer_qubit_length,
    local_vector_for_preset,
    normalize_density_matrix,
    normalize_vector,
    preset_is_product_state,
    product_state_vector,
    reject_preset_only_kwargs,
    resolve_physical_dimensions,
    validate_representation,
)


def test_validate_representation_accepts_known() -> None:
    """Known representation labels are returned unchanged."""
    assert validate_representation("vector") == "vector"


def test_validate_representation_rejects_unknown() -> None:
    """Invalid representation labels raise ValueError."""
    with pytest.raises(ValueError, match=r"Invalid representation 'bad'"):
        validate_representation("bad")


def test_resolve_physical_dimensions_defaults() -> None:
    """Default physical dimensions are qubits (2)."""
    assert resolve_physical_dimensions(3, None) == [2, 2, 2]


def test_resolve_physical_dimensions_int_broadcast() -> None:
    """Integer physical dimension is broadcast to all sites."""
    assert resolve_physical_dimensions(2, 3) == [3, 3]


def test_resolve_physical_dimensions_list_mismatch() -> None:
    """List physical_dimensions must match chain length."""
    with pytest.raises(ValueError, match="physical_dimensions length"):
        resolve_physical_dimensions(2, [2, 2, 2])


def test_infer_qubit_length_power_of_two() -> None:
    """Hilbert dimension must be a positive power of two."""
    assert infer_qubit_length(4) == 2
    with pytest.raises(ValueError, match="not a power of two"):
        infer_qubit_length(6)


def test_normalize_vector_zero_raises() -> None:
    """Zero vectors cannot be normalized."""
    with pytest.raises(ValueError, match="non-zero"):
        normalize_vector(np.zeros(2, dtype=np.complex128))


def test_normalize_density_matrix_invalid() -> None:
    """Density matrix must be square with non-zero trace."""
    with pytest.raises(ValueError, match="square 2-D"):
        normalize_density_matrix(np.ones((2, 3), dtype=np.complex128))
    with pytest.raises(ValueError, match="non-zero trace"):
        normalize_density_matrix(np.zeros((2, 2), dtype=np.complex128))


def test_normalize_density_matrix_renormalizes_trace() -> None:
    """Non-unit trace density matrices are normalized."""
    rho = 2.0 * np.eye(2, dtype=np.complex128)
    out = normalize_density_matrix(rho)
    assert np.isclose(np.trace(out), 1.0)


def test_reject_preset_only_kwargs() -> None:
    """Preset-only kwargs cannot be combined with manual state data."""
    with pytest.raises(ValueError, match="initial= and other preset"):
        reject_preset_only_kwargs(initial="ones", pad=None, basis_string=None, seed=None)
    with pytest.raises(ValueError, match="pad applies only"):
        reject_preset_only_kwargs(initial="zeros", pad=1, basis_string=None, seed=None)
    with pytest.raises(ValueError, match="basis_string applies only"):
        reject_preset_only_kwargs(initial="zeros", pad=None, basis_string="0", seed=None)
    with pytest.raises(ValueError, match="seed applies only"):
        reject_preset_only_kwargs(initial="zeros", pad=None, basis_string=None, seed=1)


def test_resolve_physical_dimensions_list() -> None:
    """Explicit per-site physical dimension lists are copied."""
    assert resolve_physical_dimensions(2, [2, 3]) == [2, 3]


def test_preset_is_product_state() -> None:
    """Product presets are distinguished from entangled presets."""
    assert preset_is_product_state("zeros")
    assert not preset_is_product_state("haar-random")


@pytest.mark.parametrize(
    "initial",
    ["zeros", "ones", "x+", "x-", "y+", "y-", "Neel", "wall"],
)
def test_local_vector_for_preset_product_states(initial: str) -> None:
    """Product presets yield normalized local vectors."""
    rng = np.random.default_rng(0)
    for site in range(4):
        vec = local_vector_for_preset(site, initial, 2, length=4, basis_string=None, rng=rng)
        assert vec.shape == (2,)
        assert np.isclose(np.linalg.norm(vec), 1.0)


def test_local_vector_for_preset_x_plus_requires_dim_two() -> None:
    """x+ preset needs local dimension at least 2."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="x\\+ preset requires"):
        local_vector_for_preset(0, "x+", 1, length=1, basis_string=None, rng=rng)


def test_local_vector_for_preset_ones_requires_dim_two() -> None:
    """Ones preset needs local dimension at least 2."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="ones preset requires"):
        local_vector_for_preset(0, "ones", 1, length=1, basis_string=None, rng=rng)


def test_local_vector_for_preset_neel_local_dim_one() -> None:
    """Neel preset falls back to |0> when local dimension is 1."""
    rng = np.random.default_rng(0)
    vec = local_vector_for_preset(0, "Neel", 1, length=2, basis_string=None, rng=rng)
    assert np.isclose(vec[0], 1.0)


def test_local_vector_for_preset_wall_local_dim_one() -> None:
    """Wall preset falls back to |0> on the right half when local dimension is 1."""
    rng = np.random.default_rng(0)
    vec = local_vector_for_preset(1, "wall", 1, length=2, basis_string=None, rng=rng)
    assert np.isclose(vec[0], 1.0)


def test_local_vector_for_preset_random_requires_dim_two() -> None:
    """Random preset needs local dimension at least 2."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="random preset requires"):
        local_vector_for_preset(0, "random", 1, length=1, basis_string=None, rng=rng)


def test_local_vector_for_preset_basis_requires_string() -> None:
    """Basis preset requires basis_string."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="basis_string must be provided"):
        local_vector_for_preset(0, "basis", 2, length=2, basis_string=None, rng=rng)


def test_local_vector_for_preset_basis_index_out_of_range() -> None:
    """Basis index must fit local dimension."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="out of range"):
        local_vector_for_preset(0, "basis", 2, length=2, basis_string="2", rng=rng)


def test_local_vector_for_preset_unknown() -> None:
    """Unknown presets raise ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Unknown product-state preset"):
        local_vector_for_preset(0, "not-a-preset", 2, length=2, basis_string=None, rng=rng)


def test_product_state_vector_neel() -> None:
    """Product-state builder matches State vector for Neel."""
    vec = product_state_vector(4, "Neel", None)
    ref = State(4, initial="Neel", representation="vector").vector
    np.testing.assert_allclose(vec, ref, atol=1e-10)


def test_product_state_vector_basis() -> None:
    """Product-state builder supports basis_string presets."""
    vec = product_state_vector(2, "basis", None, basis_string="01")
    ref = State(2, initial="basis", basis_string="01", representation="vector").vector
    np.testing.assert_allclose(vec, ref, atol=1e-10)
