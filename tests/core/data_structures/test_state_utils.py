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
from qiskit.circuit.library import CXGate
from qiskit.quantum_info import Pauli

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.simulation_parameters import Observable
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.data_structures.state_utils import (
    embed_adjacent_two_site_operator,
    embed_one_site_operator,
    embed_two_site_factors,
    infer_chain_length,
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
from mqt.yaqs.digital.utils.matrix_utils import embed_unitary


def test_validate_representation_accepts_known() -> None:
    """Known representation labels are returned unchanged."""
    assert validate_representation("mps") == "mps"
    assert validate_representation("vector") == "vector"
    assert validate_representation("density_matrix") == "density_matrix"


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


def test_resolve_physical_dimensions_rejects_nonpositive_int() -> None:
    """Integer physical_dimensions must be strictly positive."""
    with pytest.raises(ValueError, match="resolve_physical_dimensions"):
        resolve_physical_dimensions(2, 0)


def test_resolve_physical_dimensions_rejects_nonpositive_list_element() -> None:
    """Each list element must be a strictly positive integer."""
    with pytest.raises(ValueError, match=r"physical_dimensions\[1\]"):
        resolve_physical_dimensions(2, [2, -1])


def test_infer_chain_length_general_base() -> None:
    """Chain length is inferred from Hilbert dimension and local dimension."""
    assert infer_chain_length(9, physical_dimension=3) == 2


def test_infer_chain_length_rejects_nonpositive_physical_dimension() -> None:
    """physical_dimension must be a positive integer."""
    with pytest.raises(ValueError, match="physical_dimension must be a positive integer"):
        infer_chain_length(4, physical_dimension=0)


def test_infer_chain_length_rejects_nonpositive_hilbert_dim() -> None:
    """Hilbert-space dimension must be positive."""
    with pytest.raises(ValueError, match="Hilbert-space dimension"):
        infer_chain_length(0, physical_dimension=2)


def test_infer_qubit_length_power_of_two() -> None:
    """Hilbert dimension must be a positive power of two."""
    assert infer_qubit_length(4) == 2
    with pytest.raises(ValueError, match="is not physical_dimension\\*\\*length"):
        infer_qubit_length(6)


def test_normalize_vector_zero_raises() -> None:
    """Zero vectors cannot be normalized."""
    with pytest.raises(ValueError, match="non-zero"):
        normalize_vector(np.zeros(2, dtype=np.complex128))


def test_normalize_vector_unit_norm() -> None:
    """Non-zero vectors are scaled to unit norm."""
    vec = np.array([3.0, 4.0], dtype=np.complex128)
    out = normalize_vector(vec)
    assert np.isclose(np.linalg.norm(out), 1.0)


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


def test_local_vector_for_preset_random_builds_mixed_state() -> None:
    """Random preset yields a superposition of |0> and |1> with weights summing to one."""
    rng = np.random.default_rng(0)
    vec = local_vector_for_preset(0, "random", 2, length=2, basis_string=None, rng=rng)
    assert np.isclose(vec[0] + vec[1], 1.0)
    assert vec[0] >= 0.0 and vec[1] >= 0.0


def test_local_vector_for_preset_basis_requires_string() -> None:
    """Basis preset requires basis_string."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="basis_string must be provided"):
        local_vector_for_preset(0, "basis", 2, length=2, basis_string=None, rng=rng)


def test_local_vector_for_preset_basis_string_too_short() -> None:
    """basis_string must cover every site index."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="basis_string length"):
        local_vector_for_preset(2, "basis", 2, length=4, basis_string="01", rng=rng)


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


def test_product_state_vector_uniform_local_dimension() -> None:
    """product_state_vector broadcasts an integer physical dimension."""
    vec = product_state_vector(2, "zeros", 3)
    assert vec.shape == (9,)
    assert np.isclose(np.linalg.norm(vec), 1.0)
    assert np.isclose(vec[0], 1.0)


def test_normalize_density_matrix_already_normalized() -> None:
    """Unit-trace density matrices are returned unchanged."""
    rho = np.eye(2, dtype=np.complex128) / 2.0
    out = normalize_density_matrix(rho)
    np.testing.assert_allclose(out, rho, atol=1e-12)


@pytest.mark.parametrize("length", [2, 3, 4])
@pytest.mark.parametrize("site", [0, 1, 2])
@pytest.mark.parametrize("local", ["X", "Y", "Z"])
def test_embed_one_site_matches_qiskit(length: int, site: int, local: str) -> None:
    """One-site embedding agrees with Qiskit ``Operator`` layout."""
    if site >= length:
        pytest.skip("site out of range for chain length")

    local_mat = np.asarray(Pauli(local).to_matrix(), dtype=np.complex128)
    yaqs = embed_one_site_operator(local_mat, length, site)
    qiskit = embed_unitary(local_mat, [site], length)
    np.testing.assert_allclose(yaqs, qiskit, atol=1e-12)


@pytest.mark.parametrize("length", [3, 4])
@pytest.mark.parametrize("site_left", [0, 1, 2])
def test_embed_adjacent_two_site_matches_qiskit(length: int, site_left: int) -> None:
    """Adjacent two-site embedding agrees with Qiskit ``Operator`` layout."""
    if site_left + 1 >= length:
        pytest.skip("pair out of range for chain length")

    local_mat = np.asarray(CXGate().to_matrix(), dtype=np.complex128)
    yaqs = embed_adjacent_two_site_operator(local_mat, length, site_left)
    qiskit = embed_unitary(local_mat, [site_left, site_left + 1], length)
    np.testing.assert_allclose(yaqs, qiskit, atol=1e-12)


def test_embed_matches_mps_expect_on_haar() -> None:
    """Embedded Pauli expectations match ``MPS.expect`` on an entangled state."""
    length = 3
    mps = MPS(length, state="haar-random", pad=4)
    psi = mps.to_vec()
    for site in range(length):
        for name in ("x", "z"):
            obs = Observable(name, site)
            mps_val = mps.expect(obs)
            op = embed_one_site_operator(np.asarray(obs.gate.matrix, dtype=np.complex128), length, site)
            embed_val = float(np.real(np.vdot(psi, op @ psi)))
            assert mps_val == pytest.approx(embed_val, abs=1e-9)


def test_embed_one_site_non_qubit_local_dimension() -> None:
    """One-site embedding supports local dimensions larger than two."""
    length = 1
    local_dim = 3
    op = np.eye(local_dim, dtype=np.complex128)
    embedded = embed_one_site_operator(op, length, 0, physical_dimensions=[local_dim])
    np.testing.assert_allclose(embedded, op, atol=1e-12)


def test_embed_two_site_factors_non_adjacent() -> None:
    """Non-adjacent factor embedding matches sequential one-site products."""
    length = 3
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    full = embed_two_site_factors(x, z, length, 0, 2)
    expected = embed_one_site_operator(z, length, 2) @ embed_one_site_operator(x, length, 0)
    np.testing.assert_allclose(full, expected, atol=1e-12)


def test_embed_one_site_operator_rejects_out_of_range_site() -> None:
    """One-site embedding rejects invalid site indices."""
    op = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError, match="site 2 out of range"):
        embed_one_site_operator(op, 2, 2)


def test_embed_one_site_operator_rejects_wrong_shape() -> None:
    """One-site embedding rejects operators with the wrong local shape."""
    with pytest.raises(ValueError, match="op must have shape"):
        embed_one_site_operator(np.eye(3, dtype=np.complex128), 2, 0)


def test_embed_adjacent_two_site_operator_rejects_invalid_pair() -> None:
    """Adjacent two-site embedding rejects pairs outside the chain."""
    op4 = np.eye(4, dtype=np.complex128)
    with pytest.raises(ValueError, match="adjacent pair"):
        embed_adjacent_two_site_operator(op4, 2, 1)


def test_embed_adjacent_two_site_operator_rejects_wrong_shape() -> None:
    """Adjacent two-site embedding rejects operators with the wrong pair shape."""
    with pytest.raises(ValueError, match="op4 must have shape"):
        embed_adjacent_two_site_operator(np.eye(2, dtype=np.complex128), 3, 0)


def test_embed_two_site_factors_rejects_same_site() -> None:
    """Two-site factor embedding requires distinct sites."""
    op = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError, match="site1 and site2 must differ"):
        embed_two_site_factors(op, op, 3, 1, 1)


def test_embed_two_site_factors_rejects_out_of_range_site() -> None:
    """Two-site factor embedding rejects invalid site indices."""
    op = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError, match="site 3 out of range"):
        embed_two_site_factors(op, op, 3, 0, 3)


def test_embed_two_site_factors_rejects_shape_mismatch() -> None:
    """Two-site factor embedding rejects local operators with wrong shapes."""
    op2 = np.eye(2, dtype=np.complex128)
    op3 = np.eye(3, dtype=np.complex128)
    with pytest.raises(ValueError, match="local operators must match site dimensions"):
        embed_two_site_factors(op2, op3, 3, 0, 1, physical_dimensions=[3, 2, 2])
