# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :meth:`mqt.yaqs.core.data_structures.state.State.ensure_encoded`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.state import State

if TYPE_CHECKING:
    from mqt.yaqs.core.data_structures.state import Representation


def test_state_default_representation_is_mps() -> None:
    """Preset-only State defaults to MPS representation."""
    psi = State(3, initial="zeros")
    assert psi.representation == "mps"


def test_state_rejects_nonpositive_length() -> None:
    """Preset construction requires a positive chain length."""
    with pytest.raises(ValueError, match="length must be a positive integer"):
        State(0, initial="zeros")


def test_state_invalid_representation() -> None:
    """State rejects unknown representation values."""
    with pytest.raises(ValueError, match=r"Invalid representation 'tjm'"):
        State(2, representation=cast(Any, "tjm"))  # noqa: TC006


def test_ensure_encoded_without_argument_uses_state_representation() -> None:
    """ensure_encoded() with no argument materializes self.representation."""
    psi = State(3, initial="zeros", representation="vector")
    psi.ensure_encoded()
    assert psi.representation == "vector"
    with pytest.raises(RuntimeError, match="MPS is not available"):
        _ = psi.mps


@pytest.mark.parametrize(
    "representation",
    ["mps", "vector", "density_matrix"],
)
def test_ensure_encoded_materializes_representation(representation: Representation) -> None:
    """ensure_encoded materializes the requested backend representation."""
    psi = State(3, initial="zeros")
    psi.ensure_encoded(representation)
    if representation == "mps":
        _ = psi.mps
    elif representation == "vector":
        _ = psi.vector
    else:
        _ = psi.density_matrix


def test_ensure_encoded_mps_normalizes_b_form() -> None:
    """ensure_encoded('mps') leaves the underlying MPS in B-canonical form."""
    psi = State(4, initial="x+")
    psi.ensure_encoded("mps")
    assert psi.mps.check_canonical_form() != [-1]


def test_ensure_encoded_vector_caches_normalized_vector() -> None:
    """ensure_encoded('vector') stores a unit-norm dense vector."""
    psi = State(3, initial="zeros", representation="vector")
    assert np.isclose(np.linalg.norm(psi.vector), 1.0)
    with pytest.raises(RuntimeError, match="MPS is not available"):
        _ = psi.mps
    ref = MPS(3, state="zeros").to_vec()
    ref /= np.linalg.norm(ref)
    np.testing.assert_allclose(psi.vector, ref)


def test_ensure_encoded_density_matrix_from_pure_state() -> None:
    """ensure_encoded('density_matrix') caches |psi><psi| for a pure initial state."""
    psi = State(2, initial="x+", representation="density_matrix")
    with pytest.raises(RuntimeError, match="MPS is not available"):
        _ = psi.mps
    ref_vec = MPS(2, state="x+").to_vec()
    ref_vec /= np.linalg.norm(ref_vec)
    expected = np.outer(ref_vec, ref_vec.conj())
    np.testing.assert_allclose(psi.density_matrix, expected)


def test_ensure_encoded_idempotent() -> None:
    """Repeated ensure_encoded with the same representation is a no-op."""
    psi = State(3, initial="zeros", representation="vector")
    cached = psi.vector.copy()
    psi.ensure_encoded("vector")
    np.testing.assert_allclose(psi.vector, cached)


def test_ensure_encoded_invalid_representation_raises() -> None:
    """Unknown representation strings raise ValueError."""
    psi = State(2, initial="zeros")
    with pytest.raises(ValueError, match=r"Invalid representation 'invalid'"):
        psi.ensure_encoded(cast(Any, "invalid"))  # noqa: TC006


def test_initial_kwarg_builds_mps() -> None:
    """Constructor uses initial= preset names when building the MPS."""
    psi = State(2, initial="ones")
    psi.ensure_encoded("mps")
    vec = psi.mps.to_vec()
    assert np.isclose(abs(vec[-1]), 1.0)


def test_from_mps_wraps_existing() -> None:
    """from_mps preserves tensor data without rebuilding."""
    mps = MPS(2, state="zeros")
    spec = State.from_mps(mps)
    assert spec.mps is mps


def test_init_from_tensor_list() -> None:
    """List of cores is encoded as MPS at construction."""
    mps_ref = MPS(2, state="zeros")
    spec = State(tensors=list(mps_ref.tensors))
    assert spec.length == 2
    assert spec.representation == "mps"
    np.testing.assert_allclose(spec.mps.to_vec(), mps_ref.to_vec())


def test_init_from_vector() -> None:
    """1-D array is encoded as a dense vector at construction."""
    vec = np.array([1.0, 0.0], dtype=np.complex128)
    spec = State(vector=vec)
    assert spec.length == 1
    assert spec.representation == "vector"
    np.testing.assert_allclose(spec.vector, vec)


def test_init_from_density_matrix() -> None:
    """2-D array is encoded as a density matrix at construction."""
    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    spec = State(density_matrix=rho)
    assert spec.length == 1
    assert spec.representation == "density_matrix"
    np.testing.assert_allclose(spec.density_matrix, rho)


def test_manual_data_infers_representation_without_kwarg() -> None:
    """tensors/vector/density_matrix set representation; no representation= needed."""
    mps_ref = MPS(2, state="zeros")
    from_tensors = State(tensors=list(mps_ref.tensors))
    assert from_tensors.representation == "mps"

    vec = np.array([1.0, 0.0], dtype=np.complex128)
    from_vector = State(vector=vec)
    assert from_vector.representation == "vector"

    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    from_rho = State(density_matrix=rho)
    assert from_rho.representation == "density_matrix"


def test_manual_data_rejects_conflicting_representation() -> None:
    """Explicit representation= must not contradict manual data."""
    vec = np.array([1.0, 0.0], dtype=np.complex128)
    with pytest.raises(ValueError, match="inferred as 'vector' from vector="):
        State(vector=vec, representation="mps")
    rho = np.eye(2, dtype=np.complex128)
    with pytest.raises(ValueError, match="inferred as 'density_matrix' from density_matrix="):
        State(density_matrix=rho, representation="vector")


def test_state_tensors_length_mismatch() -> None:
    """length= must match len(tensors)."""
    mps_ref = MPS(2, state="zeros")
    with pytest.raises(ValueError, match="does not match len\\(tensors\\)"):
        State(tensors=list(mps_ref.tensors), length=3)


def test_state_tensors_representation_conflict() -> None:
    """representation= must not contradict tensors=."""
    mps_ref = MPS(2, state="zeros")
    with pytest.raises(ValueError, match="inferred as 'mps' from tensors="):
        State(tensors=list(mps_ref.tensors), representation="vector")


def test_state_vector_explicit_length() -> None:
    """Vector init accepts an explicit length when Hilbert dimension is known."""
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
    spec = State(vector=vec, length=2)
    assert spec.length == 2


def test_state_density_matrix_explicit_length() -> None:
    """Density-matrix init accepts an explicit length."""
    rho = np.eye(4, dtype=np.complex128) / 4.0
    spec = State(density_matrix=rho, length=2)
    assert spec.length == 2


def test_state_vector_property_unavailable_for_density() -> None:
    """Vector property raises when encoded as density_matrix."""
    spec = State(2, initial="zeros", representation="density_matrix")
    with pytest.raises(RuntimeError, match="State vector is not available"):
        _ = spec.vector


def test_state_density_matrix_property_unavailable_for_vector() -> None:
    """density_matrix property raises when encoded as vector."""
    spec = State(2, initial="zeros", representation="vector")
    with pytest.raises(RuntimeError, match="Density matrix is not available"):
        _ = spec.density_matrix


def test_ensure_encoded_density_matrix_from_vector_init() -> None:
    """Vector-initialized State can be encoded as a density matrix."""
    vec = np.array([1.0, 0.0], dtype=np.complex128)
    spec = State(vector=vec)
    spec.ensure_encoded("density_matrix")
    expected = np.outer(vec, vec.conj())
    np.testing.assert_allclose(spec.density_matrix, expected)


def test_ensure_encoded_density_matrix_from_preset() -> None:
    """Preset State can encode density_matrix without materializing MPS first."""
    spec = State(2, initial="x+", representation="mps")
    spec.ensure_encoded("density_matrix")
    ref_vec = MPS(2, state="x+").to_vec()
    ref_vec /= np.linalg.norm(ref_vec)
    expected = np.outer(ref_vec, ref_vec.conj())
    np.testing.assert_allclose(spec.density_matrix, expected)


def test_manual_init_mutually_exclusive() -> None:
    """Cannot pass more than one of tensors, vector, and density_matrix."""
    vec = np.ones(4, dtype=np.complex128)
    rho = np.eye(4, dtype=np.complex128)

    def _invalid() -> State:
        return State(2, vector=vec, density_matrix=rho)

    with pytest.raises(ValueError, match="at most one"):
        _invalid()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vector": np.array([1.0, 0.0], dtype=np.complex128), "initial": "ones"},
        {"vector": np.array([1.0, 0.0], dtype=np.complex128), "pad": 4},
        {
            "tensors": [np.zeros((1, 2, 1), dtype=np.complex128)],
            "initial": "ones",
        },
    ],
)
def test_preset_kwargs_rejected_for_manual_data(kwargs: dict) -> None:
    """Preset-only arguments cannot be passed with tensors, vector, or density_matrix."""
    with pytest.raises(ValueError, match="only to preset State construction"):
        State(**kwargs)


def test_ensure_encoded_mps_from_vector_raises() -> None:
    """Vector-initialized states cannot be encoded as MPS without tensor data."""
    spec = State(vector=np.array([1.0, 0.0], dtype=np.complex128))
    with pytest.raises(ValueError, match="Cannot build an MPS"):
        spec.ensure_encoded("mps")


@pytest.mark.parametrize(
    "initial",
    ["zeros", "ones", "x+", "x-", "y+", "y-", "Neel", "wall"],
)
def test_preset_ensure_encoded_vector_matches_mps(initial: str) -> None:
    """Product presets build the same dense vector as MPS.to_vec without materializing MPS."""
    length = 4
    spec = State(length, initial=initial, representation="vector")
    with pytest.raises(RuntimeError, match="MPS is not available"):
        _ = spec.mps
    ref = MPS(length, state=initial).to_vec()
    ref /= np.linalg.norm(ref)
    np.testing.assert_allclose(spec.vector, ref)


def test_preset_encode_basis_string() -> None:
    """Basis preset uses basis_string for the dense product vector."""
    spec = State(3, initial="basis", basis_string="010", representation="vector")
    ref = MPS(3, state="basis", basis_string="010").to_vec()
    ref /= np.linalg.norm(ref)
    np.testing.assert_allclose(spec.vector, ref)


def test_preset_random_with_seed() -> None:
    """Random preset is reproducible when seed is set on State."""
    spec_a = State(3, initial="random", seed=42, representation="vector")
    spec_b = State(3, initial="random", seed=42, representation="vector")
    np.testing.assert_allclose(spec_a.vector, spec_b.vector)


def test_state_preset_without_length_raises() -> None:
    """Preset construction requires length=."""
    with pytest.raises(ValueError, match="length is required"):
        State(initial="zeros")


def test_state_tensors_empty_raises() -> None:
    """Empty tensor list is rejected."""
    with pytest.raises(ValueError, match="non-empty list"):
        State(tensors=[])


def test_state_vector_zero_norm_raises() -> None:
    """Zero state vectors are rejected at construction."""
    with pytest.raises(ValueError, match="non-zero"):
        State(vector=np.zeros(2, dtype=np.complex128))


def test_state_density_matrix_renormalizes_trace() -> None:
    """Non-unit-trace density matrices are normalized at construction."""
    rho = 2.0 * np.eye(2, dtype=np.complex128)
    spec = State(density_matrix=rho)
    assert np.isclose(np.trace(spec.density_matrix), 1.0)


def test_state_physical_dimensions_int() -> None:
    """Integer physical_dimensions broadcasts to all sites."""
    spec = State(2, initial="zeros", physical_dimensions=2)
    assert spec.length == 2
    assert spec.mps.length == 2


@pytest.mark.parametrize("initial", ["y+", "y-", "Neel", "wall"])
def test_preset_vector_matches_product_builder(initial: str) -> None:
    """Additional product presets match dense product-state construction."""
    spec = State(4, initial=initial, representation="vector")
    ref = State(4, initial=initial, representation="vector").vector
    np.testing.assert_allclose(spec.vector, ref)


def test_haar_random_ensure_encoded_vector_uses_mps() -> None:
    """Entangled haar-random presets still require MPS for dense encoding."""
    spec = State(4, initial="haar-random", pad=4, representation="vector")
    # Entangled presets materialize via an internal MPS before dense conversion.
    spec.ensure_encoded("vector")
    assert spec.representation == "vector"
    np.testing.assert_allclose(np.linalg.norm(spec.vector), 1.0)


def test_ensure_encoded_density_matrix_from_tensors() -> None:
    """Tensor-initialized State builds density matrices via an internal MPS."""
    mps_ref = MPS(2, state="zeros")
    spec = State(tensors=list(mps_ref.tensors), representation="mps")
    spec.ensure_encoded("density_matrix")
    ref_vec = mps_ref.to_vec()
    ref_vec /= np.linalg.norm(ref_vec)
    expected = np.outer(ref_vec, ref_vec.conj())
    np.testing.assert_allclose(spec.density_matrix, expected)


def test_haar_random_ensure_encoded_density_matrix_uses_mps() -> None:
    """Entangled presets materialize density matrices through MPS.to_vec."""
    spec = State(4, initial="haar-random", pad=4, representation="mps")
    spec.ensure_encoded("density_matrix")
    assert np.isclose(np.trace(spec.density_matrix), 1.0)
