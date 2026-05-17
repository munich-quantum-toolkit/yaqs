# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :meth:`mqt.yaqs.core.data_structures.state.State.encode`."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.networks import MPS
from mqt.yaqs.core.data_structures.state import Representation, State


def test_state_default_representation_is_mps() -> None:
    """Preset-only State defaults to MPS representation."""
    psi = State(3, initial="zeros")
    assert psi.representation == "mps"


def test_state_invalid_representation() -> None:
    """State rejects unknown representation values."""
    with pytest.raises(ValueError, match=r"Invalid representation 'tjm'"):
        State(2, representation=cast(Any, "tjm"))  # noqa: TC006


def test_encode_without_argument_uses_state_representation() -> None:
    """encode() with no argument materializes self.representation."""
    psi = State(3, initial="zeros", representation="vector")
    psi.encode()
    assert psi._encoded_as == "vector"
    assert psi._mps is None


@pytest.mark.parametrize(
    "representation",
    ["mps", "vector", "density_matrix"],
)
def test_encode_sets_encoded_as(representation: Representation) -> None:
    """encode records the active representation on the state."""
    psi = State(3, initial="zeros")
    psi.encode(representation)
    assert psi._encoded_as == representation


def test_encode_mps_normalizes_b_form() -> None:
    """encode('mps') leaves the underlying MPS in B-canonical form."""
    psi = State(4, initial="x+")
    psi.encode("mps")
    assert psi.mps.check_canonical_form() != [-1]


def test_encode_vector_caches_normalized_vector() -> None:
    """encode('vector') stores a unit-norm dense vector."""
    psi = State(3, initial="zeros")
    psi.encode("vector")
    assert psi._vector is not None
    assert psi._mps is None
    assert np.isclose(np.linalg.norm(psi.vector), 1.0)
    ref = MPS(3, state="zeros").to_vec()
    ref /= np.linalg.norm(ref)
    np.testing.assert_allclose(psi.vector, ref)


def test_encode_density_matrix_from_pure_state() -> None:
    """encode('density_matrix') caches |psi><psi| for a pure initial state."""
    psi = State(2, initial="x+")
    psi.encode("density_matrix")
    assert psi._mps is None
    assert psi._vector is not None
    expected = np.outer(psi.vector, psi.vector.conj())
    np.testing.assert_allclose(psi.density_matrix, expected)


def test_encode_idempotent() -> None:
    """Repeated encode with the same representation is a no-op."""
    psi = State(3, initial="zeros")
    psi.encode("vector")
    assert psi._vector is not None
    cached = psi._vector.copy()
    psi.encode("vector")
    np.testing.assert_allclose(psi.vector, cached)


def test_encode_invalid_representation_raises() -> None:
    """Unknown representation strings raise ValueError."""
    psi = State(2, initial="zeros")
    with pytest.raises(ValueError, match=r"Invalid representation 'invalid'"):
        psi.encode("invalid")  # ty: ignore[invalid-argument-type]


def test_initial_kwarg_builds_mps() -> None:
    """Constructor uses initial= preset names when building the MPS."""
    psi = State(2, initial="ones")
    psi.encode("mps")
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
    assert spec._encoded_as == "mps"
    np.testing.assert_allclose(spec.mps.to_vec(), mps_ref.to_vec())


def test_init_from_vector() -> None:
    """1-D array is encoded as a dense vector at construction."""
    vec = np.array([1.0, 0.0], dtype=np.complex128)
    spec = State(vector=vec)
    assert spec.length == 1
    assert spec.representation == "vector"
    assert spec._encoded_as == "vector"
    np.testing.assert_allclose(spec.vector, vec)


def test_init_from_density_matrix() -> None:
    """2-D array is encoded as a density matrix at construction."""
    rho = np.diag([1.0, 0.0]).astype(np.complex128)
    spec = State(density_matrix=rho)
    assert spec.length == 1
    assert spec.representation == "density_matrix"
    assert spec._encoded_as == "density_matrix"
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


def test_encode_mps_from_vector_raises() -> None:
    """Vector-initialized states cannot be encoded as MPS without tensor data."""
    spec = State(vector=np.array([1.0, 0.0], dtype=np.complex128))
    with pytest.raises(ValueError, match="Cannot build an MPS"):
        spec.encode("mps")


@pytest.mark.parametrize(
    "initial",
    ["zeros", "ones", "x+", "x-", "y+", "y-", "Neel", "wall"],
)
def test_preset_encode_vector_matches_mps(initial: str) -> None:
    """Product presets build the same dense vector as MPS.to_vec without materializing MPS."""
    length = 4
    spec = State(length, initial=initial)
    spec.encode("vector")
    assert spec._mps is None
    ref = MPS(length, state=initial).to_vec()
    ref /= np.linalg.norm(ref)
    np.testing.assert_allclose(spec.vector, ref)


def test_preset_encode_basis_string() -> None:
    """basis preset uses basis_string for the dense product vector."""
    spec = State(3, initial="basis", basis_string="010")
    spec.encode("vector")
    ref = MPS(3, state="basis", basis_string="010").to_vec()
    ref /= np.linalg.norm(ref)
    np.testing.assert_allclose(spec.vector, ref)


def test_preset_random_with_seed() -> None:
    """random preset is reproducible when seed is set on State."""
    spec_a = State(3, initial="random", seed=42)
    spec_b = State(3, initial="random", seed=42)
    spec_a.encode("vector")
    spec_b.encode("vector")
    np.testing.assert_allclose(spec_a.vector, spec_b.vector)


def test_haar_random_encode_vector_uses_mps() -> None:
    """Entangled haar-random presets still require MPS for dense encoding."""
    spec = State(4, initial="haar-random", pad=4)
    spec.encode("vector")
    assert spec._mps is not None
