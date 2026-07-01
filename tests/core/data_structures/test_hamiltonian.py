# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

# ruff: noqa: SLF001 -- white-box tests exercise private Hamiltonian encoders

"""Tests for :class:`mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian`."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse

from mqt.yaqs import AnalogSimParams, Observable, Simulator, State
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.mpo import MPO


def _blank_hamiltonian(**attrs: object) -> Hamiltonian:
    """Construct an uninitialized Hamiltonian for error-path tests.

    Args:
        **attrs: Keyword attributes to set on the returned instance (``name=value``).

    Returns:
        A Hamiltonian instance with ``attrs`` set via :func:`setattr`.

    """
    h = Hamiltonian.__new__(Hamiltonian)
    for name, value in attrs.items():
        setattr(h, name, value)
    return h


def test_hamiltonian_requires_exactly_one_manual_source() -> None:
    """Constructor rejects zero or multiple manual data sources."""
    with pytest.raises(ValueError, match="exactly one of tensors, matrix, or sparse_matrix"):
        Hamiltonian()
    with pytest.raises(ValueError, match="exactly one of tensors, matrix, or sparse_matrix"):
        Hamiltonian(
            matrix=np.eye(2, dtype=np.complex128),
            sparse_matrix=scipy.sparse.eye(2, dtype=np.complex128),
        )


def test_hamiltonian_tensors_empty_raises() -> None:
    """Empty tensor list is rejected."""
    with pytest.raises(ValueError, match="non-empty list"):
        Hamiltonian(tensors=[])


def test_hamiltonian_tensors_length_mismatch() -> None:
    """length= must match len(tensors)."""
    mpo = MPO.ising(2, J=1.0, g=0.5)
    with pytest.raises(ValueError, match="does not match len\\(tensors\\)"):
        Hamiltonian(tensors=list(mpo.tensors), length=3)


def test_hamiltonian_tensors_rejects_conflicting_representation() -> None:
    """representation= must not contradict tensors=."""
    mpo = MPO.ising(2, J=1.0, g=0.5)
    with pytest.raises(ValueError, match="inferred as 'mpo' from tensors="):
        Hamiltonian(tensors=list(mpo.tensors), representation="sparse")


def test_hamiltonian_from_manual_tensors() -> None:
    """Manual MPO cores build an encoded Hamiltonian at construction."""
    rng = np.random.default_rng(0)
    tensors = [
        rng.random(size=(1, 2, 2, 2)).astype(np.complex128),
        rng.random(size=(2, 1, 2, 2)).astype(np.complex128),
    ]
    h = Hamiltonian(tensors=tensors)
    assert h.representation == "mpo"
    assert h.mpo.length == 2


def test_hamiltonian_matrix_explicit_length() -> None:
    """Dense matrix init accepts an explicit length."""
    h = Hamiltonian(matrix=np.eye(4, dtype=np.complex128), length=2)
    assert h.length == 2


def test_hamiltonian_matrix_infers_length_from_physical_dimension() -> None:
    """Dense matrix init infers length using physical_dimension as the local base."""
    h = Hamiltonian(matrix=np.eye(9, dtype=np.complex128), physical_dimension=3)
    assert h.length == 2


def test_hamiltonian_rejects_nonpositive_physical_dimension() -> None:
    """physical_dimension must be strictly positive."""
    with pytest.raises(ValueError, match="physical_dimension must be a positive integer"):
        Hamiltonian(matrix=np.eye(4, dtype=np.complex128), physical_dimension=0)


def test_hamiltonian_sparse_rejects_conflicting_representation() -> None:
    """representation= must not contradict sparse_matrix=."""
    sparse = scipy.sparse.eye(4, dtype=np.complex128, format="csr")
    with pytest.raises(ValueError, match="inferred as 'sparse' from sparse_matrix="):
        Hamiltonian(sparse_matrix=sparse, representation="dense")


def test_hamiltonian_sparse_explicit_length() -> None:
    """Sparse matrix init accepts an explicit length."""
    sparse = scipy.sparse.eye(4, dtype=np.complex128, format="csr")
    h = Hamiltonian(sparse_matrix=sparse, length=2)
    assert h.length == 2


def test_hamiltonian_coupled_transmon_factory() -> None:
    """Coupled transmon preset builds an MPO-backed Hamiltonian."""
    h = Hamiltonian.coupled_transmon(
        4,
        qubit_dim=2,
        resonator_dim=2,
        qubit_freq=5.0,
        resonator_freq=6.0,
        anharmonicity=0.2,
        coupling=0.1,
    )
    assert h.representation == "mpo"
    assert h.length == 4


def test_hamiltonian_matrix_property_unavailable_for_mpo() -> None:
    """Matrix property raises for MPO-only Hamiltonian."""
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    with pytest.raises(RuntimeError, match="Dense matrix is not available"):
        _ = h.matrix


def test_to_sparse_matrix_raises_without_data() -> None:
    """to_sparse_matrix raises when no backing data exists."""
    h = _blank_hamiltonian(
        representation="mpo",
        _encoded_as=None,
        _matrix=None,
        _mpo=None,
        _sparse_matrix=None,
    )
    with pytest.raises(RuntimeError, match="no materialized data"):
        h.to_sparse_matrix()


def test_encode_sparse_raises_without_data() -> None:
    """_encode('sparse') fails when no specification is available."""
    h = _blank_hamiltonian(
        representation="mpo",
        _encoded_as=None,
        _matrix=None,
        _mpo=None,
        _sparse_matrix=None,
    )
    with pytest.raises(ValueError, match="Cannot build sparse matrix"):
        h._encode("sparse")


def test_encode_dense_raises_without_data() -> None:
    """_encode('dense') fails when no specification is available."""
    h = _blank_hamiltonian(
        representation="dense",
        _encoded_as=None,
        _matrix=None,
        _mpo=None,
        _sparse_matrix=None,
    )
    with pytest.raises(ValueError, match="Cannot build dense matrix"):
        h._encode("dense")


def test_ensure_encoded_mpo_idempotent_when_already_materialized() -> None:
    """ensure_encoded('mpo') returns early when MPO is already cached."""
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    mpo = h.mpo
    h.ensure_encoded("mpo")
    assert h.mpo is mpo


def test_hamiltonian_matrix_not_square() -> None:
    """Dense matrix must be square."""
    with pytest.raises(ValueError, match="square 2-D"):
        Hamiltonian(matrix=np.ones((2, 3), dtype=np.complex128))


def test_hamiltonian_matrix_rejects_conflicting_representation() -> None:
    """representation= must not contradict matrix=."""
    with pytest.raises(ValueError, match="inferred as 'dense' from matrix="):
        Hamiltonian(matrix=np.eye(2, dtype=np.complex128), representation="mpo")


def test_hamiltonian_sparse_not_square() -> None:
    """Sparse matrix must be square."""
    coo = scipy.sparse.coo_matrix(np.ones((2, 3), dtype=np.complex128))
    with pytest.raises(ValueError, match="sparse_matrix must be square"):
        Hamiltonian(sparse_matrix=coo)


def test_hamiltonian_sparse_coo_converted_to_csr() -> None:
    """Non-CSR sparse formats are normalized to CSR at construction."""
    coo = scipy.sparse.eye(2, dtype=np.complex128, format="coo")
    h = Hamiltonian(sparse_matrix=coo)
    assert isinstance(h.sparse_matrix, scipy.sparse.csr_matrix)


def test_hamiltonian_dense_matrix_init() -> None:
    """matrix= infers dense representation and length from Hilbert dimension."""
    mat = np.eye(4, dtype=np.complex128)
    h = Hamiltonian(matrix=mat)
    assert h.representation == "dense"
    assert h.length == 2
    np.testing.assert_allclose(h.matrix, mat)


def test_hamiltonian_ising_encoded_at_init() -> None:
    """Preset classmethod encodes MPO at construction."""
    h = Hamiltonian.ising(3, J=1.0, g=0.5)
    assert h.representation == "mpo"
    assert h.mpo.length == 3


def test_hamiltonian_heisenberg_factory() -> None:
    """Heisenberg preset builds a valid MPO-backed Hamiltonian."""
    h = Hamiltonian.heisenberg(2, Jx=1.0, Jy=0.5, Jz=0.3, h=0.1)
    assert h.representation == "mpo"
    assert h.mpo.length == 2


def test_hamiltonian_pauli_factory() -> None:
    """Pauli Hamiltonian classmethod delegates to MPO."""
    h = Hamiltonian.pauli(
        length=2,
        two_body=[(-1.0, "Z", "Z")],
        one_body=[(-0.5, "X")],
    )
    assert h.mpo.length == 2


def test_hamiltonian_fermi_hubbard_factory() -> None:
    """Fermi-Hubbard preset builds a Hamiltonian."""
    h = Hamiltonian.fermi_hubbard_1d(2, t=1.0, u=0.5)
    assert h.representation == "mpo"
    assert h.length == 2


def test_hamiltonian_from_mpo() -> None:
    """from_mpo wraps without rebuilding."""
    mpo = MPO.ising(2, J=1.0, g=0.5)
    h = Hamiltonian.from_mpo(mpo)
    assert h.mpo is mpo


def test_hamiltonian_sparse_matrix_init() -> None:
    """sparse_matrix= infers sparse representation."""
    dim = 4
    sparse = scipy.sparse.eye(dim, dtype=np.complex128, format="csr")
    h = Hamiltonian(sparse_matrix=sparse)
    assert h.representation == "sparse"
    np.testing.assert_allclose(h.sparse_matrix.toarray(), sparse.toarray())


def test_ensure_encoded_mpo_to_sparse_cached() -> None:
    """Converting MPO to sparse once caches sparse_matrix for later runs."""
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    mpo = h.mpo
    with patch.object(MPO, "to_sparse_matrix", wraps=mpo.to_sparse_matrix) as mock_sparse:
        h.ensure_encoded("sparse")
        h.ensure_encoded("sparse")
    assert mock_sparse.call_count == 1
    np.testing.assert_allclose(
        h.sparse_matrix.toarray(),
        h.to_sparse_matrix().toarray(),
    )


def test_ensure_encoded_dense_from_mpo() -> None:
    """MPO Hamiltonian can materialize dense matrix without changing representation."""
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    assert h.representation == "mpo"
    h.ensure_encoded("dense")
    dense = h.matrix
    assert dense.shape == (4, 4)
    assert h.representation == "mpo"


def test_ensure_encoded_sparse_from_dense_hamiltonian() -> None:
    """Dense-init Hamiltonian can encode sparse for MCWF backends."""
    h = Hamiltonian(matrix=np.eye(4, dtype=np.complex128))
    h.ensure_encoded("sparse")
    np.testing.assert_allclose(h.sparse_matrix.toarray(), np.eye(4))


def test_hamiltonian_mpo_property_unavailable_for_dense_init() -> None:
    """Mpo property raises when only dense matrix is materialized."""
    h = Hamiltonian(matrix=np.eye(4, dtype=np.complex128))
    with pytest.raises(RuntimeError, match="MPO is not available"):
        _ = h.mpo


def test_hamiltonian_sparse_property_unavailable_for_mpo_init() -> None:
    """sparse_matrix property raises for MPO-only Hamiltonian."""
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    with pytest.raises(RuntimeError, match="Sparse matrix is not available"):
        _ = h.sparse_matrix


def test_to_matrix_from_mpo_and_sparse() -> None:
    """to_matrix converts from MPO or sparse without changing representation."""
    h_mpo = Hamiltonian.ising(2, J=1.0, g=0.5)
    ref = h_mpo.mpo.to_matrix()
    np.testing.assert_allclose(h_mpo.to_matrix(), ref, atol=1e-10)

    h_sparse = Hamiltonian(sparse_matrix=scipy.sparse.eye(4, dtype=np.complex128))
    np.testing.assert_allclose(h_sparse.to_matrix(), np.eye(4))


def test_to_matrix_returns_cached_dense_array() -> None:
    """to_matrix returns the stored dense matrix for dense-init Hamiltonians."""
    mat = np.eye(4, dtype=np.complex128)
    h = Hamiltonian(matrix=mat)
    np.testing.assert_allclose(h.to_matrix(), mat)


def test_to_sparse_matrix_from_mpo_only() -> None:
    """to_sparse_matrix converts from an MPO-backed Hamiltonian."""
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    sparse = h.to_sparse_matrix()
    np.testing.assert_allclose(sparse.toarray(), h.mpo.to_sparse_matrix().toarray())


def test_ensure_encoded_dense_idempotent() -> None:
    """ensure_encoded('dense') is a no-op when a dense matrix is already cached."""
    h = Hamiltonian(matrix=np.eye(4, dtype=np.complex128))
    cached = h.matrix.copy()
    h.ensure_encoded("dense")
    np.testing.assert_allclose(h.matrix, cached)


def test_ensure_encoded_dense_from_sparse_init() -> None:
    """Sparse-init Hamiltonian can materialize a dense matrix on demand."""
    h = Hamiltonian(sparse_matrix=scipy.sparse.eye(4, dtype=np.complex128))
    h.ensure_encoded("dense")
    np.testing.assert_allclose(h.matrix, np.eye(4))


def test_build_mpo_raises_without_tensors() -> None:
    """_build_mpo fails when no tensor specification exists."""
    h = _blank_hamiltonian(
        _mpo=None,
        _tensors=None,
        _matrix=None,
        _sparse_matrix=None,
    )
    with pytest.raises(ValueError, match="No MPO specification available"):
        h._build_mpo()


def test_to_sparse_matrix_from_dense() -> None:
    """to_sparse_matrix converts from dense matrix storage."""
    h = Hamiltonian(matrix=np.eye(4, dtype=np.complex128))
    sparse = h.to_sparse_matrix()
    np.testing.assert_allclose(sparse.toarray(), np.eye(4))


def test_to_matrix_raises_without_data() -> None:
    """to_matrix raises when no backing data exists."""
    h = _blank_hamiltonian(
        representation="mpo",
        _encoded_as=None,
        _matrix=None,
        _mpo=None,
        _sparse_matrix=None,
    )
    with pytest.raises(RuntimeError, match="no materialized data"):
        h.to_matrix()


def test_build_mpo_raises_for_matrix_only() -> None:
    """Cannot build MPO from dense-only specification."""
    h = Hamiltonian(matrix=np.eye(4, dtype=np.complex128))
    with pytest.raises(ValueError, match="Cannot build an MPO from matrix"):
        h._build_mpo()


def test_run_rejects_mpo_hamiltonian_with_mps_state() -> None:
    """TJM requires Hamiltonian stored as MPO."""
    dim = 4
    mat = np.eye(dim, dtype=np.complex128)
    h = Hamiltonian(matrix=mat)
    state = State(2, initial="zeros", representation="mps")
    params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.1,
        dt=0.1,
    )
    sim = Simulator(show_progress=False)
    with pytest.raises(ValueError, match=r"TJM simulation requires Hamiltonian\.representation='mpo'"):
        sim.run(state, h, params, None)


def test_run_hamiltonian_length_mismatch() -> None:
    """State and Hamiltonian lengths must match."""
    state = State(3, initial="zeros")
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.1,
        dt=0.1,
    )
    sim = Simulator(show_progress=False)
    with pytest.raises(ValueError, match=r"does not match Hamiltonian\.length"):
        sim.run(state, h, params, None)


def test_to_sparse_matrix_called_once_across_two_runs() -> None:
    """Outer loops reuse the same Hamiltonian without re-sparsifying."""
    state = State(2, initial="zeros", representation="vector")
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.1,
        dt=0.1,
    )
    mpo = h.mpo
    sim = Simulator(show_progress=False)
    with patch.object(MPO, "to_sparse_matrix", wraps=mpo.to_sparse_matrix) as mock_sparse:
        sim.run(state, h, params, None)
        sim.run(state, h, params, None)
    assert mock_sparse.call_count == 1
