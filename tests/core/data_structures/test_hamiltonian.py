# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :class:`mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian`."""

from __future__ import annotations

import inspect
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse

from mqt.yaqs import AnalogSimParams, Observable, Simulator, State
from mqt.yaqs.core.data_structures import hamiltonian as hamiltonian_mod
from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.mpo import MPO


def _blank_hamiltonian(**attrs: object) -> Hamiltonian:
    """Construct an uninitialized Hamiltonian for error-path tests.

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


def test_hamiltonian_rejects_representation_kwarg() -> None:
    """representation= is no longer a public constructor argument."""
    assert "representation" not in inspect.signature(Hamiltonian.__init__).parameters


def test_hamiltonian_from_manual_tensors() -> None:
    """Manual MPO cores build an encoded Hamiltonian at construction."""
    rng = np.random.default_rng(0)
    tensors = [
        rng.random(size=(1, 2, 2, 2)).astype(np.complex128),
        rng.random(size=(2, 1, 2, 2)).astype(np.complex128),
    ]
    h = Hamiltonian(tensors=tensors)
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
    assert h.length == 4
    assert h.mpo.length == 4


def test_hamiltonian_matrix_property_unavailable_for_mpo() -> None:
    """Matrix property raises for MPO-only Hamiltonian until densified."""
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    with pytest.raises(RuntimeError, match="Dense matrix is not available"):
        _ = h.matrix


def test_to_sparse_matrix_raises_without_data() -> None:
    """to_sparse_matrix raises when no backing data exists."""
    h = _blank_hamiltonian(
        _matrix=None,
        _mpo=None,
        _sparse_matrix=None,
        _tensors=None,
    )
    with pytest.raises(RuntimeError, match="no materialized data"):
        h.to_sparse_matrix()


def test_ensure_sparse_raises_without_data() -> None:
    """ensure_sparse fails when no specification is available."""
    h = _blank_hamiltonian(
        _matrix=None,
        _mpo=None,
        _sparse_matrix=None,
        _tensors=None,
    )
    with pytest.raises(ValueError, match="Cannot build sparse matrix"):
        h.ensure_sparse()


def test_ensure_mpo_raises_without_data() -> None:
    """ensure_mpo fails when no specification is available."""
    h = _blank_hamiltonian(
        _matrix=None,
        _mpo=None,
        _sparse_matrix=None,
        _tensors=None,
        physical_dimension=2,
        length=1,
    )
    with pytest.raises(ValueError, match="No Hamiltonian data available to build an MPO"):
        h.ensure_mpo()


def test_ensure_mpo_idempotent_when_already_materialized() -> None:
    """ensure_mpo returns early when MPO is already cached."""
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    mpo = h.mpo
    h.ensure_mpo()
    assert h.mpo is mpo


def test_hamiltonian_matrix_not_square() -> None:
    """Dense matrix must be square."""
    with pytest.raises(ValueError, match="square 2-D"):
        Hamiltonian(matrix=np.ones((2, 3), dtype=np.complex128))


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
    """matrix= stores dense data and infers length from Hilbert dimension."""
    mat = np.eye(4, dtype=np.complex128)
    h = Hamiltonian(matrix=mat)
    assert h.length == 2
    np.testing.assert_allclose(h.matrix, mat)


def test_hamiltonian_ising_encoded_at_init() -> None:
    """Preset classmethod encodes MPO at construction."""
    h = Hamiltonian.ising(3, J=1.0, g=0.5)
    assert h.mpo.length == 3


def test_hamiltonian_heisenberg_factory() -> None:
    """Heisenberg preset builds a valid MPO-backed Hamiltonian."""
    h = Hamiltonian.heisenberg(2, Jx=1.0, Jy=0.5, Jz=0.3, h=0.1)
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
    assert h.length == 2
    assert h.mpo.length == 2


def test_hamiltonian_from_mpo() -> None:
    """from_mpo wraps without rebuilding."""
    mpo = MPO.ising(2, J=1.0, g=0.5)
    h = Hamiltonian.from_mpo(mpo)
    assert h.mpo is mpo


def test_hamiltonian_sparse_matrix_init() -> None:
    """sparse_matrix= stores sparse data at construction."""
    dim = 4
    sparse = scipy.sparse.eye(dim, dtype=np.complex128, format="csr")
    h = Hamiltonian(sparse_matrix=sparse)
    np.testing.assert_allclose(h.sparse_matrix.toarray(), sparse.toarray())


def test_ensure_sparse_from_mpo_cached() -> None:
    """Converting MPO to sparse once caches sparse_matrix for later runs."""
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    mpo = h.mpo
    with patch.object(MPO, "to_sparse_matrix", wraps=mpo.to_sparse_matrix) as mock_sparse:
        h.ensure_sparse()
        h.ensure_sparse()
    assert mock_sparse.call_count == 1
    np.testing.assert_allclose(
        h.sparse_matrix.toarray(),
        h.to_sparse_matrix().toarray(),
    )


def test_ensure_sparse_from_dense_hamiltonian() -> None:
    """Dense-init Hamiltonian can materialize sparse for MCWF backends."""
    h = Hamiltonian(matrix=np.eye(4, dtype=np.complex128))
    h.ensure_sparse()
    np.testing.assert_allclose(h.sparse_matrix.toarray(), np.eye(4))


def test_ensure_mpo_from_dense_and_sparse() -> None:
    """Dense and sparse sources convert to MPO via MPO.from_matrix."""
    dense = Hamiltonian.ising(2, J=1.0, g=0.5).to_matrix()
    h_dense = Hamiltonian(matrix=dense.copy())
    h_dense.ensure_mpo()
    np.testing.assert_allclose(h_dense.mpo.to_matrix(), dense, atol=1e-10)

    h_sparse = Hamiltonian(sparse_matrix=scipy.sparse.csr_matrix(dense))
    h_sparse.ensure_mpo()
    np.testing.assert_allclose(h_sparse.mpo.to_matrix(), dense, atol=1e-10)
    # Sparse→MPO densifies and caches the dense form.
    np.testing.assert_allclose(h_sparse.matrix, dense, atol=1e-10)


def test_cached_forms_remain_available_after_conversions() -> None:
    """Accessors return any materialized form after ensure_mpo / ensure_sparse."""
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    mpo = h.mpo

    h.ensure_sparse()
    assert h.mpo is mpo
    np.testing.assert_allclose(h.sparse_matrix.toarray(), mpo.to_sparse_matrix().toarray())

    dense = h.to_matrix()
    assert h.mpo is mpo
    np.testing.assert_allclose(dense, mpo.to_matrix(), atol=1e-12)


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
    """to_matrix converts from MPO or sparse."""
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


def test_to_sparse_matrix_from_dense() -> None:
    """to_sparse_matrix converts from dense matrix storage."""
    h = Hamiltonian(matrix=np.eye(4, dtype=np.complex128))
    sparse = h.to_sparse_matrix()
    np.testing.assert_allclose(sparse.toarray(), np.eye(4))


def test_to_matrix_raises_without_data() -> None:
    """to_matrix raises when no backing data exists."""
    h = _blank_hamiltonian(
        _matrix=None,
        _mpo=None,
        _sparse_matrix=None,
        _tensors=None,
    )
    with pytest.raises(RuntimeError, match="no materialized data"):
        h.to_matrix()


def test_run_accepts_dense_hamiltonian_with_mps_state() -> None:
    """TJM materializes an MPO from a dense-source Hamiltonian."""
    dim = 4
    mat = np.eye(dim, dtype=np.complex128)
    h = Hamiltonian(matrix=mat)
    state = State(2, initial="zeros", representation="mps")
    params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.1,
        dt=0.1,
    )
    result = Simulator(show_progress=False).run(state, h, params, None)
    assert result.expectation_values[0].shape[0] >= 1
    _ = h.mpo


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


def test_ensure_sparse_prefers_dense_source_after_ensure_mpo() -> None:
    """After ensure_mpo, ensure_sparse still uses the original dense matrix."""
    dense = Hamiltonian.ising(2, J=1.0, g=0.5).to_matrix()
    # Perturb off-diagonals so an approximate MPO conversion can differ.
    dense = dense.copy()
    dense[0, 3] += 0.37
    dense[3, 0] += 0.37
    h = Hamiltonian(matrix=dense.copy())
    h.ensure_mpo()
    mpo = h.mpo
    with patch.object(MPO, "to_sparse_matrix", wraps=mpo.to_sparse_matrix) as mock_sparse:
        h.ensure_sparse()
    assert mock_sparse.call_count == 0
    np.testing.assert_allclose(h.sparse_matrix.toarray(), dense)


@pytest.mark.parametrize("order", ["mps_then_vector", "vector_then_mps"])
def test_dense_hamiltonian_run_order_preserves_source_fidelity(order: str) -> None:
    """Dense-source fidelity is independent of MPS vs vector run order."""
    length = 2
    dense = Hamiltonian.ising(length, J=1.0, g=0.5).to_matrix().copy()
    dense[0, 3] += 0.21
    dense[3, 0] += 0.21
    hamiltonian = Hamiltonian(matrix=dense.copy())
    sim = Simulator(show_progress=False)
    obs = Observable("z", sites=[0])
    params_mps = AnalogSimParams(observables=[obs], elapsed_time=0.2, dt=0.05, max_bond_dim=16, svd_threshold=1e-10)
    params_vec = AnalogSimParams(observables=[obs], elapsed_time=0.2, dt=0.05, num_traj=1)

    state_mps = State(length, initial="zeros", representation="mps")
    state_vec = State(length, initial="zeros", representation="vector")
    state_rho = State(length, initial="zeros", representation="density_matrix")

    if order == "mps_then_vector":
        mps_val = float(sim.run(state_mps, hamiltonian, params_mps, None).expectation_values[0][-1])
        vec_val = float(sim.run(state_vec, hamiltonian, params_vec, None).expectation_values[0][-1])
        rho_val = float(sim.run(state_rho, hamiltonian, params_vec, None).expectation_values[0][-1])
    else:
        vec_val = float(sim.run(state_vec, hamiltonian, params_vec, None).expectation_values[0][-1])
        rho_val = float(sim.run(state_rho, hamiltonian, params_vec, None).expectation_values[0][-1])
        mps_val = float(sim.run(state_mps, hamiltonian, params_mps, None).expectation_values[0][-1])

    assert vec_val == pytest.approx(rho_val, abs=1e-8)
    assert mps_val == pytest.approx(vec_val, abs=1e-4)
    np.testing.assert_allclose(hamiltonian.sparse_matrix.toarray(), dense)
    np.testing.assert_allclose(hamiltonian.matrix, dense)


def test_ensure_mpo_warns_before_large_dense_factorization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large dense→MPO conversion emits the preprocess_mcwf-style RuntimeWarning."""
    monkeypatch.setattr(hamiltonian_mod, "_LARGE_HILBERT_DIM", 2)
    h = Hamiltonian(matrix=np.eye(4, dtype=np.complex128))
    with pytest.warns(RuntimeWarning, match="factorizing a dense matrix into an MPO"):
        h.ensure_mpo()
    assert h.mpo.length == 2


def test_ensure_mpo_warns_before_large_sparse_densification(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large sparse→MPO conversion warns before densifying."""
    monkeypatch.setattr(hamiltonian_mod, "_LARGE_HILBERT_DIM", 2)
    h = Hamiltonian(sparse_matrix=scipy.sparse.eye(4, dtype=np.complex128, format="csr"))
    with pytest.warns(RuntimeWarning, match="densifying a sparse matrix to build an MPO"):
        h.ensure_mpo()
    assert h.mpo.length == 2


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


def test_piecewise_stores_static_pieces() -> None:
    """piecewise() keeps static Hamiltonians and durations."""
    first = Hamiltonian.ising(2, J=1.0, g=0.5)
    second = Hamiltonian.ising(2, J=1.0, g=2.0)
    hamiltonian = Hamiltonian.piecewise([(first, 0.1), (second, 0.2)])
    assert hamiltonian.is_piecewise
    assert hamiltonian.length == 2
    assert hamiltonian.pieces == ((first, 0.1), (second, 0.2))
    assert hamiltonian.duration == pytest.approx(0.3)


def test_static_hamiltonian_has_no_piecewise_duration() -> None:
    """Duration is defined only for piecewise Hamiltonians."""
    with pytest.raises(ValueError, match="do not have a piecewise duration"):
        _ = Hamiltonian.ising(2, J=1.0, g=0.5).duration


def test_piecewise_rejects_empty_or_nested_or_mismatched_pieces() -> None:
    """Construction rejects empty, nested, and length-mismatched pieces."""
    static = Hamiltonian.ising(2, J=1.0, g=0.5)
    with pytest.raises(ValueError, match="non-empty sequence"):
        Hamiltonian.piecewise([])
    with pytest.raises(TypeError, match="non-empty sequence"):
        Hamiltonian.piecewise(cast("Any", 0))
    with pytest.raises(TypeError, match="non-empty sequence"):
        Hamiltonian.piecewise(cast("Any", "pairs"))
    with pytest.raises(TypeError, match="non-empty sequence"):
        Hamiltonian.piecewise(cast("Any", b"pairs"))
    nested = Hamiltonian.piecewise([(static, 0.1)])
    with pytest.raises(ValueError, match="nested piecewise"):
        Hamiltonian.piecewise([(nested, 0.1)])
    with pytest.raises(TypeError, match="must be a \\(Hamiltonian, duration\\) tuple"):
        Hamiltonian.piecewise(cast("Any", [(static, 0.1, 0.0)]))
    with pytest.raises(TypeError, match="must start with a Hamiltonian"):
        Hamiltonian.piecewise(cast("Any", [(object(), 0.1)]))
    with pytest.raises(TypeError, match="duration must be a real number"):
        Hamiltonian.piecewise(cast("Any", [(static, "0.1")]))
    with pytest.raises(TypeError, match="duration must be a real number"):
        Hamiltonian.piecewise(cast("Any", [(static, True)]))
    with pytest.raises(ValueError, match="does not match piece 0 length"):
        Hamiltonian.piecewise([(static, 0.1), (Hamiltonian.ising(3, J=1.0, g=0.5), 0.1)])
    with pytest.raises(ValueError, match="finite and positive"):
        Hamiltonian.piecewise([(static, 0.0)])


def test_piecewise_cannot_materialize_a_single_operator() -> None:
    """A piecewise Hamiltonian is not one static MPO or sparse matrix."""
    hamiltonian = Hamiltonian.piecewise([(Hamiltonian.ising(2, J=1.0, g=0.5), 0.1)])
    with pytest.raises(ValueError, match="no single static operator"):
        hamiltonian.ensure_mpo()
    with pytest.raises(ValueError, match="no single static operator"):
        hamiltonian.ensure_sparse()
    with pytest.raises(ValueError, match="do not have piecewise durations"):
        _ = Hamiltonian.ising(2, J=1.0, g=0.5).pieces
