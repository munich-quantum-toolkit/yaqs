# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :class:`mqt.yaqs.core.data_structures.hamiltonian.Hamiltonian`."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import scipy.sparse

from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.simulator import run


def test_hamiltonian_ising_encoded_at_init() -> None:
    """Preset classmethod encodes MPO at construction."""
    h = Hamiltonian.ising(3, J=1.0, g=0.5)
    assert h.representation == "mpo"
    assert h.mpo.length == 3


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
        show_progress=False,
    )
    with pytest.raises(ValueError, match=r"TJM simulation requires Hamiltonian\.representation='mpo'"):
        run(state, h, params, None)


def test_run_hamiltonian_length_mismatch() -> None:
    """State and Hamiltonian lengths must match."""
    state = State(3, initial="zeros")
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.1,
        dt=0.1,
        show_progress=False,
    )
    with pytest.raises(ValueError, match=r"does not match Hamiltonian\.length"):
        run(state, h, params, None)


def test_to_sparse_matrix_called_once_across_two_runs() -> None:
    """Outer loops reuse the same Hamiltonian without re-sparsifying."""
    state = State(2, initial="zeros", representation="vector")
    h = Hamiltonian.ising(2, J=1.0, g=0.5)
    params = AnalogSimParams(
        observables=[Observable("z", sites=[0])],
        elapsed_time=0.1,
        dt=0.1,
        show_progress=False,
    )
    mpo = h.mpo
    with patch.object(MPO, "to_sparse_matrix", wraps=mpo.to_sparse_matrix) as mock_sparse:
        run(state, h, params, None)
        run(state, h, params, None)
    assert mock_sparse.call_count == 1
