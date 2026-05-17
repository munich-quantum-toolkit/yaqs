# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :mod:`mqt.yaqs.core.data_structures.hamiltonian_utils`."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from mqt.yaqs.core.data_structures.hamiltonian import Hamiltonian
from mqt.yaqs.core.data_structures.hamiltonian_utils import (
    attach_mpo,
    sparse_to_csr,
    validate_representation,
)
from mqt.yaqs.core.data_structures.mpo import MPO


def test_sparse_to_csr_returns_same_csr() -> None:
    """CSR input is returned without copying."""
    csr = scipy.sparse.eye(2, dtype=np.complex128, format="csr")
    assert sparse_to_csr(csr) is csr


def test_sparse_to_csr_converts_coo() -> None:
    """Non-CSR sparse formats are converted to CSR."""
    coo = scipy.sparse.eye(2, dtype=np.complex128, format="coo")
    out = sparse_to_csr(coo)
    assert isinstance(out, scipy.sparse.csr_matrix)


def test_validate_representation_accepts_known() -> None:
    """Known representation labels are returned unchanged."""
    assert validate_representation("mpo") == "mpo"


def test_validate_representation_rejects_unknown() -> None:
    """Invalid representation labels raise ValueError."""
    with pytest.raises(ValueError, match=r"Invalid representation 'bad'"):
        validate_representation("bad")


def test_attach_mpo_initializes_wrapped_hamiltonian() -> None:
    """attach_mpo wires an existing MPO into a blank Hamiltonian instance."""
    mpo = MPO.ising(2, J=1.0, g=0.5)
    wrapped = Hamiltonian.__new__(Hamiltonian)
    attach_mpo(wrapped, mpo)
    assert wrapped.mpo is mpo
    assert wrapped.representation == "mpo"
    assert wrapped.length == 2
