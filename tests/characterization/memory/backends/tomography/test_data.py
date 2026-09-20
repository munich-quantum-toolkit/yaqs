# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for tomography SequenceData process-tensor conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from mqt.yaqs.characterization.memory.backends.tomography.data import (
    SequenceData,
    accumulate_rank1_terms,
    assemble_upsilon,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mqt.yaqs.core.data_structures.mpo import MPO

_REF_RHO0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)


def test_to_dense_sequence_data_minimal() -> None:
    """Smoke test SequenceData.to_dense_process_tensor on minimal data."""
    rho = np.eye(2, dtype=np.complex128)
    seqs: list[tuple[int, ...]] = [(0,)]
    outputs = [rho]
    weights = [1.0]
    choi_basis = [np.eye(4, dtype=np.complex128)] * 16
    choi_indices = [(0, 0)] * 16
    choi_duals = [np.eye(4, dtype=np.complex128)] * 16
    timesteps = [0.1, 0.1]

    data = SequenceData(
        sequences=seqs,
        outputs=outputs,
        weights=weights,
        choi_basis=choi_basis,
        choi_indices=choi_indices,
        choi_duals=choi_duals,
        timesteps=timesteps,
        initial_rho=_REF_RHO0,
    )
    pt = data.to_dense_process_tensor(check=False)
    mat = pt.to_matrix()
    assert mat.shape == (2 * 4, 2 * 4)
    assert pt.timesteps == timesteps


def test_to_dense_sequence_data_zero_step_weighted() -> None:
    """num_interventions=0 reconstruction applies the scalar sequence weight before returning rho."""
    rho = np.eye(2, dtype=np.complex128)
    choi = [np.eye(4, dtype=np.complex128)] * 16
    out_vecs = rho.reshape(-1)
    seq_weights = np.array(0.25, dtype=np.float64)
    rho_w = assemble_upsilon(
        out_vecs=out_vecs,
        seq_weights=seq_weights,
        dual_ops=choi,
        basis_ops=choi,
        check=False,
        atol=1e-8,
    )
    np.testing.assert_allclose(rho_w, 0.25 * rho, atol=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"num_steps": 0}, ValueError, r"num_steps must be >= 1"),
        ({"num_steps": -1}, ValueError, r"num_steps must be >= 1"),
        ({"num_steps": False}, TypeError, r"num_steps must be an integer"),
        ({"num_steps": 1.5}, TypeError, r"num_steps must be an integer"),
        ({"num_steps": "1"}, TypeError, r"num_steps must be an integer"),
        ({"compress_every": 0}, ValueError, r"compress_every must be >= 1"),
        ({"compress_every": -1}, ValueError, r"compress_every must be >= 1"),
        ({"compress_every": False}, TypeError, r"compress_every must be an integer"),
        ({"compress_every": 1.5}, TypeError, r"compress_every must be an integer"),
        ({"compress_every": "1"}, TypeError, r"compress_every must be an integer"),
        ({"max_bond_dim": 0}, ValueError, r"max_bond_dim must be >= 1"),
        ({"max_bond_dim": -1}, ValueError, r"max_bond_dim must be >= 1"),
        ({"max_bond_dim": False}, TypeError, r"max_bond_dim must be an integer"),
        ({"max_bond_dim": 1.5}, TypeError, r"max_bond_dim must be an integer"),
        ({"max_bond_dim": "1"}, TypeError, r"max_bond_dim must be an integer"),
        ({"n_sweeps": -1}, ValueError, r"n_sweeps must be >= 0"),
        ({"n_sweeps": False}, TypeError, r"n_sweeps must be an integer"),
        ({"n_sweeps": 1.5}, TypeError, r"n_sweeps must be an integer"),
        ({"n_sweeps": "0"}, TypeError, r"n_sweeps must be an integer"),
    ],
)
def test_accumulate_rank1_terms_validates_sizes(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    """Rank-1 accumulation rejects invalid sizes before consuming terms."""

    def fail_if_consumed() -> Iterator[MPO]:
        pytest.fail("terms were consumed before size validation")
        yield from ()

    options: dict[str, object] = {"num_steps": 1, **kwargs}
    with pytest.raises(error, match=match):
        accumulate_rank1_terms(fail_if_consumed(), **options)  # ty: ignore[invalid-argument-type]


def test_accumulate_rank1_terms_accepts_zero_sweeps() -> None:
    """Zero compression sweeps are a valid direct-MPO boundary value."""
    result = accumulate_rank1_terms(
        [],
        num_steps=np.int64(1),  # ty: ignore[invalid-argument-type]
        compress_every=np.int64(1),  # ty: ignore[invalid-argument-type]
        max_bond_dim=np.int64(2),  # ty: ignore[invalid-argument-type]
        n_sweeps=np.int64(0),  # ty: ignore[invalid-argument-type]
    )
    assert result.length == 2
