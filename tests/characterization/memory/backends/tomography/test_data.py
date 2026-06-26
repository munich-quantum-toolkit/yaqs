# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for tomography SequenceData comb conversion."""

from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.memory.backends.tomography.data import SequenceData, assemble_upsilon


def test_to_dense_sequence_data_minimal() -> None:
    """Smoke test SequenceData.to_dense_comb on minimal data."""
    rho = np.eye(2, dtype=np.complex128)
    seqs: list[tuple[int, ...]] = [(0,)]
    outputs = [rho]
    weights = [1.0]
    choi_basis = [np.eye(4, dtype=np.complex128)] * 16
    choi_indices = [(0, 0)] * 16
    choi_duals = [np.eye(4, dtype=np.complex128)] * 16
    timesteps = [0.1]

    data = SequenceData(
        sequences=seqs,
        outputs=outputs,
        weights=weights,
        choi_basis=choi_basis,
        choi_indices=choi_indices,
        choi_duals=choi_duals,
        timesteps=timesteps,
    )
    comb = data.to_dense_comb(check=False)
    mat = comb.to_matrix()
    assert mat.shape == (2 * 4, 2 * 4)
    assert comb.timesteps == timesteps


def test_to_dense_sequence_data_zero_step_weighted() -> None:
    """k=0 reconstruction applies the scalar sequence weight before returning rho."""
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


def test_to_mpo_sequence_data_minimal() -> None:
    """Smoke test SequenceData.to_mpo_comb on minimal data."""
    rho = np.eye(2, dtype=np.complex128)
    seqs: list[tuple[int, ...]] = [(0,)]
    outputs = [rho]
    weights = [1.0]
    choi_basis = [np.eye(4, dtype=np.complex128)] * 16
    choi_indices = [(0, 0)] * 16
    choi_duals = [np.eye(4, dtype=np.complex128)] * 16
    timesteps = [0.1]

    data = SequenceData(
        sequences=seqs,
        outputs=outputs,
        weights=weights,
        choi_basis=choi_basis,
        choi_indices=choi_indices,
        choi_duals=choi_duals,
        timesteps=timesteps,
    )
    comb = data.to_mpo_comb(compress_every=1)
    mat = comb.to_matrix()
    assert mat.shape == (2 * 4, 2 * 4)
