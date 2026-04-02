from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.tomography.process_tensor.build import (
    build_mpo_comb,
    rank1_upsilon_mpo_term,
    sequence_to_dense_comb,
)
from mqt.yaqs.characterization.tomography.process_tensor.data import SequenceData


def test_rank1_upsilon_mpo_term_shapes() -> None:
    rho = np.eye(2, dtype=np.complex128)
    dual_ops = [np.eye(4, dtype=np.complex128)]
    mpo = rank1_upsilon_mpo_term(rho, dual_ops, weight=1.0)
    mat = mpo.to_matrix()
    assert mat.shape == (2 * 4, 2 * 4)


def test_to_dense_sequence_data_minimal() -> None:
    """Smoke test sequence_to_dense_comb on minimal SequenceData."""
    rho = np.eye(2, dtype=np.complex128)
    seqs = [(0,)]
    outputs = [rho]
    weights = [1.0]
    choi_basis = [np.eye(4, dtype=np.complex128)]
    choi_indices = [(0, 0)]
    choi_duals = [np.eye(4, dtype=np.complex128)]
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
    comb = sequence_to_dense_comb(data, check=False)
    mat = comb.to_matrix()
    assert mat.shape == (2 * 4, 2 * 4)
    assert comb.timesteps == timesteps


def test_to_mpo_sequence_data_minimal() -> None:
    """Smoke test build_mpo_comb on minimal SequenceData."""
    rho = np.eye(2, dtype=np.complex128)
    seqs = [(0,)]
    outputs = [rho]
    weights = [1.0]
    choi_basis = [np.eye(4, dtype=np.complex128)]
    choi_indices = [(0, 0)]
    choi_duals = [np.eye(4, dtype=np.complex128)]
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
    mpo = build_mpo_comb(data, compress_every=1)
    mat = mpo.to_matrix()
    assert mat.shape == (2 * 4, 2 * 4)
