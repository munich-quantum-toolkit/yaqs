from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.process_tensors.tomography.data import SequenceData


def test_to_dense_sequence_data_minimal() -> None:
    """Smoke test SequenceData.to_dense_comb on minimal data."""
    rho = np.eye(2, dtype=np.complex128)
    seqs = [(0,)]
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


def test_to_mpo_sequence_data_minimal() -> None:
    """Smoke test SequenceData.to_mpo_comb on minimal data."""
    rho = np.eye(2, dtype=np.complex128)
    seqs = [(0,)]
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
