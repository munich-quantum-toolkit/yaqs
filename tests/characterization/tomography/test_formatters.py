from __future__ import annotations

import numpy as np

from mqt.yaqs.characterization.tomography.exact.formatters import (
    _to_dense,
    _to_mpo,
    rank1_upsilon_mpo_term,
)
from mqt.yaqs.characterization.tomography.estimate.sampling import SamplingData, SequenceData


def test_rank1_upsilon_mpo_term_shapes() -> None:
    rho = np.eye(2, dtype=np.complex128)
    dual_ops = [np.eye(4, dtype=np.complex128)]
    mpo = rank1_upsilon_mpo_term(rho, dual_ops, weight=1.0)
    mat = mpo.to_matrix()
    assert mat.shape == (2 * 4, 2 * 4)


def test_to_dense_sequence_data_minimal() -> None:
    """Smoke test _to_dense on minimal SequenceData."""
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
    est = _to_dense(data)
    # Either dense_choi or tensor is populated
    assert est.dense_choi is not None or est.tensor is not None
    assert est.timesteps == timesteps


def test_to_mpo_sampling_data_minimal() -> None:
    """Smoke test _to_mpo on minimal SamplingData."""
    rho = np.eye(2, dtype=np.complex128)
    duals = [np.eye(4, dtype=np.complex128)]
    outputs = [rho]
    dual_ops = [duals]
    weights = [1.0]
    timesteps = [0.1]

    data = SamplingData(
        outputs=outputs,
        dual_ops=dual_ops,
        weights=weights,
        timesteps=timesteps,
    )
    mpo = _to_mpo(data, compress_every=1)
    mat = mpo.to_matrix()
    assert mat.shape == (2 * 4, 2 * 4)