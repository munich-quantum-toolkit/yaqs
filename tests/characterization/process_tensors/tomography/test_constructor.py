from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs import construct_process_tensor
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_construct_process_tensor_invalid_return_type_raises() -> None:
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)
    with pytest.raises(ValueError):
        construct_process_tensor(op, params, timesteps=[0.0], return_type="nope")  # type: ignore[arg-type]


def test_construct_process_tensor_returns_dense_and_mpo_smoke() -> None:
    op = MPO.ising(length=1, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, max_bond_dim=8)

    dense = construct_process_tensor(op, params, timesteps=[0.0], parallel=False, return_type="dense")
    assert dense.to_matrix().shape == (8, 8)

    mpo = construct_process_tensor(op, params, timesteps=[0.0], parallel=False, return_type="mpo", compress_every=1)
    mat = mpo.to_matrix()
    assert mat.shape == (8, 8)
    np.testing.assert_allclose(mat, dense.to_matrix(), atol=1e-8)

