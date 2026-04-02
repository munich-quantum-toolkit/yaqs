from __future__ import annotations

import numpy as np

from mqt.yaqs import construct_process_tensor
from mqt.yaqs.core.data_structures.networks import MPO
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams


def test_construct_process_tensor_returns_densecomb() -> None:
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, solver="TJM", show_progress=False, max_bond_dim=16)
    comb = construct_process_tensor(
        op,
        params,
        timesteps=[0.1],
        parallel=False,
        return_type="dense",
        check=True,
    )

    U = comb.to_matrix()
    assert U.shape == (2 * 4, 2 * 4)
    np.testing.assert_allclose(U, U.conj().T, atol=1e-8)


def test_construct_process_tensor_returns_mpocomb() -> None:
    op = MPO.ising(length=2, J=0.0, g=0.0)
    params = AnalogSimParams(dt=0.1, solver="TJM", show_progress=False, max_bond_dim=16)
    comb = construct_process_tensor(
        op,
        params,
        timesteps=[0.1],
        parallel=False,
        return_type="mpo",
        compress_every=1,
    )
    U = comb.to_matrix()
    assert U.shape == (2 * 4, 2 * 4)

