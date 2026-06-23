# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Cross-representation qubit-ordering regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs import AnalogSimParams, Hamiltonian, NoiseModel, Observable, Simulator, State
from mqt.yaqs.core.data_structures.mps import MPS
from mqt.yaqs.core.data_structures.state_utils import embed_one_site_operator


@pytest.fixture
def haar_state() -> tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Haar-random 3-qubit MPS and dense snapshots.

    Returns:
        Tuple of ``(mps, psi, rho, tensors)``.
    """
    length = 3
    mps = MPS(length, state="haar-random", pad=4)
    psi = np.asarray(mps.to_vec(), dtype=np.complex128)
    rho = np.outer(psi, psi.conj())
    tensors = [np.asarray(t, dtype=np.complex128).copy() for t in mps.tensors]
    return mps, psi, rho, tensors


def test_haar_embedded_observables_match_mps(haar_state: tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]]) -> None:
    """Dense MCWF/Lindblad embeddings agree with ``MPS.expect`` on entangled states."""
    mps, psi, _rho, _tensors = haar_state
    length = mps.length
    for site in range(length):
        for name in ("x", "z"):
            obs = Observable(name, site)
            mps_val = mps.expect(obs)
            op = embed_one_site_operator(np.asarray(obs.gate.matrix, dtype=np.complex128), length, site)
            embed_val = float(np.real(np.vdot(psi, op @ psi)))
            assert mps_val == pytest.approx(embed_val, abs=1e-9), f"{name} site {site}"


def test_noiseless_evolution_agrees_across_representations(
    haar_state: tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]],
) -> None:
    """MPS, MCWF, and Lindblad paths agree on noiseless Ising observables."""
    _mps, psi, rho, tensors = haar_state
    length = len(tensors)
    sim = Simulator(show_progress=False)
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)

    obs_list = [Observable("z", s) for s in range(length)] + [Observable("x", 0)]
    params_mps = AnalogSimParams(observables=obs_list, elapsed_time=0.5, dt=0.05, max_bond_dim=32, svd_threshold=1e-10)
    params_dense = AnalogSimParams(observables=obs_list, elapsed_time=0.5, dt=0.05, num_traj=1)

    z_x_mps = sim.run(State(length, tensors=[t.copy() for t in tensors]), hamiltonian, params_mps, None)
    z_x_vec = sim.run(State(vector=psi.copy()), hamiltonian, params_dense, None)
    z_x_rho = sim.run(State(density_matrix=rho.copy()), hamiltonian, params_dense, None)

    for idx in range(len(obs_list)):
        mps_val = float(z_x_mps.expectation_values[idx][-1])
        vec_val = float(z_x_vec.expectation_values[idx][-1])
        rho_val = float(z_x_rho.expectation_values[idx][-1])
        assert vec_val == pytest.approx(rho_val, abs=1e-8), f"obs {idx} vector vs density_matrix"
        assert mps_val == pytest.approx(vec_val, abs=1e-5), f"obs {idx} mps vs vector"


def test_noisy_short_step_mps_vs_mcwf(
    haar_state: tuple[MPS, np.ndarray, np.ndarray, list[np.ndarray]],
) -> None:
    """Noisy one-step X expectation stays aligned between MPS and MCWF."""
    _mps, psi, _rho, tensors = haar_state
    length = len(tensors)
    sim = Simulator(show_progress=False)
    hamiltonian = Hamiltonian.ising(length, J=1.0, g=0.5)
    noise = NoiseModel([{"name": "pauli_z", "sites": [i], "strength": 0.2} for i in range(length)])
    obs = Observable("x", sites=[0])
    params = AnalogSimParams(
        observables=[obs],
        elapsed_time=0.05,
        dt=0.05,
        num_traj=1,
        max_bond_dim=16,
        random_seed=7,
    )

    mps_val = float(
        sim.run(State(length, tensors=[t.copy() for t in tensors]), hamiltonian, params, noise).expectation_values[0][
            -1
        ]
    )
    vec_val = float(sim.run(State(vector=psi.copy()), hamiltonian, params, noise).expectation_values[0][-1])
    assert mps_val == pytest.approx(vec_val, abs=1e-6)
