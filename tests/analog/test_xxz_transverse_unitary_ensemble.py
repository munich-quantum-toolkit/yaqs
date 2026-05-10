# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end XXZ+transverse-field unitary-ensemble checks against dense ED."""

from __future__ import annotations

import numpy as np

from mqt.yaqs import simulator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z


def _embed_one_site_operator(length: int, site: int, op2: np.ndarray) -> np.ndarray:
    left_dim = 2**site
    right_dim = 2 ** (length - site - 1)
    return np.kron(np.kron(np.eye(left_dim, dtype=np.complex128), op2), np.eye(right_dim, dtype=np.complex128))


def _embed_two_site_operator(length: int, site_left: int, op4: np.ndarray) -> np.ndarray:
    left_dim = 2**site_left
    right_dim = 2 ** (length - site_left - 2)
    return np.kron(np.kron(np.eye(left_dim, dtype=np.complex128), op4), np.eye(right_dim, dtype=np.complex128))


def _build_xxz_tf_open_chain_dense(length: int, j_xy: float, delta: float, h_x: float) -> np.ndarray:
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    xx = np.kron(x, x)
    yy = np.kron(y, y)
    zz = np.kron(z, z)
    dim = 2**length
    h_dense = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(length - 1):
        h_dense += 0.25 * j_xy * _embed_two_site_operator(length, i, xx)
        h_dense += 0.25 * j_xy * _embed_two_site_operator(length, i, yy)
        h_dense += 0.25 * delta * _embed_two_site_operator(length, i, zz)
    for r in range(length):
        h_dense += 0.5 * h_x * _embed_one_site_operator(length, r, x)
    return h_dense


def _ed_first_k_basis_two_time_mean(
    *,
    length: int,
    j_xy: float,
    delta: float,
    h_x: float,
    times: np.ndarray,
    k: int,
    site: int,
    a2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Return ``(1/k) sum_n <n|U^†(t) A U(t) B|n>`` over first ``k`` basis kets."""
    dim = 2**length
    k_eff = min(k, dim)
    h_dense = _build_xxz_tf_open_chain_dense(length, j_xy, delta, h_x)
    a = _embed_one_site_operator(length, site, a2)
    b = _embed_one_site_operator(length, site, b2)

    evals, evecs = np.linalg.eigh(h_dense)
    evecs_h = evecs.conj().T
    out = np.zeros(len(times), dtype=np.complex128)
    for t_idx, t in enumerate(times):
        phases = np.exp(-1j * evals * t)
        u_t = (evecs * phases[np.newaxis, :]) @ evecs_h
        m = u_t.conj().T @ a @ u_t @ b
        out[t_idx] = np.sum(np.diag(m)[:k_eff]) / float(k_eff)
    return out


def test_xxz_transverse_unitary_ensemble_pauli_and_two_time_vs_ed() -> None:
    """L=6, first-20 basis-state ensemble should match dense ED for 3 Pauli and one mixed two-time pair."""
    length = 6
    j_xy = 1.0
    delta = 0.7
    h_x = 0.5
    t_final = 2.0
    dt = 0.05
    k = 64
    mid = length // 2

    h_mpo = MPO.hamiltonian(
        length=length,
        two_body=[(0.25 * j_xy, "X", "X"), (0.25 * j_xy, "Y", "Y"), (0.25 * delta, "Z", "Z")],
        one_body=[(0.5 * h_x, "X")],
        bc="open",
    )
    states = [MPS(length, state="basis", basis_string=format(i, f"0{length}b")) for i in range(k)]

    ox = Observable(X(), mid)
    oy = Observable(Y(), mid)
    oz = Observable(Z(), mid)
    pairs = [(ox, ox), (oy, oy), (oz, oz), (oz, ox)]

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=t_final,
        dt=dt,
        max_bond_dim=4096,
        threshold=1e-12,
        sample_timesteps=True,
        show_progress=False,
        two_time_correlators=pairs,
    )
    simulator.run(states, h_mpo, sim_params, noise_model=None, parallel=False)
    assert sim_params.two_time_correlator_results is not None
    yaqs = np.asarray(sim_params.two_time_correlator_results, dtype=np.complex128)
    times = np.asarray(sim_params.two_time_correlator_times, dtype=np.float64)

    x2 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y2 = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    z2 = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)

    ed_xx = _ed_first_k_basis_two_time_mean(
        length=length, j_xy=j_xy, delta=delta, h_x=h_x, times=times, k=k, site=mid, a2=x2, b2=x2
    )
    ed_yy = _ed_first_k_basis_two_time_mean(
        length=length, j_xy=j_xy, delta=delta, h_x=h_x, times=times, k=k, site=mid, a2=y2, b2=y2
    )
    ed_zz = _ed_first_k_basis_two_time_mean(
        length=length, j_xy=j_xy, delta=delta, h_x=h_x, times=times, k=k, site=mid, a2=z2, b2=z2
    )
    ed_zx = _ed_first_k_basis_two_time_mean(
        length=length, j_xy=j_xy, delta=delta, h_x=h_x, times=times, k=k, site=mid, a2=z2, b2=x2
    )
    ed = np.vstack([ed_xx, ed_yy, ed_zz, ed_zx])

    max_abs_per_pair = np.max(np.abs(yaqs - ed), axis=1)
    assert np.all(max_abs_per_pair < 1e-5), f"pairwise max errors were {max_abs_per_pair!r}"
