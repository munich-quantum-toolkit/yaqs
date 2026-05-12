# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""End-to-end spin-chain ensemble checks against dense ED and worker paths."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs import simulator
from mqt.yaqs.analog.ensemble import ensemble_member_worker
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import BaseGate, X, Y, Z


def _embed_one_site_operator(length: int, site: int, op2: np.ndarray) -> np.ndarray:
    left_dim = 2**site
    right_dim = 2 ** (length - site - 1)
    return np.kron(np.kron(np.eye(left_dim, dtype=np.complex128), op2), np.eye(right_dim, dtype=np.complex128))


def _embed_two_site_operator(length: int, site_left: int, op4: np.ndarray) -> np.ndarray:
    left_dim = 2**site_left
    right_dim = 2 ** (length - site_left - 2)
    return np.kron(np.kron(np.eye(left_dim, dtype=np.complex128), op4), np.eye(right_dim, dtype=np.complex128))


def _swap_gate_4() -> np.ndarray:
    """Two-qubit SWAP in lexicographic ``|ab>`` basis.

    Returns:
        The ``4x4`` SWAP matrix in the lexicographic two-qubit basis.
    """
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)


def _permuted_periodic_wrap_gate(gate4: np.ndarray) -> np.ndarray:
    """Permute wrap gate from ``|q_{L-1}, q_0>`` into merged nearest-neighbor ordering.

    Args:
        gate4: Two-qubit gate matrix in the periodic-wrap bond convention.

    Returns:
        The permuted ``4x4`` matrix acting on the merged nearest-neighbor basis at ``(L-2, L-1)``.
    """
    p_perm = np.zeros((4, 4), dtype=np.complex128)
    for a in range(2):
        for b in range(2):
            idx_merged = 2 * a + b
            idx_bond = 2 * b + a
            p_perm[idx_bond, idx_merged] = 1.0
    g = np.asarray(gate4, dtype=np.complex128)
    return p_perm.conj().T @ g @ p_perm


def _dense_embed_periodic_wrap_two_site(length: int, gate4: np.ndarray) -> np.ndarray:
    """Dense embedding of a two-site gate on periodic bond ``(q_{L-1}, q_0)``.

    Args:
        length: Number of sites.
        gate4: Two-qubit ``4 x 4`` gate on the wrap bond.

    Returns:
        Dense ``2^L`` by ``2^L`` complex matrix.
    """
    g = np.asarray(gate4, dtype=np.complex128)
    if length <= 2:
        return np.asarray(g, dtype=np.complex128)
    dim = 2**length
    sw = _swap_gate_4()
    u_fwd = np.eye(dim, dtype=np.complex128)
    for i in range(length - 2):
        u_fwd = _embed_two_site_operator(length, i, sw) @ u_fwd
    g_merged = _permuted_periodic_wrap_gate(g)
    g_nn = _embed_two_site_operator(length, length - 2, g_merged)
    return np.asarray(u_fwd.conj().T @ g_nn @ u_fwd, dtype=np.complex128)


def _spin_current_bond_matrix(j_coupling: float) -> np.ndarray:
    """Construct XY-derived spin-current bond operator.

    Args:
        j_coupling: XY coupling strength scaling the bond operator.

    Returns:
        The ``4x4`` complex spin-current bond matrix.
    """
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    return 0.25 * j_coupling * (np.kron(x, y) - np.kron(y, x))


def _periodic_bond_endpoints(length: int) -> list[tuple[int, int]]:
    """Generate nearest-neighbor bond pairs with periodic boundary conditions.

    Args:
        length: Number of sites.

    Returns:
        List of ``(site_a, site_b)`` tuples with ``site_b = (site_a + 1) % length``.
    """
    return [(i, (i + 1) % length) for i in range(length)]


def _spin_current_observable_for_periodic_bond(site_a: int, site_b: int, j_xy: float) -> Observable:
    """Create an Observable for spin-current on one periodic bond.

    Args:
        site_a: First site index.
        site_b: Second site index.
        j_xy: XY coupling strength.

    Returns:
        Observable wrapping the spin-current bond matrix on ``(site_a, site_b)``.
    """
    return Observable(BaseGate(_spin_current_bond_matrix(j_xy)), sites=[site_a, site_b])


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
    r"""Compute the dense ED mean two-time correlator over the first ``k`` basis states.

    Args:
        length: Chain length.
        j_xy: XY coupling strength.
        delta: ZZ anisotropy coupling.
        h_x: Transverse-field strength.
        times: Time grid for evaluating the correlator.
        k: Number of computational-basis states to average.
        site: Site index where one-site operators are embedded.
        a2: Dense ``2x2`` probe operator :math:`A`.
        b2: Dense ``2x2`` kick operator :math:`B`.

    Returns:
        numpy.ndarray: Complex vector with values of
        :math:`(1/k)\\sum_n \\langle n|U^\\dagger(t) A U(t) B|n\rangle`
        at each entry of ``times``.
    """
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
    """Match YAQS two-time correlators against dense ED for a transverse-field XXZ setup."""
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
    assert sim_params.two_time_correlator_times is not None, "Expected two-time correlator time grid to be set."
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


def test_two_time_correlator_probe_row_diagonal_matches_expectation_at_t0() -> None:
    """At ``t=0``, two-time block diagonal entry ``(r,r)`` must match dense ``<psi|j_r j_r|psi>``."""
    length = 4
    j_xy = 1.0
    delta = 0.8
    mps = MPS(length, state="random", pad=8)
    mps.normalize("B")

    row = tuple(_spin_current_observable_for_periodic_bond(a, b, j_xy) for a, b in _periodic_bond_endpoints(length))
    s_index = 1
    obs_s = row[s_index]
    pairs = [(obs_r, obs_s) for obs_r in row]

    h = MPO.hamiltonian(
        length=length,
        two_body=[(0.25 * j_xy, "X", "X"), (0.25 * j_xy, "Y", "Y"), (0.25 * delta, "Z", "Z")],
        bc="periodic",
    )

    sim_params = AnalogSimParams(
        observables=[],
        elapsed_time=0.0,
        dt=0.5,
        max_bond_dim=32,
        threshold=1e-10,
        order=1,
        sample_timesteps=True,
        show_progress=False,
        compute_autocorrelator=False,
        two_time_correlators=pairs,
    )

    _, _, mat = ensemble_member_worker((0, mps, sim_params, h))
    assert mat is not None
    val_worker = float(np.real(mat[s_index, 0]))

    psi = np.asarray(mps.to_vec(), dtype=np.complex128)
    j_bond = _spin_current_bond_matrix(j_xy)
    jr = _embed_two_site_operator(length, 1, j_bond)
    expected = float(np.real(np.vdot(psi, jr @ jr @ psi)))
    assert val_worker == pytest.approx(expected, rel=0, abs=1e-6)
