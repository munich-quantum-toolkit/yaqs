# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for periodic-wrap two-site application in the analog autocorrelator."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.analog.autocorrelator import apply_observable_inplace, mixed_expectation
from mqt.yaqs.analog.unitary_ensemble import unitary_ensemble_member_worker
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import BaseGate


def _swap_gate_4() -> np.ndarray:
    """Two-qubit SWAP in lexicographic ``|ab⟩`` basis.

    Returns:
        The ``4x4`` SWAP matrix.
    """
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)


def _permuted_periodic_wrap_gate(gate4: np.ndarray) -> np.ndarray:
    """Same convention as :func:`mqt.yaqs.analog.autocorrelator._permuted_periodic_wrap_gate`.

    Returns:
        Permuted ``4x4`` matrix acting on the merged NN basis at ``(L-2, L-1)``.
    """
    p_perm = np.zeros((4, 4), dtype=np.complex128)
    for a in range(2):
        for b in range(2):
            idx_merged = 2 * a + b
            idx_bond = 2 * b + a
            p_perm[idx_bond, idx_merged] = 1.0
    g = np.asarray(gate4, dtype=np.complex128)
    return p_perm.conj().T @ g @ p_perm


def dense_embed_adjacent_two_site(length: int, site_left: int, gate4: np.ndarray) -> np.ndarray:
    r"""Embed a ``4 x 4`` NN gate on ``(site_left, site_left+1)`` into ``(C^2)^{\otimes L}``.

    Uses the same qubit order as :meth:`MPS.to_vec` (site ``0`` leftmost).

    Returns:
        Dense ``2^L`` by ``2^L`` complex matrix.
    """
    left_dim = 2**site_left
    right_dim = 2 ** (length - site_left - 2)
    op4 = np.asarray(gate4, dtype=np.complex128)
    return np.asarray(
        np.kron(
            np.kron(np.eye(left_dim, dtype=np.complex128), op4),
            np.eye(right_dim, dtype=np.complex128),
        ),
        dtype=np.complex128,
    )


def dense_embed_periodic_wrap_two_site(length: int, gate4: np.ndarray) -> np.ndarray:
    """Dense embedding of ``gate4`` on ``(q_{L-1}, q_0)``.

    Matches :func:`~mqt.yaqs.analog.autocorrelator.apply_observable_inplace` conventions.

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
        u_fwd = dense_embed_adjacent_two_site(length, i, sw) @ u_fwd
    g_merged = _permuted_periodic_wrap_gate(g)
    g_nn = dense_embed_adjacent_two_site(length, length - 2, g_merged)
    return np.asarray(u_fwd.conj().T @ g_nn @ u_fwd, dtype=np.complex128)


def _spin_current_bond_matrix(j_coupling: float) -> np.ndarray:
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    return 0.25 * j_coupling * (np.kron(x, y) - np.kron(y, x))


def _periodic_bond_endpoints(length: int) -> list[tuple[int, int]]:
    return [(i, (i + 1) % length) for i in range(length)]


def _spin_current_observable_for_periodic_bond(site_a: int, site_b: int, j_xy: float) -> Observable:
    return Observable(BaseGate(_spin_current_bond_matrix(j_xy)), sites=[site_a, site_b])


def test_wrap_expectation_matches_dense() -> None:
    """Periodic wrap observable on `(L-1, 0)` should match dense expectation."""
    length = 5
    j_xy = 1.1
    mps = MPS(length, state="random", pad=16)
    mps.normalize("B")
    psi = np.asarray(mps.to_vec(), dtype=np.complex128)
    j_mat = _spin_current_bond_matrix(j_xy)
    j_dense = dense_embed_periodic_wrap_two_site(length, j_mat)
    obs = Observable(BaseGate(j_mat), sites=[length - 1, 0])
    ex_dense = float(np.real(np.vdot(psi, j_dense @ psi)))
    ex_mps = float(np.real(mixed_expectation(mps, mps, obs)))
    assert ex_mps == pytest.approx(ex_dense, rel=0, abs=1e-6)


def test_two_time_correlator_probe_row_diagonal_matches_expectation_at_t0() -> None:
    """At ``t=0``, entry ``(r,r)`` of the two-time block matches ``⟨ψ|j_r j_r|ψ⟩`` (bulk bond)."""
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

    _, _, mat = unitary_ensemble_member_worker((0, mps, sim_params, h))
    assert mat is not None
    val_worker = float(np.real(mat[s_index, 0]))

    psi = np.asarray(mps.to_vec(), dtype=np.complex128)
    j_bond = _spin_current_bond_matrix(j_xy)
    jr = dense_embed_adjacent_two_site(length, 1, j_bond)
    expected = float(np.real(np.vdot(psi, jr @ jr @ psi)))
    assert val_worker == pytest.approx(expected, rel=0, abs=1e-6)


def test_apply_observable_inplace_non_adjacent_two_site_raises() -> None:
    """Two-site 4x4 observables must be nearest neighbors (or the supported periodic wrap)."""
    length = 4
    mps = MPS(length, state="random", pad=4)
    mps.normalize("B")
    gate4 = np.eye(4, dtype=np.complex128)
    obs = Observable(BaseGate(gate4), sites=[0, 2])
    with pytest.raises(ValueError, match="Only nearest-neighbor two-site observables are currently implemented"):
        apply_observable_inplace(mps, obs)


def test_apply_observable_inplace_unsupported_gate_dimension_raises() -> None:
    """Only one-site (2x2) and two-site (4x4) gates are supported in the autocorrelator helper."""
    length = 3
    mps = MPS(length, state="random", pad=4)
    mps.normalize("B")
    obs = Observable(BaseGate(np.eye(8, dtype=np.complex128)), sites=[0, 1, 2])
    with pytest.raises(ValueError, match="Autocorrelator observable must be one-site or nearest-neighbor two-site"):
        apply_observable_inplace(mps, obs)
