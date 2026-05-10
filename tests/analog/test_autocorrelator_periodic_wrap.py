# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for periodic-wrap two-site application in the analog autocorrelator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mqt.yaqs.analog.autocorrelator import mixed_expectation
from mqt.yaqs.analog.unitary_ensemble import unitary_ensemble_member_worker
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import BaseGate

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH_SCRIPT = _REPO_ROOT / "scripts" / "bench_xxz_spin_current_autocorrelator_yaqs_vs_ed.py"


def _load_bench_module():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("yaqs_bench_xxz_spin_current", _BENCH_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrap_expectation_matches_dense() -> None:
    """Periodic wrap observable on `(L-1, 0)` should match dense expectation."""
    mod = _load_bench_module()
    length = 5
    j_xy = 1.1
    mps = MPS(length, state="random", pad=16)
    mps.normalize("B")
    psi = np.asarray(mps.to_vec(), dtype=np.complex128)
    j_mat = mod.spin_current_bond_matrix(j_xy)
    j_dense = mod.embed_two_site_operator_periodic(length, length - 1, 0, j_mat)
    obs = Observable(BaseGate(j_mat), sites=[length - 1, 0])
    ex_dense = float(np.real(np.vdot(psi, j_dense @ psi)))
    ex_mps = float(np.real(mixed_expectation(mps, mps, obs)))
    assert ex_mps == pytest.approx(ex_dense, rel=0, abs=1e-6)


def test_two_time_correlator_probe_row_diagonal_matches_expectation_at_t0() -> None:
    """At ``t=0``, entry ``(r,r)`` of the two-time block matches ``⟨ψ|j_r j_r|ψ⟩`` (bulk bond)."""
    mod = _load_bench_module()
    length = 4
    j_xy = 1.0
    delta = 0.8
    mps = MPS(length, state="random", pad=8)
    mps.normalize("B")

    row = tuple(
        mod.spin_current_observable_for_periodic_bond(length, a, b, j_xy)
        for a, b in mod.periodic_bond_endpoints(length)
    )
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
    j_bond = mod.spin_current_bond_matrix(j_xy)
    jr = mod.embed_two_site_operator(length, 1, j_bond)
    expected = float(np.real(np.vdot(psi, jr @ jr @ psi)))
    assert val_worker == pytest.approx(expected, rel=0, abs=1e-6)
