# Copyright (c) 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Focused unit tests for structural projector helpers."""

from __future__ import annotations

import numpy as np
import pytest

from .config import CHI, GENERATOR_SEED, REL_TOL, N
from .dense_projectors import (
    FixtureError,
    apply_two_site_op,
    build_p1_full,
    build_p2_full,
    compute_schmidt,
    localized_p1_action,
    localized_p2_action,
    make_generic_generator,
    projector_diagnostics,
    random_exact_rank_state,
)


def test_generic_generator_properties() -> None:
    h = make_generic_generator(GENERATOR_SEED)
    assert h.shape == (4, 4)
    assert np.allclose(h, h.conj().T)
    assert abs(np.trace(h)) < 1e-12
    assert abs(np.linalg.norm(h, 2) - 1.0) < 1e-12


def test_schmidt_fixture_conditioned() -> None:
    psi = random_exact_rank_state(101, n=N, chi=CHI)
    sch = compute_schmidt(psi, n=N, chi=CHI)
    assert sch.profile[0] == 1
    assert sch.profile[-1] == 1
    assert max(sch.profile) == CHI


def test_projectors_hermitian_idempotent() -> None:
    psi = random_exact_rank_state(102, n=N, chi=CHI)
    sch = compute_schmidt(psi, n=N, chi=CHI)
    for builder in (build_p1_full, build_p2_full):
        diag = projector_diagnostics(builder(sch))
        assert diag["hermitian_rel"] < REL_TOL
        assert diag["idempotent_rel"] < REL_TOL


def test_fixed_rank_locality_interior() -> None:
    psi = random_exact_rank_state(101, n=N, chi=CHI)
    sch = compute_schmidt(psi, n=N, chi=CHI)
    h = make_generic_generator()
    q0, q1 = 2, 5
    x = apply_two_site_op(psi, h, q0, q1, n=N)
    full = build_p1_full(sch) @ x
    windowed = localized_p1_action(x, q0, q1, sch)
    rel = float(np.linalg.norm(full - windowed) / np.linalg.norm(full))
    assert rel < REL_TOL


def test_two_site_locality_separated() -> None:
    psi = random_exact_rank_state(101, n=N, chi=CHI)
    sch = compute_schmidt(psi, n=N, chi=CHI)
    h = make_generic_generator()
    q0, q1 = 2, 5
    x = apply_two_site_op(psi, h, q0, q1, n=N)
    full = build_p2_full(sch) @ x
    windowed = localized_p2_action(x, q0, q1, sch)
    rel = float(np.linalg.norm(full - windowed) / np.linalg.norm(full))
    assert rel < REL_TOL


def test_bad_seed_raises_rather_than_resampling() -> None:
    psi = random_exact_rank_state(101, n=N, chi=CHI)
    with pytest.raises(FixtureError):
        compute_schmidt(np.zeros_like(psi), n=N, chi=CHI)
