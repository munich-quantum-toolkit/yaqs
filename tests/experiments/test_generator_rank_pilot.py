# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT
"""Smoke tests for the generator-rank pilot (D_H << D_U screening experiment)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PILOT = Path(__file__).resolve().parents[2] / "experiments" / "generator_rank_pilot"
sys.path.insert(0, str(PILOT))

import pilot_lib as pl  # noqa: E402
from run_pilot import validate  # noqa: E402


def test_snake_ordering() -> None:
    """Snake ordering reverses odd rows."""
    assert [pl.snake_index(0, c, 3) for c in range(3)] == [0, 1, 2]
    assert [pl.snake_index(1, c, 3) for c in range(3)] == [5, 4, 3]
    assert len(pl.grid_edges(4)) == 2 * 4 * 3


def test_oat_gate_angle_convention() -> None:
    """Rxx pair angle is 2*kappa/(N-1) and both orderings cover all pairs once."""
    n, kappa = 6, 0.8
    for ordering in ("lexicographic", "by_distance"):
        gates = pl.oat_gate_list(n, kappa, ordering)
        assert len(gates) == n * (n - 1) // 2
        assert all(np.isclose(theta, 2 * kappa / (n - 1)) for _, theta, _, _ in gates)
        assert len({(min(a, b), max(a, b)) for _, _, a, b in gates}) == len(gates)


def test_generator_mpo_ranks_compact() -> None:
    """Generator MPOs realize the intended compact D_H."""
    assert pl.mpo_max_bond(pl.oat_generator_mpo(8, 0.5)) == 3
    assert pl.mpo_max_bond(pl.qaoa_generator_mpo(3, 0.15)) <= 5


def test_tiny_instance_validations() -> None:
    """Full tiny-instance validation suite (expm vs gates, MPO routes, zero angle, TDVP)."""
    report = validate()
    assert report["to_vec_little_endian"] is True
    assert report["qaoa_expm_vs_gates_maxabs"] < 1e-10
    assert report["oat_expm_vs_gates_maxabs"] < 1e-10
