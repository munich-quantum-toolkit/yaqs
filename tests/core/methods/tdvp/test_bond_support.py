# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for :mod:`mqt.yaqs.core.methods.tdvp.bond_support`."""

from __future__ import annotations

import copy

import pytest

from mqt.yaqs.core.data_structures.simulation_parameters import StrongSimParams
from mqt.yaqs.core.data_structures.state import State
from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.core.methods.tdvp import tdvp
from mqt.yaqs.core.methods.tdvp.bond_support import (
    crossed_bonds_for_two_site_gate,
    prepare_lr_tdvp_seed_bonds,
    prepare_support_bonds,
    protected_bonds_for_two_site_gate,
    select_protected_seed_bonds,
    select_seed_bonds,
)
from mqt.yaqs.digital.digital_tjm import apply_window, construct_generator_mpo
from tests.core.methods.tdvp.conftest import _tdvp_params, assert_mps_bond_invariants

pytestmark = pytest.mark.tdvp_regression


# --- protected_bonds_for_two_site_gate ---


@pytest.mark.parametrize(
    ("site0", "site1", "window_left", "window_right", "expected"),
    [
        (0, 7, 0, 7, frozenset({0, 1, 2, 3, 4, 5, 6})),
        (0, 7, 0, 9, frozenset({0, 1, 2, 3, 4, 5, 6})),
        (2, 7, 0, 9, frozenset({2, 3, 4, 5, 6})),
        (2, 7, 1, 8, frozenset({1, 2, 3, 4, 5})),
        (1, 8, 0, 9, frozenset({1, 2, 3, 4, 5, 6, 7})),
        (3, 4, 0, 5, frozenset({3})),
    ],
)
def test_protected_bonds_geometry(
    site0: int,
    site1: int,
    window_left: int,
    window_right: int,
    expected: frozenset[int],
) -> None:
    """Protected bonds follow gate-crossed bonds in the TDVP window."""
    assert protected_bonds_for_two_site_gate(site0, site1, window_left, window_right) == expected


def test_internal_pair_no_protection_outside_crossed_interval() -> None:
    """Bonds outside the gate-crossed interval are not in the protected set."""
    sites = (2, 7)
    window = (1, 8)
    protected = protected_bonds_for_two_site_gate(sites[0], sites[1], window[0], window[1])
    assert 0 not in protected
    assert 6 not in protected
    assert protected == frozenset({1, 2, 3, 4, 5})


# --- prepare_support_bonds ---


def test_crossed_bonds_alias_matches_protected() -> None:
    """Legacy protected_bonds name matches crossed_bonds geometry."""
    assert crossed_bonds_for_two_site_gate(2, 7, 0, 9) == protected_bonds_for_two_site_gate(2, 7, 0, 9)


def test_prepare_support_bonds_returns_crossed_for_lab() -> None:
    """Lab helper returns crossed bonds after seed prep; production uses support_bonds=None."""
    length = 10
    sites = (2, 7)
    prep = copy.deepcopy(State(length, initial="x+").mps)
    gate = GateLibrary.rzz([0.3])
    gate.set_sites(*sites)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    short_state, _short_mpo, win = apply_window(prep, mpo, first_site, last_site, 1)
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=8,
        tdvp_sweeps=4,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    expected = crossed_bonds_for_two_site_gate(sites[0], sites[1], win[0], win[1])
    support = prepare_support_bonds(short_state, sites[0], sites[1], (win[0], win[1]), params)
    assert support is not None
    assert support == expected


def test_prepare_support_bonds_none_when_chi_one() -> None:
    """max_bond_dim=1 disables support retention without padding to χ=2."""
    length = 8
    sites = (0, length - 1)
    prep = copy.deepcopy(State(length, initial="x+").mps)
    gate = GateLibrary.rzz([0.3])
    gate.set_sites(*sites)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    short_state, _short_mpo, win = apply_window(prep, mpo, first_site, last_site, 1)
    params = StrongSimParams(
        preset="exact",
        get_state=True,
        max_bond_dim=1,
        tdvp_sweeps=4,
        svd_threshold=1e-14,
        krylov_tol=1e-12,
    )
    assert prepare_support_bonds(short_state, sites[0], sites[1], (win[0], win[1]), params) is None


def test_bond_invariants_after_seed_prep_and_dynamic_sweep() -> None:
    """Seed prep + dynamic TDVP without support_bonds preserve neighbor bond shapes."""
    length = 8
    theta = 0.3
    gate = GateLibrary.rzz([theta])
    gate.set_sites(0, length - 1)
    prep = copy.deepcopy(State(length, initial="x+").mps)
    mpo, first_site, last_site = construct_generator_mpo(gate, length)
    window_state, window_mpo, window = apply_window(prep, mpo, first_site, last_site, 1)
    params = _tdvp_params(max_bond_dim=8, tdvp_sweeps=4)
    prepare_lr_tdvp_seed_bonds(window_state, 0, length - 1, (window[0], window[1]), params)
    assert_mps_bond_invariants(window_state, max_bond_dim=8)
    tdvp(window_state, window_mpo, params, support_bonds=None)
    assert_mps_bond_invariants(window_state, max_bond_dim=8)


# --- select_protected_seed_bonds ---


def test_select_seed_bonds_alias_matches_protected_seed() -> None:
    """select_seed_bonds is an alias for select_protected_seed_bonds."""
    crossed = frozenset({2, 3, 4, 5, 6})
    assert select_seed_bonds(crossed) == select_protected_seed_bonds(crossed)


@pytest.mark.parametrize(
    ("protected", "expected"),
    [
        (frozenset(), frozenset()),
        (frozenset({0}), frozenset({0})),
        (frozenset({0, 1}), frozenset({0})),
        (frozenset({0, 1, 2}), frozenset({0})),
        (frozenset({0, 1, 2, 3}), frozenset({0, 1})),
        (frozenset({2, 3, 4, 5, 6}), frozenset({2, 3})),
        (frozenset({0, 1, 2, 3, 4, 5, 6}), frozenset({0, 1, 2})),
        (frozenset({1, 2, 3, 4, 5}), frozenset({1, 2})),
        (frozenset({3}), frozenset({3})),
    ],
)
def test_select_protected_seed_bonds_deterministic(protected: frozenset[int], expected: frozenset[int]) -> None:
    """Left-half seeding is deterministic and always a subset of protected bonds."""
    selected = select_protected_seed_bonds(protected)
    assert selected == expected
    assert selected <= protected
