# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for named observable definitions."""

from __future__ import annotations

import numpy as np
import pytest

from mqt.yaqs.core.libraries.gate_library import GateLibrary
from mqt.yaqs.core.libraries.observable_library import ObservableLibrary


@pytest.mark.parametrize(
    ("name", "interaction"),
    [
        ("x", 1),
        ("y", 1),
        ("z", 1),
        ("h", 1),
        ("id", 1),
        ("p0", 1),
        ("p1", 1),
        ("xx", 2),
        ("yy", 2),
        ("zz", 2),
        ("cx", 2),
        ("cz", 2),
        ("swap", 2),
    ],
)
def test_named_operators_are_hermitian(name: str, interaction: int) -> None:
    """Each built-in operator has the declared site count and is Hermitian."""
    definition = ObservableLibrary.resolve(name)

    assert definition.name == name
    assert definition.interaction == interaction
    assert definition.kind == "operator"
    assert definition.matrix is not None
    np.testing.assert_allclose(definition.matrix, definition.matrix.conj().T)


@pytest.mark.parametrize("alias", ["i", "iden"])
def test_identity_aliases_use_canonical_name(alias: str) -> None:
    """Identity aliases resolve to the canonical ``id`` definition."""
    assert ObservableLibrary.resolve(alias).name == "id"


def test_definitions_do_not_share_mutable_matrix_data() -> None:
    """Changing one returned matrix does not change later definitions."""
    first = ObservableLibrary.x()
    assert first.matrix is not None
    first.matrix[0, 0] = 7

    second = ObservableLibrary.x()

    assert second.matrix is not None
    assert second.matrix[0, 0] == 0


def test_diagnostics_have_no_placeholder_operator() -> None:
    """State diagnostics are distinct from linear operators."""
    definition = ObservableLibrary.schmidt_spectrum()

    assert definition.kind == "diagnostic"
    assert definition.matrix is None
    assert definition.interaction == 0


@pytest.mark.parametrize(
    "name",
    ["xx", "yy", "zz", "p0", "p1", "pvm", "local", "position", "entropy", "schmidt_spectrum"],
)
def test_observable_definitions_are_not_in_gate_library(name: str) -> None:
    """GateLibrary does not expose observable-only factories."""
    assert not hasattr(GateLibrary, name)
