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

from mqt.yaqs.core.libraries.observable_library import ObservableLibrary


@pytest.mark.parametrize(
    ("name", "interaction"),
    [
        ("x", 1),
        ("y", 1),
        ("z", 1),
        ("id", 1),
        ("p0", 1),
        ("p1", 1),
        ("xx", 2),
        ("yy", 2),
        ("zz", 2),
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


@pytest.mark.parametrize("name", ["i", "iden", "h", "cx", "cz", "swap", "rx", "unknown"])
def test_unsupported_names_are_unknown(name: str) -> None:
    """The observable library rejects each name that it does not define."""
    with pytest.raises(ValueError, match=f"Unknown observable {name!r}"):
        ObservableLibrary.resolve(name)
