# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the NoiseModel class.

This module provides unit tests for the NoiseModel class.
It verifies that a NoiseModel is created correctly when valid processes and strengths are provided,
raises an AssertionError when the lengths of the processes and strengths lists differ,
and handles empty noise models appropriately.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.libraries.noise_library import PauliX, PauliY, PauliZ


def _allclose(a: np.ndarray, b: np.ndarray) -> bool:
    return np.allclose(a, b, atol=1e-12)


def test_noise_model_creation() -> None:
    """Test that NoiseModel is created correctly with valid process dicts.

    This test constructs a NoiseModel with two single-site processes
    ("lowering" and "pauli_z") and corresponding strengths.
    It verifies that:
      - Each process is stored as a dictionary with correct fields.
      - The number of processes is correct.
      - Each process contains a jump_operator with the expected shape (2x2).
    """
    processes: list[dict[str, Any]] = [
        {"name": "lowering", "sites": [0], "strength": 0.1},
        {"name": "pauli_z", "sites": [1], "strength": 0.05},
    ]

    model = NoiseModel(processes)

    assert len(model.processes) == 2
    assert model.processes[0]["name"] == "lowering"
    assert model.processes[1]["name"] == "pauli_z"
    assert model.processes[0]["strength"] == 0.1
    assert model.processes[1]["strength"] == 0.05
    assert model.processes[0]["sites"] == [0]
    assert model.processes[1]["sites"] == [1]
    assert model.processes[0]["matrix"].shape == (2, 2)
    assert model.processes[1]["matrix"].shape == (2, 2)


def test_noise_model_assertion() -> None:
    """Test that NoiseModel raises an AssertionError when a process dict is missing required fields.

    This test constructs a process list where one entry is missing the 'strength' field,
    which should cause the NoiseModel initialization to fail.
    """
    # Missing 'strength' in the second dict
    processes: list[dict[str, Any]] = [
        {"name": "lowering", "sites": [0], "strength": 0.1},
        {"name": "pauli_z", "sites": [1]},  # Missing strength
    ]

    with pytest.raises(AssertionError):
        _ = NoiseModel(processes)


def test_noise_model_empty() -> None:
    """Test that NoiseModel handles an empty list of processes without error.

    This test initializes a NoiseModel with an empty list of process dictionaries and verifies that the resulting
    model has empty `processes` and `jump_operators` lists.
    """
    model = NoiseModel()

    assert model.processes == []


def test_noise_model_none() -> None:
    """Test that NoiseModel handles a None input without error.

    This test initializes a NoiseModel with `None` and verifies that the resulting
    model has no processes.
    """
    model = NoiseModel(None)

    assert model.processes == []


def test_one_site_matrix_auto() -> None:
    """Test that one-site processes auto-fill a 2x2 'matrix'.

    This verifies that providing name/sites/strength for a single-site process
    produces a process with a 2x2 operator populated from the library.
    """
    nm = NoiseModel([{"name": "pauli_x", "sites": [1], "strength": 0.1}])
    assert len(nm.processes) == 1
    p = nm.processes[0]
    assert "matrix" in p, "1-site process should have matrix auto-filled"
    assert p["matrix"].shape == (2, 2)
    assert _allclose(p["matrix"], PauliX.matrix)


def test_adjacent_two_site_matrix_auto() -> None:
    """Test that adjacent two-site processes auto-fill a 4x4 'matrix'.

    This checks that nearest-neighbor crosstalk uses the library matrix (kron)
    and requires no explicit operator in the process dict.
    """
    nm = NoiseModel([{"name": "crosstalk_xz", "sites": [1, 2], "strength": 0.2}])
    p = nm.processes[0]
    assert "matrix" in p, "Adjacent 2-site process should have matrix auto-filled"
    assert p["matrix"].shape == (4, 4)
    expected = np.kron(PauliX.matrix, PauliZ.matrix)
    assert _allclose(p["matrix"], expected)


def test_longrange_two_site_factors_auto() -> None:
    """Test that long-range two-site processes auto-fill 'factors' only.

    Using the canonical 'longrange_crosstalk_{ab}' name, the model should attach
    per-site 2x2 factors (A,B) and omit any large Kronecker 'matrix'.
    """
    nm = NoiseModel([{"name": "longrange_crosstalk_xy", "sites": [0, 2], "strength": 0.3}])
    p = nm.processes[0]
    assert "factors" in p, "Long-range 2-site process should have factors auto-filled"
    a_op, b_op = p["factors"]
    assert a_op.shape == (2, 2)
    assert b_op.shape == (2, 2)
    assert _allclose(a_op, PauliX.matrix)
    assert _allclose(b_op, PauliY.matrix)
    assert "matrix" not in p, "Long-range processes should not attach a full matrix"


def test_longrange_two_site_factors_explicit() -> None:
    """Test that explicit 'factors' for long-range are accepted and sites normalize.

    Supplying (A,B) and unsorted endpoints should result in stored ascending sites,
    preserving factors and omitting a full 'matrix'.
    """
    nm = NoiseModel([
        {
            "name": "custom_longrange_xy",
            "sites": [3, 1],  # intentionally unsorted
            "strength": 0.3,
            "factors": (PauliX.matrix, PauliY.matrix),
        }
    ])
    p = nm.processes[0]
    # Sites must be normalized to ascending order
    assert p["sites"] == [1, 3]
    assert "factors" in p
    assert len(p["factors"]) == 2
    a_op, b_op = p["factors"]
    assert _allclose(a_op, PauliX.matrix)
    assert _allclose(b_op, PauliY.matrix)
    assert "matrix" not in p


def test_longrange_unknown_label_without_factors_raises() -> None:
    """Test that unknown long-range labels without 'factors' raise.

    If the name is not 'longrange_crosstalk_{ab}' and no factors are provided,
    initialization must fail to avoid guessing operators.

    Raises:
        AssertionError: If the model accepts an unknown long-range label without factors.
    """
    try:
        # Name is not a recognized non-adjacent 'crosstalk_{ab}' and no factors provided
        _ = NoiseModel([{"name": "foo_bar", "sites": [0, 2], "strength": 0.1}])
    except AssertionError:
        return
    msg = "Expected AssertionError for unknown long-range label without factors."
    raise AssertionError(msg)
