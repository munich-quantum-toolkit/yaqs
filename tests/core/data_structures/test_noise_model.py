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

import pytest

from mqt.yaqs.core.data_structures.noise_model import NoiseModel


def test_noise_model_creation() -> None:
    """Test that NoiseModel is created correctly with valid process dicts.

    This test constructs a NoiseModel with two single-site processes
    ("relaxation" and "dephasing") and corresponding strengths.
    It verifies that:
      - Each process is stored as a dictionary with correct fields.
      - The number of processes is correct.
      - Each process contains a jump_operator with the expected shape (2x2).
    """
    processes: list[dict[str, Any]] = [
        {"name": "relaxation", "sites": [0], "strength": 0.1},
        {"name": "dephasing", "sites": [1], "strength": 0.05},
    ]

    model = NoiseModel(processes)

    assert len(model.processes) == 2
    assert model.processes[0]["name"] == "relaxation"
    assert model.processes[1]["name"] == "dephasing"
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
        {"name": "relaxation", "sites": [0], "strength": 0.1},
        {"name": "dephasing", "sites": [1]},  # Missing strength
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
