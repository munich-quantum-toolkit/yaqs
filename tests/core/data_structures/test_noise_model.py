# Copyright (c) 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

from __future__ import annotations

import pytest

from mqt.yaqs.core.data_structures.noise_model import NoiseModel


def test_noise_model_creation() -> None:
    """Test that NoiseModel is created correctly with valid processes and strengths.

    This test constructs a NoiseModel with two processes ("relaxation" and "dephasing") and corresponding strengths.
    It verifies that:
      - The processes and strengths are stored correctly.
      - The number of jump operators equals the number of processes.
      - The first jump operator has the expected shape (2x2), which is typical for the 'dephasing' process.
    """
    processes = ["relaxation", "dephasing"]
    strengths = [0.1, 0.05]

    model = NoiseModel(processes, strengths)

    assert model.processes == processes
    assert model.strengths == strengths
    assert len(model.jump_operators) == len(processes)
    assert model.jump_operators[0].shape == (2, 2), "First jump operator should be 2x2 for 'dephasing'."


def test_noise_model_assertion() -> None:
    """Test that NoiseModel raises an AssertionError when provided with mismatched process and strength lists.

    This test creates a scenario where the list of processes and the list of strengths have different lengths.
    An AssertionError is expected in this case.
    """
    processes = ["relaxation", "dephasing"]
    strengths = [0.1]
    with pytest.raises(AssertionError):
        _ = NoiseModel(processes, strengths)


def test_noise_model_empty() -> None:
    """Test that NoiseModel handles an empty process list without error.

    This test initializes a NoiseModel with empty lists for processes and strengths, and verifies that the resulting
    model has empty processes, strengths, and jump_operators lists.
    """
    model = NoiseModel([], [])
    assert model.processes == []
    assert model.strengths == []
    assert model.jump_operators == []
