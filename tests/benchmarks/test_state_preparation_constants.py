# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for shared state-preparation benchmark identifiers."""

from __future__ import annotations

from benchmarks.generate_state_preparation_targets import QUBIT_COUNTS as GENERATOR_QUBIT_COUNTS
from benchmarks.generate_state_preparation_targets import TARGET_FIXTURE_FORMAT as GENERATOR_TARGET_FIXTURE_FORMAT
from benchmarks.generate_state_preparation_targets import TARGET_IDS as GENERATOR_TARGET_IDS
from benchmarks.state_preparation import (
    BALLARIN_NOISE_ID,
    DEPHASING_NOISE_IDS,
    DEPOLARIZING_NOISE_IDS,
    NOISE_IDS,
    NOISELESS_NOISE_ID,
    STANDARD_NOISE_IDS,
    SUPPORTED_QUBIT_COUNTS,
    TARGET_FIXTURE_FORMAT,
    TARGET_GENERATION_SEEDS,
    TARGET_IDS,
)


def test_target_generator_uses_shared_target_constants() -> None:
    """The target generator and benchmark package expose the same identifiers."""
    assert GENERATOR_QUBIT_COUNTS is SUPPORTED_QUBIT_COUNTS
    assert GENERATOR_TARGET_IDS is TARGET_IDS
    assert GENERATOR_TARGET_FIXTURE_FORMAT == TARGET_FIXTURE_FORMAT
    assert tuple(TARGET_GENERATION_SEEDS) == TARGET_IDS


def test_noise_identifier_groups_are_complete_and_disjoint() -> None:
    """The combined noise tuple contains every family exactly once in stable order."""
    assert STANDARD_NOISE_IDS == DEPHASING_NOISE_IDS + DEPOLARIZING_NOISE_IDS
    assert (NOISELESS_NOISE_ID, BALLARIN_NOISE_ID, *STANDARD_NOISE_IDS) == NOISE_IDS
    assert len(NOISE_IDS) == 12
    assert len(set(NOISE_IDS)) == len(NOISE_IDS)
