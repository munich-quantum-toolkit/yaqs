# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the sequences backend package re-exports."""

from __future__ import annotations

import mqt.yaqs.characterization.memory.backends.sequences as sequences_pkg
from mqt.yaqs.characterization.memory.backends.sequences import simulate_sequences


def test_sequences_package_reexports_simulate_sequences() -> None:
    """The public sequences package exposes simulate_sequences."""
    assert "simulate_sequences" in sequences_pkg.__all__
    assert callable(simulate_sequences)
