# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the sequences backend package exports."""

from __future__ import annotations

import mqt.yaqs.characterization.memory.backends.sequences as sequences_pkg


def test_sequences_package_has_no_public_execution_helpers() -> None:
    """Sequence execution is provided through MemoryCharacterizer."""
    assert sequences_pkg.__all__ == []
    assert not hasattr(sequences_pkg, "simulate_sequences")
