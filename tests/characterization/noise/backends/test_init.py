# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for noise backend package exports."""

from mqt.yaqs.characterization.noise import backends


def test_noise_backends_have_no_public_optimizer_shortcut() -> None:
    """Noise fitting is provided through NoiseCharacterizer."""
    assert backends.__all__ == []
    assert not hasattr(backends, "cma_opt")
