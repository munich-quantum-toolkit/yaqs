# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Shared fixtures for noise characterization tests."""

from __future__ import annotations

import pytest

from .fixtures import NoiseTestConfig, build_propagator

__all__ = ["NoiseTestConfig", "build_propagator", "noise_test_config"]


@pytest.fixture
def noise_test_config() -> NoiseTestConfig:
    """Default open-system geometry for noise characterization tests.

    Returns:
        Shared :class:`NoiseTestConfig` instance for parametrized smoke tests.
    """
    return NoiseTestConfig()
