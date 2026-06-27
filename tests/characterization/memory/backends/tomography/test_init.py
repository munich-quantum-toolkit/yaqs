# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Smoke tests for tomography package re-exports."""

from __future__ import annotations

from mqt.yaqs.characterization.memory.backends import tomography


def test_tomography_package_reexports() -> None:
    """Public symbols resolve through the package import path."""
    assert tomography.DenseComb is not None
    assert tomography.MPOComb is not None
    assert tomography.build_process_tensor is not None
    assert tomography.TomographyBasis is not None
