# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for optional PyTorch import helpers."""

from __future__ import annotations

from torch_support import import_torch, torch_importable


def test_torch_importable_matches_import_torch() -> None:
    """When torch imports successfully, helpers report it as available."""
    if not torch_importable():
        return
    module = import_torch()
    assert module.__name__ == "torch"
