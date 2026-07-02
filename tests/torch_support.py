# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helpers for optional PyTorch-dependent tests."""

from __future__ import annotations

import importlib
import importlib.util
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


def torch_importable() -> bool:
    """Return whether ``torch`` can be imported in the current environment.

    ``importlib.util.find_spec`` alone is insufficient: legacy CUDA wheels can be
    present on disk yet fail at import time when CUDA libraries are unavailable.
    """
    if importlib.util.find_spec("torch") is None:
        return False
    try:
        importlib.import_module("torch")
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError):
        return False
    else:
        return True


def import_torch() -> ModuleType:
    """Import ``torch`` or skip the current test when unavailable.

    Returns:
        The imported :mod:`torch` module.
    """
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    try:
        return importlib.import_module("torch")
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"torch not available: {exc}")


requires_torch = pytest.mark.skipif(
    not torch_importable(),
    reason="torch not available",
)
