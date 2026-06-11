# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helpers for exercising library-private APIs in tests."""

from __future__ import annotations

from typing import Any


def call_private(obj: object, name: str, /, *args: Any, **kwargs: Any) -> Any:
    """Call ``name`` on ``obj`` without direct private-attribute syntax."""
    return getattr(obj, name)(*args, **kwargs)
