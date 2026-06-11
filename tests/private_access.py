# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Helpers for exercising library-private APIs in tests."""

from __future__ import annotations


def call_private(obj: object, name: str, /, *args: object, **kwargs: object) -> object:
    """Call ``name`` on ``obj`` without direct private-attribute syntax.

    Args:
        obj: Object exposing the callable attribute.
        name: Attribute name to resolve with :func:`getattr`.
        *args: Positional arguments forwarded to the callable.
        **kwargs: Keyword arguments forwarded to the callable.

    Returns:
        Whatever the invoked callable returns.

    """
    return getattr(obj, name)(*args, **kwargs)
