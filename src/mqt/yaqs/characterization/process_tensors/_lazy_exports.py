# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Lazy attribute resolution for optional surrogate exports."""

from __future__ import annotations

import importlib

_LAZY_EXPORTS = {
    "TransformerComb": ("mqt.yaqs.characterization.process_tensors.surrogates.model", "TransformerComb"),
    "create_surrogate": ("mqt.yaqs.characterization.process_tensors.surrogates.workflow", "create_surrogate"),
    "generate_data": ("mqt.yaqs.characterization.process_tensors.surrogates.workflow", "generate_data"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_path), attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
