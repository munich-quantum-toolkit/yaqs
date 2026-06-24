# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Characterization entry point for YAQS."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from mqt.yaqs.characterization.memory.combs.tomography import DenseComb, MPOComb, construct_process_tensor

if TYPE_CHECKING:
    from typing import TypeAlias

    from mqt.yaqs.characterization.memory.combs.surrogates.model import TransformerComb
    from mqt.yaqs.characterization.memory.combs.surrogates.workflow import (
        create_surrogate,
        generate_data,
    )

    Comb: TypeAlias = DenseComb | MPOComb | TransformerComb

_LAZY_EXPORTS = {
    "TransformerComb": ("mqt.yaqs.characterization.memory.combs.surrogates.model", "TransformerComb"),
    "create_surrogate": ("mqt.yaqs.characterization.memory.combs.surrogates.workflow", "create_surrogate"),
    "generate_data": ("mqt.yaqs.characterization.memory.combs.surrogates.workflow", "generate_data"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        return getattr(importlib.import_module(module_path), attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


__all__ = [
    "DenseComb",
    "MPOComb",
    "TransformerComb",
    "construct_process_tensor",
    "create_surrogate",
    "generate_data",
]
