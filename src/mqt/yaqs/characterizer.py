# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# SPDX-License-Identifier: MIT

"""Characterization entry point for YAQS."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mqt.yaqs.characterization.process_tensors.tomography import DenseComb, MPOComb, construct_process_tensor

if TYPE_CHECKING:
    from typing import TypeAlias

    from mqt.yaqs.characterization.process_tensors.surrogates.model import TransformerComb
    from mqt.yaqs.characterization.process_tensors.surrogates.workflow import (
        create_surrogate,
        generate_data,
    )

    Comb: TypeAlias = DenseComb | MPOComb | TransformerComb

_LAZY_EXPORTS = {
    "TransformerComb": ("mqt.yaqs.characterization.process_tensors.surrogates.model", "TransformerComb"),
    "create_surrogate": ("mqt.yaqs.characterization.process_tensors.surrogates.workflow", "create_surrogate"),
    "generate_data": ("mqt.yaqs.characterization.process_tensors.surrogates.workflow", "generate_data"),
}

__all__ = [
    "TransformerComb",
    "construct_process_tensor",
    "create_surrogate",
    "generate_data",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_path, attr = _LAZY_EXPORTS[name]
        import importlib

        return getattr(importlib.import_module(module_path), attr)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
