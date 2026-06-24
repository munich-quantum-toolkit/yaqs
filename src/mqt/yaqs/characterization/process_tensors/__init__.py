# Copyright (c) 2025 - 2026 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Process-tensors characterization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._lazy_exports import __getattr__
from .diagnostics.probe import probe_process
from .tomography.constructor import construct_process_tensor

if TYPE_CHECKING:
    from .surrogates.model import TransformerComb
    from .surrogates.workflow import create_surrogate, generate_data

__all__ = [
    "TransformerComb",
    "__getattr__",
    "construct_process_tensor",
    "create_surrogate",
    "generate_data",
    "probe_process",
]
